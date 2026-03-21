"""
Aphelion Lab — Backtest Engine
Bar-by-bar simulation with realistic fills and full metrics.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger("aphelion.engine")


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    id: int
    side: Side
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    size: float = 0.01
    sl: Optional[float] = None
    tp: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""
    commission: float = 0.0


@dataclass
class Position:
    trade: Trade
    unrealized_pnl: float = 0.0

    def update(self, price: float):
        if self.trade.side == Side.BUY:
            self.unrealized_pnl = (price - self.trade.entry_price) * self.trade.size * 100
        else:
            self.unrealized_pnl = (self.trade.entry_price - price) * self.trade.size * 100


@dataclass
class BacktestConfig:
    initial_capital: float = 5000.0
    spread_pips: float = 2.0
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0
    pip_value: float = 0.01  # for gold: 0.01 per pip
    lot_multiplier: float = 100  # 1 lot = 100 oz for gold


class StrategyContext:
    """Context passed to strategy on each bar."""

    def __init__(self, engine: "BacktestEngine"):
        self._engine = engine

    @property
    def bar(self) -> pd.Series:
        return self._engine._current_bar

    @property
    def bars(self) -> pd.DataFrame:
        """All bars up to current (inclusive)."""
        return self._engine._bars_so_far

    @property
    def position(self) -> Optional[Position]:
        return self._engine._position

    @property
    def equity(self) -> float:
        return self._engine._equity

    @property
    def bar_index(self) -> int:
        return self._engine._bar_idx

    def buy(self, size: float = 0.01, sl: float = None, tp: float = None):
        self._engine._open_trade(Side.BUY, size, sl, tp)

    def sell(self, size: float = 0.01, sl: float = None, tp: float = None):
        self._engine._open_trade(Side.SELL, size, sl, tp)

    def close(self, reason: str = "signal"):
        self._engine._close_trade(reason)

    @property
    def has_position(self) -> bool:
        return self._engine._position is not None

    # Indicator helpers
    def sma(self, period: int, col: str = "close") -> float:
        s = self.bars[col]
        if len(s) < period: return float("nan")
        return s.iloc[-period:].mean()

    def ema(self, period: int, col: str = "close") -> float:
        s = self.bars[col]
        if len(s) < period: return float("nan")
        return s.ewm(span=period, adjust=False).mean().iloc[-1]

    def rsi(self, period: int = 14, col: str = "close") -> float:
        s = self.bars[col]
        if len(s) < period + 1: return float("nan")
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        return float(100 - 100 / (1 + rs.iloc[-1]))

    def atr(self, period: int = 14) -> float:
        b = self.bars
        if len(b) < period + 1: return float("nan")
        h, l, c = b["high"], b["low"], b["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def bbands(self, period: int = 20, std: float = 2.0):
        s = self.bars["close"]
        if len(s) < period: return float("nan"), float("nan"), float("nan")
        mid = s.rolling(period).mean().iloc[-1]
        sd = s.rolling(period).std().iloc[-1]
        return float(mid + std * sd), float(mid), float(mid - std * sd)


class BacktestEngine:
    """Event-driven bar-by-bar backtest engine."""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self._trades: list[Trade] = []
        self._position: Optional[Position] = None
        self._equity = self.config.initial_capital
        self._peak_equity = self.config.initial_capital
        self._equity_curve: list[float] = []
        self._drawdown_curve: list[float] = []
        self._timestamps: list[pd.Timestamp] = []
        self._trade_counter = 0
        self._current_bar = None
        self._bars_so_far = None
        self._bar_idx = 0
        self._data = None

    def run(self, data: pd.DataFrame, strategy) -> "BacktestResult":
        """Run backtest on data with given strategy."""
        self._reset()
        self._data = data
        ctx = StrategyContext(self)

        if hasattr(strategy, "on_init"):
            strategy.on_init(ctx)

        for i in range(len(data)):
            self._bar_idx = i
            self._current_bar = data.iloc[i]
            self._bars_so_far = data.iloc[:i + 1]

            # Check SL/TP
            self._check_stops()

            # Call strategy
            try:
                strategy.on_bar(ctx)
            except Exception as e:
                logger.error(f"Strategy error at bar {i}: {e}")

            # Update equity
            if self._position:
                self._position.update(self._current_bar["close"])
                eq = self._equity + self._position.unrealized_pnl
            else:
                eq = self._equity

            self._equity_curve.append(eq)
            self._peak_equity = max(self._peak_equity, eq)
            dd = (eq - self._peak_equity) / self._peak_equity * 100 if self._peak_equity > 0 else 0
            self._drawdown_curve.append(dd)
            self._timestamps.append(self._current_bar.name)

        # Close any remaining position
        if self._position:
            self._close_trade("end_of_data")

        return BacktestResult(
            trades=self._trades,
            equity_curve=self._equity_curve,
            drawdown_curve=self._drawdown_curve,
            timestamps=self._timestamps,
            config=self.config,
            data=data,
        )

    def _reset(self):
        self._trades = []
        self._position = None
        self._equity = self.config.initial_capital
        self._peak_equity = self.config.initial_capital
        self._equity_curve = []
        self._drawdown_curve = []
        self._timestamps = []
        self._trade_counter = 0

    def _open_trade(self, side: Side, size: float, sl: float, tp: float):
        if self._position is not None:
            return  # Already in a position

        bar = self._current_bar
        spread = self.config.spread_pips * self.config.pip_value
        slip = self.config.slippage_pips * self.config.pip_value

        if side == Side.BUY:
            price = bar["close"] + spread / 2 + slip
        else:
            price = bar["close"] - spread / 2 - slip

        commission = self.config.commission_per_lot * size
        self._equity -= commission

        self._trade_counter += 1
        trade = Trade(
            id=self._trade_counter, side=side,
            entry_time=bar.name, entry_price=price,
            size=size, sl=sl, tp=tp, commission=commission,
        )
        self._position = Position(trade=trade)

    def _close_trade(self, reason: str = "signal"):
        if self._position is None:
            return

        bar = self._current_bar
        trade = self._position.trade
        spread = self.config.spread_pips * self.config.pip_value
        slip = self.config.slippage_pips * self.config.pip_value

        if trade.side == Side.BUY:
            price = bar["close"] - spread / 2 - slip
        else:
            price = bar["close"] + spread / 2 + slip

        if trade.side == Side.BUY:
            pnl = (price - trade.entry_price) * trade.size * self.config.lot_multiplier
        else:
            pnl = (trade.entry_price - price) * trade.size * self.config.lot_multiplier

        trade.exit_time = bar.name
        trade.exit_price = price
        trade.pnl = pnl
        trade.pnl_pct = pnl / self._equity * 100 if self._equity > 0 else 0
        trade.bars_held = self._bar_idx - self._data.index.get_loc(trade.entry_time) if trade.entry_time in self._data.index else 0
        trade.exit_reason = reason

        self._equity += pnl
        self._trades.append(trade)
        self._position = None

    def _check_stops(self):
        if self._position is None:
            return
        trade = self._position.trade
        bar = self._current_bar
        h, l = bar["high"], bar["low"]

        if trade.side == Side.BUY:
            if trade.sl and l <= trade.sl:
                self._close_trade("stop_loss")
            elif trade.tp and h >= trade.tp:
                self._close_trade("take_profit")
        else:
            if trade.sl and h >= trade.sl:
                self._close_trade("stop_loss")
            elif trade.tp and l <= trade.tp:
                self._close_trade("take_profit")


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[float]
    drawdown_curve: list[float]
    timestamps: list[pd.Timestamp]
    config: BacktestConfig
    data: pd.DataFrame

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losers(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def long_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.side == Side.BUY]

    @property
    def short_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.side == Side.SELL]

    @property
    def win_rate(self) -> float:
        if not self.trades: return 0
        return len(self.winners) / len(self.trades) * 100

    @property
    def net_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def gross_profit(self) -> float:
        return sum(t.pnl for t in self.winners)

    @property
    def gross_loss(self) -> float:
        return abs(sum(t.pnl for t in self.losers))

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0: return float("inf") if self.gross_profit > 0 else 0
        return self.gross_profit / self.gross_loss

    @property
    def avg_win(self) -> float:
        return np.mean([t.pnl for t in self.winners]) if self.winners else 0

    @property
    def avg_loss(self) -> float:
        return np.mean([t.pnl for t in self.losers]) if self.losers else 0

    @property
    def avg_rr(self) -> float:
        if self.avg_loss == 0: return 0
        return abs(self.avg_win / self.avg_loss)

    @property
    def max_drawdown(self) -> float:
        if not self.drawdown_curve: return 0
        return min(self.drawdown_curve)

    @property
    def max_drawdown_usd(self) -> float:
        if not self.equity_curve: return 0
        peak = self.config.initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = eq - peak
            max_dd = min(max_dd, dd)
        return max_dd

    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2: return 0
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if returns.std() == 0: return 0
        return float(returns.mean() / returns.std() * np.sqrt(252 * 24))  # annualized hourly

    @property
    def sortino_ratio(self) -> float:
        if len(self.equity_curve) < 2: return 0
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0: return 0
        return float(returns.mean() / downside.std() * np.sqrt(252 * 24))

    @property
    def expectancy(self) -> float:
        if not self.trades: return 0
        return self.net_pnl / len(self.trades)

    @property
    def total_commission(self) -> float:
        return sum(t.commission for t in self.trades)

    @property
    def avg_bars_held(self) -> float:
        if not self.trades: return 0
        return np.mean([t.bars_held for t in self.trades])

    @property
    def largest_win(self) -> float:
        return max((t.pnl for t in self.trades), default=0)

    @property
    def largest_loss(self) -> float:
        return min((t.pnl for t in self.trades), default=0)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.config.initial_capital

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity - self.config.initial_capital) / self.config.initial_capital * 100

    def to_metrics_dict(self) -> dict:
        return {
            "Net P&L": f"${self.net_pnl:.2f}",
            "Net P&L %": f"{self.total_return_pct:.2f}%",
            "Total Trades": str(self.total_trades),
            "Win Rate": f"{self.win_rate:.1f}%",
            "Profit Factor": f"{self.profit_factor:.3f}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "Max Drawdown": f"{self.max_drawdown:.2f}%",
            "Max DD ($)": f"${self.max_drawdown_usd:.2f}",
            "Avg Win": f"${self.avg_win:.2f}",
            "Avg Loss": f"${self.avg_loss:.2f}",
            "Avg Win/Loss": f"{self.avg_rr:.3f}",
            "Expectancy": f"${self.expectancy:.2f}",
            "Largest Win": f"${self.largest_win:.2f}",
            "Largest Loss": f"${self.largest_loss:.2f}",
            "Avg Bars Held": f"{self.avg_bars_held:.1f}",
            "Commission": f"${self.total_commission:.2f}",
            "Winners": f"{len(self.winners)}",
            "Losers": f"{len(self.losers)}",
            "Long Trades": f"{len(self.long_trades)}",
            "Short Trades": f"{len(self.short_trades)}",
            "Initial Capital": f"${self.config.initial_capital:.2f}",
            "Final Equity": f"${self.final_equity:.2f}",
        }

    def trades_df(self) -> pd.DataFrame:
        if not self.trades: return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "#": t.id,
                "Side": t.side.value,
                "Entry Time": str(t.entry_time)[:19],
                "Exit Time": str(t.exit_time)[:19] if t.exit_time else "",
                "Entry": f"{t.entry_price:.2f}",
                "Exit": f"{t.exit_price:.2f}" if t.exit_price else "",
                "Size": t.size,
                "P&L": f"{t.pnl:.2f}",
                "P&L %": f"{t.pnl_pct:.2f}%",
                "Bars": t.bars_held,
                "Reason": t.exit_reason,
            })
        return pd.DataFrame(rows)
