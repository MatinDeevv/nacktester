"""
Aphelion Lab — Backtest Engine
Bar-by-bar simulation with realistic fills and full metrics.
"""

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional
from enum import Enum

import pandas as pd
import numpy as np

from aphelion_lab.execution import (
    SlippageMode, calc_slippage,
    CommissionMode, calc_commission,
    dynamic_spread,
    intrabar_stop_priority,
    OrderType, PendingOrder, OrderManager,
    TrailingStopMode, TrailingStopState,
    check_breakeven,
    PartialTP, check_partial_tps,
    SizingMode, calc_position_size,
    CooldownTracker,
    session_allows_entry,
    RiskGuard, should_force_close,
)
from aphelion_lab.market_structure import enrich_dataframe
from aphelion_lab.metrics import PerformanceStats, compute_performance_stats

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

    # ── Execution-realism extensions (all default to legacy behaviour) ──
    slippage_mode: SlippageMode = SlippageMode.FIXED
    commission_mode: CommissionMode = CommissionMode.PER_LOT
    dynamic_spread_enabled: bool = False
    intrabar_stop_priority: bool = False  # D39: smarter SL/TP sequencing

    # D40: pending orders
    pending_orders_enabled: bool = False

    # D41: trailing stop (default=off)
    trailing_stop_mode: TrailingStopMode = TrailingStopMode.NONE
    trailing_stop_distance: float = 0.0

    # D42: break-even
    breakeven_trigger_pips: float = 0.0  # 0 = disabled
    breakeven_lock_pips: float = 0.0

    # D43: partial take-profit (strategy sets levels)
    partial_tp_enabled: bool = False

    # D44: position sizing
    sizing_mode: SizingMode = SizingMode.FIXED
    risk_pct: float = 1.0
    target_risk_atr: float = 1.5

    # D45: cooldown
    cooldown_bars: int = 0

    # D46: session filter
    allowed_sessions: list = field(default_factory=list)

    # D47: risk guards
    max_daily_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    max_trades_per_day: int = 0
    max_open_positions: int = 1

    # D48: max holding time
    max_holding_bars: int = 0

    # D49: multi-position
    multi_position: bool = False

    # Data preparation
    auto_enrich_data: bool = True
    regime_features_enabled: bool = True
    periods_per_year: Optional[float] = None
    risk_free_rate_annual: float = 0.0
    return_method: str = "simple"


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

    # ── Pending-order API (D40) ──────────────────────────────────────────
    def place_limit(self, side: str, price: float, size: float = 0.01,
                    sl: float = None, tp: float = None,
                    expiry_bars: int = None, tag: str = ""):
        """Place a limit order (buy below / sell above current price)."""
        s = Side.BUY if side.upper() == "BUY" else Side.SELL
        self._engine._order_mgr.add(PendingOrder(
            OrderType.LIMIT, s.value, price, size, sl, tp, expiry_bars, tag=tag))

    def place_stop_order(self, side: str, price: float, size: float = 0.01,
                         sl: float = None, tp: float = None,
                         expiry_bars: int = None, tag: str = ""):
        """Place a stop order (buy above / sell below current price)."""
        s = Side.BUY if side.upper() == "BUY" else Side.SELL
        self._engine._order_mgr.add(PendingOrder(
            OrderType.STOP, s.value, price, size, sl, tp, expiry_bars, tag=tag))

    def cancel_orders(self, tag: str = ""):
        if tag:
            self._engine._order_mgr.cancel_by_tag(tag)
        else:
            self._engine._order_mgr.cancel_all()

    @property
    def pending_orders(self) -> list:
        return self._engine._order_mgr.pending

    # ── D50: modify existing position ────────────────────────────────────
    def modify_sl(self, new_sl: float):
        if self._engine._position:
            self._engine._position.trade.sl = new_sl

    def modify_tp(self, new_tp: float):
        if self._engine._position:
            self._engine._position.trade.tp = new_tp

    # ── Sizing helper (D44) ──────────────────────────────────────────────
    def calc_size(self, sl_distance: float = 0.0) -> float:
        """Calculate position size using engine's sizing model."""
        cfg = self._engine.config
        return calc_position_size(
            mode=cfg.sizing_mode, fixed_size=0.01,
            equity=self._engine._equity, risk_pct=cfg.risk_pct,
            sl_distance=sl_distance, pip_value=cfg.pip_value,
            lot_multiplier=cfg.lot_multiplier,
            win_rate=self._engine._running_win_rate,
            avg_rr=self._engine._running_avg_rr,
            atr=self.atr(), target_risk_atr=cfg.target_risk_atr,
        )

    # ── Market-structure accessors ───────────────────────────────────────
    @property
    def session(self) -> str:
        return self.bar.get("session", "")

    @property
    def day_of_week(self) -> int:
        return int(self.bar.get("day_of_week", self.bar.name.weekday()))

    @property
    def spread(self) -> float:
        return float(self.bar.get("spread", self._engine.config.spread_pips))

    @property
    def volatility_regime(self) -> str:
        return str(self.bar.get("volatility_regime", "normal"))

    @property
    def market_regime(self) -> str:
        return str(self.bar.get("market_regime", "range"))

    @property
    def entropy(self) -> float:
        return float(self.bar.get("entropy_64", float("nan")))

    @property
    def hurst(self) -> float:
        return float(self.bar.get("hurst_128", float("nan")))

    @property
    def jump_intensity(self) -> float:
        return float(self.bar.get("jump_intensity", float("nan")))

    @property
    def distribution_shift(self) -> float:
        return float(self.bar.get("distribution_shift_norm", float("nan")))

    # ── Pre-computed indicator lookup ────────────────────────────────────
    def ind(self, col: str):
        """Look up a pre-computed indicator column at current bar."""
        return self._engine._current_bar.get(col, float("nan"))

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

        # Execution-realism state
        self._order_mgr = OrderManager()
        self._cooldown = CooldownTracker(self.config.cooldown_bars)
        self._risk_guard = RiskGuard(
            max_daily_loss_pct=self.config.max_daily_loss_pct,
            max_drawdown_pct=self.config.max_drawdown_pct,
            max_trades_per_day=self.config.max_trades_per_day,
            max_open_positions=self.config.max_open_positions,
        )
        self._trailing: Optional[TrailingStopState] = None
        self._breakeven_active = False
        self._partials: list[PartialTP] = []
        # Running stats for Kelly sizing
        self._running_win_rate = 0.5
        self._running_avg_rr = 1.5

    def run(self, data: pd.DataFrame, strategy, on_progress=None, progress_interval=50) -> "BacktestResult":
        """Run backtest on data with given strategy."""
        self._reset()
        data = self._prepare_data(data)
        self._data = data
        ctx = StrategyContext(self)

        if hasattr(strategy, "on_init"):
            strategy.on_init(ctx)

        for i in range(len(data)):
            self._bar_idx = i
            self._current_bar = data.iloc[i]
            self._bars_so_far = data.iloc[:i + 1]

            # Daily risk-guard reset
            self._risk_guard.reset_if_new_day(self._current_bar.name.date())

            # D48: max holding time
            if self._position and self.config.max_holding_bars > 0:
                entry_idx = self._data.index.get_loc(self._position.trade.entry_time) \
                    if self._position.trade.entry_time in self._data.index else 0
                if should_force_close(entry_idx, i, self.config.max_holding_bars):
                    self._close_trade("max_holding_time")

            # D41: trailing stop update
            if self._position and self._trailing and self._trailing.mode != TrailingStopMode.NONE:
                atr_val = self._quick_atr(i, data, 14)
                new_sl = self._trailing.update(
                    self._current_bar, self._position.trade.side.value,
                    self.config.pip_value, atr_val)
                if new_sl is not None:
                    self._position.trade.sl = new_sl

            # D42: breakeven
            if self._position and self.config.breakeven_trigger_pips > 0 and not self._breakeven_active:
                activated, new_sl = check_breakeven(
                    self._position.trade.entry_price,
                    self._current_bar["high"], self._current_bar["low"],
                    self._position.trade.side.value,
                    self.config.breakeven_trigger_pips,
                    self.config.breakeven_lock_pips,
                    self.config.pip_value, self._breakeven_active)
                if activated and new_sl is not None:
                    self._breakeven_active = True
                    self._position.trade.sl = new_sl

            # D43: partial take-profit
            if self._position and self._partials:
                triggered = check_partial_tps(
                    self._partials, self._current_bar,
                    self._position.trade.side.value)
                for pt in triggered:
                    self._apply_partial_tp(pt)

            # Check SL/TP (D39: smarter priority when configured)
            self._check_stops()

            # D40: check pending order fills
            if self.config.pending_orders_enabled:
                filled = self._order_mgr.check_fills(self._current_bar)
                for order in filled:
                    side = Side.BUY if order.side == "BUY" else Side.SELL
                    self._open_trade(side, order.size, order.sl, order.tp,
                                     fill_price=order.trigger_price)

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

            # Progress callback for live replay
            if on_progress is not None and (i % progress_interval == 0 or i == len(data) - 1):
                on_progress(i, len(data), list(self._trades))

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
        self._order_mgr = OrderManager()
        self._cooldown = CooldownTracker(self.config.cooldown_bars)
        self._risk_guard = RiskGuard(
            max_daily_loss_pct=self.config.max_daily_loss_pct,
            max_drawdown_pct=self.config.max_drawdown_pct,
            max_trades_per_day=self.config.max_trades_per_day,
            max_open_positions=self.config.max_open_positions,
        )
        self._trailing = None
        self._breakeven_active = False
        self._partials = []
        self._running_win_rate = 0.5
        self._running_avg_rr = 1.5

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        prepared = data.copy()
        if not self.config.auto_enrich_data:
            return prepared
        if not isinstance(prepared.index, pd.DatetimeIndex):
            return prepared
        return enrich_dataframe(
            prepared,
            fixed_spread=self.config.spread_pips * self.config.pip_value,
            add_regimes=self.config.regime_features_enabled,
        )

    def _quick_atr(self, idx: int, data: pd.DataFrame, period: int) -> float:
        """Fast ATR lookup without recomputing full series."""
        if idx < period:
            return 0.0
        sl = data.iloc[max(0, idx - period):idx + 1]
        tr = pd.concat([sl["high"] - sl["low"],
                        (sl["high"] - sl["close"].shift()).abs(),
                        (sl["low"] - sl["close"].shift()).abs()], axis=1).max(axis=1)
        return float(tr.mean())

    def _open_trade(self, side: Side, size: float, sl: float, tp: float,
                    fill_price: float = None):
        if self._position is not None and not self.config.multi_position:
            return  # Already in a position

        # D45: cooldown check
        if not self._cooldown.can_trade(self._bar_idx):
            return

        # D46: session filter
        if self.config.allowed_sessions:
            if not session_allows_entry(self._current_bar, self.config.allowed_sessions):
                return

        # D47: risk guard
        allowed, reason = self._risk_guard.can_open_trade(
            self._equity, self.config.initial_capital,
            1 if self._position else 0)
        if not allowed:
            logger.debug(f"Trade blocked: {reason}")
            return

        bar = self._current_bar

        # D38: dynamic spread
        if self.config.dynamic_spread_enabled:
            eff_spread = dynamic_spread(
                self.config.spread_pips,
                bar.get("session", ""),
                bar.get("volatility_regime", "normal"))
        else:
            eff_spread = self.config.spread_pips

        spread = eff_spread * self.config.pip_value

        # D36: slippage
        atr_val = self._quick_atr(self._bar_idx, self._data, 14) if \
            self.config.slippage_mode == SlippageMode.VOL_SCALED else 0.0
        slip = calc_slippage(self.config.slippage_mode, self.config.slippage_pips,
                             self.config.pip_value, bar["close"], atr_val)

        if fill_price is not None:
            price = fill_price
        elif side == Side.BUY:
            price = bar["close"] + spread / 2 + slip
        else:
            price = bar["close"] - spread / 2 - slip

        # D44: position sizing
        if self.config.sizing_mode != SizingMode.FIXED:
            sl_dist = abs(price - sl) if sl else 0.0
            size = calc_position_size(
                mode=self.config.sizing_mode, fixed_size=size,
                equity=self._equity, risk_pct=self.config.risk_pct,
                sl_distance=sl_dist, pip_value=self.config.pip_value,
                lot_multiplier=self.config.lot_multiplier,
                win_rate=self._running_win_rate,
                avg_rr=self._running_avg_rr,
                atr=self._quick_atr(self._bar_idx, self._data, 14),
                target_risk_atr=self.config.target_risk_atr,
            )

        # D37: commission
        notional = price * size * self.config.lot_multiplier
        commission = calc_commission(self.config.commission_mode,
                                     self.config.commission_per_lot,
                                     size, notional)
        self._equity -= commission

        self._trade_counter += 1
        trade = Trade(
            id=self._trade_counter, side=side,
            entry_time=bar.name, entry_price=price,
            size=size, sl=sl, tp=tp, commission=commission,
        )
        self._position = Position(trade=trade)

        # D41: initialise trailing stop for new position
        if self.config.trailing_stop_mode != TrailingStopMode.NONE:
            self._trailing = TrailingStopState(
                mode=self.config.trailing_stop_mode,
                distance=self.config.trailing_stop_distance,
                best_price=price,
            )
        else:
            self._trailing = None

        self._breakeven_active = False
        self._partials = []

    def _close_trade(self, reason: str = "signal", fill_price: float = None):
        if self._position is None:
            return

        bar = self._current_bar
        trade = self._position.trade

        if fill_price is not None:
            price = fill_price
        else:
            # D38: dynamic spread
            if self.config.dynamic_spread_enabled:
                eff_spread = dynamic_spread(
                    self.config.spread_pips,
                    bar.get("session", ""),
                    bar.get("volatility_regime", "normal"))
            else:
                eff_spread = self.config.spread_pips

            spread = eff_spread * self.config.pip_value
            atr_val = self._quick_atr(self._bar_idx, self._data, 14) if \
                self.config.slippage_mode == SlippageMode.VOL_SCALED else 0.0
            slip = calc_slippage(self.config.slippage_mode, self.config.slippage_pips,
                                 self.config.pip_value, bar["close"], atr_val)

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
        self._trailing = None
        self._breakeven_active = False
        self._partials = []

        # D47: record daily PnL
        self._risk_guard.record_trade_pnl(pnl)
        # D45: record exit for cooldown
        self._cooldown.record_exit(self._bar_idx)
        # Update running stats for Kelly
        self._update_running_stats()

    def _apply_partial_tp(self, pt: PartialTP):
        """Close a fraction of the position at the partial-TP price."""
        if self._position is None:
            return
        trade = self._position.trade
        close_size = trade.size * pt.close_fraction
        if close_size < 0.01:
            return

        if trade.side == Side.BUY:
            partial_pnl = (pt.price - trade.entry_price) * close_size * self.config.lot_multiplier
        else:
            partial_pnl = (trade.entry_price - pt.price) * close_size * self.config.lot_multiplier

        self._equity += partial_pnl
        trade.size -= close_size
        trade.pnl += partial_pnl  # accumulated partial
        if trade.size < 0.01:
            # Entire position closed via partials
            trade.exit_time = self._current_bar.name
            trade.exit_price = pt.price
            trade.exit_reason = "partial_tp_full"
            trade.bars_held = self._bar_idx - self._data.index.get_loc(trade.entry_time) \
                if trade.entry_time in self._data.index else 0
            self._trades.append(trade)
            self._position = None
            self._trailing = None
            self._breakeven_active = False
            self._partials = []
            self._risk_guard.record_trade_pnl(trade.pnl)
            self._cooldown.record_exit(self._bar_idx)
            self._update_running_stats()

    def _check_stops(self):
        if self._position is None:
            return
        trade = self._position.trade
        bar = self._current_bar
        h, l = bar["high"], bar["low"]

        if self.config.intrabar_stop_priority:
            # D39: smarter priority based on distance from open
            result, fill_price = intrabar_stop_priority(
                bar, trade.side.value, trade.entry_price, trade.sl, trade.tp)
            if result == "sl":
                self._close_trade("stop_loss", fill_price=fill_price)
            elif result == "tp":
                self._close_trade("take_profit", fill_price=fill_price)
        else:
            # Legacy behaviour
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

    def _update_running_stats(self):
        """Maintain rolling win-rate & avg-RR for Kelly sizing."""
        if len(self._trades) < 2:
            return
        recent = self._trades[-min(50, len(self._trades)):]
        wins = [t for t in recent if t.pnl > 0]
        losses = [t for t in recent if t.pnl <= 0]
        self._running_win_rate = len(wins) / len(recent) if recent else 0.5
        avg_w = np.mean([t.pnl for t in wins]) if wins else 0
        avg_l = abs(np.mean([t.pnl for t in losses])) if losses else 1
        self._running_avg_rr = avg_w / avg_l if avg_l > 0 else 1.5


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[float]
    drawdown_curve: list[float]
    timestamps: list[pd.Timestamp]
    config: BacktestConfig
    data: pd.DataFrame

    def _metrics_index(self):
        if isinstance(self.data.index, pd.DatetimeIndex) and len(self.data.index) > 0:
            return self.data.index
        if self.timestamps:
            return pd.DatetimeIndex(self.timestamps)
        return None

    @cached_property
    def performance_stats(self) -> PerformanceStats:
        return compute_performance_stats(
            self.equity_curve,
            initial_equity=self.config.initial_capital,
            periods_per_year=self.config.periods_per_year,
            index=self._metrics_index(),
            rf_annual=self.config.risk_free_rate_annual,
            return_method=self.config.return_method,
            trade_pnls=self.trade_results_after_costs,
        )

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
        return self.performance_stats.net_profit

    @property
    def trade_results_after_costs(self) -> list[float]:
        return [t.pnl - t.commission for t in self.trades]

    @property
    def net_pnl_after_costs(self) -> float:
        return self.net_pnl

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
    def max_drawdown_pct(self) -> float:
        return self.performance_stats.max_drawdown_pct

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
        return self.performance_stats.sharpe

    @property
    def sortino_ratio(self) -> float:
        return self.performance_stats.sortino

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
        return self.performance_stats.total_return_pct

    @property
    def cagr_pct(self) -> float:
        return self.performance_stats.cagr_pct

    @property
    def annual_volatility_pct(self) -> float:
        return self.performance_stats.annual_volatility_pct

    @property
    def periods_per_year(self) -> float:
        return self.performance_stats.periods_per_year

    # ── Extended analytics ────────────────────────────────────────────────

    @property
    def calmar_ratio(self) -> float:
        return self.performance_stats.calmar

    @property
    def consecutive_wins(self) -> int:
        return self._max_streak(True)

    @property
    def consecutive_losses(self) -> int:
        return self._max_streak(False)

    def _max_streak(self, wins: bool) -> int:
        streak = best = 0
        for t in self.trades:
            if (t.pnl > 0) == wins:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        return best

    @property
    def avg_holding_time_minutes(self) -> float:
        if not self.trades: return 0
        deltas = []
        for t in self.trades:
            if t.entry_time and t.exit_time:
                deltas.append((t.exit_time - t.entry_time).total_seconds() / 60)
        return np.mean(deltas) if deltas else 0

    @property
    def recovery_factor(self) -> float:
        if self.max_drawdown_usd == 0: return 0
        return abs(self.net_pnl / self.max_drawdown_usd)

    @property
    def sl_exits(self) -> int:
        return sum(1 for t in self.trades if t.exit_reason == "stop_loss")

    @property
    def tp_exits(self) -> int:
        return sum(1 for t in self.trades if t.exit_reason == "take_profit")

    @property
    def signal_exits(self) -> int:
        return sum(1 for t in self.trades if t.exit_reason == "signal")

    def monte_carlo(self, config=None):
        from aphelion_lab.monte_carlo import MonteCarloConfig, simulate_trade_sequence

        mc_config = config or MonteCarloConfig()
        return simulate_trade_sequence(
            self.trade_results_after_costs,
            self.config.initial_capital,
            mc_config,
        )

    def monte_carlo_metrics(self, config=None) -> dict:
        return self.monte_carlo(config).to_metrics_dict()

    def to_metrics_dict(self) -> dict:
        return {
            "Net P&L": f"${self.net_pnl:.2f}",
            "Net P&L %": f"{self.total_return_pct:.2f}%",
            "Total Trades": str(self.total_trades),
            "Win Rate": f"{self.win_rate:.1f}%",
            "Profit Factor": f"{self.profit_factor:.3f}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "CAGR": f"{self.cagr_pct:.2f}%",
            "Annual Volatility": f"{self.annual_volatility_pct:.2f}%",
            "Max Drawdown": f"{self.max_drawdown_pct:.2f}%",
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
            "Periods/Year": f"{self.periods_per_year:.1f}",
            "Calmar Ratio": f"{self.calmar_ratio:.3f}",
            "Recovery Factor": f"{self.recovery_factor:.3f}",
            "Consec. Wins": str(self.consecutive_wins),
            "Consec. Losses": str(self.consecutive_losses),
            "Avg Hold (min)": f"{self.avg_holding_time_minutes:.0f}",
            "SL Exits": str(self.sl_exits),
            "TP Exits": str(self.tp_exits),
            "Signal Exits": str(self.signal_exits),
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
