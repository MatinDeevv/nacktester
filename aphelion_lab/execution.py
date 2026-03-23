"""
Aphelion Lab — Execution Realism
Pluggable execution components for the backtest engine.
Keeps the engine loop clean; each component is opt-in via BacktestConfig.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("aphelion.execution")


# ─── D36: Slippage Model ───────────────────────────────────────────────────

class SlippageMode(str, Enum):
    FIXED = "fixed"
    PCT = "pct"            # percentage of price
    VOL_SCALED = "vol"    # ATR-proportional


def calc_slippage(mode: SlippageMode, base_pips: float, pip_value: float,
                  price: float = 0.0, atr: float = 0.0) -> float:
    """Return slippage in price units."""
    if mode == SlippageMode.FIXED:
        return base_pips * pip_value
    elif mode == SlippageMode.PCT:
        return price * base_pips / 10000.0  # base_pips treated as basis-points
    elif mode == SlippageMode.VOL_SCALED:
        return atr * base_pips / 100.0  # base_pips treated as % of ATR
    return base_pips * pip_value


# ─── D37: Commission Model ─────────────────────────────────────────────────

class CommissionMode(str, Enum):
    PER_LOT = "per_lot"
    PER_TRADE = "per_trade"
    PCT = "pct"


def calc_commission(mode: CommissionMode, rate: float, size: float,
                    notional: float = 0.0) -> float:
    """Return total commission for a trade."""
    if mode == CommissionMode.PER_LOT:
        return rate * size / 0.01  # size in lots (0.01 = micro)
    elif mode == CommissionMode.PER_TRADE:
        return rate
    elif mode == CommissionMode.PCT:
        return notional * rate / 100.0
    return rate * size / 0.01


# ─── D38: Dynamic Spread ───────────────────────────────────────────────────

def dynamic_spread(base_spread_pips: float, session: str = "",
                   volatility_regime: str = "normal") -> float:
    """Widen spread during off-hours or extreme volatility."""
    mult = 1.0
    if session in ("off_hours", ""):
        mult *= 1.8
    elif session == "asia":
        mult *= 1.3
    if volatility_regime == "extreme":
        mult *= 1.5
    elif volatility_regime == "high":
        mult *= 1.2
    elif volatility_regime == "low":
        mult *= 0.9
    return base_spread_pips * mult


# ─── D39: Intrabar SL/TP priority ──────────────────────────────────────────

def intrabar_stop_priority(bar: pd.Series, side: str, entry_price: float,
                           sl: Optional[float], tp: Optional[float]):
    """Determine which was hit first (SL or TP) using proximity heuristic.
    Returns ('sl', fill_price), ('tp', fill_price), or (None, None).
    For BUY: SL triggers if low <= sl; TP if high >= tp.
    Ambiguity (both triggered): use distance from open to determine sequence."""
    h, l, o = bar["high"], bar["low"], bar["open"]
    sl_hit = False
    tp_hit = False

    if side == "BUY":
        if sl is not None and l <= sl:
            sl_hit = True
        if tp is not None and h >= tp:
            tp_hit = True
    else:  # SELL
        if sl is not None and h >= sl:
            sl_hit = True
        if tp is not None and l <= tp:
            tp_hit = True

    if sl_hit and tp_hit:
        # Use distance from open as heuristic for which was hit first
        if side == "BUY":
            dist_sl = abs(o - sl) if sl else float("inf")
            dist_tp = abs(o - tp) if tp else float("inf")
        else:
            dist_sl = abs(o - sl) if sl else float("inf")
            dist_tp = abs(o - tp) if tp else float("inf")
        if dist_sl <= dist_tp:
            return "sl", sl
        else:
            return "tp", tp
    elif sl_hit:
        return "sl", sl
    elif tp_hit:
        return "tp", tp
    return None, None


# ─── D40: Pending Orders ───────────────────────────────────────────────────

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class PendingOrder:
    order_type: OrderType
    side: str  # 'BUY' or 'SELL'
    trigger_price: float
    size: float = 0.01
    sl: Optional[float] = None
    tp: Optional[float] = None
    expiry_bars: Optional[int] = None  # cancel after N bars
    bars_alive: int = 0
    tag: str = ""


class OrderManager:
    """Manages pending limit/stop orders."""

    def __init__(self):
        self._orders: list[PendingOrder] = []

    def add(self, order: PendingOrder):
        self._orders.append(order)

    def cancel_all(self):
        self._orders.clear()

    def cancel_by_tag(self, tag: str):
        self._orders = [o for o in self._orders if o.tag != tag]

    @property
    def pending(self) -> list[PendingOrder]:
        return list(self._orders)

    def check_fills(self, bar: pd.Series) -> list[PendingOrder]:
        """Check which pending orders should fill on this bar. Returns filled orders."""
        filled = []
        remaining = []
        h, l = bar["high"], bar["low"]

        for order in self._orders:
            order.bars_alive += 1
            # Check expiry
            if order.expiry_bars is not None and order.bars_alive > order.expiry_bars:
                continue  # expired, drop

            trigger = False
            if order.order_type == OrderType.LIMIT:
                if order.side == "BUY" and l <= order.trigger_price:
                    trigger = True
                elif order.side == "SELL" and h >= order.trigger_price:
                    trigger = True
            elif order.order_type == OrderType.STOP:
                if order.side == "BUY" and h >= order.trigger_price:
                    trigger = True
                elif order.side == "SELL" and l <= order.trigger_price:
                    trigger = True

            if trigger:
                filled.append(order)
            else:
                remaining.append(order)

        self._orders = remaining
        return filled


# ─── D41: Trailing Stop ────────────────────────────────────────────────────

class TrailingStopMode(str, Enum):
    NONE = "none"
    FIXED = "fixed"       # fixed pips distance
    ATR = "atr"           # ATR-based distance
    PERCENT = "percent"   # percentage of price


@dataclass
class TrailingStopState:
    mode: TrailingStopMode = TrailingStopMode.NONE
    distance: float = 0.0   # pips, ATR mult, or percent
    best_price: float = 0.0
    current_stop: Optional[float] = None

    def update(self, bar: pd.Series, side: str, pip_value: float = 0.01,
               atr: float = 0.0) -> Optional[float]:
        """Update trailing stop; returns new SL level or None if unchanged."""
        if self.mode == TrailingStopMode.NONE:
            return self.current_stop

        if side == "BUY":
            self.best_price = max(self.best_price, bar["high"])
            offset = self._calc_offset(self.best_price, pip_value, atr)
            new_stop = self.best_price - offset
            if self.current_stop is None or new_stop > self.current_stop:
                self.current_stop = new_stop
        else:
            self.best_price = min(self.best_price, bar["low"])
            offset = self._calc_offset(self.best_price, pip_value, atr)
            new_stop = self.best_price + offset
            if self.current_stop is None or new_stop < self.current_stop:
                self.current_stop = new_stop

        return self.current_stop

    def _calc_offset(self, price: float, pip_value: float, atr: float) -> float:
        if self.mode == TrailingStopMode.FIXED:
            return self.distance * pip_value
        elif self.mode == TrailingStopMode.ATR:
            return self.distance * atr
        elif self.mode == TrailingStopMode.PERCENT:
            return price * self.distance / 100.0
        return 0.0


# ─── D42: Break-Even Logic ─────────────────────────────────────────────────

def check_breakeven(entry_price: float, current_high: float, current_low: float,
                    side: str, trigger_pips: float, lock_pips: float,
                    pip_value: float = 0.01, activated: bool = False) -> tuple[bool, Optional[float]]:
    """Check if breakeven should activate.
    Returns (activated, new_sl). Once activated stays on."""
    if activated:
        return True, None  # already applied

    trigger_offset = trigger_pips * pip_value
    lock_offset = lock_pips * pip_value

    if side == "BUY":
        if current_high >= entry_price + trigger_offset:
            return True, entry_price + lock_offset
    else:
        if current_low <= entry_price - trigger_offset:
            return True, entry_price - lock_offset

    return False, None


# ─── D43: Partial Take-Profit ──────────────────────────────────────────────

@dataclass
class PartialTP:
    """Define a partial take-profit level."""
    price: float
    close_fraction: float  # 0.0 – 1.0 of remaining position
    triggered: bool = False


def check_partial_tps(partials: list[PartialTP], bar: pd.Series,
                      side: str) -> list[PartialTP]:
    """Return list of partials that triggered on this bar."""
    triggered = []
    for pt in partials:
        if pt.triggered:
            continue
        if side == "BUY" and bar["high"] >= pt.price:
            pt.triggered = True
            triggered.append(pt)
        elif side == "SELL" and bar["low"] <= pt.price:
            pt.triggered = True
            triggered.append(pt)
    return triggered


# ─── D44: Position Sizing ──────────────────────────────────────────────────

class SizingMode(str, Enum):
    FIXED = "fixed"
    RISK_PCT = "risk_pct"       # risk N% of equity per trade
    KELLY = "kelly"             # Kelly criterion based
    VOL_ADJUSTED = "vol_adj"    # ATR-normalised sizing


def calc_position_size(mode: SizingMode, fixed_size: float = 0.01,
                       equity: float = 5000.0, risk_pct: float = 1.0,
                       sl_distance: float = 0.0, pip_value: float = 0.01,
                       lot_multiplier: float = 100.0,
                       win_rate: float = 0.5, avg_rr: float = 1.5,
                       atr: float = 0.0, target_risk_atr: float = 1.0) -> float:
    """Calculate position size based on sizing model. Returns lot size."""
    if mode == SizingMode.FIXED:
        return fixed_size

    elif mode == SizingMode.RISK_PCT:
        if sl_distance <= 0:
            return fixed_size
        risk_amount = equity * risk_pct / 100.0
        size_lots = risk_amount / (sl_distance / pip_value * lot_multiplier * pip_value)
        return max(0.01, round(size_lots, 2))

    elif mode == SizingMode.KELLY:
        # Kelly fraction = W - (1-W)/R
        if avg_rr <= 0:
            return fixed_size
        kelly_f = win_rate - (1 - win_rate) / avg_rr
        kelly_f = max(0, min(kelly_f, 0.25))  # cap at 25%
        risk_amount = equity * kelly_f
        if sl_distance <= 0:
            return fixed_size
        size_lots = risk_amount / (sl_distance / pip_value * lot_multiplier * pip_value)
        return max(0.01, round(size_lots, 2))

    elif mode == SizingMode.VOL_ADJUSTED:
        if atr <= 0:
            return fixed_size
        risk_amount = equity * risk_pct / 100.0
        target_sl = atr * target_risk_atr
        size_lots = risk_amount / (target_sl / pip_value * lot_multiplier * pip_value)
        return max(0.01, round(size_lots, 2))

    return fixed_size


# ─── D45: Cooldown ──────────────────────────────────────────────────────────

class CooldownTracker:
    """Enforce minimum bars between trades."""

    def __init__(self, cooldown_bars: int = 0):
        self.cooldown_bars = cooldown_bars
        self._last_exit_bar: Optional[int] = None

    def record_exit(self, bar_index: int):
        self._last_exit_bar = bar_index

    def can_trade(self, bar_index: int) -> bool:
        if self.cooldown_bars <= 0:
            return True
        if self._last_exit_bar is None:
            return True
        return (bar_index - self._last_exit_bar) >= self.cooldown_bars


# ─── D46: Session Filter ───────────────────────────────────────────────────

def session_allows_entry(bar: pd.Series, allowed_sessions: Optional[list[str]] = None) -> bool:
    """Check if the current bar's session is in the allowed list.
    If allowed_sessions is None or empty, all sessions allowed."""
    if not allowed_sessions:
        return True
    session = bar.get("session", "")
    return session in allowed_sessions


# ─── D47: Risk Guards ──────────────────────────────────────────────────────

@dataclass
class RiskGuard:
    max_daily_loss_pct: float = 0.0    # 0 = disabled
    max_drawdown_pct: float = 0.0      # 0 = disabled
    max_trades_per_day: int = 0        # 0 = unlimited
    max_open_positions: int = 1        # for multi-position mode

    # Internal tracking (reset daily or per-session)
    _daily_pnl: float = 0.0
    _daily_trade_count: int = 0
    _current_date: object = None

    def reset_if_new_day(self, bar_date):
        if bar_date != self._current_date:
            self._daily_pnl = 0.0
            self._daily_trade_count = 0
            self._current_date = bar_date

    def record_trade_pnl(self, pnl: float):
        self._daily_pnl += pnl
        self._daily_trade_count += 1

    def can_open_trade(self, equity: float, initial_capital: float,
                       open_positions: int = 0) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        if self.max_daily_loss_pct > 0:
            if self._daily_pnl < 0:
                daily_loss_pct = abs(self._daily_pnl) / equity * 100
                if daily_loss_pct >= self.max_daily_loss_pct:
                    return False, "daily_loss_limit"

        if self.max_drawdown_pct > 0:
            dd_pct = (initial_capital - equity) / initial_capital * 100
            if equity < initial_capital and dd_pct >= self.max_drawdown_pct:
                return False, "max_drawdown_limit"

        if self.max_trades_per_day > 0 and self._daily_trade_count >= self.max_trades_per_day:
            return False, "max_trades_per_day"

        if open_positions >= self.max_open_positions:
            return False, "max_open_positions"

        return True, ""


# ─── D48: Max Holding Time ──────────────────────────────────────────────────

def should_force_close(entry_bar_idx: int, current_bar_idx: int,
                       max_bars: int = 0) -> bool:
    """Force-close position after max_bars. 0 = disabled."""
    if max_bars <= 0:
        return False
    return (current_bar_idx - entry_bar_idx) >= max_bars


# ─── D49/D50: Multi-position & order modification helpers ──────────────────

@dataclass
class PositionSlot:
    """A position in a multi-position context."""
    trade_id: int
    side: str
    entry_price: float
    entry_bar_idx: int
    size: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    trailing: Optional[TrailingStopState] = None
    breakeven_active: bool = False
    partials: list = field(default_factory=list)
    tag: str = ""

    def modify_sl(self, new_sl: float):
        self.sl = new_sl

    def modify_tp(self, new_tp: float):
        self.tp = new_tp


class MultiPositionManager:
    """Manages multiple simultaneous positions."""

    def __init__(self, max_positions: int = 1):
        self.max_positions = max_positions
        self._slots: list[PositionSlot] = []

    @property
    def count(self) -> int:
        return len(self._slots)

    @property
    def slots(self) -> list[PositionSlot]:
        return list(self._slots)

    def can_open(self) -> bool:
        return len(self._slots) < self.max_positions

    def add(self, slot: PositionSlot):
        if len(self._slots) < self.max_positions:
            self._slots.append(slot)

    def remove(self, trade_id: int) -> Optional[PositionSlot]:
        for i, s in enumerate(self._slots):
            if s.trade_id == trade_id:
                return self._slots.pop(i)
        return None

    def get_by_tag(self, tag: str) -> list[PositionSlot]:
        return [s for s in self._slots if s.tag == tag]

    def close_all(self) -> list[PositionSlot]:
        closed = self._slots[:]
        self._slots.clear()
        return closed
