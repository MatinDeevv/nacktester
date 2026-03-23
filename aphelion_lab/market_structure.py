"""
Aphelion Lab — Market Structure
Data enrichment: sessions, day-of-week, gaps, HTF cache, symbol metadata,
spread/bid-ask synthesis, news event markers.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import time as dtime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from aphelion_lab.regime_detection import add_regime_features

logger = logging.getLogger("aphelion.market")

# ─── A4: Session definitions (UTC hours) ────────────────────────────────────

SESSION_DEFS = {
    "asia":     (dtime(0, 0),  dtime(8, 0)),
    "london":   (dtime(7, 0),  dtime(16, 0)),
    "new_york": (dtime(12, 0), dtime(21, 0)),
    "overlap":  (dtime(12, 0), dtime(16, 0)),
}


def label_session(ts: pd.Timestamp) -> str:
    """Return the primary session label for a UTC timestamp."""
    t = ts.time()
    if dtime(12, 0) <= t < dtime(16, 0):
        return "overlap"
    if dtime(7, 0) <= t < dtime(16, 0):
        return "london"
    if dtime(12, 0) <= t < dtime(21, 0):
        return "new_york"
    if dtime(0, 0) <= t < dtime(8, 0):
        return "asia"
    return "off_hours"


def add_session_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'session' column to DataFrame with UTC-indexed bars."""
    hours = df.index.hour
    minutes = df.index.minute
    total = hours * 60 + minutes
    conditions = [
        (total >= 720) & (total < 960),   # overlap 12:00-16:00
        (total >= 420) & (total < 960),    # london 07:00-16:00
        (total >= 720) & (total < 1260),   # new_york 12:00-21:00
        (total >= 0) & (total < 480),      # asia 00:00-08:00
    ]
    choices = ["overlap", "london", "new_york", "asia"]
    df["session"] = np.select(conditions, choices, default="off_hours")
    return df


# ─── A5: Day-of-week labels ────────────────────────────────────────────────

DOW_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def add_dow_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'dow' (0=Mon) and 'dow_name' columns."""
    df["dow"] = df.index.dayofweek
    df["dow_name"] = df["dow"].map(DOW_NAMES)
    return df


# ─── A1: Tick volume support ───────────────────────────────────────────────

def ensure_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'volume' column exists (fill with 0 if missing)."""
    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        else:
            df["volume"] = 0
    return df


# ─── A2/A3: Spread and bid/ask candle synthesis ────────────────────────────

def add_spread_columns(df: pd.DataFrame, fixed_spread: float = 0.0) -> pd.DataFrame:
    """Add spread, bid_close, ask_close columns.
    If 'spread' already exists in data (MT5 provides it), use it.
    Otherwise apply fixed_spread (in price units, not pips)."""
    if "spread" not in df.columns:
        df["spread"] = fixed_spread
    else:
        # MT5 spread is in points; keep raw for now
        df["spread"] = df["spread"].fillna(fixed_spread)
    half = df["spread"] / 2.0
    df["bid_close"] = df["close"] - half
    df["ask_close"] = df["close"] + half
    df["bid_open"] = df["open"] - half
    df["ask_open"] = df["open"] + half
    return df


# ─── A8: Gap detection ─────────────────────────────────────────────────────

def detect_gaps(df: pd.DataFrame, atr_mult: float = 3.0, atr_period: int = 14) -> pd.DataFrame:
    """Add 'gap' column: True where open is abnormally far from prior close."""
    if len(df) < atr_period + 1:
        df["gap"] = False
        return df
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    gap_size = (df["open"] - df["close"].shift()).abs()
    df["gap"] = gap_size > (atr * atr_mult)
    df["gap"] = df["gap"].fillna(False)
    return df


# ─── A9: Partial/incomplete candle awareness ───────────────────────────────

def mark_last_bar_partial(df: pd.DataFrame) -> pd.DataFrame:
    """Mark last bar as potentially partial (live data).
    In backtest on completed data this is always False except the very last bar."""
    df["is_partial"] = False
    if len(df) > 0:
        df.iloc[-1, df.columns.get_loc("is_partial")] = True
    return df


# ─── A10: Symbol metadata ──────────────────────────────────────────────────

@dataclass
class SymbolMeta:
    """Execution-related metadata for a trading instrument."""
    name: str = "XAUUSD"
    point: float = 0.01       # smallest price increment
    pip_size: float = 0.01    # pip size (for gold = 0.01)
    lot_size: float = 100.0   # contract size per lot (100 oz for gold)
    min_lot: float = 0.01
    max_lot: float = 100.0
    lot_step: float = 0.01
    currency_base: str = "XAU"
    currency_profit: str = "USD"
    trading_hours: str = "Mon 01:00 - Fri 23:00 UTC"
    swap_long: float = 0.0
    swap_short: float = 0.0
    digits: int = 2


# Preset metadata for common instruments
SYMBOL_META = {
    "XAUUSD": SymbolMeta(name="XAUUSD", point=0.01, pip_size=0.01, lot_size=100, digits=2,
                         currency_base="XAU", currency_profit="USD"),
    "XAGUSD": SymbolMeta(name="XAGUSD", point=0.001, pip_size=0.001, lot_size=5000, digits=3,
                         currency_base="XAG", currency_profit="USD"),
    "EURUSD": SymbolMeta(name="EURUSD", point=0.00001, pip_size=0.0001, lot_size=100000, digits=5,
                         currency_base="EUR", currency_profit="USD"),
    "GBPUSD": SymbolMeta(name="GBPUSD", point=0.00001, pip_size=0.0001, lot_size=100000, digits=5,
                         currency_base="GBP", currency_profit="USD"),
    "USDJPY": SymbolMeta(name="USDJPY", point=0.001, pip_size=0.01, lot_size=100000, digits=3,
                         currency_base="USD", currency_profit="JPY"),
    "US500":  SymbolMeta(name="US500", point=0.01, pip_size=0.01, lot_size=1, digits=2,
                         currency_base="US500", currency_profit="USD"),
    "BTCUSD": SymbolMeta(name="BTCUSD", point=0.01, pip_size=0.01, lot_size=1, digits=2,
                         currency_base="BTC", currency_profit="USD"),
    "XTIUSD": SymbolMeta(name="XTIUSD", point=0.01, pip_size=0.01, lot_size=1000, digits=2,
                         currency_base="XTI", currency_profit="USD"),
}


def get_symbol_meta(symbol: str) -> SymbolMeta:
    """Get symbol metadata, falling back to XAUUSD defaults."""
    return SYMBOL_META.get(symbol, SymbolMeta(name=symbol))


# ─── A6: News event markers ────────────────────────────────────────────────

@dataclass
class NewsEvent:
    timestamp: pd.Timestamp
    currency: str
    impact: str  # "low", "medium", "high"
    title: str


def load_news_events(path: str) -> list[NewsEvent]:
    """Load news events from a JSON or CSV file.
    JSON format: [{"timestamp": "...", "currency": "USD", "impact": "high", "title": "NFP"}]
    CSV format: timestamp,currency,impact,title"""
    p = Path(path)
    if not p.exists():
        logger.warning(f"News file not found: {path}")
        return []
    events = []
    if p.suffix == ".json":
        raw = json.loads(p.read_text())
        for r in raw:
            events.append(NewsEvent(
                timestamp=pd.Timestamp(r["timestamp"], tz="UTC"),
                currency=r.get("currency", ""),
                impact=r.get("impact", "medium"),
                title=r.get("title", ""),
            ))
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            events.append(NewsEvent(
                timestamp=pd.Timestamp(r["timestamp"], tz="UTC"),
                currency=str(r.get("currency", "")),
                impact=str(r.get("impact", "medium")),
                title=str(r.get("title", "")),
            ))
    return events


def mark_news_bars(df: pd.DataFrame, events: list[NewsEvent],
                   window_before_mins: int = 15, window_after_mins: int = 15) -> pd.DataFrame:
    """Add 'news_nearby' and 'news_impact' columns."""
    df["news_nearby"] = False
    df["news_impact"] = ""
    if not events:
        return df
    for ev in events:
        start = ev.timestamp - pd.Timedelta(minutes=window_before_mins)
        end = ev.timestamp + pd.Timedelta(minutes=window_after_mins)
        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, "news_nearby"] = True
        # Keep highest impact
        current = df.loc[mask, "news_impact"]
        priority = {"high": 3, "medium": 2, "low": 1, "": 0}
        for idx in current.index:
            if priority.get(ev.impact, 0) > priority.get(current[idx], 0):
                df.at[idx, "news_impact"] = ev.impact
    return df


# ─── A7: Higher-timeframe cache ────────────────────────────────────────────

TF_RESAMPLE_MAP = {
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def build_htf_bar(group: pd.DataFrame) -> pd.Series:
    """Aggregate a group of bars into one OHLCV bar."""
    if len(group) == 0:
        return pd.Series(dtype=float)
    return pd.Series({
        "open": group["open"].iloc[0],
        "high": group["high"].max(),
        "low": group["low"].min(),
        "close": group["close"].iloc[-1],
        "volume": group["volume"].sum(),
    })


class HTFCache:
    """Derive and cache higher-timeframe bars from the base (e.g. 5m) stream."""

    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}

    def build(self, base_df: pd.DataFrame, htf_list: list[str] = None):
        """Pre-build all HTF DataFrames from the base data."""
        htf_list = htf_list or ["M15", "H1", "H4", "D1"]
        self._cache.clear()
        for tf in htf_list:
            rule = TF_RESAMPLE_MAP.get(tf)
            if rule is None:
                continue
            resampled = base_df.resample(rule).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna(subset=["open"])
            self._cache[tf] = resampled

    def get(self, tf: str) -> Optional[pd.DataFrame]:
        return self._cache.get(tf)

    def get_current(self, tf: str, current_ts: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Get HTF bars up to current_ts (no lookahead)."""
        df = self._cache.get(tf)
        if df is None:
            return None
        return df[df.index <= current_ts]

    def get_last_bar(self, tf: str, current_ts: pd.Timestamp) -> Optional[pd.Series]:
        """Get the most recent completed HTF bar."""
        df = self.get_current(tf, current_ts)
        if df is None or len(df) < 2:
            return None
        # The last bar at current_ts may be incomplete; return the one before it
        return df.iloc[-2]

    @property
    def available_timeframes(self) -> list[str]:
        return list(self._cache.keys())


# ─── Enrichment pipeline ───────────────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame, symbol: str = "XAUUSD",
                     fixed_spread: float = 0.0,
                     news_events: list[NewsEvent] = None,
                     add_regimes: bool = True) -> pd.DataFrame:
    """Apply all data enrichment steps to a raw OHLCV DataFrame."""
    df = df.copy()
    df = ensure_volume(df)
    meta = get_symbol_meta(symbol)
    spread = fixed_spread if fixed_spread > 0 else meta.pip_size * 2
    df = add_spread_columns(df, fixed_spread=spread)
    df = add_session_labels(df)
    df = add_dow_labels(df)
    df = detect_gaps(df)
    if news_events:
        df = mark_news_bars(df, news_events)
    if add_regimes:
        df = add_regime_features(df)
    return df
