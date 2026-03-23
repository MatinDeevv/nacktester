"""
Aphelion Lab — Price-Action / Structure Primitives
Vectorised where possible; all accept pd.DataFrame with OHLCV + DatetimeIndex.
"""

import numpy as np
import pandas as pd


# ─── C26: Pivot Points ─────────────────────────────────────────────────────

def pivot_points(df: pd.DataFrame, method: str = "classic") -> pd.DataFrame:
    """Daily pivot points projected onto intraday bars.
    method: 'classic', 'fibonacci', 'camarilla', 'woodie'
    Uses *previous* day's OHLC to avoid look-ahead."""
    daily = df.resample("D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    # Shift so today uses yesterday's values
    prev = daily.shift(1).dropna()

    pp = (prev["high"] + prev["low"] + prev["close"]) / 3.0

    if method == "classic":
        r1 = 2 * pp - prev["low"]
        s1 = 2 * pp - prev["high"]
        r2 = pp + (prev["high"] - prev["low"])
        s2 = pp - (prev["high"] - prev["low"])
        r3 = prev["high"] + 2 * (pp - prev["low"])
        s3 = prev["low"] - 2 * (prev["high"] - pp)
    elif method == "fibonacci":
        rng = prev["high"] - prev["low"]
        r1 = pp + 0.382 * rng
        r2 = pp + 0.618 * rng
        r3 = pp + rng
        s1 = pp - 0.382 * rng
        s2 = pp - 0.618 * rng
        s3 = pp - rng
    elif method == "camarilla":
        rng = prev["high"] - prev["low"]
        r1 = prev["close"] + rng * 1.1 / 12
        r2 = prev["close"] + rng * 1.1 / 6
        r3 = prev["close"] + rng * 1.1 / 4
        s1 = prev["close"] - rng * 1.1 / 12
        s2 = prev["close"] - rng * 1.1 / 6
        s3 = prev["close"] - rng * 1.1 / 4
    elif method == "woodie":
        pp = (prev["high"] + prev["low"] + 2 * prev["close"]) / 4.0
        r1 = 2 * pp - prev["low"]
        s1 = 2 * pp - prev["high"]
        r2 = pp + (prev["high"] - prev["low"])
        s2 = pp - (prev["high"] - prev["low"])
        r3 = prev["high"] + 2 * (pp - prev["low"])
        s3 = prev["low"] - 2 * (prev["high"] - pp)
    else:
        raise ValueError(f"Unknown pivot method: {method}")

    pivot_df = pd.DataFrame({"pp": pp, "r1": r1, "r2": r2, "r3": r3,
                              "s1": s1, "s2": s2, "s3": s3})
    # Reindex onto intraday bars (forward fill within each day)
    pivot_df.index = pivot_df.index.normalize()
    joined = df[[]].copy()
    joined["_date"] = joined.index.normalize()
    for col in pivot_df.columns:
        mapping = pivot_df[col].to_dict()
        joined[col] = joined["_date"].map(mapping)
    joined.drop(columns="_date", inplace=True)
    return joined.ffill()


# ─── C27: Inside / Outside Bars ────────────────────────────────────────────

def inside_bars(df: pd.DataFrame) -> pd.Series:
    """True when bar's range is fully contained in previous bar's range."""
    result = (df["high"] <= df["high"].shift(1)) & (df["low"] >= df["low"].shift(1))
    result.name = "inside_bar"
    return result.fillna(False)


def outside_bars(df: pd.DataFrame) -> pd.Series:
    """True when bar fully engulfs previous bar's range."""
    result = (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
    result.name = "outside_bar"
    return result.fillna(False)


# ─── C28: NR4 / NR7 ────────────────────────────────────────────────────────

def narrow_range(df: pd.DataFrame, lookback: int = 4) -> pd.Series:
    """True when current bar has narrowest high-low range in last `lookback` bars.
    Use lookback=4 for NR4, 7 for NR7."""
    rng = df["high"] - df["low"]
    rolling_min = rng.rolling(lookback).min()
    result = rng <= rolling_min
    result.name = f"nr{lookback}"
    return result.fillna(False)


# ─── C29: Range helpers (ATR-normalised) ───────────────────────────────────

def bar_range_atr(df: pd.DataFrame, atr_period: int = 14) -> pd.Series:
    """Current bar range expressed as a multiple of ATR."""
    from aphelion_lab.indicators import atr_series
    atr_val = atr_series(df, atr_period)
    rng = df["high"] - df["low"]
    result = rng / atr_val.replace(0, 1e-10)
    result.name = "range_atr"
    return result


def body_ratio(df: pd.DataFrame) -> pd.Series:
    """Ratio of body size to total range (0 = doji, 1 = marubozu)."""
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, 1e-10)
    result = body / rng
    result.name = "body_ratio"
    return result


def upper_wick_ratio(df: pd.DataFrame) -> pd.Series:
    """Upper wick as proportion of total range."""
    top = df[["open", "close"]].max(axis=1)
    rng = (df["high"] - df["low"]).replace(0, 1e-10)
    result = (df["high"] - top) / rng
    result.name = "upper_wick_ratio"
    return result


def lower_wick_ratio(df: pd.DataFrame) -> pd.Series:
    """Lower wick as proportion of total range."""
    bot = df[["open", "close"]].min(axis=1)
    rng = (df["high"] - df["low"]).replace(0, 1e-10)
    result = (bot - df["low"]) / rng
    result.name = "lower_wick_ratio"
    return result


# ─── C30: Trend Classifier ─────────────────────────────────────────────────

def trend_classifier(df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50,
                     adx_period: int = 14, adx_threshold: float = 25.0) -> pd.Series:
    """Classify bars: 'strong_up', 'weak_up', 'strong_down', 'weak_down', 'range'.
    Uses dual-EMA slope + ADX strength."""
    from aphelion_lab.indicators import adx as calc_adx
    fast_ema = df["close"].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow_period, adjust=False).mean()
    adx_df = calc_adx(df, adx_period)
    adx_val = adx_df["adx"]

    bullish = fast_ema > slow_ema
    strong = adx_val >= adx_threshold

    conditions = [
        bullish & strong,
        bullish & ~strong,
        ~bullish & strong,
        ~bullish & ~strong,
    ]
    choices = ["strong_up", "weak_up", "strong_down", "weak_down"]
    result = pd.Series(np.select(conditions, choices, default="range"),
                       index=df.index, name="trend")
    return result


# ─── C31: Volatility Classifier ────────────────────────────────────────────

def volatility_classifier(df: pd.DataFrame, atr_period: int = 14,
                          short_window: int = 5) -> pd.Series:
    """Classify volatility as 'low', 'normal', 'high', 'extreme' based on ATR percentile."""
    from aphelion_lab.indicators import atr_series
    atr_val = atr_series(df, atr_period)
    atr_pct = atr_val.rolling(100, min_periods=20).rank(pct=True)

    conditions = [
        atr_pct <= 0.25,
        atr_pct <= 0.50,
        atr_pct <= 0.75,
        atr_pct > 0.75,
    ]
    labels = ["low", "normal", "high", "extreme"]
    result = pd.Series(np.select(conditions, labels, default="normal"),
                       index=df.index, name="volatility_regime")
    return result


# ─── C32: Bar Patterns ─────────────────────────────────────────────────────

def bar_patterns(df: pd.DataFrame, doji_threshold: float = 0.05,
                 hammer_wick_ratio: float = 0.6) -> pd.DataFrame:
    """Detect common single-bar patterns: doji, hammer, inverted_hammer, engulfing_bull, engulfing_bear."""
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, 1e-10)
    br = body / rng
    top = df[["open", "close"]].max(axis=1)
    bot = df[["open", "close"]].min(axis=1)
    upper_w = (df["high"] - top) / rng
    lower_w = (bot - df["low"]) / rng

    doji = br <= doji_threshold
    hammer = (lower_w >= hammer_wick_ratio) & (upper_w < 0.2) & (br >= doji_threshold)
    inv_hammer = (upper_w >= hammer_wick_ratio) & (lower_w < 0.2) & (br >= doji_threshold)

    prev_body = (df["close"].shift(1) - df["open"].shift(1))
    curr_body = df["close"] - df["open"]
    engulf_bull = (prev_body < 0) & (curr_body > 0) & (df["close"] > df["open"].shift(1)) & (df["open"] < df["close"].shift(1))
    engulf_bear = (prev_body > 0) & (curr_body < 0) & (df["close"] < df["open"].shift(1)) & (df["open"] > df["close"].shift(1))

    return pd.DataFrame({
        "doji": doji,
        "hammer": hammer,
        "inverted_hammer": inv_hammer,
        "engulfing_bull": engulf_bull.fillna(False),
        "engulfing_bear": engulf_bear.fillna(False),
    }, index=df.index)


# ─── C33: Distance Helpers ─────────────────────────────────────────────────

def distance_from_level(price: pd.Series, level: pd.Series) -> pd.Series:
    """Absolute distance of price from a reference level, in same units."""
    result = price - level
    result.name = "dist_from_level"
    return result


def distance_from_level_atr(price: pd.Series, level: pd.Series,
                             atr_val: pd.Series) -> pd.Series:
    """Distance normalised by ATR (how many ATRs away)."""
    result = (price - level) / atr_val.replace(0, 1e-10)
    result.name = "dist_atr"
    return result


def distance_from_high(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Distance of current close from rolling high, as fraction of range."""
    rolling_hi = df["high"].rolling(lookback).max()
    rolling_lo = df["low"].rolling(lookback).min()
    rng = (rolling_hi - rolling_lo).replace(0, 1e-10)
    result = (df["close"] - rolling_lo) / rng
    result.name = "dist_from_high"
    return result


# ─── C34: Breakout Validation ──────────────────────────────────────────────

def breakout_quality(df: pd.DataFrame, lookback: int = 20,
                     volume_mult: float = 1.5) -> pd.DataFrame:
    """Score breakout quality: range expansion + volume surge + close near extreme."""
    hi = df["high"].rolling(lookback).max().shift(1)
    lo = df["low"].rolling(lookback).min().shift(1)
    vol_avg = df["volume"].rolling(lookback).mean()

    broke_high = df["close"] > hi
    broke_low = df["close"] < lo
    vol_surge = df["volume"] > vol_avg * volume_mult

    # Close position within bar range (1 = closed at high, 0 = at low)
    bar_rng = (df["high"] - df["low"]).replace(0, 1e-10)
    close_pos = (df["close"] - df["low"]) / bar_rng

    bull_quality = broke_high.astype(float) + vol_surge.astype(float) + close_pos
    bear_quality = broke_low.astype(float) + vol_surge.astype(float) + (1 - close_pos)

    return pd.DataFrame({
        "bull_breakout": broke_high,
        "bear_breakout": broke_low,
        "bull_quality": bull_quality,
        "bear_quality": bear_quality,
    }, index=df.index)


# ─── C35: Liquidity Sweeps ─────────────────────────────────────────────────

def liquidity_sweeps(df: pd.DataFrame, lookback: int = 20,
                     rejection_bars: int = 2) -> pd.DataFrame:
    """Detect potential liquidity sweep patterns:
    Price spikes beyond recent high/low but closes back inside,
    suggesting a stop-hunt / liquidity grab."""
    prev_high = df["high"].rolling(lookback).max().shift(1)
    prev_low = df["low"].rolling(lookback).min().shift(1)

    # Sweep high: wick above prev high but close below
    sweep_high = (df["high"] > prev_high) & (df["close"] < prev_high)
    # Sweep low: wick below prev low but close above
    sweep_low = (df["low"] < prev_low) & (df["close"] > prev_low)

    # Confirm with rejection: next N bars stay inside
    confirmed_high = sweep_high.copy()
    confirmed_low = sweep_low.copy()
    for lag in range(1, rejection_bars + 1):
        confirmed_high = confirmed_high & (df["high"].shift(-lag) < prev_high).fillna(False)
        confirmed_low = confirmed_low & (df["low"].shift(-lag) > prev_low).fillna(False)

    return pd.DataFrame({
        "sweep_high": sweep_high.fillna(False),
        "sweep_low": sweep_low.fillna(False),
        "confirmed_sweep_high": confirmed_high.fillna(False),
        "confirmed_sweep_low": confirmed_low.fillna(False),
    }, index=df.index)
