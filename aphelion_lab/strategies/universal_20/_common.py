from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import (
    adx as adx_indicator,
    atr_bands as atr_bands_indicator,
    cci as cci_indicator,
    cmf as cmf_indicator,
    keltner as keltner_indicator,
    macd as macd_indicator,
    mfi as mfi_indicator,
    obv as obv_indicator,
    parabolic_sar as sar_indicator,
    roc as roc_indicator,
    stoch_rsi as stoch_rsi_indicator,
    supertrend as supertrend_indicator,
)


def tail(series_or_df, n: int):
    return series_or_df.iloc[-n:] if len(series_or_df) > n else series_or_df


def enough(*values) -> bool:
    return all(v == v and np.isfinite(v) for v in values)


def ema(series: pd.Series, period: int) -> float:
    s = tail(series, max(period * 4, period + 5))
    if len(s) < period:
        return float("nan")
    return float(s.ewm(span=period, adjust=False).mean().iloc[-1])


def ema_prev(series: pd.Series, period: int, back: int = 1) -> float:
    s = tail(series, max(period * 4 + back + 5, period + back + 5))
    if len(s) < period + back:
        return float("nan")
    return float(s.ewm(span=period, adjust=False).mean().iloc[-1 - back])


def sma(series: pd.Series, period: int) -> float:
    s = tail(series, period)
    if len(s) < period:
        return float("nan")
    return float(s.mean())


def atr(bars: pd.DataFrame, period: int = 14) -> float:
    b = tail(bars[["high", "low", "close"]], max(period * 4, period + 5))
    if len(b) < period + 1:
        return float("nan")
    h = b["high"].to_numpy(dtype=float)
    l = b["low"].to_numpy(dtype=float)
    c = b["close"].to_numpy(dtype=float)
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    return float(pd.Series(tr).rolling(period).mean().iloc[-1])


def rsi(series: pd.Series, period: int = 14) -> float:
    s = tail(series, max(period * 4, period + 5))
    if len(s) < period + 1:
        return float("nan")
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def highest(series: pd.Series, period: int, exclude_current: bool = False) -> float:
    s = series.iloc[:-1] if exclude_current and len(series) > 1 else series
    s = tail(s, period)
    if len(s) < period:
        return float("nan")
    return float(s.max())


def lowest(series: pd.Series, period: int, exclude_current: bool = False) -> float:
    s = series.iloc[:-1] if exclude_current and len(series) > 1 else series
    s = tail(s, period)
    if len(s) < period:
        return float("nan")
    return float(s.min())


def zscore(series: pd.Series, period: int = 20) -> float:
    s = tail(series, period)
    if len(s) < period:
        return float("nan")
    sd = float(s.std())
    if sd == 0:
        return 0.0
    return float((s.iloc[-1] - s.mean()) / sd)


def volume_ratio(bars: pd.DataFrame, period: int = 20) -> float:
    if len(bars) < period + 1:
        return float("nan")
    hist = bars["volume"].iloc[-period - 1:-1]
    mean_vol = float(hist.mean())
    if mean_vol <= 0:
        return float("nan")
    return float(bars["volume"].iloc[-1] / mean_vol)


def range_ratio(bars: pd.DataFrame, period: int = 20) -> float:
    if len(bars) < period + 1:
        return float("nan")
    hist = (bars["high"] - bars["low"]).iloc[-period - 1:-1]
    mean_rng = float(hist.mean())
    if mean_rng <= 0:
        return float("nan")
    current_rng = float(bars["high"].iloc[-1] - bars["low"].iloc[-1])
    return current_rng / mean_rng


def candle_metrics(bar: pd.Series) -> tuple[float, float, float, float]:
    rng = float(bar["high"] - bar["low"])
    body = float(abs(bar["close"] - bar["open"]))
    upper = float(bar["high"] - max(bar["open"], bar["close"]))
    lower = float(min(bar["open"], bar["close"]) - bar["low"])
    return rng, body, upper, lower


def body_ratio(bar: pd.Series) -> float:
    rng, body, _, _ = candle_metrics(bar)
    if rng <= 0:
        return 0.0
    return body / rng


def spread_ok(ctx, max_spread: float = 0.12) -> bool:
    return ctx.spread <= max_spread


def trend_regime(ctx) -> bool:
    return (
        ctx.market_regime in ("trend", "transition")
        and ctx.hurst >= 0.50
        and ctx.entropy <= 0.84
        and ctx.jump_intensity < 0.30
    )


def range_regime(ctx) -> bool:
    return (
        ctx.market_regime in ("range", "mean_revert")
        or (ctx.hurst < 0.50 and ctx.entropy >= 0.70 and ctx.jump_intensity < 0.20)
    )


def current_side(ctx) -> str:
    if ctx.position is None:
        return ""
    return ctx.position.trade.side.value


def entry_size(ctx, price: float, sl: float, floor: float = 0.01, cap: float = 0.05) -> float:
    risk = abs(price - sl)
    if not enough(risk) or risk <= 0:
        return floor
    try:
        size = float(ctx.calc_size(risk))
    except Exception:
        size = floor
    if not enough(size) or size <= 0:
        size = floor
    return float(min(cap, max(floor, size)))


def open_trade(ctx, side: str, sl: float, tp: float | None = None, floor: float = 0.01, cap: float = 0.05):
    price = float(ctx.bar["close"])
    if not enough(price, sl):
        return
    size = entry_size(ctx, price, sl, floor=floor, cap=cap)
    if side == "BUY":
        ctx.buy(size=size, sl=sl, tp=tp)
    else:
        ctx.sell(size=size, sl=sl, tp=tp)


def trail_to_level(ctx, level: float, buffer: float = 0.0):
    if ctx.position is None or not enough(level):
        return
    side = current_side(ctx)
    price = float(ctx.bar["close"])
    current_sl = ctx.position.trade.sl
    if side == "BUY":
        candidate = min(price - buffer, level)
        if enough(candidate) and candidate < price and (current_sl is None or candidate > current_sl):
            ctx.modify_sl(candidate)
    elif side == "SELL":
        candidate = max(price + buffer, level)
        if enough(candidate) and candidate > price and (current_sl is None or candidate < current_sl):
            ctx.modify_sl(candidate)


def tighten_tp(ctx, level: float):
    if ctx.position is None or not enough(level):
        return
    side = current_side(ctx)
    price = float(ctx.bar["close"])
    current_tp = ctx.position.trade.tp
    if side == "BUY":
        if level > price and (current_tp is None or level < current_tp):
            ctx.modify_tp(level)
    elif side == "SELL":
        if level < price and (current_tp is None or level > current_tp):
            ctx.modify_tp(level)


def risk_target(price: float, sl: float, tp_rr: float, side: str) -> float:
    risk = abs(price - sl)
    return price + risk * tp_rr if side == "BUY" else price - risk * tp_rr


def _indicator_bars(bars: pd.DataFrame, n: int = 220) -> pd.DataFrame:
    cols = [c for c in ("open", "high", "low", "close", "volume", "session") if c in bars.columns]
    return tail(bars[cols], n).copy()


def adx_state(bars: pd.DataFrame, period: int = 14, back: int = 0) -> tuple[float, float, float]:
    b = _indicator_bars(bars, max(period * 6 + back + 10, 120))
    if len(b) < period + back + 5:
        return float("nan"), float("nan"), float("nan")
    result = adx_indicator(b[["high", "low", "close"]], period)
    row = result.iloc[-1 - back]
    return float(row["plus_di"]), float(row["minus_di"]), float(row["adx"])


def macd_state(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, back: int = 0) -> tuple[float, float, float]:
    s = tail(series, max(slow * 6 + signal + back + 10, 150))
    if len(s) < slow + signal + back + 5:
        return float("nan"), float("nan"), float("nan")
    result = macd_indicator(s, fast=fast, slow=slow, signal=signal)
    row = result.iloc[-1 - back]
    return float(row["macd"]), float(row["signal"]), float(row["histogram"])


def keltner_levels(bars: pd.DataFrame, ema_period: int = 20, atr_period: int = 14, mult: float = 1.8, back: int = 0):
    b = _indicator_bars(bars, max(ema_period * 6 + atr_period + back + 10, 160))
    if len(b) < max(ema_period, atr_period) + back + 5:
        return float("nan"), float("nan"), float("nan")
    result = keltner_indicator(b[["high", "low", "close"]], ema_period=ema_period, atr_period=atr_period, mult=mult)
    row = result.iloc[-1 - back]
    return float(row["kc_upper"]), float(row["kc_mid"]), float(row["kc_lower"])


def supertrend_state(bars: pd.DataFrame, period: int = 10, mult: float = 3.0, back: int = 0):
    b = _indicator_bars(bars, max(period * 8 + back + 10, 160))
    if len(b) < period + back + 10:
        return float("nan"), float("nan")
    result = supertrend_indicator(b[["high", "low", "close"]], period=period, mult=mult)
    row = result.iloc[-1 - back]
    return float(row["supertrend"]), float(row["st_direction"])


def stoch_state(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14, back: int = 0):
    s = tail(series, max((rsi_period + stoch_period) * 6 + back + 10, 180))
    if len(s) < rsi_period + stoch_period + back + 10:
        return float("nan"), float("nan")
    result = stoch_rsi_indicator(s, rsi_period=rsi_period, stoch_period=stoch_period)
    row = result.iloc[-1 - back]
    return float(row["stoch_rsi_k"]), float(row["stoch_rsi_d"])


def cci_value(bars: pd.DataFrame, period: int = 20, back: int = 0) -> float:
    b = _indicator_bars(bars, max(period * 6 + back + 10, 140))
    if len(b) < period + back + 5:
        return float("nan")
    result = cci_indicator(b[["high", "low", "close"]], period=period)
    return float(result.iloc[-1 - back])


def roc_value(series: pd.Series, period: int = 12, back: int = 0) -> float:
    s = tail(series, max(period * 6 + back + 10, 100))
    if len(s) < period + back + 5:
        return float("nan")
    result = roc_indicator(s, period=period)
    return float(result.iloc[-1 - back])


def obv_state(bars: pd.DataFrame, fast: int = 10, slow: int = 30):
    b = _indicator_bars(bars, max(slow * 6 + 20, 180))
    if len(b) < slow + 5:
        return float("nan"), float("nan"), float("nan"), float("nan")
    series = obv_indicator(b[["close", "volume"]])
    current = float(series.iloc[-1])
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    slope = float(series.iloc[-1] - series.iloc[-6]) if len(series) >= 6 else float("nan")
    return current, fast_ema, slow_ema, slope


def mfi_value(bars: pd.DataFrame, period: int = 14, back: int = 0) -> float:
    b = _indicator_bars(bars, max(period * 6 + back + 10, 120))
    if len(b) < period + back + 5:
        return float("nan")
    result = mfi_indicator(b[["high", "low", "close", "volume"]], period=period)
    return float(result.iloc[-1 - back])


def cmf_value(bars: pd.DataFrame, period: int = 20, back: int = 0) -> float:
    b = _indicator_bars(bars, max(period * 6 + back + 10, 140))
    if len(b) < period + back + 5:
        return float("nan")
    result = cmf_indicator(b[["high", "low", "close", "volume"]], period=period)
    return float(result.iloc[-1 - back])


def sar_state(bars: pd.DataFrame, back: int = 0):
    b = _indicator_bars(bars, 180)
    if len(b) < back + 8:
        return float("nan"), float("nan")
    result = sar_indicator(b[["high", "low", "close"]])
    row = result.iloc[-1 - back]
    return float(row["sar"]), float(row["sar_direction"])


def atr_band_levels(bars: pd.DataFrame, period: int = 14, mult: float = 2.0, back: int = 0):
    b = _indicator_bars(bars, max(period * 6 + back + 10, 140))
    if len(b) < period + back + 5:
        return float("nan"), float("nan"), float("nan")
    result = atr_bands_indicator(b[["high", "low", "close"]], period=period, mult=mult)
    row = result.iloc[-1 - back]
    return float(row["atr_upper"]), float(row["atr_mid"]), float(row["atr_lower"])
