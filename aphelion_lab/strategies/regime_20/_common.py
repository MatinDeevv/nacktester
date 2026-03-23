from __future__ import annotations

import numpy as np
import pandas as pd


def tail(series_or_df, n: int):
    return series_or_df.iloc[-n:] if len(series_or_df) > n else series_or_df


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


def daily_vwap(bars: pd.DataFrame) -> float:
    if len(bars) == 0:
        return float("nan")
    current_day = bars.index[-1].normalize()
    day = bars.loc[bars.index.normalize() == current_day]
    if len(day) == 0:
        return float("nan")
    vol = day["volume"].to_numpy(dtype=float)
    denom = float(vol.sum())
    if denom <= 0:
        return float(day["close"].mean())
    typical = ((day["high"] + day["low"] + day["close"]) / 3.0).to_numpy(dtype=float)
    return float((typical * vol).sum() / denom)


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


def session_slice(bars: pd.DataFrame, start_hour: int, end_hour: int, same_day_only: bool = True):
    if len(bars) == 0:
        return bars.iloc[:0]
    idx = bars.index
    mask = (idx.hour >= start_hour) & (idx.hour < end_hour)
    if same_day_only:
        mask &= idx.normalize() == idx[-1].normalize()
    return bars.loc[mask]


def enough(*values) -> bool:
    return all(v == v and np.isfinite(v) for v in values)


def market_is(ctx, *names: str) -> bool:
    return ctx.market_regime in names


def volatility_is(ctx, *names: str) -> bool:
    return ctx.volatility_regime in names


def session_is(ctx, *names: str) -> bool:
    return ctx.session in names


def allow_trend(ctx) -> bool:
    return market_is(ctx, "trend", "transition") and ctx.entropy < 0.80 and ctx.jump_intensity < 0.22


def allow_reversion(ctx) -> bool:
    return market_is(ctx, "range", "mean_revert") and ctx.jump_intensity < 0.18


def spread_ok(ctx, max_spread: float = 0.08) -> bool:
    return ctx.spread <= max_spread


def current_side(ctx) -> str:
    if ctx.position is None:
        return ""
    return ctx.position.trade.side.value


def entry_size(ctx, price: float, sl: float, floor: float = 0.01, cap: float = 0.04) -> float:
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


def open_trade(ctx, side: str, sl: float, tp: float | None = None, floor: float = 0.01, cap: float = 0.04):
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


def risk_target(price: float, sl: float, tp_rr: float, side: str) -> float:
    risk = abs(price - sl)
    return price + risk * tp_rr if side == "BUY" else price - risk * tp_rr
