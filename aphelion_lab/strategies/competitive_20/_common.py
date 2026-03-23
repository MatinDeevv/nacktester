import pandas as pd
import numpy as np


def tail(series_or_df, n: int):
    return series_or_df.iloc[-n:] if len(series_or_df) > n else series_or_df


def ema(series: pd.Series, period: int) -> float:
    s = tail(series, max(period * 4, period + 5))
    if len(s) < period:
        return float('nan')
    return float(s.ewm(span=period, adjust=False).mean().iloc[-1])


def ema_prev(series: pd.Series, period: int, back: int = 1) -> float:
    s = tail(series, max(period * 4 + back + 5, period + back + 5))
    if len(s) < period + back:
        return float('nan')
    return float(s.ewm(span=period, adjust=False).mean().iloc[-1 - back])


def sma(series: pd.Series, period: int) -> float:
    s = tail(series, period)
    if len(s) < period:
        return float('nan')
    return float(s.mean())


def rolling_std(series: pd.Series, period: int) -> float:
    s = tail(series, period)
    if len(s) < period:
        return float('nan')
    return float(s.std())


def rsi(series: pd.Series, period: int = 14) -> float:
    s = tail(series, max(period * 4, period + 5))
    if len(s) < period + 1:
        return float('nan')
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def rsi_prev(series: pd.Series, period: int = 14, back: int = 1) -> float:
    s = tail(series, max(period * 4 + back + 5, period + back + 5))
    if len(s) < period + 1 + back:
        return float('nan')
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    out = (100 - 100 / (1 + rs))
    return float(out.iloc[-1 - back])


def atr(bars: pd.DataFrame, period: int = 14) -> float:
    b = tail(bars[['high', 'low', 'close']], max(period * 4, period + 5))
    if len(b) < period + 1:
        return float('nan')
    h = b['high'].to_numpy(dtype=float)
    l = b['low'].to_numpy(dtype=float)
    c = b['close'].to_numpy(dtype=float)
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    return float(pd.Series(tr).rolling(period).mean().iloc[-1])


def atr_prev(bars: pd.DataFrame, period: int = 14, back: int = 1) -> float:
    b = tail(bars[['high', 'low', 'close']], max(period * 4 + back + 5, period + back + 5))
    if len(b) < period + 1 + back:
        return float('nan')
    h = b['high'].to_numpy(dtype=float)
    l = b['low'].to_numpy(dtype=float)
    c = b['close'].to_numpy(dtype=float)
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    out = pd.Series(tr).rolling(period).mean()
    return float(out.iloc[-1 - back])


def highest(series: pd.Series, period: int, exclude_current: bool = False) -> float:
    s = series.iloc[:-1] if exclude_current and len(series) > 1 else series
    s = tail(s, period)
    if len(s) < period:
        return float('nan')
    return float(s.max())


def lowest(series: pd.Series, period: int, exclude_current: bool = False) -> float:
    s = series.iloc[:-1] if exclude_current and len(series) > 1 else series
    s = tail(s, period)
    if len(s) < period:
        return float('nan')
    return float(s.min())


def zscore(series: pd.Series, period: int = 20) -> float:
    s = tail(series, period)
    if len(s) < period:
        return float('nan')
    sd = float(s.std())
    if sd == 0:
        return 0.0
    return float((s.iloc[-1] - s.mean()) / sd)


def stochastic_k(bars: pd.DataFrame, period: int = 14) -> float:
    b = tail(bars[['high', 'low', 'close']], period)
    if len(b) < period:
        return float('nan')
    hh = float(b['high'].max())
    ll = float(b['low'].min())
    if hh == ll:
        return 50.0
    return float((b['close'].iloc[-1] - ll) / (hh - ll) * 100)


def stochastic_k_prev(bars: pd.DataFrame, period: int = 14, back: int = 1) -> float:
    b = bars.iloc[:-back] if len(bars) > back else bars.iloc[:0]
    if len(b) < period:
        return float('nan')
    return stochastic_k(b, period)


def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = tail(series, max(slow * 4 + signal * 2, slow + signal + 10))
    if len(s) < slow + signal:
        return float('nan'), float('nan')
    fast_ema = s.ewm(span=fast, adjust=False).mean()
    slow_ema = s.ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return float(hist.iloc[-1]), float(hist.iloc[-2])


def crossed_above(prev_a, curr_a, prev_b, curr_b) -> bool:
    return prev_a <= prev_b and curr_a > curr_b


def crossed_below(prev_a, curr_a, prev_b, curr_b) -> bool:
    return prev_a >= prev_b and curr_a < curr_b


def candle_metrics(bar):
    rng = float(bar['high'] - bar['low'])
    body = float(abs(bar['close'] - bar['open']))
    upper = float(bar['high'] - max(bar['open'], bar['close']))
    lower = float(min(bar['open'], bar['close']) - bar['low'])
    return rng, body, upper, lower


def session_slice(bars: pd.DataFrame, start_hour: int, end_hour: int, same_day_only: bool = True):
    if len(bars) == 0:
        return bars.iloc[:0]
    idx = bars.index
    hours = idx.hour
    mask = (hours >= start_hour) & (hours < end_hour)
    if same_day_only:
        current_date = idx[-1].date()
        mask &= (idx.normalize() == pd.Timestamp(current_date, tz=idx.tz))
    return bars.loc[mask]


def daily_vwap(bars: pd.DataFrame) -> float:
    if len(bars) == 0:
        return float('nan')
    idx = bars.index
    day = bars.loc[idx.normalize() == idx[-1].normalize()]
    if len(day) == 0:
        return float('nan')
    vol = day['volume'].to_numpy(dtype=float)
    denom = float(vol.sum())
    if denom <= 0:
        return float(day['close'].mean())
    typical = ((day['high'] + day['low'] + day['close']) / 3.0).to_numpy(dtype=float)
    return float((typical * vol).sum() / denom)


def liquid_hours(ts) -> bool:
    return 6 <= ts.hour <= 20


def enough(value) -> bool:
    return value == value and np.isfinite(value)
