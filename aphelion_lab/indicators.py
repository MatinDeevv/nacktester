"""
Aphelion Lab — Indicators
Vectorized indicator library for the backtest engine.
All functions accept pd.Series or pd.DataFrame and return values/Series.
Designed for one-shot vectorized computation on the full dataset, then
lookup by index during bar-by-bar iteration.
"""

import numpy as np
import pandas as pd


# ─── B11: VWAP ──────────────────────────────────────────────────────────────

def vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP, reset at each calendar day boundary.
    Assumes UTC DatetimeIndex and columns: high, low, close, volume."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)
    cum_tp_vol = (typical * vol).groupby(df.index.date).cumsum()
    cum_vol = vol.groupby(df.index.date).cumsum()
    result = cum_tp_vol / cum_vol
    result = result.fillna(typical)  # fallback when volume is 0
    result.name = "vwap"
    return result


# ─── B12: Anchored VWAP ────────────────────────────────────────────────────

def anchored_vwap(df: pd.DataFrame, anchor: str = "day") -> pd.Series:
    """VWAP anchored to session/day/week/custom timestamp.
    anchor: 'day', 'week', 'session_asia', 'session_london', 'session_ny',
            or a timestamp string."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)
    tp_vol = typical * vol

    if anchor == "day":
        grouper = df.index.date
    elif anchor == "week":
        grouper = df.index.isocalendar().week.values
    elif anchor.startswith("session_"):
        # Group by session start boundaries
        if "session" in df.columns:
            target = anchor.replace("session_", "")
            session_map = {"asia": "asia", "london": "london", "ny": "new_york"}
            sess = session_map.get(target, target)
            is_target = df["session"] == sess
            grouper = is_target.ne(is_target.shift()).cumsum()
        else:
            grouper = df.index.date  # fallback
    else:
        # Custom timestamp anchor: single anchor point
        anchor_ts = pd.Timestamp(anchor)
        mask = df.index >= anchor_ts
        cum_tp = tp_vol[mask].cumsum()
        cum_v = vol[mask].cumsum()
        result = pd.Series(np.nan, index=df.index, name="avwap")
        result[mask] = cum_tp / cum_v
        return result.fillna(method=None)

    cum_tp = tp_vol.groupby(grouper).cumsum()
    cum_v = vol.groupby(grouper).cumsum()
    result = cum_tp / cum_v
    result = result.fillna(typical)
    result.name = "avwap"
    return result


# ─── B13: MACD ──────────────────────────────────────────────────────────────

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram) as pd.DataFrame."""
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": hist},
                        index=series.index)


# ─── B14: ADX / DMI ─────────────────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Return DataFrame with columns: plus_di, minus_di, adx."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
                   axis=1).max(axis=1)

    atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_s
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_val}, index=df.index)


# ─── B15: Donchian Channels ────────────────────────────────────────────────

def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Return upper, mid, lower Donchian channels."""
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    mid = (upper + lower) / 2.0
    return pd.DataFrame({"dc_upper": upper, "dc_mid": mid, "dc_lower": lower}, index=df.index)


# ─── B16: Keltner Channels ─────────────────────────────────────────────────

def keltner(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14,
            mult: float = 2.0) -> pd.DataFrame:
    """Return upper, mid, lower Keltner channels."""
    mid = df["close"].ewm(span=ema_period, adjust=False).mean()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(span=atr_period, adjust=False).mean()
    return pd.DataFrame({
        "kc_upper": mid + mult * atr_s,
        "kc_mid": mid,
        "kc_lower": mid - mult * atr_s,
    }, index=df.index)


# ─── B17: Supertrend ───────────────────────────────────────────────────────

def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Return supertrend value and direction (+1 up, -1 down)."""
    hl2 = (df["high"] + df["low"]) / 2.0
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(span=period, adjust=False).mean()

    up = hl2 - mult * atr_s
    dn = hl2 + mult * atr_s
    close = df["close"].values
    st = np.empty(len(df))
    direction = np.ones(len(df))
    st[0] = up.iloc[0]

    up_arr = up.values
    dn_arr = dn.values

    for i in range(1, len(df)):
        if close[i - 1] > st[i - 1]:
            # was uptrend
            st[i] = max(up_arr[i], st[i - 1]) if close[i] > up_arr[i] else dn_arr[i]
        else:
            st[i] = min(dn_arr[i], st[i - 1]) if close[i] < dn_arr[i] else up_arr[i]
        direction[i] = 1.0 if close[i] > st[i] else -1.0

    return pd.DataFrame({"supertrend": st, "st_direction": direction}, index=df.index)


# ─── B18: Stochastic RSI ───────────────────────────────────────────────────

def stoch_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
              k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    """Return %K and %D of Stochastic RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi_s = 100.0 - 100.0 / (1.0 + rs)

    rsi_min = rsi_s.rolling(stoch_period).min()
    rsi_max = rsi_s.rolling(stoch_period).max()
    rng = (rsi_max - rsi_min).replace(0, 1e-10)
    k = ((rsi_s - rsi_min) / rng * 100).rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return pd.DataFrame({"stoch_rsi_k": k, "stoch_rsi_d": d}, index=series.index)


# ─── B19: CCI ──────────────────────────────────────────────────────────────

def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    result = (tp - sma) / (0.015 * mad.replace(0, 1e-10))
    result.name = "cci"
    return result


# ─── B20: ROC / Momentum ───────────────────────────────────────────────────

def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change (percentage)."""
    shifted = series.shift(period)
    result = (series - shifted) / shifted.replace(0, 1e-10) * 100
    result.name = "roc"
    return result


def momentum(series: pd.Series, period: int = 12) -> pd.Series:
    """Absolute momentum (price difference)."""
    result = series - series.shift(period)
    result.name = "momentum"
    return result


# ─── B21: OBV ──────────────────────────────────────────────────────────────

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    sign = np.sign(df["close"].diff())
    sign.iloc[0] = 0
    result = (sign * df["volume"]).cumsum()
    result.name = "obv"
    return result


# ─── B22: MFI ──────────────────────────────────────────────────────────────

def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]
    pos_mf = pd.Series(np.where(tp > tp.shift(), mf, 0.0), index=df.index)
    neg_mf = pd.Series(np.where(tp < tp.shift(), mf, 0.0), index=df.index)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    ratio = pos_sum / neg_sum.replace(0, 1e-10)
    result = 100.0 - 100.0 / (1.0 + ratio)
    result.name = "mfi"
    return result


# ─── B23: CMF ──────────────────────────────────────────────────────────────

def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    hl_range = (df["high"] - df["low"]).replace(0, 1e-10)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
    mfv = clv * df["volume"]
    result = mfv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, 1e-10)
    result.name = "cmf"
    return result


# ─── B24: Parabolic SAR ────────────────────────────────────────────────────

def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02,
                  af_max: float = 0.20) -> pd.DataFrame:
    """Parabolic SAR. Returns sar value and direction (+1/-1)."""
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    sar = np.empty(n)
    direction = np.ones(n)
    af = af_start
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if direction[i - 1] == 1:  # uptrend
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1])
            if i >= 2:
                sar[i] = min(sar[i], low[i - 2])
            if low[i] < sar[i]:
                direction[i] = -1
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                direction[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:  # downtrend
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high[i - 1])
            if i >= 2:
                sar[i] = max(sar[i], high[i - 2])
            if high[i] > sar[i]:
                direction[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                direction[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return pd.DataFrame({"sar": sar, "sar_direction": direction}, index=df.index)


# ─── B25: ATR Bands / ATR Channels ─────────────────────────────────────────

def atr_bands(df: pd.DataFrame, period: int = 14, mult: float = 2.0,
              source: str = "ema", source_period: int = 20) -> pd.DataFrame:
    """ATR-based envelope/channels around an EMA or SMA midline."""
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.rolling(period).mean()
    if source == "ema":
        mid = df["close"].ewm(span=source_period, adjust=False).mean()
    else:
        mid = df["close"].rolling(source_period).mean()
    return pd.DataFrame({
        "atr_upper": mid + mult * atr_s,
        "atr_mid": mid,
        "atr_lower": mid - mult * atr_s,
    }, index=df.index)


# ─── Helper: True Range series (reusable) ──────────────────────────────────

def true_range(df: pd.DataFrame) -> pd.Series:
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    tr.name = "tr"
    return tr


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    result = true_range(df).rolling(period).mean()
    result.name = "atr"
    return result
