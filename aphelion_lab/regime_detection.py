"""
Rolling market-state and regime features for backtest data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


VOLATILITY_REGIMES = ("low", "normal", "high", "extreme")
MARKET_REGIMES = ("range", "trend", "mean_revert", "transition", "stress")


def rolling_shannon_entropy(series: pd.Series, window: int = 64, bins: int = 8) -> pd.Series:
    values = series.astype(float).to_numpy()
    out = np.full(len(values), np.nan, dtype=float)
    if window <= 1:
        return pd.Series(out, index=series.index, name="entropy")

    norm = math.log2(max(bins, 2))
    for idx in range(window - 1, len(values)):
        sample = values[idx - window + 1:idx + 1]
        sample = sample[np.isfinite(sample)]
        if sample.size < max(8, bins):
            continue
        lo = float(sample.min())
        hi = float(sample.max())
        if math.isclose(lo, hi):
            out[idx] = 0.0
            continue
        counts, _ = np.histogram(sample, bins=bins, range=(lo, hi))
        total = counts.sum()
        if total <= 0:
            continue
        probs = counts[counts > 0] / total
        out[idx] = float(-(probs * np.log2(probs)).sum() / norm)
    return pd.Series(out, index=series.index, name=f"entropy_{window}")


def rolling_hurst_exponent(
    series: pd.Series,
    window: int = 128,
    min_lag: int = 2,
    max_lag: int = 20,
) -> pd.Series:
    values = series.astype(float).to_numpy()
    out = np.full(len(values), np.nan, dtype=float)
    if window <= max_lag + 2:
        return pd.Series(out, index=series.index, name="hurst")

    lags = np.arange(min_lag, max_lag + 1, dtype=int)
    for idx in range(window - 1, len(values)):
        sample = values[idx - window + 1:idx + 1]
        sample = sample[np.isfinite(sample)]
        if sample.size <= max_lag + 1:
            continue
        tau = []
        valid_lags = []
        for lag in lags:
            diff = sample[lag:] - sample[:-lag]
            sigma = float(np.std(diff))
            if sigma > 0:
                tau.append(sigma)
                valid_lags.append(lag)
        if len(tau) < 3:
            continue
        slope, _ = np.polyfit(np.log(valid_lags), np.log(tau), 1)
        out[idx] = float(np.clip(slope, 0.0, 1.0))
    return pd.Series(out, index=series.index, name=f"hurst_{window}")


def rolling_wasserstein_shift(
    series: pd.Series,
    window: int = 64,
    reference_window: int | None = None,
) -> pd.Series:
    values = series.astype(float).to_numpy()
    ref_window = reference_window or window
    out = np.full(len(values), np.nan, dtype=float)
    start = window + ref_window - 1
    if window <= 1 or ref_window <= 1:
        return pd.Series(out, index=series.index, name="distribution_shift")

    for idx in range(start, len(values)):
        ref = values[idx - window - ref_window + 1:idx - window + 1]
        cur = values[idx - window + 1:idx + 1]
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]
        if ref.size < 4 or cur.size < 4:
            continue
        out[idx] = _wasserstein_1d(ref, cur)

    return pd.Series(out, index=series.index, name=f"distribution_shift_{window}")


def add_regime_features(
    df: pd.DataFrame,
    entropy_window: int = 64,
    hurst_window: int = 128,
    shift_window: int = 64,
    vol_window: int = 32,
    jump_window: int = 64,
    jump_sigma: float = 2.5,
) -> pd.DataFrame:
    close = df["close"].astype(float)
    log_close = np.log(close.replace(0, np.nan))
    log_return = log_close.diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    realized_vol = log_return.rolling(vol_window, min_periods=max(5, vol_window // 2)).std()
    entropy = rolling_shannon_entropy(log_return, window=entropy_window)
    hurst = rolling_hurst_exponent(close, window=hurst_window)
    distribution_shift = rolling_wasserstein_shift(log_return, window=shift_window)

    vol_floor = realized_vol.replace(0, np.nan)
    jump_score = (log_return.abs() / vol_floor).replace([np.inf, -np.inf], np.nan)
    jump_event = (jump_score > jump_sigma).fillna(False)
    jump_intensity = jump_event.astype(float).rolling(
        jump_window,
        min_periods=max(5, jump_window // 4),
    ).mean()

    shift_scale = realized_vol.rolling(shift_window, min_periods=max(5, shift_window // 2)).mean()
    normalized_shift = (distribution_shift / shift_scale.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    vol_baseline = realized_vol.rolling(hurst_window, min_periods=max(10, hurst_window // 2)).mean()
    vol_std = realized_vol.rolling(hurst_window, min_periods=max(10, hurst_window // 2)).std()
    vol_zscore = ((realized_vol - vol_baseline) / vol_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    df["log_return"] = log_return
    df["realized_vol"] = realized_vol
    df[f"entropy_{entropy_window}"] = entropy
    df[f"hurst_{hurst_window}"] = hurst
    df["jump_score"] = jump_score
    df["jump_event"] = jump_event.astype(bool)
    df["jump_intensity"] = jump_intensity
    df["distribution_shift"] = distribution_shift
    df["distribution_shift_norm"] = normalized_shift
    df["volatility_zscore"] = vol_zscore
    df["volatility_regime"] = _classify_volatility_regime(vol_zscore)
    df["market_regime"] = _classify_market_regime(
        entropy=entropy,
        hurst=hurst,
        shift_norm=normalized_shift,
        jump_intensity=jump_intensity,
        vol_regime=df["volatility_regime"],
    )
    return df


def _classify_volatility_regime(vol_zscore: pd.Series) -> pd.Series:
    regime = pd.Series("normal", index=vol_zscore.index, dtype=object)
    regime = regime.mask(vol_zscore <= -0.75, "low")
    regime = regime.mask((vol_zscore > 0.75) & (vol_zscore <= 1.5), "high")
    regime = regime.mask(vol_zscore > 1.5, "extreme")
    return regime


def _classify_market_regime(
    entropy: pd.Series,
    hurst: pd.Series,
    shift_norm: pd.Series,
    jump_intensity: pd.Series,
    vol_regime: pd.Series,
) -> pd.Series:
    regime = pd.Series("range", index=entropy.index, dtype=object)
    regime = regime.mask(
        (vol_regime == "extreme") | (jump_intensity.fillna(0.0) >= 0.18) | (shift_norm.fillna(0.0) >= 1.8),
        "stress",
    )
    regime = regime.mask(
        (regime == "range") & (shift_norm.fillna(0.0) >= 1.1),
        "transition",
    )
    regime = regime.mask(
        (regime == "range") & (hurst >= 0.58) & (entropy <= 0.72),
        "trend",
    )
    regime = regime.mask(
        (regime == "range") & (hurst <= 0.44) & (entropy <= 0.80),
        "mean_revert",
    )
    return regime


def _wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    if left.size == right.size:
        return float(np.mean(np.abs(left - right)))

    quantiles = np.linspace(0.0, 1.0, max(left.size, right.size))
    left_q = np.quantile(left, quantiles)
    right_q = np.quantile(right, quantiles)
    return float(np.mean(np.abs(left_q - right_q)))
