import numpy as np
import pandas as pd

from aphelion_lab.backtest_engine import BacktestConfig, BacktestEngine
from aphelion_lab.market_structure import enrich_dataframe
from aphelion_lab.regime_detection import MARKET_REGIMES, VOLATILITY_REGIMES, add_regime_features


def _make_bars(n: int = 320, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-02 00:00", periods=n, freq="15min")
    close = 2000.0 + np.cumsum(rng.randn(n) * 1.7)
    data = pd.DataFrame(
        {
            "open": close + rng.randn(n) * 0.4,
            "high": close + rng.uniform(0.4, 2.4, n),
            "low": close - rng.uniform(0.4, 2.4, n),
            "close": close,
            "volume": rng.randint(100, 10000, n).astype(float),
        },
        index=dates,
    )
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)
    return data


def test_add_regime_features_adds_expected_columns():
    bars = add_regime_features(_make_bars())

    expected = {
        "log_return",
        "realized_vol",
        "entropy_64",
        "hurst_128",
        "jump_score",
        "jump_event",
        "jump_intensity",
        "distribution_shift",
        "distribution_shift_norm",
        "volatility_zscore",
        "volatility_regime",
        "market_regime",
    }
    assert expected.issubset(bars.columns)
    assert set(bars["volatility_regime"].dropna().unique()).issubset(set(VOLATILITY_REGIMES))
    assert set(bars["market_regime"].dropna().unique()).issubset(set(MARKET_REGIMES))
    assert bars["jump_event"].dtype == bool


def test_enrich_dataframe_includes_regime_and_market_structure_features():
    result = enrich_dataframe(_make_bars())

    assert "session" in result.columns
    assert "spread" in result.columns
    assert "entropy_64" in result.columns
    assert "hurst_128" in result.columns
    assert "market_regime" in result.columns
    assert "volatility_regime" in result.columns


def test_engine_auto_enriches_and_exposes_regime_context():
    bars = _make_bars()
    snapshots = []

    class ReaderStrategy:
        def on_bar(self, ctx):
            if ctx.bar_index >= 180 and len(snapshots) < 5:
                snapshots.append(
                    (
                        ctx.session,
                        ctx.market_regime,
                        ctx.volatility_regime,
                        ctx.hurst,
                        ctx.entropy,
                        ctx.jump_intensity,
                        ctx.distribution_shift,
                    )
                )

    engine = BacktestEngine(BacktestConfig(initial_capital=5000))
    result = engine.run(bars, ReaderStrategy())

    assert "market_regime" in result.data.columns
    assert "volatility_regime" in result.data.columns
    assert snapshots
    for session, market_regime, volatility_regime, hurst, entropy, jump_intensity, shift in snapshots:
        assert session in {"asia", "london", "new_york", "overlap", "off_hours"}
        assert market_regime in MARKET_REGIMES
        assert volatility_regime in VOLATILITY_REGIMES
        assert np.isfinite(hurst)
        assert np.isfinite(entropy)
        assert np.isfinite(jump_intensity)
        assert np.isfinite(shift)


def test_engine_can_disable_auto_enrichment():
    bars = _make_bars()

    class NoopStrategy:
        def on_bar(self, ctx):
            return None

    result = BacktestEngine(
        BacktestConfig(initial_capital=5000, auto_enrich_data=False)
    ).run(bars, NoopStrategy())

    assert "market_regime" not in result.data.columns
    assert "session" not in result.data.columns
