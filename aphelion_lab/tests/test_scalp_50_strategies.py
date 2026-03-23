import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aphelion_lab.backtest_engine import BacktestConfig, BacktestEngine
from aphelion_lab.strategy_runtime import StrategyLoader

ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = ROOT / "aphelion_lab" / "strategies" / "scalp_50"
STRATEGY_PATHS = sorted(STRATEGY_DIR.glob("s_*.py"))
FREQUENCIES = [("5min", 540, 11), ("15min", 420, 19)]
EXPECTED_COLUMNS = {
    "session",
    "spread",
    "market_regime",
    "volatility_regime",
    "entropy_64",
    "hurst_128",
    "jump_intensity",
    "distribution_shift_norm",
}


def _make_intraday_bars(freq: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-02 00:00", periods=n, freq=freq)
    block = n // 6
    returns = np.concatenate(
        [
            rng.normal(0.12, 0.22, block),
            rng.normal(-0.05, 0.15, block),
            rng.normal(0.20, 0.42, block),
            rng.normal(-0.18, 0.46, block),
            rng.normal(0.03, 0.18, block),
            rng.normal(0.09, 0.64, n - block * 5),
        ]
    )
    drift_wave = np.sin(np.linspace(0.0, 22.0, n)) * 0.18
    impulse = np.where(np.arange(n) % 37 == 0, rng.normal(0.0, 0.9, n), 0.0)
    close = 1950.0 + np.cumsum(returns + drift_wave + impulse)
    open_ = np.concatenate(([close[0]], close[:-1])) + rng.normal(0.0, 0.08, n)
    high = np.maximum(open_, close) + rng.uniform(0.04, 0.65, n)
    low = np.minimum(open_, close) - rng.uniform(0.04, 0.65, n)
    volume = np.concatenate(
        [
            rng.randint(180, 900, block),
            rng.randint(90, 450, block),
            rng.randint(500, 1600, block),
            rng.randint(700, 2400, block),
            rng.randint(120, 700, block),
            rng.randint(300, 2200, n - block * 5),
        ]
    ).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def test_scalp_50_strategy_files_exist():
    assert len(STRATEGY_PATHS) == 50


@pytest.mark.parametrize("strategy_path", STRATEGY_PATHS, ids=lambda path: path.stem)
def test_scalp_50_strategies_are_fixed_rr_three(strategy_path):
    loader = StrategyLoader()
    strategy = loader.load(str(strategy_path))

    assert strategy is not None, loader.error
    assert float(strategy.tp_rr) == pytest.approx(3.0)


@pytest.mark.parametrize("strategy_path", STRATEGY_PATHS, ids=lambda path: path.stem)
@pytest.mark.parametrize("freq,n,seed", FREQUENCIES, ids=[item[0] for item in FREQUENCIES])
def test_scalp_50_strategies_smoke_run_on_m5_and_m15(strategy_path, freq, n, seed, caplog):
    loader = StrategyLoader()
    strategy = loader.load(str(strategy_path))

    assert strategy is not None, loader.error

    data = _make_intraday_bars(freq, n, seed)
    engine = BacktestEngine(
        BacktestConfig(
            auto_enrich_data=True,
            regime_features_enabled=True,
            dynamic_spread_enabled=True,
        )
    )

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        result = engine.run(data, strategy)

    strategy_errors = [
        record.getMessage()
        for record in caplog.records
        if record.name == "aphelion.engine" and "Strategy error at bar" in record.getMessage()
    ]
    assert not strategy_errors, strategy_errors[0]
    assert len(result.equity_curve) == len(data)
    assert EXPECTED_COLUMNS.issubset(result.data.columns)
