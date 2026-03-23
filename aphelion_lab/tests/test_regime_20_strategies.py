import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aphelion_lab.backtest_engine import BacktestConfig, BacktestEngine
from aphelion_lab.strategy_runtime import StrategyLoader

ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = ROOT / "aphelion_lab" / "strategies" / "regime_20"
STRATEGY_PATHS = sorted(STRATEGY_DIR.glob("r_*.py"))
EXPECTED_REGIME_COLUMNS = {
    "session",
    "spread",
    "gap",
    "market_regime",
    "volatility_regime",
    "entropy_64",
    "hurst_128",
    "jump_intensity",
    "distribution_shift_norm",
}


def _make_regime_bars(n: int = 864, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-02 00:00", periods=n, freq="5min")

    block = n // 4
    returns = np.concatenate(
        [
            rng.normal(0.18, 0.45, block),
            rng.normal(-0.03, 0.18, block),
            rng.normal(-0.22, 0.55, block),
            rng.normal(0.05, 0.95, n - (block * 3)),
        ]
    )
    close = 1950.0 + np.cumsum(returns)
    open_ = np.concatenate(([close[0]], close[:-1])) + rng.normal(0.0, 0.12, n)
    high = np.maximum(open_, close) + rng.uniform(0.08, 0.90, n)
    low = np.minimum(open_, close) - rng.uniform(0.08, 0.90, n)
    volume = np.concatenate(
        [
            rng.randint(300, 1200, block),
            rng.randint(120, 700, block),
            rng.randint(700, 2200, block),
            rng.randint(900, 3200, n - (block * 3)),
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


def test_regime_20_strategy_files_exist():
    assert len(STRATEGY_PATHS) == 20


@pytest.mark.parametrize("strategy_path", STRATEGY_PATHS, ids=lambda path: path.stem)
def test_regime_20_strategies_load_and_smoke_run(strategy_path, caplog):
    loader = StrategyLoader()
    strategy = loader.load(str(strategy_path))

    assert strategy is not None, loader.error

    engine = BacktestEngine(
        BacktestConfig(
            auto_enrich_data=True,
            regime_features_enabled=True,
            dynamic_spread_enabled=True,
        )
    )
    data = _make_regime_bars()

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
    assert EXPECTED_REGIME_COLUMNS.issubset(result.data.columns)
