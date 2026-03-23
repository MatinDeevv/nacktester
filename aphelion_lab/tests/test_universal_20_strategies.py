import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aphelion_lab.backtest_engine import BacktestConfig, BacktestEngine
from aphelion_lab.strategy_runtime import StrategyLoader

ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = ROOT / "aphelion_lab" / "strategies" / "universal_20"
STRATEGY_PATHS = sorted(STRATEGY_DIR.glob("u_*.py"))
FREQUENCIES = [("5min", 420, 11), ("1h", 420, 17), ("1D", 360, 23)]
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


def _make_multi_tf_bars(freq: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2021-01-04 00:00", periods=n, freq=freq)
    block = n // 4
    returns = np.concatenate(
        [
            rng.normal(0.14, 0.32, block),
            rng.normal(-0.05, 0.18, block),
            rng.normal(-0.18, 0.42, block),
            rng.normal(0.06, 0.60, n - (block * 3)),
        ]
    )
    wave = np.sin(np.linspace(0.0, 18.0, n)) * 1.6
    close = 1000.0 + np.cumsum(returns + (wave / max(1, n // 60)))
    open_ = np.concatenate(([close[0]], close[:-1])) + rng.normal(0.0, 0.10, n)
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.75, n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.75, n)
    volume = np.concatenate(
        [
            rng.randint(250, 1000, block),
            rng.randint(120, 650, block),
            rng.randint(700, 2000, block),
            rng.randint(500, 2400, n - (block * 3)),
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


def test_universal_20_strategy_files_exist():
    assert len(STRATEGY_PATHS) == 20


@pytest.mark.parametrize("strategy_path", STRATEGY_PATHS, ids=lambda path: path.stem)
@pytest.mark.parametrize("freq,n,seed", FREQUENCIES, ids=[item[0] for item in FREQUENCIES])
def test_universal_20_strategies_smoke_run_across_timeframes(strategy_path, freq, n, seed, caplog):
    loader = StrategyLoader()
    strategy = loader.load(str(strategy_path))

    assert strategy is not None, loader.error

    data = _make_multi_tf_bars(freq, n, seed)
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
