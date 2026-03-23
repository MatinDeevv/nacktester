#!/usr/bin/env python3
"""Verify that the Universal 20 strategy pack loads and smoke-runs."""

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

from aphelion_lab.backtest_engine import BacktestConfig, BacktestEngine
from aphelion_lab.strategy_runtime import StrategyLoader

ROOT = Path(__file__).resolve().parent
STRATEGY_DIR = ROOT / "aphelion_lab" / "strategies" / "universal_20"
STRATEGIES = sorted(STRATEGY_DIR.glob("u_*.py"))
FREQUENCIES = [("5min", 420, 11), ("1h", 420, 17), ("1D", 360, 23)]


def make_multi_tf_bars(freq: str, n: int, seed: int) -> pd.DataFrame:
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


def main() -> int:
    print("=" * 118)
    print("UNIVERSAL 20 PACK CHECK  |  synthetic multi-timeframe sample with default engine-compatible settings")
    print("=" * 118)
    print(f"{'freq':6s} {'file':42s} {'trades':>6s} {'net':>10s} {'ret%':>8s} {'wr%':>8s} {'pf':>8s}")
    print("-" * 118)

    success_count = 0
    total_runs = len(STRATEGIES) * len(FREQUENCIES)

    for freq, n, seed in FREQUENCIES:
        data = make_multi_tf_bars(freq, n, seed)
        for path in STRATEGIES:
            loader = StrategyLoader()
            strategy = loader.load(str(path))
            name = path.name
            if strategy is None:
                print(f"{freq:6s} {name:42s} LOAD FAIL")
                continue

            result = BacktestEngine(
                BacktestConfig(
                    initial_capital=5000,
                    auto_enrich_data=True,
                    regime_features_enabled=True,
                    dynamic_spread_enabled=True,
                )
            ).run(data, strategy)
            pf = "inf" if not math.isfinite(result.profit_factor) else f"{float(result.profit_factor):.3f}"
            print(
                f"{freq:6s} {name:42s} {result.total_trades:6d} {float(result.net_pnl):10.2f} "
                f"{float(result.total_return_pct):8.2f} {float(result.win_rate):8.1f} {pf:>8s}"
            )
            success_count += 1

    print("-" * 118)
    print(f"RESULT: {success_count}/{total_runs} strategy-timeframe smoke runs completed")
    print("=" * 118)
    return 0 if success_count == total_runs else 1


if __name__ == "__main__":
    sys.exit(main())
