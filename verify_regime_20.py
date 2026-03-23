#!/usr/bin/env python3
"""Verify that the Regime 20 strategy pack loads and smoke-runs."""

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

from aphelion_lab.backtest_engine import BacktestConfig, BacktestEngine
from aphelion_lab.strategy_runtime import StrategyLoader

ROOT = Path(__file__).resolve().parent
STRATEGY_DIR = ROOT / "aphelion_lab" / "strategies" / "regime_20"
STRATEGIES = sorted(STRATEGY_DIR.glob("r_*.py"))


def make_regime_bars(n: int = 864, seed: int = 7) -> pd.DataFrame:
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


def main() -> int:
    print("=" * 112)
    print("REGIME 20 PACK CHECK  |  synthetic multi-regime sample with engine auto-enrichment")
    print("=" * 112)
    print(f"{'file':42s} {'trades':>6s} {'net':>10s} {'ret%':>8s} {'wr%':>8s} {'pf':>8s}")
    print("-" * 112)

    data = make_regime_bars()
    success_count = 0

    for path in STRATEGIES:
        loader = StrategyLoader()
        strategy = loader.load(str(path))
        name = path.name
        if strategy is None:
            print(f"{name:42s} LOAD FAIL")
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
            f"{name:42s} {result.total_trades:6d} {float(result.net_pnl):10.2f} "
            f"{float(result.total_return_pct):8.2f} {float(result.win_rate):8.1f} {pf:>8s}"
        )
        success_count += 1

    print("-" * 112)
    print(f"RESULT: {success_count}/{len(STRATEGIES)} strategies loaded and smoke-ran")
    print("=" * 112)
    return 0 if success_count == len(STRATEGIES) else 1


if __name__ == "__main__":
    sys.exit(main())
