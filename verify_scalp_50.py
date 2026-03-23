#!/usr/bin/env python3
"""Verify that the Scalp 50 strategy pack loads and smoke-runs on M5/M15 data."""

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

from aphelion_lab.backtest_engine import BacktestConfig, BacktestEngine
from aphelion_lab.strategy_runtime import StrategyLoader

ROOT = Path(__file__).resolve().parent
STRATEGY_DIR = ROOT / "aphelion_lab" / "strategies" / "scalp_50"
STRATEGIES = sorted(STRATEGY_DIR.glob("s_*.py"))
FREQUENCIES = [("5min", 540, 11), ("15min", 420, 19)]


def make_intraday_bars(freq: str, n: int, seed: int) -> pd.DataFrame:
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


def main() -> int:
    print("=" * 118)
    print("SCALP 50 PACK CHECK  |  synthetic 5min and 15min sample with fixed 1:3 risk-reward strategies")
    print("=" * 118)
    print(f"{'freq':6s} {'file':44s} {'rr':>5s} {'trades':>6s} {'net':>10s} {'ret%':>8s} {'pf':>8s}")
    print("-" * 118)

    success_count = 0
    total_runs = len(STRATEGIES) * len(FREQUENCIES)

    for freq, n, seed in FREQUENCIES:
        data = make_intraday_bars(freq, n, seed)
        for path in STRATEGIES:
            loader = StrategyLoader()
            strategy = loader.load(str(path))
            name = path.name
            if strategy is None:
                print(f"{freq:6s} {name:44s} LOAD FAIL")
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
                f"{freq:6s} {name:44s} {float(strategy.tp_rr):5.2f} {result.total_trades:6d} "
                f"{float(result.net_pnl):10.2f} {float(result.total_return_pct):8.2f} {pf:>8s}"
            )
            success_count += 1

    print("-" * 118)
    print(f"RESULT: {success_count}/{total_runs} strategy-timeframe smoke runs completed")
    print("=" * 118)
    return 0 if success_count == total_runs else 1


if __name__ == "__main__":
    sys.exit(main())
