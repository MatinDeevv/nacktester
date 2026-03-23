#!/usr/bin/env python3
"""Smoke-test the backtest engine with a packaged strategy."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from aphelion_lab.backtest_engine import BacktestConfig, BacktestEngine
from aphelion_lab.strategy_runtime import StrategyLoader

ROOT = Path(__file__).resolve().parent
STRATEGY_PATH = ROOT / "aphelion_lab" / "strategies" / "st_01_sma_crossover.py"


def _build_sample_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    closes = 100 + rng.standard_normal(100).cumsum()
    data = pd.DataFrame(
        {
            "open": closes + rng.standard_normal(100) * 0.5,
            "high": closes + np.abs(rng.standard_normal(100) * 0.8),
            "low": closes - np.abs(rng.standard_normal(100) * 0.8),
            "close": closes,
            "volume": rng.integers(1000, 10000, 100),
        },
        index=dates,
    )
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)
    return data


def main() -> int:
    loader = StrategyLoader()
    strategy = loader.load(str(STRATEGY_PATH))
    if strategy is None:
        print(f"[ERROR] Failed to load strategy: {loader.error}", file=sys.stderr)
        return 1

    data = _build_sample_data()
    engine = BacktestEngine(BacktestConfig(initial_capital=5000))
    result = engine.run(data, strategy)

    print("[OK] Backtest ran successfully")
    print(f"[OK] Trades: {result.total_trades}")
    print(f"[OK] P&L: ${result.net_pnl:.2f}")
    print(f"[OK] Return: {result.total_return_pct:.2f}%")
    print(f"[OK] Sharpe: {result.sharpe_ratio:.3f}")
    print()
    print("All systems operational!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
