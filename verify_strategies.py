#!/usr/bin/env python3
"""Verify that the packaged strategies load correctly."""

from pathlib import Path
import sys

from aphelion_lab.strategy_runtime import StrategyLoader

ROOT = Path(__file__).resolve().parent
STRATEGY_DIR = ROOT / "aphelion_lab" / "strategies"
STRATEGIES = [
    "st_01_sma_crossover.py",
    "st_02_rsi_mean_reversion.py",
    "st_03_ema_ribbon.py",
    "st_04_bollinger_breakout.py",
    "st_05_stochastic.py",
    "st_06_adx_trend.py",
    "st_07_donchian_breakout.py",
    "st_08_macd_crossover.py",
    "st_09_volume_price.py",
    "st_10_mean_reversion.py",
]


def main() -> int:
    print("=" * 70)
    print("TESTING ALL 10 STRATEGIES")
    print("=" * 70)

    success_count = 0

    for index, filename in enumerate(STRATEGIES, 1):
        loader = StrategyLoader()
        strategy = loader.load(str(STRATEGY_DIR / filename))

        if strategy is not None:
            status = "[OK]"
            name = loader.strategy_name
            success_count += 1
        else:
            status = "[ERROR]"
            name = f"Failed to load: {loader.error[:40]}"

        print(f"{index:2d}. {status:10s} {name:40s}")

    print("=" * 70)
    print(f"RESULT: {success_count}/10 strategies loaded successfully")
    if success_count == len(STRATEGIES):
        print("[OK] ALL SYSTEMS OPERATIONAL")
        print("=" * 70)
        return 0

    print(f"[ERROR] {len(STRATEGIES) - success_count} strategies failed")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
