#!/usr/bin/env python3
"""Verify all 10 strategies load correctly"""

from aphelion_lab.strategy_runtime import StrategyLoader

strategies = [
    "st_01_sma_crossover.py",
    "st_02_rsi_mean_reversion.py",
    "st_03_ema_ribbon.py",
    "st_04_bollinger_breakout.py",
    "st_05_stochastic.py",
    "st_06_adx_trend.py",
    "st_07_donchian_breakout.py",
    "st_08_macd_crossover.py",
    "st_09_volume_price.py",
    "st_10_mean_reversion.py"
]

print("=" * 70)
print("TESTING ALL 10 STRATEGIES")
print("=" * 70)

loader = StrategyLoader()
success_count = 0

for i, strat_file in enumerate(strategies, 1):
    path = f"aphelion_lab/strategies/{strat_file}"
    strat = loader.load(path)
    
    if strat:
        status = "✓ OK"
        name = loader.strategy_name
        success_count += 1
    else:
        status = "✗ ERROR"
        name = f"Failed to load: {loader.error[:40]}"
    
    print(f"{i:2d}. {status:10s} {name:40s}")

print("=" * 70)
print(f"RESULT: {success_count}/10 strategies loaded successfully")
if success_count == 10:
    print("✓ ALL SYSTEMS OPERATIONAL")
else:
    print(f"✗ {10 - success_count} strategies failed")
print("=" * 70)
