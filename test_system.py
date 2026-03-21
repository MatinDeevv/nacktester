#!/usr/bin/env python3
"""Test the backtest engine with a strategy"""

from aphelion_lab.backtest_engine import BacktestEngine, BacktestConfig
from aphelion_lab.strategy_runtime import StrategyLoader
import pandas as pd
import numpy as np

# Quick test
loader = StrategyLoader()
loader.load('aphelion_lab/strategies/st_01_sma_crossover.py')

# Create test data
dates = pd.date_range('2024-01-01', periods=100, freq='h')
closes = 100 + np.random.randn(100).cumsum()
data = pd.DataFrame({
    'open': closes + np.random.randn(100)*0.5,
    'high': closes + abs(np.random.randn(100)*0.8),
    'low': closes - abs(np.random.randn(100)*0.8),
    'close': closes,
    'volume': np.random.randint(1000, 10000, 100)
}, index=dates)

config = BacktestConfig(initial_capital=5000)
engine = BacktestEngine(config)
result = engine.run(data, loader.strategy)

print(f"✓ Backtest ran successfully")
print(f"✓ Trades: {result.total_trades}")
print(f"✓ P&L: ${result.net_pnl:.2f}")
print(f"✓ Return: {result.total_return_pct:.2f}%")
print(f"✓ Sharpe: {result.sharpe_ratio:.3f}")
print()
print("All systems operational!")
