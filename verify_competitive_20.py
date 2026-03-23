#!/usr/bin/env python3
import sys, os, glob, math
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, 'aphelion_lab') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'aphelion_lab'))

from aphelion_lab.backtest_engine import BacktestEngine, BacktestConfig
from aphelion_lab.strategy_runtime import StrategyLoader

pack_dir = os.path.join(ROOT, 'aphelion_lab', 'strategies', 'competitive_20')
data_path = os.path.join(ROOT, 'cache', 'XAUUSD_M5.parquet')

table = pq.read_table(data_path)
df = table.to_pandas(ignore_metadata=True).set_index('timestamp').sort_index()[['open','high','low','close','volume']]
df = df.iloc[-800:]

print('=' * 96)
print('COMPETITIVE 20 PACK CHECK  |  sample: last 800 XAUUSD M5 bars from cache')
print('=' * 96)
print(f"{'file':38s} {'trades':>6s} {'net':>10s} {'ret%':>8s} {'wr%':>8s} {'pf':>8s}")
print('-' * 96)
for path in sorted(glob.glob(os.path.join(pack_dir, 'q_*.py'))):
    loader = StrategyLoader()
    strat = loader.load(path)
    name = os.path.basename(path)
    if not strat:
        print(f"{name:38s} LOAD FAIL")
        continue
    res = BacktestEngine(BacktestConfig(initial_capital=5000)).run(df, strat)
    pf = 'inf' if not math.isfinite(res.profit_factor) else f"{float(res.profit_factor):.3f}"
    print(f"{name:38s} {res.total_trades:6d} {float(res.net_pnl):10.2f} {float(res.total_return_pct):8.2f} {float(res.win_rate):8.1f} {pf:>8s}")
print('-' * 96)
print('Done.')
