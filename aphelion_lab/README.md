# APHELION LAB — Visual Backtesting Laboratory

A local-first visual backtesting lab for rapid strategy iteration.

**Download once → Code strategy → Click Refresh → Instantly see results**

## Setup

```bash
# Install dependencies
pip install PySide6 matplotlib mplfinance pandas numpy pyarrow MetaTrader5

# Or use requirements.txt
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Quick Start

1. Open the app
2. Select symbol (XAUUSD) and timeframe (M5)
3. Click **⬇ Download Data** — fetches history from MT5, caches locally
4. Click **📂 Load Strategy** — pick a .py file from `strategies/`
5. Click **▶ Run Backtest** — see chart, equity curve, metrics, trades
6. Edit your strategy file in any editor
7. Click **🔄 Refresh** — strategy hot-reloads, backtest reruns instantly
8. Repeat 6-7 forever

## Writing Strategies

```python
from strategy_runtime import Strategy

class MyStrategy(Strategy):
    name = "My Edge"
    
    # Parameters
    fast = 10
    slow = 30
    
    def on_bar(self, ctx):
        # ctx.bar          → current bar (open, high, low, close, volume)
        # ctx.bars         → all bars up to now
        # ctx.position     → current position or None
        # ctx.has_position → bool
        # ctx.equity       → current equity
        # ctx.bar_index    → current bar number
        
        # Indicators
        # ctx.sma(period)  → Simple Moving Average
        # ctx.ema(period)  → Exponential Moving Average
        # ctx.rsi(period)  → RSI
        # ctx.atr(period)  → Average True Range
        # ctx.bbands(period, std) → (upper, mid, lower)
        
        # Actions
        # ctx.buy(size=0.01, sl=None, tp=None)
        # ctx.sell(size=0.01, sl=None, tp=None)
        # ctx.close(reason="signal")
        
        if not ctx.has_position:
            if ctx.sma(self.fast) > ctx.sma(self.slow):
                ctx.buy(size=0.01)
        else:
            if ctx.sma(self.fast) < ctx.sma(self.slow):
                ctx.close()
```

## Example Strategies

- `strategies/example_sma.py` — SMA Crossover with ATR stops
- `strategies/example_rsi.py` — RSI Mean Reversion
- `strategies/example_london_breakout.py` — London Session Breakout

## Data Cache

Downloaded data is stored as Parquet files in `cache/`.
Once downloaded, data loads instantly — no re-downloading needed.
Check the **Data Cache** tab to see what's available.

## Architecture

```
main.py              → Entry point
data_manager.py      → MT5 download + Parquet cache
backtest_engine.py   → Bar-by-bar simulation engine
strategy_runtime.py  → Strategy base class + hot reload
gui_app.py           → PySide6 GUI application
strategies/          → Your strategy files
cache/               → Downloaded data (Parquet)
```

## Notes

- MT5 terminal must be running for data download
- Backtesting uses cached data — MT5 not needed after download
- Hot reload reloads the Python file fresh — edit and refresh
- All times are UTC
- Gold pip value: $0.01 | Lot multiplier: 100 oz
