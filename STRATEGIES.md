# APHELION LAB — 10 Trading Strategies

All strategies are located in `aphelion_lab/strategies/` and are ready to use in the GUI.

## Strategy List

### 1. **SMA Crossover** (`st_01_sma_crossover.py`)
**Description:** Classic moving average crossover strategy  
**How it works:**
- Buy when fast SMA (10) crosses above slow SMA (30)
- Sell when fast SMA crosses below slow SMA
- Uses ATR for stop loss (2x) and take profit (3x)
- **Best for:** Trending markets, intraday to daily timeframes
- **Parameters:** `fast_period=10`, `slow_period=30`, `atr_mult_sl=2.0`, `atr_mult_tp=3.0`

### 2. **RSI Mean Reversion** (`st_02_rsi_mean_reversion.py`)
**Description:** Buy oversold, sell overbought RSI levels  
**How it works:**
- Buy when RSI(14) drops below 30 (oversold)
- Sell when RSI rises above 70 (overbought)
- Exit when RSI normalizes to 50
- **Best for:** Range-bound markets, momentum reversals
- **Parameters:** `rsi_period=14`, `oversold=30`, `overbought=70`

### 3. **EMA Ribbon** (`st_03_ema_ribbon.py`)
**Description:** Trend-following using multiple exponential moving averages  
**How it works:**
- Monitors 4 EMAs (5, 10, 20, 40)
- Buy when all EMAs are in ascending order and price > longest EMA
- Sell when all EMAs are in descending order and price < longest EMA
- Exit if ribbon breaks
- **Best for:** Strong trending markets, avoiding choppy periods
- **Parameters:** `ema_periods=[5, 10, 20, 40]`

### 4. **Bollinger Bands Breakout** (`st_04_bollinger_breakout.py`)
**Description:** Trade breakouts from Bollinger Bands  
**How it works:**
- Buy on breakout above upper band
- Sell on breakout below lower band
- Exit when price returns to middle band
- **Best for:** Volatile markets, breakout traders
- **Parameters:** `bb_period=20`, `bb_std=2.0`

### 5. **Stochastic Oscillator** (`st_05_stochastic.py`)
**Description:** Trade stochastic turning points  
**How it works:**
- Buy when Stochastic K-line crosses above 20 (oversold)
- Sell when Stochastic K-line crosses below 80 (overbought)
- Exit at opposite extremes
- **Best for:** Mean reversion, oscillating markets
- **Parameters:** `period=14`, `oversold=20`, `overbought=80`

### 6. **ADX Trend Strength** (`st_06_adx_trend.py`)
**Description:** Trade strong trends using Average Directional Index  
**How it works:**
- Only trade when ADX > 25 (strong trend)
- Buy when +DI > -DI (uptrend)
- Sell when -DI > +DI (downtrend)
- Exit when ADX weakens below 20
- **Best for:** Trend confirmation, avoiding choppy ranges
- **Parameters:** `adx_period=14`, `adx_threshold=25`

### 7. **Donchian Channel Breakout** (`st_07_donchian_breakout.py`)
**Description:** Trade breakouts from highest high and lowest low  
**How it works:**
- Buy on breakout above 20-period highest high
- Sell on breakout below 20-period lowest low
- Target: range size, Stop: channel extremes
- **Best for:** Breakout traders, institutional supply/demand
- **Parameters:** `channel_period=20`

### 8. **MACD Crossover** (`st_08_macd_crossover.py`)
**Description:** Traditional MACD signal line crossover  
**How it works:**
- Buy when MACD crosses above signal line
- Sell when MACD crosses below signal line
- Exit on reverse crossover
- **Best for:** Swing trades, trend changes
- **Parameters:** `fast_ema=12`, `slow_ema=26`, `signal_line=9`

### 9. **Volume Price Action** (`st_09_volume_price.py`)
**Description:** Trade based on volume confirmation and price structure  
**How it works:**
- Buy when price closes above previous close + high volume (>1.5x avg)
- Sell when price closes below previous close + high volume
- Exit when volume dries up (<0.7x avg)
- **Best for:** Intraday, capturing institutional moves
- **Parameters:** Volume ratio threshold: 1.5x average

### 10. **Mean Reversion Range** (`st_10_mean_reversion.py`)
**Description:** Trade when price deviates from recent range  
**How it works:**
- Buy when price < (SMA - 1.5 std dev) - expects reversion up
- Sell when price > (SMA + 1.5 std dev) - expects reversion down
- Exit at mean (SMA)
- **Best for:** Range-bound, statistical arbitrage
- **Parameters:** `lookback=14`, `std_mult=1.5`

---

## Loading Strategies in the GUI

1. Open the application: `python main.py`
2. Click **📂 Load Strategy**
3. Navigate to `aphelion_lab/strategies/`
4. Select any `st_XX_*.py` file
5. Click **▶ Run Backtest**
6. Edit parameters in the strategy file
7. Hit **🔄 Refresh** to hot-reload and retest instantly

## System Fixes Applied

✅ **Consolidated duplicate code** — Removed root-level strategy files and outdated modules  
✅ **Fixed strategy imports** — Updated to use dynamic loader with proper Strategy class injection  
✅ **Fixed chart rendering** — Corrected matplotlib date conversion for proper axes display  
✅ **Updated GUI strategy paths** — Points to `aphelion_lab/strategies/` directory  
✅ **Created 10 production-ready strategies** — All tested and working with the unified API  
✅ **All packages installed** — PySide6, matplotlib, pandas, numpy, pyarrow, MetaTrader5  

## Testing

Run the test file to verify system integrity:
```bash
python test_system.py
```

Expected output:
```
✓ Backtest ran successfully
✓ Trades: [N]
✓ P&L: $[amount]
✓ Return: [%]
✓ Sharpe: [ratio]
All systems operational!
```

---

**Aphelion Lab Version 1.0**  
*Visual Backtesting Laboratory for Rapid Strategy Iteration*
