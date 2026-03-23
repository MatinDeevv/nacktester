Competitive 20 Strategy Pack

Files live in aphelion_lab/strategies/competitive_20/
Each file contains exactly one Strategy subclass and only uses engine-compatible data/features:
- OHLCV history from ctx.bars
- Current bar/time from ctx.bar
- ctx.buy / ctx.sell / ctx.close
- Built-in bbands in one strategy
- pandas/numpy math only

Pack contents:
- q_01_ema_pullback_continuation.py
- q_02_adaptive_trend_breakout.py
- q_03_rsi2_trend_snapback.py
- q_04_bollinger_reentry_reversion.py
- q_05_bollinger_squeeze_breakout.py
- q_06_donchian_retest_breakout.py
- q_07_nr7_expansion.py
- q_08_inside_bar_breakout.py
- q_09_three_bar_reversal.py
- q_10_london_orb.py
- q_11_newyork_orb.py
- q_12_asian_range_fade.py
- q_13_vwap_reclaim.py
- q_14_vwap_pullback_trend.py
- q_15_macd_hist_reversal.py
- q_16_keltner_channel_breakout.py
- q_17_zscore_mean_reversion.py
- q_18_stochastic_trend_reentry.py
- q_19_range_break_volume.py
- q_20_atr_compression_expansion.py

Helper module required:
- _common.py

Reference bundle:
- ../competitive_20_bundle.py
