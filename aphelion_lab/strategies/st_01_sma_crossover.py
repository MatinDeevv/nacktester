"""
Strategy 1: SMA Crossover
Classic moving average crossover strategy with ATR-based stops.
"""
from strategy_runtime import Strategy


class SMA_Crossover(Strategy):
    name = "SMA Crossover"
    
    fast_period = 10
    slow_period = 30
    size = 0.01
    atr_mult_sl = 2.0
    atr_mult_tp = 3.0
    
    def on_bar(self, ctx):
        if ctx.bar_index < self.slow_period + 1:
            return
        
        fast = ctx.sma(self.fast_period)
        slow = ctx.sma(self.slow_period)
        atr = ctx.atr(14)
        price = ctx.bar["close"]
        
        if atr != atr:  # NaN check
            return
        
        # Get previous values
        prev_fast = ctx.bars["close"].rolling(self.fast_period).mean().iloc[-2]
        prev_slow = ctx.bars["close"].rolling(self.slow_period).mean().iloc[-2]
        
        # Bullish crossover
        if prev_fast <= prev_slow and fast > slow:
            if not ctx.has_position:
                sl = price - atr * self.atr_mult_sl
                tp = price + atr * self.atr_mult_tp
                ctx.buy(size=self.size, sl=sl, tp=tp)
        
        # Bearish crossover
        elif prev_fast >= prev_slow and fast < slow:
            if ctx.has_position:
                ctx.close("sma_cross_below")
