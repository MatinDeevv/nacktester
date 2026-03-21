"""
Strategy 7: Donchian Channel Breakout
Trade breakouts from highest high and lowest low.
"""
from strategy_runtime import Strategy


class Donchian_Breakout(Strategy):
    name = "Donchian Breakout"
    
    channel_period = 20
    size = 0.01
    atr_mult = 2.5
    
    def on_bar(self, ctx):
        if ctx.bar_index < self.channel_period:
            return
        
        bars = ctx.bars
        highest = bars["high"].rolling(self.channel_period).max()
        lowest = bars["low"].rolling(self.channel_period).min()
        
        highest_val = float(highest.iloc[-1])
        lowest_val = float(lowest.iloc[-1])
        price = ctx.bar["close"]
        high = ctx.bar["high"]
        low = ctx.bar["low"]
        atr = ctx.atr(14)
        
        if atr != atr:
            return
        
        if not ctx.has_position:
            # Breakout above highest high
            if high >= highest_val:
                sl = lowest_val
                ctx.buy(size=self.size, sl=sl, tp=price + (highest_val - lowest_val))
            # Breakout below lowest low
            elif low <= lowest_val:
                sl = highest_val
                ctx.sell(size=self.size, sl=sl, tp=price - (highest_val - lowest_val))
        else:
            # Stop loss already handles exit via channel
            pass
