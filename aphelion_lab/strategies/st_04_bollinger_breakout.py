"""
Strategy 4: Bollinger Bands Breakout
Trade breakouts from Bollinger Bands.
"""
from strategy_runtime import Strategy
import math


class BollingerBands_Breakout(Strategy):
    name = "Bollinger Bands Breakout"
    
    bb_period = 20
    bb_std = 2.0
    size = 0.01
    atr_mult = 2.5
    
    def on_bar(self, ctx):
        if ctx.bar_index < self.bb_period + 1:
            return
        
        upper, mid, lower = ctx.bbands(self.bb_period, self.bb_std)
        atr = ctx.atr(14)
        price = ctx.bar["close"]
        high = ctx.bar["high"]
        low = ctx.bar["low"]
        
        if upper != upper or atr != atr:  # NaN check
            return
        
        if not ctx.has_position:
            # Breakout above upper band
            if high > upper:
                sl = price - atr * self.atr_mult
                ctx.buy(size=self.size, sl=sl, tp=price + (upper - lower))
            # Breakout below lower band
            elif low < lower:
                sl = price + atr * self.atr_mult
                ctx.sell(size=self.size, sl=sl, tp=price - (upper - lower))
        else:
            # Exit if returns to middle band
            if ctx.position.trade.side.value == "BUY" and price < mid:
                ctx.close("bb_mid_cross")
            elif ctx.position.trade.side.value == "SELL" and price > mid:
                ctx.close("bb_mid_cross")
