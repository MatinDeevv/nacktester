"""
Strategy 9: Volume-Weighted Price Action
Trade based on volume confirmation and price structure.
"""
from strategy_runtime import Strategy


class VolumePrice_Action(Strategy):
    name = "Volume Price Action"
    
    size = 0.01
    atr_mult = 2.0
    
    def on_bar(self, ctx):
        if ctx.bar_index < 5:
            return
        
        bars = ctx.bars
        close = bars["close"]
        volume = bars["volume"]
        
        avg_vol = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / avg_vol.iloc[-1]
        
        # Get price structure
        price = ctx.bar["close"]
        prev_close = close.iloc[-2]
        atr = ctx.atr(14)
        
        if atr != atr or vol_ratio != vol_ratio:
            return
        
        # Strong volume confirmation
        high_volume = vol_ratio > 1.5
        
        if not ctx.has_position:
            # Bullish: price above prev close with high volume
            if price > prev_close and high_volume:
                sl = price - atr * self.atr_mult
                ctx.buy(size=self.size, sl=sl, tp=price + atr * 3)
            # Bearish: price below prev close with high volume
            elif price < prev_close and high_volume:
                sl = price + atr * self.atr_mult
                ctx.sell(size=self.size, sl=sl, tp=price - atr * 3)
        else:
            # Exit if volume dries up or reverses
            if vol_ratio < 0.7:
                ctx.close("low_volume_exit")
