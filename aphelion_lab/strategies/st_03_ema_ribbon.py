"""
Strategy 3: EMA Ribbon
Trend-following using multiple exponential moving averages.
"""
from strategy_runtime import Strategy


class EMA_Ribbon(Strategy):
    name = "EMA Ribbon"
    
    ema_periods = [5, 10, 20, 40]
    size = 0.01
    atr_mult = 2.0
    
    def on_bar(self, ctx):
        if ctx.bar_index < max(self.ema_periods) + 1:
            return
        
        emas = [ctx.ema(p) for p in self.ema_periods]
        atr = ctx.atr(14)
        price = ctx.bar["close"]
        
        if atr != atr or any(e != e for e in emas):
            return
        
        # Check if all EMAs are in bullish order (ascending)
        bullish = all(emas[i] < emas[i+1] for i in range(len(emas)-1))
        # Check if all EMAs are in bearish order (descending)
        bearish = all(emas[i] > emas[i+1] for i in range(len(emas)-1))
        
        if not ctx.has_position:
            if bullish and price > emas[-1]:  # Above the longest EMA
                ctx.buy(size=self.size, sl=price - atr * self.atr_mult)
            elif bearish and price < emas[-1]:  # Below the longest EMA
                ctx.sell(size=self.size, sl=price + atr * self.atr_mult)
        else:
            # Exit if ribbon breaks
            if ctx.position.trade.side.value == "BUY" and price < emas[0]:
                ctx.close("ribbon_break")
            elif ctx.position.trade.side.value == "SELL" and price > emas[0]:
                ctx.close("ribbon_break")
