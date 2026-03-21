"""
Strategy 2: RSI Mean Reversion
Buy oversold, sell overbought conditions.
"""
from strategy_runtime import Strategy


class RSI_MeanReversion(Strategy):
    name = "RSI Mean Reversion"
    
    rsi_period = 14
    oversold = 30
    overbought = 70
    size = 0.01
    atr_period = 14
    sl_mult = 2.0
    tp_mult = 3.0
    
    def on_bar(self, ctx):
        if ctx.bar_index < max(self.rsi_period, self.atr_period) + 2:
            return
        
        rsi = ctx.rsi(self.rsi_period)
        atr = ctx.atr(self.atr_period)
        price = ctx.bar["close"]
        
        if rsi != rsi or atr != atr:
            return
        
        if not ctx.has_position:
            # Buy oversold
            if rsi < self.oversold:
                sl = price - atr * self.sl_mult
                tp = price + atr * self.tp_mult
                ctx.buy(size=self.size, sl=sl, tp=tp)
            # Sell overbought
            elif rsi > self.overbought:
                sl = price + atr * self.sl_mult
                tp = price - atr * self.tp_mult
                ctx.sell(size=self.size, sl=sl, tp=tp)
        else:
            # Exit when RSI normalizes
            if ctx.position.trade.side.value == "BUY" and rsi > 50:
                ctx.close("rsi_normalized")
            elif ctx.position.trade.side.value == "SELL" and rsi < 50:
                ctx.close("rsi_normalized")
