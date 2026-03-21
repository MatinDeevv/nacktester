"""
Strategy 8: MACD Crossover
Traditional MACD signal line crossover strategy.
"""
from strategy_runtime import Strategy


class MACD_Crossover(Strategy):
    name = "MACD Crossover"
    
    fast_ema = 12
    slow_ema = 26
    signal_line = 9
    size = 0.01
    atr_mult = 2.0
    
    def on_bar(self, ctx):
        if ctx.bar_index < self.slow_ema + self.signal_line + 1:
            return
        
        close = ctx.bars["close"]
        
        # Calculate MACD
        ema_fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_ema, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal_line, adjust=False).mean()
        
        macd_val = float(macd.iloc[-1])
        signal_val = float(signal.iloc[-1])
        macd_prev = float(macd.iloc[-2]) if len(macd) > 1 else macd_val
        signal_prev = float(signal.iloc[-2]) if len(signal) > 1 else signal_val
        
        atr = ctx.atr(14)
        price = ctx.bar["close"]
        
        if macd_val != macd_val or atr != atr:
            return
        
        if not ctx.has_position:
            # MACD crosses above signal line (bullish)
            if macd_prev <= signal_prev and macd_val > signal_val:
                sl = price - atr * self.atr_mult
                ctx.buy(size=self.size, sl=sl)
            # MACD crosses below signal line (bearish)
            elif macd_prev >= signal_prev and macd_val < signal_val:
                sl = price + atr * self.atr_mult
                ctx.sell(size=self.size, sl=sl)
        else:
            # Exit on opposite crossover
            if ctx.position.trade.side.value == "BUY" and macd_val < signal_val:
                ctx.close("macd_cross_down")
            elif ctx.position.trade.side.value == "SELL" and macd_val > signal_val:
                ctx.close("macd_cross_up")
