"""
Strategy 5: Stochastic Oscillator
Trade stochastic turning points.
"""
from strategy_runtime import Strategy


class Stochastic_Oscillator(Strategy):
    name = "Stochastic Oscillator"
    
    period = 14
    smooth_k = 3
    smooth_d = 3
    oversold = 20
    overbought = 80
    size = 0.01
    atr_mult = 2.0
    
    def on_bar(self, ctx):
        if ctx.bar_index < self.period + self.smooth_k + self.smooth_d + 2:
            return
        
        # Calculate stochastic
        bars = ctx.bars
        lows = bars["low"].rolling(self.period).min()
        highs = bars["high"].rolling(self.period).max()
        range_hl = highs - lows
        k_raw = 100 * (bars["close"] - lows) / range_hl.replace(0, 1)
        
        # Smooth K
        k_line = k_raw.rolling(self.smooth_k).mean()
        # D line (SMA of K)
        d_line = k_line.rolling(self.smooth_d).mean()
        
        k = float(k_line.iloc[-1])
        d = float(d_line.iloc[-1])
        k_prev = float(k_line.iloc[-2]) if len(k_line) > 1 else k
        
        atr = ctx.atr(14)
        price = ctx.bar["close"]
        
        if k != k or d != d or atr != atr:
            return
        
        if not ctx.has_position:
            # Buy: K crosses above oversold
            if k_prev <= self.oversold and k > self.oversold:
                sl = price - atr * self.atr_mult
                ctx.buy(size=self.size, sl=sl)
            # Sell: K crosses below overbought
            elif k_prev >= self.overbought and k < self.overbought:
                sl = price + atr * self.atr_mult
                ctx.sell(size=self.size, sl=sl)
        else:
            # Exit at opposite extreme
            if ctx.position.trade.side.value == "BUY" and k > self.overbought:
                ctx.close("stoch_overbought")
            elif ctx.position.trade.side.value == "SELL" and k < self.oversold:
                ctx.close("stoch_oversold")
