"""
Strategy 6: ADX Trend Strength
Trade strong trends using ADX indicator.
"""
from strategy_runtime import Strategy


class ADX_TrendStrength(Strategy):
    name = "ADX Trend Strength"
    
    adx_period = 14
    adx_threshold = 25
    size = 0.01
    atr_mult = 2.0
    
    def on_bar(self, ctx):
        if ctx.bar_index < self.adx_period + 5:
            return
        
        bars = ctx.bars
        high = bars["high"]
        low = bars["low"]
        close = bars["close"]
        
        # Calculate plus and minus DM
        up = high.diff()
        down = -low.diff()
        
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        
        # ATR
        tr = bars["close"]
        tr.iloc[0] = high.iloc[0] - low.iloc[0]
        for i in range(1, len(tr)):
            tr.iloc[i] = max(
                high.iloc[i] - low.iloc[i],
                abs(high.iloc[i] - close.iloc[i-1]),
                abs(low.iloc[i] - close.iloc[i-1])
            )
        
        atr_val = tr.rolling(self.adx_period).mean()
        
        # DI
        plus_di = 100 * plus_dm.rolling(self.adx_period).sum() / atr_val.replace(0, 1)
        minus_di = 100 * minus_dm.rolling(self.adx_period).sum() / atr_val.replace(0, 1)
        
        # DX
        di_sum = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, 1)
        adx = dx.rolling(self.adx_period).mean()
        
        adx_val = float(adx.iloc[-1])
        plus_di_val = float(plus_di.iloc[-1])
        minus_di_val = float(minus_di.iloc[-1])
        atr_cur = ctx.atr(14)
        price = ctx.bar["close"]
        
        if adx_val == adx_val and atr_cur == atr_cur:
            if not ctx.has_position:
                # Strong uptrend
                if adx_val > self.adx_threshold and plus_di_val > minus_di_val:
                    ctx.buy(size=self.size, sl=price - atr_cur * self.atr_mult)
                # Strong downtrend
                elif adx_val > self.adx_threshold and minus_di_val > plus_di_val:
                    ctx.sell(size=self.size, sl=price + atr_cur * self.atr_mult)
            else:
                # Exit when ADX weakens
                if adx_val < 20:
                    ctx.close("adx_weak")
