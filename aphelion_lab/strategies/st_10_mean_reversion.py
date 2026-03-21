"""
Strategy 10: Mean Reversion with Range
Trade when price deviates from recent range.
"""
from strategy_runtime import Strategy


class MeanReversion_Range(Strategy):
    name = "Mean Reversion Range"
    
    lookback = 14
    std_mult = 1.5
    size = 0.01
    atr_mult = 2.0
    
    def on_bar(self, ctx):
        if ctx.bar_index < self.lookback + 1:
            return
        
        close_prices = ctx.bars["close"]
        
        # Calculate mean and std dev
        sma = close_prices.rolling(self.lookback).mean()
        std = close_prices.rolling(self.lookback).std()
        
        sma_val = float(sma.iloc[-1])
        std_val = float(std.iloc[-1])
        price = ctx.bar["close"]
        atr = ctx.atr(14)
        
        if sma_val != sma_val or std_val != std_val or atr != atr:
            return
        
        upper_band = sma_val + std_val * self.std_mult
        lower_band = sma_val - std_val * self.std_mult
        
        if not ctx.has_position:
            # Price too high, mean revert down
            if price > upper_band:
                sl = price + atr * self.atr_mult
                tp = sma_val
                ctx.sell(size=self.size, sl=sl, tp=tp)
            # Price too low, mean revert up
            elif price < lower_band:
                sl = price - atr * self.atr_mult
                tp = sma_val
                ctx.buy(size=self.size, sl=sl, tp=tp)
        else:
            # Exit at mean or if reverse hits
            if ctx.position.trade.side.value == "BUY" and price > sma_val:
                ctx.close("mean_revert_exit")
            elif ctx.position.trade.side.value == "SELL" and price < sma_val:
                ctx.close("mean_revert_exit")
