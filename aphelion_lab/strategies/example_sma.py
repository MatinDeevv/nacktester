"""
Example: SMA Crossover Strategy
Buy when fast SMA crosses above slow SMA, sell when it crosses below.
"""

from strategy_runtime import Strategy


class SMACrossover(Strategy):
    name = "SMA Crossover"

    # Parameters — edit these and hit Refresh
    fast_period = 10
    slow_period = 30
    size = 0.01
    atr_sl_mult = 2.0
    atr_tp_mult = 3.0

    def on_bar(self, ctx):
        if ctx.bar_index < self.slow_period + 1:
            return

        fast = ctx.sma(self.fast_period)
        slow = ctx.sma(self.slow_period)
        prev_fast = ctx.bars["close"].rolling(self.fast_period).mean().iloc[-2]
        prev_slow = ctx.bars["close"].rolling(self.slow_period).mean().iloc[-2]
        atr = ctx.atr(14)

        if atr != atr:  # NaN check
            return

        price = ctx.bar["close"]

        # Cross above → buy
        if prev_fast <= prev_slow and fast > slow:
            if not ctx.has_position:
                sl = price - atr * self.atr_sl_mult
                tp = price + atr * self.atr_tp_mult
                ctx.buy(size=self.size, sl=sl, tp=tp)

        # Cross below → sell
        elif prev_fast >= prev_slow and fast < slow:
            if ctx.has_position:
                ctx.close("sma_cross_down")
            else:
                sl = price + atr * self.atr_sl_mult
                tp = price - atr * self.atr_tp_mult
                ctx.sell(size=self.size, sl=sl, tp=tp)
