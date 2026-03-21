"""
Example: RSI Mean Reversion Strategy
Buy oversold, sell overbought with ATR-based stops.
Edit parameters and hit Refresh to re-run instantly.
"""

from strategy_runtime import Strategy


class RSIMeanReversion(Strategy):
    name = "RSI Mean Reversion"

    # ── Tweak these and hit Refresh ──
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
            if rsi < self.oversold:
                ctx.buy(
                    size=self.size,
                    sl=price - atr * self.sl_mult,
                    tp=price + atr * self.tp_mult,
                )
            elif rsi > self.overbought:
                ctx.sell(
                    size=self.size,
                    sl=price + atr * self.sl_mult,
                    tp=price - atr * self.tp_mult,
                )
        else:
            # Exit on RSI normalization
            if ctx.position.trade.side.value == "BUY" and rsi > 50:
                ctx.close("rsi_normalized")
            elif ctx.position.trade.side.value == "SELL" and rsi < 50:
                ctx.close("rsi_normalized")
