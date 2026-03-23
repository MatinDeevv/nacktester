from strategy_runtime import Strategy
from strategies.competitive_20._common import atr, atr_prev, highest, lowest, candle_metrics, enough


class ATRCompressionExpansion(Strategy):
    name = 'Q20 ATR Compression Expansion'
    size = 0.01
    trigger_period = 12
    stop_atr = 1.2
    tp_rr = 2.5

    def on_bar(self, ctx):
        if ctx.bar_index < 80:
            return
        bars = ctx.bars
        price = float(ctx.bar['close'])
        a_short = atr(bars, 10)
        a_long = atr(bars, 30)
        hh = highest(bars['high'], self.trigger_period, exclude_current=True)
        ll = lowest(bars['low'], self.trigger_period, exclude_current=True)
        rng, body, _, _ = candle_metrics(ctx.bar)
        if not all(map(enough, [a_short, a_long, hh, ll])) or rng <= 0:
            return
        compressed = a_short < a_long * 0.78
        strong = body / rng > 0.6
        if not ctx.has_position and compressed and strong:
            if price > hh:
                sl = price - a_short * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif price < ll:
                sl = price + a_short * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            if ctx.position and ctx.position.trade.side.value == 'BUY' and price < hh:
                ctx.close('compression_break_fail')
            elif ctx.position and ctx.position.trade.side.value == 'SELL' and price > ll:
                ctx.close('compression_break_fail')
