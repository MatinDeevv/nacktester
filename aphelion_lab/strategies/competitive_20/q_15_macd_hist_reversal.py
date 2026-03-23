from strategy_runtime import Strategy
from strategies.competitive_20._common import ema, macd_hist, atr, enough


class MACDHistogramReversal(Strategy):
    name = 'Q15 MACD Histogram Reversal'
    size = 0.01
    stop_atr = 1.25
    tp_rr = 2.1

    def on_bar(self, ctx):
        if ctx.bar_index < 60:
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        hist_now, hist_prev = macd_hist(close, 12, 26, 9)
        e50 = ema(close, 50)
        a = atr(bars, 14)
        if not all(map(enough, [hist_now, hist_prev, e50, a])):
            return
        if not ctx.has_position:
            if hist_prev <= 0 < hist_now and price > e50:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif hist_prev >= 0 > hist_now and price < e50:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and hist_now < 0:
                ctx.close('hist_flip')
            elif side == 'SELL' and hist_now > 0:
                ctx.close('hist_flip')
