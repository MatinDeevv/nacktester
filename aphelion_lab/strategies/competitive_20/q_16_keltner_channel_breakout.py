from strategy_runtime import Strategy
from strategies.competitive_20._common import ema, atr, enough


class KeltnerChannelBreakout(Strategy):
    name = 'Q16 Keltner Channel Breakout'
    size = 0.01
    stop_atr = 1.35
    tp_rr = 2.4

    def on_bar(self, ctx):
        if ctx.bar_index < 70:
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        e20 = ema(close, 20)
        e50 = ema(close, 50)
        a = atr(bars, 20)
        if not all(map(enough, [e20, e50, a])):
            return
        upper = e20 + a * 1.5
        lower = e20 - a * 1.5
        if not ctx.has_position:
            if price > upper and e20 > e50:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif price < lower and e20 < e50:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            if pos.trade.side.value == 'BUY' and price < e20:
                ctx.close('back_inside_keltner')
            elif pos.trade.side.value == 'SELL' and price > e20:
                ctx.close('back_inside_keltner')
