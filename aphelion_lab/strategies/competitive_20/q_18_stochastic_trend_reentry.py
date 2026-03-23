from strategy_runtime import Strategy
from strategies.competitive_20._common import ema, rsi, rsi_prev, atr, enough


class RSI50TrendReentry(Strategy):
    name = 'Q18 RSI50 Trend Reentry'
    size = 0.01
    stop_atr = 1.25
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 220:
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        e50 = ema(close, 50)
        e200 = ema(close, 200)
        r_now = rsi(close, 14)
        r_prev = rsi_prev(close, 14, 1)
        a = atr(bars, 14)
        if not all(map(enough, [e50, e200, r_now, r_prev, a])):
            return
        if not ctx.has_position:
            if price > e50 > e200 and r_prev <= 48 < r_now:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif price < e50 < e200 and r_prev >= 52 > r_now:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and (r_now > 68 or price < e50):
                ctx.close('rsi_reentry_done')
            elif side == 'SELL' and (r_now < 32 or price > e50):
                ctx.close('rsi_reentry_done')
