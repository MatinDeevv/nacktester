from strategy_runtime import Strategy
from strategies.competitive_20._common import ema, ema_prev, atr, atr_prev, highest, lowest, enough


class AdaptiveTrendBreakout(Strategy):
    name = 'Q02 Adaptive Trend Breakout'
    size = 0.01
    trend_period = 50
    breakout_period = 20
    stop_atr = 1.4
    tp_rr = 2.4

    def on_bar(self, ctx):
        if ctx.bar_index < self.trend_period + self.breakout_period + 10:
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        e_now = ema(close, self.trend_period)
        e_prev = ema_prev(close, self.trend_period, back=3)
        a_now = atr(bars, 14)
        a_prev = atr_prev(bars, 14, back=8)
        breakout_high = highest(bars['high'], self.breakout_period, exclude_current=True)
        breakout_low = lowest(bars['low'], self.breakout_period, exclude_current=True)
        if not all(map(enough, [e_now, e_prev, a_now, a_prev, breakout_high, breakout_low])):
            return
        expanding = a_now > a_prev * 1.08
        if not ctx.has_position:
            if price > breakout_high and price > e_now and e_now > e_prev and expanding:
                sl = price - a_now * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif price < breakout_low and price < e_now and e_now < e_prev and expanding:
                sl = price + a_now * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and price < e_now:
                ctx.close('ema_fail')
            elif side == 'SELL' and price > e_now:
                ctx.close('ema_fail')
