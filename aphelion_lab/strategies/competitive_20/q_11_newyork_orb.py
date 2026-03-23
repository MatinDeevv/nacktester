from strategy_runtime import Strategy
from strategies.competitive_20._common import session_slice, atr, enough, daily_vwap


class NewYorkORB(Strategy):
    name = 'Q11 New York ORB'
    size = 0.01
    stop_atr = 1.25
    tp_rr = 2.2

    def on_init(self, ctx):
        self._day = None
        self._traded = False

    def on_bar(self, ctx):
        ts = ctx.bar.name
        day = ts.date()
        if self._day != day:
            self._day = day
            self._traded = False
        if ctx.bar_index < 100:
            return
        bars = ctx.bars
        a = atr(bars, 14)
        vwap = daily_vwap(bars)
        if not all(map(enough, [a, vwap])):
            return
        pre = session_slice(bars, 12, 14, same_day_only=True)
        if len(pre) < 6:
            return
        orb_high = float(pre['high'].max())
        orb_low = float(pre['low'].min())
        price = float(ctx.bar['close'])
        hour = ts.hour
        if not ctx.has_position and not self._traded and 14 <= hour <= 17:
            if price > orb_high and price > vwap:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
                self._traded = True
            elif price < orb_low and price < vwap:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
                self._traded = True
        elif ctx.has_position and hour >= 20:
            ctx.close('ny_session_end')
