from strategy_runtime import Strategy
from strategies.competitive_20._common import session_slice, atr, enough


class LondonORB(Strategy):
    name = 'Q10 London ORB'
    size = 0.01
    stop_atr = 1.3
    tp_rr = 2.0

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
        if not enough(a):
            return
        asian = session_slice(bars, 0, 6, same_day_only=True)
        if len(asian) < 12:
            return
        asian_high = float(asian['high'].max())
        asian_low = float(asian['low'].min())
        rng = asian_high - asian_low
        price = float(ctx.bar['close'])
        hour = ts.hour
        if not ctx.has_position and not self._traded and 6 <= hour <= 10 and rng > a * 1.2:
            if price > asian_high:
                sl = max(asian_low, price - a * self.stop_atr)
                risk = price - sl
                if risk > a * 0.3:
                    ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
                    self._traded = True
            elif price < asian_low:
                sl = min(asian_high, price + a * self.stop_atr)
                risk = sl - price
                if risk > a * 0.3:
                    ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
                    self._traded = True
        elif ctx.has_position and hour >= 15:
            ctx.close('london_session_end')
