from strategy_runtime import Strategy
from strategies.competitive_20._common import session_slice, daily_vwap, atr, enough


class LondonMomentumBurst(Strategy):
    name = 'Q12 London Momentum Burst'
    size = 0.01
    stop_atr = 1.2
    tp_rr = 2.1

    def on_init(self, ctx):
        self._day = None
        self._traded = False

    def on_bar(self, ctx):
        ts = ctx.bar.name
        day = ts.date()
        if self._day != day:
            self._day = day
            self._traded = False
        if ctx.bar_index < 80:
            return
        bars = ctx.bars
        opening = session_slice(bars, 6, 7, same_day_only=True)
        a = atr(bars, 14)
        vwap = daily_vwap(bars)
        if len(opening) < 6 or not all(map(enough, [a, vwap])):
            return
        burst_high = float(opening['high'].max())
        burst_low = float(opening['low'].min())
        vol_avg = float(opening['volume'].astype(float).mean())
        price = float(ctx.bar['close'])
        vol_now = float(ctx.bar['volume'])
        hour = ts.hour
        if not ctx.has_position and not self._traded and 7 <= hour <= 11:
            if price > burst_high and price > vwap and vol_now >= vol_avg * 0.9:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
                self._traded = True
            elif price < burst_low and price < vwap and vol_now >= vol_avg * 0.9:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
                self._traded = True
        elif ctx.has_position and hour >= 14:
            ctx.close('london_burst_end')
