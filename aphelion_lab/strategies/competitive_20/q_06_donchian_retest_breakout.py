from strategy_runtime import Strategy
from strategies.competitive_20._common import highest, lowest, atr, enough


class DonchianRetestBreakout(Strategy):
    name = 'Q06 Donchian Retest Breakout'
    size = 0.01
    channel = 30
    stop_atr = 1.5
    tp_rr = 2.3

    def on_init(self, ctx):
        self._pending_long = False
        self._pending_short = False
        self._level = None

    def on_bar(self, ctx):
        if ctx.bar_index < self.channel + 5:
            return
        bars = ctx.bars
        price = float(ctx.bar['close'])
        high = float(ctx.bar['high'])
        low = float(ctx.bar['low'])
        a = atr(bars, 14)
        hh = highest(bars['high'], self.channel, exclude_current=True)
        ll = lowest(bars['low'], self.channel, exclude_current=True)
        if not all(map(enough, [a, hh, ll])):
            return
        if not ctx.has_position:
            if self._pending_long and low <= self._level + a * 0.2 and price > self._level:
                sl = min(low, price - a * self.stop_atr)
                risk = price - sl
                if risk > a * 0.35:
                    ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
                self._pending_long = False
            elif self._pending_short and high >= self._level - a * 0.2 and price < self._level:
                sl = max(high, price + a * self.stop_atr)
                risk = sl - price
                if risk > a * 0.35:
                    ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
                self._pending_short = False
            else:
                self._pending_long = high > hh
                self._pending_short = low < ll
                self._level = hh if self._pending_long else (ll if self._pending_short else self._level)
        else:
            pos = ctx.position
            if pos is None:
                return
            if pos.trade.side.value == 'BUY' and price < self._level:
                ctx.close('failed_retest')
            elif pos.trade.side.value == 'SELL' and price > self._level:
                ctx.close('failed_retest')
