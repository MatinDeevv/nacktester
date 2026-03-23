from strategy_runtime import Strategy
from strategies.competitive_20._common import ema, atr, lowest, highest, rsi, enough


class EMAPullbackContinuation(Strategy):
    name = 'Q01 EMA Pullback Continuation'
    size = 0.01
    fast = 20
    mid = 50
    slow = 100
    pullback_lookback = 6
    stop_atr = 1.6
    tp_rr = 2.2

    def on_bar(self, ctx):
        if ctx.bar_index < self.slow + 10:
            return
        bars = ctx.bars
        close = bars['close']
        low = bars['low']
        high = bars['high']
        price = float(ctx.bar['close'])
        e1 = ema(close, self.fast)
        e2 = ema(close, self.mid)
        e3 = ema(close, self.slow)
        rs = rsi(close, 14)
        a = atr(bars, 14)
        if not all(map(enough, [e1, e2, e3, rs, a])):
            return
        prev_low = lowest(low, self.pullback_lookback, exclude_current=True)
        prev_high = highest(high, self.pullback_lookback, exclude_current=True)
        if not ctx.has_position:
            if e1 > e2 > e3 and price > e1 and float(ctx.bar['low']) <= e1 + a * 0.15 and rs > 52:
                sl = min(prev_low, price - a * self.stop_atr)
                risk = price - sl
                if risk > a * 0.35:
                    ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif e1 < e2 < e3 and price < e1 and float(ctx.bar['high']) >= e1 - a * 0.15 and rs < 48:
                sl = max(prev_high, price + a * self.stop_atr)
                risk = sl - price
                if risk > a * 0.35:
                    ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and (price < e2 or rs < 45):
                ctx.close('trend_lost')
            elif side == 'SELL' and (price > e2 or rs > 55):
                ctx.close('trend_lost')
