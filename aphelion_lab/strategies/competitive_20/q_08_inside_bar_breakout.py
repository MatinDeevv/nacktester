from strategy_runtime import Strategy
from strategies.competitive_20._common import ema, atr, enough


class InsideBarBreakout(Strategy):
    name = 'Q08 Inside Bar Breakout'
    size = 0.01
    stop_atr = 1.1
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 100:
            return
        bars = ctx.bars.iloc[-3:]
        mother = bars.iloc[-3]
        inside = bars.iloc[-2]
        current = bars.iloc[-1]
        price = float(current['close'])
        e34 = ema(ctx.bars['close'], 34)
        e100 = ema(ctx.bars['close'], 100)
        a = atr(ctx.bars, 14)
        if not all(map(enough, [e34, e100, a])):
            return
        is_inside = float(inside['high']) <= float(mother['high']) and float(inside['low']) >= float(mother['low'])
        if not ctx.has_position and is_inside:
            if is_inside and price > float(mother['high']) and e34 > e100:
                sl = min(float(mother['low']), price - a * self.stop_atr)
                risk = price - sl
                if risk > a * 0.25:
                    ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif is_inside and price < float(mother['low']) and e34 < e100:
                sl = max(float(mother['high']), price + a * self.stop_atr)
                risk = sl - price
                if risk > a * 0.25:
                    ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        elif ctx.has_position:
            pos = ctx.position
            if pos is None:
                return
            if pos.trade.side.value == 'BUY' and price < e34:
                ctx.close('inside_bar_fail')
            elif pos.trade.side.value == 'SELL' and price > e34:
                ctx.close('inside_bar_fail')
