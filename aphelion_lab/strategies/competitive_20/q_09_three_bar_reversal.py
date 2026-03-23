from strategy_runtime import Strategy
from strategies.competitive_20._common import ema, atr, enough, rsi


class ThreeBarReversal(Strategy):
    name = 'Q09 Three Bar Reversal'
    size = 0.01
    stop_atr = 1.1

    def on_bar(self, ctx):
        if ctx.bar_index < 40:
            return
        bars = ctx.bars.iloc[-4:]
        c = bars['close']
        price = float(ctx.bar['close'])
        e21 = ema(ctx.bars['close'], 21)
        e50 = ema(ctx.bars['close'], 50)
        rs = rsi(ctx.bars['close'], 5)
        a = atr(ctx.bars, 14)
        if not all(map(enough, [e21, e50, rs, a])):
            return
        flatish = abs(e21 - e50) < a * 1.1
        up_stretch = flatish and c.iloc[-4] < c.iloc[-3] < c.iloc[-2] and price < c.iloc[-2] and c.iloc[-2] - e21 > a * 0.9 and rs > 78
        down_stretch = flatish and c.iloc[-4] > c.iloc[-3] > c.iloc[-2] and price > c.iloc[-2] and e21 - c.iloc[-2] > a * 0.9 and rs < 22
        if not ctx.has_position:
            if up_stretch:
                ctx.sell(size=self.size, sl=price + a * self.stop_atr, tp=e21)
            elif down_stretch:
                ctx.buy(size=self.size, sl=price - a * self.stop_atr, tp=e21)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and price >= e21:
                ctx.close('reversion_done')
            elif side == 'SELL' and price <= e21:
                ctx.close('reversion_done')
