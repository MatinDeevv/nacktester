from strategy_runtime import Strategy
from strategies.competitive_20._common import ema, atr, candle_metrics, enough


class NR7Expansion(Strategy):
    name = 'Q07 NR7 Expansion'
    size = 0.01
    stop_atr = 1.2
    tp_rr = 2.5

    def on_bar(self, ctx):
        if ctx.bar_index < 15:
            return
        bars = ctx.bars.iloc[-8:]
        ranges = bars['high'] - bars['low']
        current = bars.iloc[-1]
        prev = bars.iloc[-2]
        is_nr7 = float(ranges.iloc[-2]) == float(ranges.iloc[:-1].min())
        e34 = ema(ctx.bars['close'], 34)
        e89 = ema(ctx.bars['close'], 89)
        a = atr(ctx.bars, 14)
        if not all(map(enough, [e34, e89, a])):
            return
        if not ctx.has_position and is_nr7:
            if float(current['close']) > float(prev['high']) and e34 > e89:
                price = float(current['close'])
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif float(current['close']) < float(prev['low']) and e34 < e89:
                price = float(current['close'])
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        elif ctx.has_position:
            price = float(ctx.bar['close'])
            pos = ctx.position
            if pos is None:
                return
            if pos.trade.side.value == 'BUY' and price < e34:
                ctx.close('post_nr7_fail')
            elif pos.trade.side.value == 'SELL' and price > e34:
                ctx.close('post_nr7_fail')
