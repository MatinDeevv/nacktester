from strategy_runtime import Strategy
from strategies.competitive_20._common import daily_vwap, rsi, atr, enough, liquid_hours


class VWAPReclaim(Strategy):
    name = 'Q13 VWAP Reclaim'
    size = 0.01
    stop_atr = 1.1
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 40:
            return
        ts = ctx.bar.name
        if not liquid_hours(ts):
            return
        bars = ctx.bars
        price = float(ctx.bar['close'])
        prev_close = float(bars['close'].iloc[-2])
        vwap = daily_vwap(bars)
        rs = rsi(bars['close'], 14)
        a = atr(bars, 14)
        if not all(map(enough, [vwap, rs, a])):
            return
        if not ctx.has_position:
            if prev_close < vwap and price > vwap and rs > 54:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif prev_close > vwap and price < vwap and rs < 46:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and price < vwap:
                ctx.close('lost_vwap')
            elif side == 'SELL' and price > vwap:
                ctx.close('lost_vwap')
