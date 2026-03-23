from strategy_runtime import Strategy
from strategies.competitive_20._common import daily_vwap, ema, atr, enough, liquid_hours


class VWAPPullbackTrend(Strategy):
    name = 'Q14 VWAP Pullback Trend'
    size = 0.01
    stop_atr = 1.25
    tp_rr = 2.3

    def on_bar(self, ctx):
        if ctx.bar_index < 100:
            return
        ts = ctx.bar.name
        if not liquid_hours(ts):
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        low = float(ctx.bar['low'])
        high = float(ctx.bar['high'])
        vwap = daily_vwap(bars)
        e20 = ema(close, 20)
        e50 = ema(close, 50)
        a = atr(bars, 14)
        if not all(map(enough, [vwap, e20, e50, a])):
            return
        if not ctx.has_position:
            if price > vwap and e20 > e50 and low <= vwap + a * 0.15:
                sl = min(low, price - a * self.stop_atr)
                risk = price - sl
                if risk > a * 0.25:
                    ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif price < vwap and e20 < e50 and high >= vwap - a * 0.15:
                sl = max(high, price + a * self.stop_atr)
                risk = sl - price
                if risk > a * 0.25:
                    ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and price < e20:
                ctx.close('pullback_failed')
            elif side == 'SELL' and price > e20:
                ctx.close('pullback_failed')
