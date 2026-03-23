from strategy_runtime import Strategy
from strategies.competitive_20._common import highest, lowest, ema, atr, candle_metrics, enough


class RangeBreakVolume(Strategy):
    name = 'Q19 Range Break + Volume'
    size = 0.01
    range_period = 20
    stop_atr = 1.3
    tp_rr = 2.2

    def on_bar(self, ctx):
        if ctx.bar_index < self.range_period + 30:
            return
        bars = ctx.bars
        close = bars['close']
        volume = bars['volume'].astype(float)
        price = float(ctx.bar['close'])
        hh = highest(bars['high'], self.range_period, exclude_current=True)
        ll = lowest(bars['low'], self.range_period, exclude_current=True)
        e50 = ema(close, 50)
        a = atr(bars, 14)
        avg_vol = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float('nan')
        vol_now = float(volume.iloc[-1])
        rng, body, _, _ = candle_metrics(ctx.bar)
        if not all(map(enough, [hh, ll, e50, a, avg_vol])) or rng <= 0:
            return
        strong_body = body / rng > 0.55
        vol_ok = vol_now > avg_vol * 1.2
        if not ctx.has_position:
            if price > hh and price > e50 and strong_body and vol_ok:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif price < ll and price < e50 and strong_body and vol_ok:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            if pos.trade.side.value == 'BUY' and price < hh:
                ctx.close('lost_range_break')
            elif pos.trade.side.value == 'SELL' and price > ll:
                ctx.close('lost_range_break')
