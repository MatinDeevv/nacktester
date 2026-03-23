from strategy_runtime import Strategy
from strategies.competitive_20._common import ema, atr, rolling_std, enough


class BollingerSqueezeBreakout(Strategy):
    name = 'Q05 Bollinger Squeeze Breakout'
    size = 0.01
    bb_period = 20
    stop_atr = 1.3
    tp_rr = 2.6

    def on_bar(self, ctx):
        if ctx.bar_index < 60:
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        upper, mid, lower = ctx.bbands(self.bb_period, 2.0)
        a = atr(bars, 14)
        width = upper - lower if enough(upper) and enough(lower) else float('nan')
        avg_width = close.rolling(self.bb_period).std().iloc[-20:].mean() * 4 if len(close) >= 40 else float('nan')
        e20 = ema(close, 20)
        e50 = ema(close, 50)
        if not all(map(enough, [width, avg_width, a, e20, e50])):
            return
        squeezed = width < avg_width * 0.85
        if not ctx.has_position:
            if squeezed and price > upper and e20 > e50:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif squeezed and price < lower and e20 < e50:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and price < mid:
                ctx.close('back_inside_band')
            elif side == 'SELL' and price > mid:
                ctx.close('back_inside_band')
