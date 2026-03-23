from strategy_runtime import Strategy
from strategies.competitive_20._common import zscore, ema, atr, rsi, enough


class ZScoreMeanReversion(Strategy):
    name = 'Q17 Z-Score Mean Reversion'
    size = 0.01
    stop_atr = 1.15

    def on_bar(self, ctx):
        if ctx.bar_index < 60:
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        z = zscore(close, 30)
        e20 = ema(close, 20)
        e50 = ema(close, 50)
        rs = rsi(close, 14)
        a = atr(bars, 14)
        if not all(map(enough, [z, e20, e50, rs, a])):
            return
        flatish = abs(e20 - e50) < a * 0.45
        if not ctx.has_position and flatish:
            if z < -2.4 and rs < 30:
                ctx.buy(size=self.size, sl=price - a * self.stop_atr, tp=e20)
            elif z > 2.4 and rs > 70:
                ctx.sell(size=self.size, sl=price + a * self.stop_atr, tp=e20)
        else:
            pos = ctx.position
            if pos is None:
                return
            if pos.trade.side.value == 'BUY' and price >= e20:
                ctx.close('z_reverted')
            elif pos.trade.side.value == 'SELL' and price <= e20:
                ctx.close('z_reverted')
