from strategy_runtime import Strategy
from strategies.competitive_20._common import ema, rsi, enough


class RSI2TrendSnapback(Strategy):
    name = 'Q03 RSI2 Trend Snapback'
    size = 0.01
    trend_fast = 50
    trend_slow = 200
    pullback_rsi = 2

    def on_bar(self, ctx):
        if ctx.bar_index < self.trend_slow + 5:
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        e_fast = ema(close, self.trend_fast)
        e_slow = ema(close, self.trend_slow)
        r2 = rsi(close, self.pullback_rsi)
        r14 = rsi(close, 14)
        e_exit = ema(close, 10)
        if not all(map(enough, [e_fast, e_slow, r2, r14, e_exit])):
            return
        if not ctx.has_position:
            if price > e_fast > e_slow and r2 < 12 and r14 > 45:
                sl = min(float(ctx.bar['low']), price - max(price * 0.0012, 2.0))
                ctx.buy(size=self.size, sl=sl)
            elif price < e_fast < e_slow and r2 > 88 and r14 < 55:
                sl = max(float(ctx.bar['high']), price + max(price * 0.0012, 2.0))
                ctx.sell(size=self.size, sl=sl)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and (price >= e_exit or r2 > 60):
                ctx.close('snapback_done')
            elif side == 'SELL' and (price <= e_exit or r2 < 40):
                ctx.close('snapback_done')
