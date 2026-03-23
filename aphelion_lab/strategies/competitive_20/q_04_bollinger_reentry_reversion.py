from strategy_runtime import Strategy
from strategies.competitive_20._common import ema, atr, enough


class BollingerReentryReversion(Strategy):
    name = 'Q04 Bollinger Reentry Reversion'
    size = 0.01
    bb_period = 20
    bb_std = 2.2
    stop_atr = 1.2

    def on_init(self, ctx):
        self._state = None

    def on_bar(self, ctx):
        if ctx.bar_index < self.bb_period + 5:
            return
        upper, mid, lower = ctx.bbands(self.bb_period, self.bb_std)
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        a = atr(bars, 14)
        e50 = ema(close, 50)
        e100 = ema(close, 100)
        if not all(map(enough, [upper, mid, lower, a, e50, e100])):
            return
        regime_flat = abs(e50 - e100) < a * 1.2
        prev_close = float(close.iloc[-2])
        prev_upper, _, prev_lower = ctx.bbands(self.bb_period, self.bb_std)
        if not ctx.has_position:
            if not regime_flat:
                return
            if prev_close < prev_lower and price > lower:
                ctx.buy(size=self.size, sl=price - a * self.stop_atr, tp=mid)
            elif prev_close > prev_upper and price < upper:
                ctx.sell(size=self.size, sl=price + a * self.stop_atr, tp=mid)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and price >= mid:
                ctx.close('mean_hit')
            elif side == 'SELL' and price <= mid:
                ctx.close('mean_hit')
