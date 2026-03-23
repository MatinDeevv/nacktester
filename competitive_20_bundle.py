from strategy_runtime import Strategy
from strategies.competitive_20._common import *

# ===== q_01_ema_pullback_continuation.py =====


class EMAPullbackContinuation(Strategy):
    name = 'Q01 EMA Pullback Continuation'
    size = 0.01
    fast = 20
    mid = 50
    slow = 100
    pullback_lookback = 6
    stop_atr = 1.6
    tp_rr = 2.2

    def on_bar(self, ctx):
        if ctx.bar_index < self.slow + 10:
            return
        bars = ctx.bars
        close = bars['close']
        low = bars['low']
        high = bars['high']
        price = float(ctx.bar['close'])
        e1 = ema(close, self.fast)
        e2 = ema(close, self.mid)
        e3 = ema(close, self.slow)
        rs = rsi(close, 14)
        a = atr(bars, 14)
        if not all(map(enough, [e1, e2, e3, rs, a])):
            return
        prev_low = lowest(low, self.pullback_lookback, exclude_current=True)
        prev_high = highest(high, self.pullback_lookback, exclude_current=True)
        if not ctx.has_position:
            if e1 > e2 > e3 and price > e1 and float(ctx.bar['low']) <= e1 + a * 0.15 and rs > 52:
                sl = min(prev_low, price - a * self.stop_atr)
                risk = price - sl
                if risk > a * 0.35:
                    ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif e1 < e2 < e3 and price < e1 and float(ctx.bar['high']) >= e1 - a * 0.15 and rs < 48:
                sl = max(prev_high, price + a * self.stop_atr)
                risk = sl - price
                if risk > a * 0.35:
                    ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and (price < e2 or rs < 45):
                ctx.close('trend_lost')
            elif side == 'SELL' and (price > e2 or rs > 55):
                ctx.close('trend_lost')

# ===== q_02_adaptive_trend_breakout.py =====


class AdaptiveTrendBreakout(Strategy):
    name = 'Q02 Adaptive Trend Breakout'
    size = 0.01
    trend_period = 50
    breakout_period = 20
    stop_atr = 1.4
    tp_rr = 2.4

    def on_bar(self, ctx):
        if ctx.bar_index < self.trend_period + self.breakout_period + 10:
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        e_now = ema(close, self.trend_period)
        e_prev = ema_prev(close, self.trend_period, back=3)
        a_now = atr(bars, 14)
        a_prev = atr_prev(bars, 14, back=8)
        breakout_high = highest(bars['high'], self.breakout_period, exclude_current=True)
        breakout_low = lowest(bars['low'], self.breakout_period, exclude_current=True)
        if not all(map(enough, [e_now, e_prev, a_now, a_prev, breakout_high, breakout_low])):
            return
        expanding = a_now > a_prev * 1.08
        if not ctx.has_position:
            if price > breakout_high and price > e_now and e_now > e_prev and expanding:
                sl = price - a_now * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif price < breakout_low and price < e_now and e_now < e_prev and expanding:
                sl = price + a_now * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and price < e_now:
                ctx.close('ema_fail')
            elif side == 'SELL' and price > e_now:
                ctx.close('ema_fail')

# ===== q_03_rsi2_trend_snapback.py =====


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

# ===== q_04_bollinger_reentry_reversion.py =====


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

# ===== q_05_bollinger_squeeze_breakout.py =====


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

# ===== q_06_donchian_retest_breakout.py =====


class DonchianRetestBreakout(Strategy):
    name = 'Q06 Donchian Retest Breakout'
    size = 0.01
    channel = 30
    stop_atr = 1.5
    tp_rr = 2.3

    def on_init(self, ctx):
        self._pending_long = False
        self._pending_short = False
        self._level = None

    def on_bar(self, ctx):
        if ctx.bar_index < self.channel + 5:
            return
        bars = ctx.bars
        price = float(ctx.bar['close'])
        high = float(ctx.bar['high'])
        low = float(ctx.bar['low'])
        a = atr(bars, 14)
        hh = highest(bars['high'], self.channel, exclude_current=True)
        ll = lowest(bars['low'], self.channel, exclude_current=True)
        if not all(map(enough, [a, hh, ll])):
            return
        if not ctx.has_position:
            if self._pending_long and low <= self._level + a * 0.2 and price > self._level:
                sl = min(low, price - a * self.stop_atr)
                risk = price - sl
                if risk > a * 0.35:
                    ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
                self._pending_long = False
            elif self._pending_short and high >= self._level - a * 0.2 and price < self._level:
                sl = max(high, price + a * self.stop_atr)
                risk = sl - price
                if risk > a * 0.35:
                    ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
                self._pending_short = False
            else:
                self._pending_long = high > hh
                self._pending_short = low < ll
                self._level = hh if self._pending_long else (ll if self._pending_short else self._level)
        else:
            pos = ctx.position
            if pos is None:
                return
            if pos.trade.side.value == 'BUY' and price < self._level:
                ctx.close('failed_retest')
            elif pos.trade.side.value == 'SELL' and price > self._level:
                ctx.close('failed_retest')

# ===== q_07_nr7_expansion.py =====


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

# ===== q_08_inside_bar_breakout.py =====


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

# ===== q_09_three_bar_reversal.py =====


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

# ===== q_10_london_orb.py =====


class LondonORB(Strategy):
    name = 'Q10 London ORB'
    size = 0.01
    stop_atr = 1.3
    tp_rr = 2.0

    def on_init(self, ctx):
        self._day = None
        self._traded = False

    def on_bar(self, ctx):
        ts = ctx.bar.name
        day = ts.date()
        if self._day != day:
            self._day = day
            self._traded = False
        if ctx.bar_index < 100:
            return
        bars = ctx.bars
        a = atr(bars, 14)
        if not enough(a):
            return
        asian = session_slice(bars, 0, 6, same_day_only=True)
        if len(asian) < 12:
            return
        asian_high = float(asian['high'].max())
        asian_low = float(asian['low'].min())
        rng = asian_high - asian_low
        price = float(ctx.bar['close'])
        hour = ts.hour
        if not ctx.has_position and not self._traded and 6 <= hour <= 10 and rng > a * 1.2:
            if price > asian_high:
                sl = max(asian_low, price - a * self.stop_atr)
                risk = price - sl
                if risk > a * 0.3:
                    ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
                    self._traded = True
            elif price < asian_low:
                sl = min(asian_high, price + a * self.stop_atr)
                risk = sl - price
                if risk > a * 0.3:
                    ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
                    self._traded = True
        elif ctx.has_position and hour >= 15:
            ctx.close('london_session_end')

# ===== q_11_newyork_orb.py =====


class NewYorkORB(Strategy):
    name = 'Q11 New York ORB'
    size = 0.01
    stop_atr = 1.25
    tp_rr = 2.2

    def on_init(self, ctx):
        self._day = None
        self._traded = False

    def on_bar(self, ctx):
        ts = ctx.bar.name
        day = ts.date()
        if self._day != day:
            self._day = day
            self._traded = False
        if ctx.bar_index < 100:
            return
        bars = ctx.bars
        a = atr(bars, 14)
        vwap = daily_vwap(bars)
        if not all(map(enough, [a, vwap])):
            return
        pre = session_slice(bars, 12, 14, same_day_only=True)
        if len(pre) < 6:
            return
        orb_high = float(pre['high'].max())
        orb_low = float(pre['low'].min())
        price = float(ctx.bar['close'])
        hour = ts.hour
        if not ctx.has_position and not self._traded and 14 <= hour <= 17:
            if price > orb_high and price > vwap:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
                self._traded = True
            elif price < orb_low and price < vwap:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
                self._traded = True
        elif ctx.has_position and hour >= 20:
            ctx.close('ny_session_end')

# ===== q_12_asian_range_fade.py =====


class LondonMomentumBurst(Strategy):
    name = 'Q12 London Momentum Burst'
    size = 0.01
    stop_atr = 1.2
    tp_rr = 2.1

    def on_init(self, ctx):
        self._day = None
        self._traded = False

    def on_bar(self, ctx):
        ts = ctx.bar.name
        day = ts.date()
        if self._day != day:
            self._day = day
            self._traded = False
        if ctx.bar_index < 80:
            return
        bars = ctx.bars
        opening = session_slice(bars, 6, 7, same_day_only=True)
        a = atr(bars, 14)
        vwap = daily_vwap(bars)
        if len(opening) < 6 or not all(map(enough, [a, vwap])):
            return
        burst_high = float(opening['high'].max())
        burst_low = float(opening['low'].min())
        vol_avg = float(opening['volume'].astype(float).mean())
        price = float(ctx.bar['close'])
        vol_now = float(ctx.bar['volume'])
        hour = ts.hour
        if not ctx.has_position and not self._traded and 7 <= hour <= 11:
            if price > burst_high and price > vwap and vol_now >= vol_avg * 0.9:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
                self._traded = True
            elif price < burst_low and price < vwap and vol_now >= vol_avg * 0.9:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
                self._traded = True
        elif ctx.has_position and hour >= 14:
            ctx.close('london_burst_end')

# ===== q_13_vwap_reclaim.py =====


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

# ===== q_14_vwap_pullback_trend.py =====


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

# ===== q_15_macd_hist_reversal.py =====


class MACDHistogramReversal(Strategy):
    name = 'Q15 MACD Histogram Reversal'
    size = 0.01
    stop_atr = 1.25
    tp_rr = 2.1

    def on_bar(self, ctx):
        if ctx.bar_index < 60:
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        hist_now, hist_prev = macd_hist(close, 12, 26, 9)
        e50 = ema(close, 50)
        a = atr(bars, 14)
        if not all(map(enough, [hist_now, hist_prev, e50, a])):
            return
        if not ctx.has_position:
            if hist_prev <= 0 < hist_now and price > e50:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif hist_prev >= 0 > hist_now and price < e50:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and hist_now < 0:
                ctx.close('hist_flip')
            elif side == 'SELL' and hist_now > 0:
                ctx.close('hist_flip')

# ===== q_16_keltner_channel_breakout.py =====


class KeltnerChannelBreakout(Strategy):
    name = 'Q16 Keltner Channel Breakout'
    size = 0.01
    stop_atr = 1.35
    tp_rr = 2.4

    def on_bar(self, ctx):
        if ctx.bar_index < 70:
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        e20 = ema(close, 20)
        e50 = ema(close, 50)
        a = atr(bars, 20)
        if not all(map(enough, [e20, e50, a])):
            return
        upper = e20 + a * 1.5
        lower = e20 - a * 1.5
        if not ctx.has_position:
            if price > upper and e20 > e50:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif price < lower and e20 < e50:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            if pos.trade.side.value == 'BUY' and price < e20:
                ctx.close('back_inside_keltner')
            elif pos.trade.side.value == 'SELL' and price > e20:
                ctx.close('back_inside_keltner')

# ===== q_17_zscore_mean_reversion.py =====


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

# ===== q_18_stochastic_trend_reentry.py =====


class RSI50TrendReentry(Strategy):
    name = 'Q18 RSI50 Trend Reentry'
    size = 0.01
    stop_atr = 1.25
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 220:
            return
        bars = ctx.bars
        close = bars['close']
        price = float(ctx.bar['close'])
        e50 = ema(close, 50)
        e200 = ema(close, 200)
        r_now = rsi(close, 14)
        r_prev = rsi_prev(close, 14, 1)
        a = atr(bars, 14)
        if not all(map(enough, [e50, e200, r_now, r_prev, a])):
            return
        if not ctx.has_position:
            if price > e50 > e200 and r_prev <= 48 < r_now:
                sl = price - a * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif price < e50 < e200 and r_prev >= 52 > r_now:
                sl = price + a * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            pos = ctx.position
            if pos is None:
                return
            side = pos.trade.side.value
            if side == 'BUY' and (r_now > 68 or price < e50):
                ctx.close('rsi_reentry_done')
            elif side == 'SELL' and (r_now < 32 or price > e50):
                ctx.close('rsi_reentry_done')

# ===== q_19_range_break_volume.py =====


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

# ===== q_20_atr_compression_expansion.py =====


class ATRCompressionExpansion(Strategy):
    name = 'Q20 ATR Compression Expansion'
    size = 0.01
    trigger_period = 12
    stop_atr = 1.2
    tp_rr = 2.5

    def on_bar(self, ctx):
        if ctx.bar_index < 80:
            return
        bars = ctx.bars
        price = float(ctx.bar['close'])
        a_short = atr(bars, 10)
        a_long = atr(bars, 30)
        hh = highest(bars['high'], self.trigger_period, exclude_current=True)
        ll = lowest(bars['low'], self.trigger_period, exclude_current=True)
        rng, body, _, _ = candle_metrics(ctx.bar)
        if not all(map(enough, [a_short, a_long, hh, ll])) or rng <= 0:
            return
        compressed = a_short < a_long * 0.78
        strong = body / rng > 0.6
        if not ctx.has_position and compressed and strong:
            if price > hh:
                sl = price - a_short * self.stop_atr
                risk = price - sl
                ctx.buy(size=self.size, sl=sl, tp=price + risk * self.tp_rr)
            elif price < ll:
                sl = price + a_short * self.stop_atr
                risk = sl - price
                ctx.sell(size=self.size, sl=sl, tp=price - risk * self.tp_rr)
        else:
            if ctx.position and ctx.position.trade.side.value == 'BUY' and price < hh:
                ctx.close('compression_break_fail')
            elif ctx.position and ctx.position.trade.side.value == 'SELL' and price > ll:
                ctx.close('compression_break_fail')
