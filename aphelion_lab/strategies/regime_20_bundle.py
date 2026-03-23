from strategy_runtime import Strategy
from strategies.regime_20._common import (
    allow_reversion,
    allow_trend,
    atr,
    body_ratio,
    current_side,
    daily_vwap,
    ema,
    enough,
    highest,
    lowest,
    market_is,
    open_trade,
    range_ratio,
    risk_target,
    rsi,
    session_is,
    session_slice,
    spread_ok,
    trail_to_level,
    volume_ratio,
    volatility_is,
    zscore,
)


class TrendEntropyPullback(Strategy):
    name = "R01 Trend Entropy Pullback"
    floor_size = 0.01
    fast = 21
    slow = 55
    stop_atr = 1.5
    tp_rr = 2.4

    def on_bar(self, ctx):
        if ctx.bar_index < self.slow + 10:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        e_fast = ema(bars["close"], self.fast)
        e_slow = ema(bars["close"], self.slow)
        a = atr(bars, 14)
        if not enough(price, low, high, e_fast, e_slow, a):
            return
        if not ctx.has_position:
            if allow_trend(ctx) and spread_ok(ctx, 0.08) and ctx.hurst > 0.57 and ctx.entropy < 0.72:
                if e_fast > e_slow and price > e_slow and low <= e_fast + a * 0.15:
                    sl = min(lowest(bars["low"], 6, exclude_current=True), price - a * self.stop_atr)
                    if enough(sl) and (price - sl) > a * 0.35:
                        open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif e_fast < e_slow and price < e_slow and high >= e_fast - a * 0.15:
                    sl = max(highest(bars["high"], 6, exclude_current=True), price + a * self.stop_atr)
                    if enough(sl) and (sl - price) > a * 0.35:
                        open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            side = current_side(ctx)
            if side == "BUY":
                trail_to_level(ctx, e_fast - a * 0.20, buffer=a * 0.05)
                if price < e_slow or market_is(ctx, "range", "mean_revert", "stress"):
                    ctx.close("trend_lost")
            elif side == "SELL":
                trail_to_level(ctx, e_fast + a * 0.20, buffer=a * 0.05)
                if price > e_slow or market_is(ctx, "range", "mean_revert", "stress"):
                    ctx.close("trend_lost")


class DistributionShiftBreakout(Strategy):
    name = "R02 Distribution Shift Breakout"
    floor_size = 0.01
    channel = 20
    stop_atr = 1.35
    tp_rr = 2.2

    def on_bar(self, ctx):
        if ctx.bar_index < self.channel + 30:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        a = atr(bars, 14)
        breakout_high = highest(bars["high"], self.channel, exclude_current=True)
        breakout_low = lowest(bars["low"], self.channel, exclude_current=True)
        vr = volume_ratio(bars, 20)
        rr = range_ratio(bars, 20)
        br = body_ratio(ctx.bar)
        if not all(map(enough, [price, a, breakout_high, breakout_low, vr, rr, br])):
            return
        if not ctx.has_position:
            if (
                session_is(ctx, "london", "overlap", "new_york")
                and spread_ok(ctx, 0.07)
                and volatility_is(ctx, "high", "extreme")
                and ctx.distribution_shift > 1.05
                and ctx.jump_intensity < 0.24
            ):
                if price > breakout_high and vr > 1.10 and rr > 1.05 and br > 0.55:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < breakout_low and vr > 1.10 and rr > 1.05 and br > 0.55:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            side = current_side(ctx)
            if side == "BUY" and (price < breakout_high or ctx.distribution_shift < 0.65):
                ctx.close("shift_failed")
            elif side == "SELL" and (price > breakout_low or ctx.distribution_shift < 0.65):
                ctx.close("shift_failed")


class HurstRibbonReentry(Strategy):
    name = "R03 Hurst Ribbon Reentry"
    floor_size = 0.01
    stop_atr = 1.4
    tp_rr = 2.5

    def on_bar(self, ctx):
        if ctx.bar_index < 70:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        e8 = ema(bars["close"], 8)
        e21 = ema(bars["close"], 21)
        e55 = ema(bars["close"], 55)
        a = atr(bars, 14)
        if not all(map(enough, [price, low, high, e8, e21, e55, a])):
            return
        if not ctx.has_position:
            if allow_trend(ctx) and ctx.hurst > 0.60 and ctx.entropy < 0.70:
                if e8 > e21 > e55 and price > e55 and low <= e21 + a * 0.10:
                    sl = min(lowest(bars["low"], 5, exclude_current=True), price - a * self.stop_atr)
                    if enough(sl):
                        open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif e8 < e21 < e55 and price < e55 and high >= e21 - a * 0.10:
                    sl = max(highest(bars["high"], 5, exclude_current=True), price + a * self.stop_atr)
                    if enough(sl):
                        open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            side = current_side(ctx)
            if side == "BUY":
                trail_to_level(ctx, e55, buffer=a * 0.10)
                if price < e55 or ctx.hurst < 0.54:
                    ctx.close("hurst_fade")
            elif side == "SELL":
                trail_to_level(ctx, e55, buffer=a * 0.10)
                if price > e55 or ctx.hurst < 0.54:
                    ctx.close("hurst_fade")


class GapReclaimTrend(Strategy):
    name = "R04 Gap Reclaim Trend"
    floor_size = 0.01
    stop_atr = 1.2
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 60:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        opn = float(ctx.bar["open"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        is_gap = bool(ctx.ind("gap"))
        e13 = ema(bars["close"], 13)
        e50 = ema(bars["close"], 50)
        a = atr(bars, 14)
        if not all(map(enough, [price, opn, low, high, e13, e50, a])):
            return
        if not ctx.has_position:
            if is_gap and allow_trend(ctx) and session_is(ctx, "london", "new_york", "overlap"):
                if e13 > e50 and price > max(opn, e13) and low < opn and ctx.distribution_shift > 0.60:
                    sl = min(low, price - a * self.stop_atr)
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif e13 < e50 and price < min(opn, e13) and high > opn and ctx.distribution_shift > 0.60:
                    sl = max(high, price + a * self.stop_atr)
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            side = current_side(ctx)
            if side == "BUY" and (price < opn or market_is(ctx, "range", "mean_revert")):
                ctx.close("gap_failed")
            elif side == "SELL" and (price > opn or market_is(ctx, "range", "mean_revert")):
                ctx.close("gap_failed")


class JumpCompressionExpansion(Strategy):
    name = "R05 Jump Compression Expansion"
    floor_size = 0.01
    stop_atr = 1.35
    tp_rr = 2.1

    def on_bar(self, ctx):
        if ctx.bar_index < 90:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        a = atr(bars, 14)
        prev_jump = float(bars["jump_intensity"].iloc[-2])
        rr = range_ratio(bars, 18)
        hh = highest(bars["high"], 12, exclude_current=True)
        ll = lowest(bars["low"], 12, exclude_current=True)
        if not all(map(enough, [price, a, prev_jump, rr, hh, ll])):
            return
        if not ctx.has_position:
            if (
                volatility_is(ctx, "high", "extreme")
                and ctx.distribution_shift > 0.90
                and prev_jump > 0.10
                and ctx.jump_intensity < prev_jump
                and rr > 1.05
            ):
                if price > hh and market_is(ctx, "transition", "trend", "stress"):
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < ll and market_is(ctx, "transition", "trend", "stress"):
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if ctx.jump_intensity > 0.20:
                ctx.close("jump_returned")


class RangeEntropyFade(Strategy):
    name = "R06 Range Entropy Fade"
    floor_size = 0.01
    stop_atr = 1.05

    def on_bar(self, ctx):
        if ctx.bar_index < 40:
            return
        price = float(ctx.bar["close"])
        a = atr(ctx.bars, 14)
        rs = rsi(ctx.bars["close"], 5)
        upper, mid, lower = ctx.bbands(20, 2.0)
        if not all(map(enough, [price, a, rs, upper, mid, lower])):
            return
        if not ctx.has_position:
            if allow_reversion(ctx) and ctx.hurst < 0.48 and ctx.entropy > 0.76 and spread_ok(ctx, 0.09):
                if price < lower and rs < 30:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, mid, floor=self.floor_size)
                elif price > upper and rs > 70:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, mid, floor=self.floor_size)
        else:
            if market_is(ctx, "trend", "stress"):
                ctx.close("regime_flip")


class VWAPMeanReclaim(Strategy):
    name = "R07 VWAP Mean Reclaim"
    floor_size = 0.01
    stop_atr = 1.1

    def on_bar(self, ctx):
        if ctx.bar_index < 60:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        vwap = daily_vwap(bars)
        rs = rsi(bars["close"], 6)
        a = atr(bars, 14)
        if not all(map(enough, [price, low, high, vwap, rs, a])):
            return
        if not ctx.has_position:
            if allow_reversion(ctx) and ctx.distribution_shift < 1.00 and ctx.jump_intensity < 0.10:
                if low < vwap - a * 0.60 and price > vwap - a * 0.20 and rs < 45:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, vwap, floor=self.floor_size)
                elif high > vwap + a * 0.60 and price < vwap + a * 0.20 and rs > 55:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, vwap, floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY" and price >= vwap:
                ctx.close("vwap_hit")
            elif current_side(ctx) == "SELL" and price <= vwap:
                ctx.close("vwap_hit")


class LondonExpansionSpreadGate(Strategy):
    name = "R08 London Expansion Spread Gate"
    floor_size = 0.01
    stop_atr = 1.3
    tp_rr = 2.0

    def on_init(self, ctx):
        self._traded_day = None

    def on_bar(self, ctx):
        if ctx.bar_index < 100:
            return
        ts = ctx.bar.name
        if self._traded_day != ts.date() and ts.hour < 7:
            self._traded_day = None
        if ctx.has_position and ts.hour >= 15:
            ctx.close("london_end")
            return
        bars = ctx.bars
        asian = session_slice(bars, 0, 7, same_day_only=True)
        price = float(ctx.bar["close"])
        a = atr(bars, 14)
        vr = volume_ratio(bars, 20)
        if len(asian) < 4 or not all(map(enough, [price, a, vr])):
            return
        rng_high = float(asian["high"].max())
        rng_low = float(asian["low"].min())
        if not ctx.has_position and self._traded_day is None:
            if (
                session_is(ctx, "london", "overlap")
                and spread_ok(ctx, 0.05)
                and allow_trend(ctx)
                and not volatility_is(ctx, "low")
            ):
                if price > rng_high and vr > 1.05:
                    sl = min(rng_low, price - a * self.stop_atr)
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                    self._traded_day = ts.date()
                elif price < rng_low and vr > 1.05:
                    sl = max(rng_high, price + a * self.stop_atr)
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
                    self._traded_day = ts.date()


class NewYorkShockFade(Strategy):
    name = "R09 New York Shock Fade"
    floor_size = 0.01
    stop_atr = 1.2

    def on_bar(self, ctx):
        if ctx.bar_index < 40:
            return
        if not session_is(ctx, "new_york", "overlap"):
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        rr = range_ratio(bars, 20)
        rs = rsi(bars["close"], 5)
        e21 = ema(bars["close"], 21)
        a = atr(bars, 14)
        if not all(map(enough, [price, rr, rs, e21, a])):
            return
        if not ctx.has_position:
            if (market_is(ctx, "stress") or ctx.jump_intensity > 0.16) and rr > 1.35 and ctx.entropy > 0.72:
                if rs > 78:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, e21, floor=self.floor_size)
                elif rs < 22:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, e21, floor=self.floor_size)
        else:
            if ctx.jump_intensity < 0.08:
                ctx.close("shock_cooled")


class EntropyCompressionBreakout(Strategy):
    name = "R10 Entropy Compression Breakout"
    floor_size = 0.01
    stop_atr = 1.25
    tp_rr = 2.3

    def on_bar(self, ctx):
        if ctx.bar_index < 90:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        a = atr(bars, 14)
        shift_prev = float(bars["distribution_shift_norm"].iloc[-2])
        hh = highest(bars["high"], 30, exclude_current=True)
        ll = lowest(bars["low"], 30, exclude_current=True)
        vr = volume_ratio(bars, 20)
        if not all(map(enough, [price, a, shift_prev, hh, ll, vr])):
            return
        if not ctx.has_position:
            if (
                ctx.entropy < 0.60
                and volatility_is(ctx, "low", "normal")
                and ctx.distribution_shift > shift_prev + 0.15
                and market_is(ctx, "transition", "trend")
            ):
                if price > hh and vr > 1.0:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < ll and vr > 1.0:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY" and price < hh:
                ctx.close("compression_fail")
            elif current_side(ctx) == "SELL" and price > ll:
                ctx.close("compression_fail")


class TransitionReacceleration(Strategy):
    name = "R11 Transition Reacceleration"
    floor_size = 0.01
    stop_atr = 1.35
    tp_rr = 2.1

    def on_bar(self, ctx):
        if ctx.bar_index < 90:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        e21 = ema(bars["close"], 21)
        e55 = ema(bars["close"], 55)
        prev_hurst = float(bars["hurst_128"].iloc[-2])
        a = atr(bars, 14)
        vr = volume_ratio(bars, 20)
        if not all(map(enough, [price, low, high, e21, e55, prev_hurst, a, vr])):
            return
        if not ctx.has_position:
            if market_is(ctx, "transition") and ctx.hurst > prev_hurst > 0.50 and ctx.distribution_shift > 0.90 and vr > 1.05:
                if e21 > e55 and price > e21 and low <= e21 + a * 0.10:
                    sl = min(low, price - a * self.stop_atr)
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif e21 < e55 and price < e21 and high >= e21 - a * 0.10:
                    sl = max(high, price + a * self.stop_atr)
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY":
                trail_to_level(ctx, e21, buffer=a * 0.08)
                if ctx.hurst < prev_hurst:
                    ctx.close("reaccel_lost")
            elif current_side(ctx) == "SELL":
                trail_to_level(ctx, e21, buffer=a * 0.08)
                if ctx.hurst < prev_hurst:
                    ctx.close("reaccel_lost")


class VolatilityChannelBreakout(Strategy):
    name = "R12 Volatility Channel Breakout"
    floor_size = 0.01
    channel = 20
    stop_atr = 1.25
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < self.channel + 40:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        e10 = ema(bars["close"], 10)
        a = atr(bars, 14)
        hh = highest(bars["high"], self.channel, exclude_current=True)
        ll = lowest(bars["low"], self.channel, exclude_current=True)
        if not all(map(enough, [price, e10, a, hh, ll])):
            return
        if not ctx.has_position:
            if volatility_is(ctx, "high", "extreme") and ctx.jump_intensity < 0.25 and ctx.distribution_shift > 0.80:
                if price > hh:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < ll:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY":
                trail_to_level(ctx, e10, buffer=a * 0.08)
                if price < e10:
                    ctx.close("vol_channel_fail")
            elif current_side(ctx) == "SELL":
                trail_to_level(ctx, e10, buffer=a * 0.08)
                if price > e10:
                    ctx.close("vol_channel_fail")


class ZScoreMeanReversionGuarded(Strategy):
    name = "R13 ZScore Mean Reversion Guarded"
    floor_size = 0.01
    stop_atr = 1.1

    def on_bar(self, ctx):
        if ctx.bar_index < 60:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        z = zscore(bars["close"], 30)
        e20 = ema(bars["close"], 20)
        a = atr(bars, 14)
        if not all(map(enough, [price, z, e20, a])):
            return
        if not ctx.has_position:
            if allow_reversion(ctx) and ctx.distribution_shift < 1.05 and ctx.jump_intensity < 0.14:
                if z < -2.0:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, e20, floor=self.floor_size)
                elif z > 2.0:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, e20, floor=self.floor_size)
        else:
            if market_is(ctx, "trend", "stress"):
                ctx.close("z_guard_exit")


class SessionRotationHybrid(Strategy):
    name = "R14 Session Rotation Hybrid"
    floor_size = 0.01
    stop_atr = 1.25
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 90:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        e13 = ema(bars["close"], 13)
        a = atr(bars, 14)
        vwap = daily_vwap(bars)
        hh = highest(bars["high"], 15, exclude_current=True)
        ll = lowest(bars["low"], 15, exclude_current=True)
        if not all(map(enough, [price, e13, a, vwap, hh, ll])):
            return
        if not ctx.has_position:
            if session_is(ctx, "london", "overlap") and allow_trend(ctx) and ctx.hurst > 0.55:
                if price > hh:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < ll:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
            elif session_is(ctx, "new_york") and allow_reversion(ctx) and ctx.hurst < 0.48:
                if price < vwap - a * 0.7:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, vwap, floor=self.floor_size)
                elif price > vwap + a * 0.7:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, vwap, floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY":
                trail_to_level(ctx, e13, buffer=a * 0.08)
            elif current_side(ctx) == "SELL":
                trail_to_level(ctx, e13, buffer=a * 0.08)
            if session_is(ctx, "off_hours") or market_is(ctx, "stress"):
                ctx.close("rotation_exit")


class HurstFlipReversal(Strategy):
    name = "R15 Hurst Flip Reversal"
    floor_size = 0.01
    stop_atr = 1.2
    tp_rr = 1.8

    def on_bar(self, ctx):
        if ctx.bar_index < 160:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        prev_hurst = float(bars["hurst_128"].iloc[-2])
        e13 = ema(bars["close"], 13)
        e34 = ema(bars["close"], 34)
        rs = rsi(bars["close"], 6)
        a = atr(bars, 14)
        if not all(map(enough, [price, prev_hurst, e13, e34, rs, a])):
            return
        if not ctx.has_position:
            if prev_hurst > 0.58 and ctx.hurst < 0.50 and ctx.distribution_shift > 0.90 and price < e13 < e34 and rs < 45:
                sl = price + a * self.stop_atr
                open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
            elif prev_hurst < 0.44 and ctx.hurst > 0.54 and ctx.distribution_shift > 0.90 and price > e13 > e34 and rs > 55:
                sl = price - a * self.stop_atr
                open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY" and ctx.hurst < 0.50:
                ctx.close("hurst_flip_done")
            elif current_side(ctx) == "SELL" and ctx.hurst > 0.54:
                ctx.close("hurst_flip_done")


class JumpFilteredVWAPPullback(Strategy):
    name = "R16 Jump Filtered VWAP Pullback"
    floor_size = 0.01
    stop_atr = 1.3
    tp_rr = 2.1

    def on_bar(self, ctx):
        if ctx.bar_index < 100:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        e21 = ema(bars["close"], 21)
        e55 = ema(bars["close"], 55)
        vwap = daily_vwap(bars)
        a = atr(bars, 14)
        if not all(map(enough, [price, low, high, e21, e55, vwap, a])):
            return
        if not ctx.has_position:
            if allow_trend(ctx) and ctx.jump_intensity < 0.10 and session_is(ctx, "london", "new_york", "overlap"):
                if price > vwap and e21 > e55 and low <= max(vwap, e21) + a * 0.10:
                    sl = min(low, price - a * self.stop_atr)
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < vwap and e21 < e55 and high >= min(vwap, e21) - a * 0.10:
                    sl = max(high, price + a * self.stop_atr)
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY":
                trail_to_level(ctx, max(vwap, e55), buffer=a * 0.06)
                if price < e55:
                    ctx.close("vwap_pullback_fail")
            elif current_side(ctx) == "SELL":
                trail_to_level(ctx, min(vwap, e55), buffer=a * 0.06)
                if price > e55:
                    ctx.close("vwap_pullback_fail")


class StressEscapeBreakout(Strategy):
    name = "R17 Stress Escape Breakout"
    floor_size = 0.01
    stop_atr = 1.1
    tp_rr = 1.7

    def on_bar(self, ctx):
        if ctx.bar_index < 120:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        a = atr(bars, 14)
        hh = highest(bars["high"], 40, exclude_current=True)
        ll = lowest(bars["low"], 40, exclude_current=True)
        vr = volume_ratio(bars, 20)
        br = body_ratio(ctx.bar)
        if not all(map(enough, [price, a, hh, ll, vr, br])):
            return
        if not ctx.has_position:
            if (
                market_is(ctx, "stress")
                and volatility_is(ctx, "high", "extreme")
                and ctx.distribution_shift > 1.40
                and 0.08 < ctx.jump_intensity < 0.28
            ):
                if price > hh and vr > 1.20 and br > 0.60:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < ll and vr > 1.20 and br > 0.60:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if ctx.distribution_shift < 0.80:
                ctx.close("stress_release")


class DistributionShiftExhaustion(Strategy):
    name = "R18 Distribution Shift Exhaustion"
    floor_size = 0.01
    stop_atr = 1.1

    def on_bar(self, ctx):
        if ctx.bar_index < 60:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        prev_shift = float(bars["distribution_shift_norm"].iloc[-2])
        prev_close = float(bars["close"].iloc[-2])
        rs = rsi(bars["close"], 5)
        e21 = ema(bars["close"], 21)
        a = atr(bars, 14)
        rr = range_ratio(bars, 14)
        if not all(map(enough, [price, prev_shift, prev_close, rs, e21, a, rr])):
            return
        if not ctx.has_position:
            if prev_shift > 1.70 and ctx.distribution_shift < prev_shift * 0.85 and rr > 1.05:
                if price < prev_close and rs > 65:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, e21, floor=self.floor_size)
                elif price > prev_close and rs < 35:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, e21, floor=self.floor_size)
        else:
            if ctx.distribution_shift < 0.80:
                ctx.close("shift_exhausted")


class EntropyHurstDualMode(Strategy):
    name = "R19 Entropy Hurst Dual Mode"
    floor_size = 0.01
    stop_atr = 1.2
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 90:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        a = atr(bars, 14)
        e21 = ema(bars["close"], 21)
        hh = highest(bars["high"], 15, exclude_current=True)
        ll = lowest(bars["low"], 15, exclude_current=True)
        z = zscore(bars["close"], 25)
        if not all(map(enough, [price, a, e21, hh, ll, z])):
            return
        if not ctx.has_position:
            if allow_trend(ctx) and ctx.hurst > 0.62 and ctx.entropy < 0.65:
                if price > hh:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < ll:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
            elif allow_reversion(ctx) and ctx.hurst < 0.45 and ctx.entropy > 0.82:
                if z < -1.8:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, e21, floor=self.floor_size)
                elif z > 1.8:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, e21, floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY":
                trail_to_level(ctx, e21, buffer=a * 0.08)
            elif current_side(ctx) == "SELL":
                trail_to_level(ctx, e21, buffer=a * 0.08)
            if market_is(ctx, "stress"):
                ctx.close("dual_mode_stress")


class AdaptiveRegimeSwitch(Strategy):
    name = "R20 Adaptive Regime Switch"
    floor_size = 0.01
    stop_atr = 1.25
    tp_rr = 2.1

    def on_bar(self, ctx):
        if ctx.bar_index < 120:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        e21 = ema(bars["close"], 21)
        e55 = ema(bars["close"], 55)
        vwap = daily_vwap(bars)
        a = atr(bars, 14)
        hh = highest(bars["high"], 25, exclude_current=True)
        ll = lowest(bars["low"], 25, exclude_current=True)
        z = zscore(bars["close"], 25)
        if not all(map(enough, [price, low, high, e21, e55, vwap, a, hh, ll, z])):
            return
        if not ctx.has_position and spread_ok(ctx, 0.08):
            if market_is(ctx, "stress"):
                if ctx.jump_intensity < 0.22 and ctx.distribution_shift > 1.30 and volatility_is(ctx, "high", "extreme"):
                    if price > hh:
                        sl = price - a * self.stop_atr
                        open_trade(ctx, "BUY", sl, risk_target(price, sl, 1.7, "BUY"), floor=self.floor_size)
                    elif price < ll:
                        sl = price + a * self.stop_atr
                        open_trade(ctx, "SELL", sl, risk_target(price, sl, 1.7, "SELL"), floor=self.floor_size)
            elif allow_trend(ctx) and ctx.hurst > 0.56 and ctx.entropy < 0.74:
                if e21 > e55 and price > e55 and low <= e21 + a * 0.10:
                    sl = min(low, price - a * self.stop_atr)
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif e21 < e55 and price < e55 and high >= e21 - a * 0.10:
                    sl = max(high, price + a * self.stop_atr)
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
            elif allow_reversion(ctx) and ctx.hurst < 0.46 and ctx.distribution_shift < 1.00:
                if z < -2.0 and price < vwap - a * 0.50:
                    sl = price - a * 1.1
                    open_trade(ctx, "BUY", sl, vwap, floor=self.floor_size)
                elif z > 2.0 and price > vwap + a * 0.50:
                    sl = price + a * 1.1
                    open_trade(ctx, "SELL", sl, vwap, floor=self.floor_size)
            elif market_is(ctx, "transition") and ctx.distribution_shift > 1.10 and volatility_is(ctx, "high", "extreme"):
                if price > hh:
                    sl = price - a * 1.2
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, 2.0, "BUY"), floor=self.floor_size)
                elif price < ll:
                    sl = price + a * 1.2
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, 2.0, "SELL"), floor=self.floor_size)
        elif ctx.has_position:
            side = current_side(ctx)
            if side == "BUY":
                trail_level = max(e21, min(vwap, price - a * 0.05))
                trail_to_level(ctx, trail_level, buffer=a * 0.05)
                if (market_is(ctx, "range", "mean_revert") and price < e21) or (market_is(ctx, "stress") and ctx.jump_intensity > 0.24):
                    ctx.close("adaptive_exit")
            elif side == "SELL":
                trail_level = min(e21, max(vwap, price + a * 0.05))
                trail_to_level(ctx, trail_level, buffer=a * 0.05)
                if (market_is(ctx, "range", "mean_revert") and price > e21) or (market_is(ctx, "stress") and ctx.jump_intensity > 0.24):
                    ctx.close("adaptive_exit")
