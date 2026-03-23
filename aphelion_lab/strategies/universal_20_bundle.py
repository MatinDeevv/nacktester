from strategy_runtime import Strategy

from strategies.universal_20._common import (
    adx_state,
    atr,
    atr_band_levels,
    body_ratio,
    cci_value,
    cmf_value,
    current_side,
    ema,
    enough,
    highest,
    keltner_levels,
    lowest,
    macd_state,
    mfi_value,
    obv_state,
    open_trade,
    range_ratio,
    range_regime,
    risk_target,
    roc_value,
    rsi,
    sar_state,
    spread_ok,
    stoch_state,
    supertrend_state,
    tighten_tp,
    trail_to_level,
    trend_regime,
    volume_ratio,
    zscore,
)


class EMARibbonTrendContinuation(Strategy):
    name = "U01 EMA Ribbon Trend Continuation"
    floor_size = 0.01
    stop_atr = 1.5
    tp_rr = 2.4

    def on_bar(self, ctx):
        if ctx.bar_index < 170:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        e21 = ema(bars["close"], 21)
        e55 = ema(bars["close"], 55)
        e144 = ema(bars["close"], 144)
        a = atr(bars, 14)
        rr = range_ratio(bars, 20)
        if not enough(price, low, high, e21, e55, e144, a, rr):
            return
        if not ctx.has_position:
            if trend_regime(ctx) and spread_ok(ctx) and rr > 0.75:
                if e21 > e55 > e144 and price > e55 and low <= e21 + a * 0.15:
                    sl = min(lowest(bars["low"], 7, exclude_current=True), price - a * self.stop_atr)
                    if enough(sl) and price - sl > a * 0.35:
                        open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif e21 < e55 < e144 and price < e55 and high >= e21 - a * 0.15:
                    sl = max(highest(bars["high"], 7, exclude_current=True), price + a * self.stop_atr)
                    if enough(sl) and sl - price > a * 0.35:
                        open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            side = current_side(ctx)
            if side == "BUY":
                trail_to_level(ctx, e55, buffer=a * 0.08)
                if price < e55 or not trend_regime(ctx):
                    ctx.close("trend_lost")
            elif side == "SELL":
                trail_to_level(ctx, e55, buffer=a * 0.08)
                if price > e55 or not trend_regime(ctx):
                    ctx.close("trend_lost")


class ADXDonchianBreakout(Strategy):
    name = "U02 ADX Donchian Breakout"
    floor_size = 0.01
    channel = 20
    stop_atr = 1.35
    tp_rr = 2.2

    def on_bar(self, ctx):
        if ctx.bar_index < self.channel + 60:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        a = atr(bars, 14)
        plus_di, minus_di, adx_now = adx_state(bars, 14)
        _, _, adx_prev = adx_state(bars, 14, back=1)
        hh = highest(bars["high"], self.channel, exclude_current=True)
        ll = lowest(bars["low"], self.channel, exclude_current=True)
        mid = (hh + ll) / 2.0 if enough(hh, ll) else float("nan")
        vr = volume_ratio(bars, 20)
        if not enough(price, a, plus_di, minus_di, adx_now, adx_prev, hh, ll, mid, vr):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and adx_now > 18 and adx_now > adx_prev and ctx.distribution_shift > 0.75 and vr > 0.95:
                if price > hh and plus_di > minus_di:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < ll and minus_di > plus_di:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            side = current_side(ctx)
            if side == "BUY" and (price < mid or adx_now < adx_prev):
                ctx.close("breakout_faded")
            elif side == "SELL" and (price > mid or adx_now < adx_prev):
                ctx.close("breakout_faded")


class SupertrendPullback(Strategy):
    name = "U03 Supertrend Pullback"
    floor_size = 0.01
    stop_atr = 1.4
    tp_rr = 2.1

    def on_bar(self, ctx):
        if ctx.bar_index < 120:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        e34 = ema(bars["close"], 34)
        e89 = ema(bars["close"], 89)
        st_value, st_dir = supertrend_state(bars, 10, 3.0)
        a = atr(bars, 14)
        if not enough(price, low, high, e34, e89, st_value, st_dir, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and trend_regime(ctx):
                if st_dir > 0 and e34 > e89 and price > st_value and low <= e34 + a * 0.10:
                    sl = min(low, price - a * self.stop_atr)
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif st_dir < 0 and e34 < e89 and price < st_value and high >= e34 - a * 0.10:
                    sl = max(high, price + a * self.stop_atr)
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            side = current_side(ctx)
            trail_to_level(ctx, st_value, buffer=a * 0.05)
            if side == "BUY" and (st_dir < 0 or price < e89):
                ctx.close("supertrend_flip")
            elif side == "SELL" and (st_dir > 0 or price > e89):
                ctx.close("supertrend_flip")


class ATRCompressionExpansion(Strategy):
    name = "U04 ATR Compression Expansion"
    floor_size = 0.01
    stop_atr = 1.25
    tp_rr = 2.3

    def on_bar(self, ctx):
        if ctx.bar_index < 120:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        a_short = atr(bars, 10)
        a_long = atr(bars, 40)
        hh = highest(bars["high"], 15, exclude_current=True)
        ll = lowest(bars["low"], 15, exclude_current=True)
        rr = range_ratio(bars, 15)
        if not enough(price, a_short, a_long, hh, ll, rr):
            return
        compressed = a_short < a_long * 0.82
        if not ctx.has_position:
            if spread_ok(ctx) and compressed and ctx.entropy < 0.78 and rr > 1.02:
                if price > hh and ctx.distribution_shift > 0.70:
                    sl = price - a_short * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < ll and ctx.distribution_shift > 0.70:
                    sl = price + a_short * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            side = current_side(ctx)
            if side == "BUY" and price < hh:
                ctx.close("compression_failed")
            elif side == "SELL" and price > ll:
                ctx.close("compression_failed")


class MACDTrendReacceleration(Strategy):
    name = "U05 MACD Trend Reacceleration"
    floor_size = 0.01
    stop_atr = 1.4
    tp_rr = 2.2

    def on_bar(self, ctx):
        if ctx.bar_index < 120:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        e34 = ema(bars["close"], 34)
        e89 = ema(bars["close"], 89)
        macd_now, signal_now, hist_now = macd_state(bars["close"])
        _, _, hist_prev = macd_state(bars["close"], back=1)
        a = atr(bars, 14)
        if not enough(price, low, high, e34, e89, macd_now, signal_now, hist_now, hist_prev, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and trend_regime(ctx):
                if e34 > e89 and price > e34 and hist_now > 0 and hist_now > hist_prev and low <= e34 + a * 0.10:
                    sl = min(low, price - a * self.stop_atr)
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif e34 < e89 and price < e34 and hist_now < 0 and hist_now < hist_prev and high >= e34 - a * 0.10:
                    sl = max(high, price + a * self.stop_atr)
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            side = current_side(ctx)
            if side == "BUY":
                trail_to_level(ctx, e34, buffer=a * 0.08)
                if hist_now < 0 or price < e89:
                    ctx.close("macd_rollover")
            elif side == "SELL":
                trail_to_level(ctx, e34, buffer=a * 0.08)
                if hist_now > 0 or price > e89:
                    ctx.close("macd_rollover")


class KeltnerImpulseBreakout(Strategy):
    name = "U06 Keltner Impulse Breakout"
    floor_size = 0.01
    stop_atr = 1.25
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 100:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        upper, mid, lower = keltner_levels(bars, 20, 14, 1.8)
        a = atr(bars, 14)
        br = body_ratio(ctx.bar)
        if not enough(price, upper, mid, lower, a, br):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and ctx.distribution_shift > 0.85 and ctx.jump_intensity < 0.22 and br > 0.50:
                if price > upper and not range_regime(ctx):
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < lower and not range_regime(ctx):
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            side = current_side(ctx)
            trail_to_level(ctx, mid, buffer=a * 0.06)
            if side == "BUY" and price < mid:
                ctx.close("keltner_fail")
            elif side == "SELL" and price > mid:
                ctx.close("keltner_fail")


class BollingerTrendRide(Strategy):
    name = "U07 Bollinger Trend Ride"
    floor_size = 0.01
    stop_atr = 1.2
    tp_rr = 1.8

    def on_bar(self, ctx):
        if ctx.bar_index < 90:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        upper, mid, lower = ctx.bbands(20, 2.0)
        e20 = ema(bars["close"], 20)
        e55 = ema(bars["close"], 55)
        rs = rsi(bars["close"], 14)
        a = atr(bars, 14)
        if not enough(price, upper, mid, lower, e20, e55, rs, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and trend_regime(ctx):
                if price > upper and e20 > e55 and rs > 56:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < lower and e20 < e55 and rs < 44:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY":
                trail_to_level(ctx, e20, buffer=a * 0.08)
                if price < mid or rs < 50:
                    ctx.close("band_reversion")
            elif current_side(ctx) == "SELL":
                trail_to_level(ctx, e20, buffer=a * 0.08)
                if price > mid or rs > 50:
                    ctx.close("band_reversion")


class StochRSITrendSnapback(Strategy):
    name = "U08 Stoch RSI Trend Snapback"
    floor_size = 0.01
    stop_atr = 1.3
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 110:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        e21 = ema(bars["close"], 21)
        e55 = ema(bars["close"], 55)
        k_now, d_now = stoch_state(bars["close"])
        k_prev, d_prev = stoch_state(bars["close"], back=1)
        a = atr(bars, 14)
        if not enough(price, e21, e55, k_now, d_now, k_prev, d_prev, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and trend_regime(ctx):
                if e21 > e55 and price > e55 and k_prev <= d_prev and k_now > d_now and k_now < 35:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif e21 < e55 and price < e55 and k_prev >= d_prev and k_now < d_now and k_now > 65:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY":
                trail_to_level(ctx, e21, buffer=a * 0.06)
                if k_now > 85 or price < e55:
                    ctx.close("snapback_complete")
            elif current_side(ctx) == "SELL":
                trail_to_level(ctx, e21, buffer=a * 0.06)
                if k_now < 15 or price > e55:
                    ctx.close("snapback_complete")


class CCIZeroLineTrend(Strategy):
    name = "U09 CCI Zero Line Trend"
    floor_size = 0.01
    stop_atr = 1.35
    tp_rr = 2.1

    def on_bar(self, ctx):
        if ctx.bar_index < 110:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        cci_now = cci_value(bars, 20)
        cci_prev = cci_value(bars, 20, back=1)
        e34 = ema(bars["close"], 34)
        e89 = ema(bars["close"], 89)
        a = atr(bars, 14)
        if not enough(price, cci_now, cci_prev, e34, e89, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and trend_regime(ctx):
                if e34 > e89 and cci_prev <= 0 < cci_now and price > e34:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif e34 < e89 and cci_prev >= 0 > cci_now and price < e34:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY":
                trail_to_level(ctx, e34, buffer=a * 0.08)
                if cci_now < 0 or price < e34:
                    ctx.close("cci_reversal")
            elif current_side(ctx) == "SELL":
                trail_to_level(ctx, e34, buffer=a * 0.08)
                if cci_now > 0 or price > e34:
                    ctx.close("cci_reversal")


class ROCAccelerationTrend(Strategy):
    name = "U10 ROC Acceleration Trend"
    floor_size = 0.01
    stop_atr = 1.3
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 100:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        roc_now = roc_value(bars["close"], 12)
        roc_prev = roc_value(bars["close"], 12, back=1)
        e21 = ema(bars["close"], 21)
        e55 = ema(bars["close"], 55)
        a = atr(bars, 14)
        rr = range_ratio(bars, 20)
        if not enough(price, roc_now, roc_prev, e21, e55, a, rr):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and trend_regime(ctx) and rr > 0.90:
                if e21 > e55 and price > e21 and roc_now > 0.40 and roc_now > roc_prev:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif e21 < e55 and price < e21 and roc_now < -0.40 and roc_now < roc_prev:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY":
                trail_to_level(ctx, e21, buffer=a * 0.06)
                if roc_now < 0 or price < e21:
                    ctx.close("roc_decay")
            elif current_side(ctx) == "SELL":
                trail_to_level(ctx, e21, buffer=a * 0.06)
                if roc_now > 0 or price > e21:
                    ctx.close("roc_decay")


class ATRBandExpansion(Strategy):
    name = "U11 ATR Band Expansion"
    floor_size = 0.01
    stop_atr = 1.2
    tp_rr = 1.9

    def on_bar(self, ctx):
        if ctx.bar_index < 90:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        upper, mid, lower = atr_band_levels(bars, 14, 2.0)
        a = atr(bars, 14)
        if not enough(price, upper, mid, lower, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and ctx.distribution_shift > 0.80 and ctx.jump_intensity < 0.22:
                if price > upper and not range_regime(ctx):
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif price < lower and not range_regime(ctx):
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            trail_to_level(ctx, mid, buffer=a * 0.05)
            if current_side(ctx) == "BUY" and price < mid:
                ctx.close("atr_band_fail")
            elif current_side(ctx) == "SELL" and price > mid:
                ctx.close("atr_band_fail")


class ParabolicSARFlip(Strategy):
    name = "U12 Parabolic SAR Flip"
    floor_size = 0.01
    stop_atr = 1.2
    tp_rr = 1.9

    def on_bar(self, ctx):
        if ctx.bar_index < 90:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        sar_now, dir_now = sar_state(bars)
        _, dir_prev = sar_state(bars, back=1)
        e50 = ema(bars["close"], 50)
        plus_di, minus_di, adx_now = adx_state(bars, 14)
        a = atr(bars, 14)
        if not enough(price, sar_now, dir_now, dir_prev, e50, plus_di, minus_di, adx_now, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and adx_now > 16:
                if dir_prev < 0 < dir_now and plus_di > minus_di and price > e50:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif dir_prev > 0 > dir_now and minus_di > plus_di and price < e50:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            trail_to_level(ctx, sar_now, buffer=a * 0.05)
            if current_side(ctx) == "BUY" and (dir_now < 0 or price < e50):
                ctx.close("sar_flip")
            elif current_side(ctx) == "SELL" and (dir_now > 0 or price > e50):
                ctx.close("sar_flip")


class OBVPressureTrend(Strategy):
    name = "U13 OBV Pressure Trend"
    floor_size = 0.01
    stop_atr = 1.35
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 110:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        e21 = ema(bars["close"], 21)
        e55 = ema(bars["close"], 55)
        _, obv_fast, obv_slow, obv_slope = obv_state(bars, 10, 30)
        hh = highest(bars["high"], 12, exclude_current=True)
        ll = lowest(bars["low"], 12, exclude_current=True)
        a = atr(bars, 14)
        if not enough(price, e21, e55, obv_fast, obv_slow, obv_slope, hh, ll, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx):
                if trend_regime(ctx) and e21 > e55 and obv_fast > obv_slow and obv_slope > 0 and price > hh:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif trend_regime(ctx) and e21 < e55 and obv_fast < obv_slow and obv_slope < 0 and price < ll:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY" and (obv_fast < obv_slow or price < e55):
                ctx.close("obv_rollover")
            elif current_side(ctx) == "SELL" and (obv_fast > obv_slow or price > e55):
                ctx.close("obv_rollover")


class CMFChannelBreakout(Strategy):
    name = "U14 CMF Channel Breakout"
    floor_size = 0.01
    stop_atr = 1.3
    tp_rr = 2.1

    def on_bar(self, ctx):
        if ctx.bar_index < 100:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        cmf_now = cmf_value(bars, 20)
        hh = highest(bars["high"], 20, exclude_current=True)
        ll = lowest(bars["low"], 20, exclude_current=True)
        mid = (hh + ll) / 2.0 if enough(hh, ll) else float("nan")
        vr = volume_ratio(bars, 20)
        a = atr(bars, 14)
        if not enough(price, cmf_now, hh, ll, mid, vr, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and vr > 0.95:
                if cmf_now > 0.08 and price > hh:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif cmf_now < -0.08 and price < ll:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY" and (cmf_now < 0 or price < mid):
                ctx.close("cmf_reversal")
            elif current_side(ctx) == "SELL" and (cmf_now > 0 or price > mid):
                ctx.close("cmf_reversal")


class MFIPullbackContinuation(Strategy):
    name = "U15 MFI Pullback Continuation"
    floor_size = 0.01
    stop_atr = 1.35
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 100:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        e21 = ema(bars["close"], 21)
        e55 = ema(bars["close"], 55)
        mfi_now = mfi_value(bars, 14)
        mfi_prev = mfi_value(bars, 14, back=1)
        a = atr(bars, 14)
        if not enough(price, low, high, e21, e55, mfi_now, mfi_prev, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and trend_regime(ctx):
                if e21 > e55 and price > e55 and 35 <= mfi_now <= 60 and mfi_now > mfi_prev and low <= e21 + a * 0.10:
                    sl = min(low, price - a * self.stop_atr)
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif e21 < e55 and price < e55 and 40 <= mfi_now <= 65 and mfi_now < mfi_prev and high >= e21 - a * 0.10:
                    sl = max(high, price + a * self.stop_atr)
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if current_side(ctx) == "BUY":
                trail_to_level(ctx, e21, buffer=a * 0.06)
                if mfi_now > 80 or price < e55:
                    ctx.close("mfi_complete")
            elif current_side(ctx) == "SELL":
                trail_to_level(ctx, e21, buffer=a * 0.06)
                if mfi_now < 20 or price > e55:
                    ctx.close("mfi_complete")


class ZScoreRangeReversion(Strategy):
    name = "U16 Z-Score Range Reversion"
    floor_size = 0.01
    stop_atr = 1.1

    def on_bar(self, ctx):
        if ctx.bar_index < 80:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        z = zscore(bars["close"], 25)
        e20 = ema(bars["close"], 20)
        a = atr(bars, 14)
        if not enough(price, z, e20, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and range_regime(ctx) and ctx.jump_intensity < 0.18:
                if z < -2.1 and price < e20 - a * 0.35:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, e20, floor=self.floor_size)
                elif z > 2.1 and price > e20 + a * 0.35:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, e20, floor=self.floor_size)
        else:
            if not range_regime(ctx):
                ctx.close("range_lost")


class RSIBollingerReversion(Strategy):
    name = "U17 RSI Bollinger Reversion"
    floor_size = 0.01
    stop_atr = 1.05

    def on_bar(self, ctx):
        if ctx.bar_index < 60:
            return
        price = float(ctx.bar["close"])
        upper, mid, lower = ctx.bbands(20, 2.0)
        rs = rsi(ctx.bars["close"], 5)
        a = atr(ctx.bars, 14)
        if not enough(price, upper, mid, lower, rs, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and range_regime(ctx):
                if price < lower and rs < 28 and ctx.jump_intensity < 0.18:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, mid, floor=self.floor_size)
                elif price > upper and rs > 72 and ctx.jump_intensity < 0.18:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, mid, floor=self.floor_size)
        else:
            if not range_regime(ctx):
                ctx.close("regime_flip")


class EntropyHurstModeSwitcher(Strategy):
    name = "U18 Entropy Hurst Mode Switcher"
    floor_size = 0.01
    stop_atr = 1.2
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 120:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        a = atr(bars, 14)
        e21 = ema(bars["close"], 21)
        hh = highest(bars["high"], 15, exclude_current=True)
        ll = lowest(bars["low"], 15, exclude_current=True)
        z = zscore(bars["close"], 20)
        if not enough(price, a, e21, hh, ll, z):
            return
        if not ctx.has_position:
            if spread_ok(ctx):
                if trend_regime(ctx) and ctx.hurst > 0.58 and ctx.entropy < 0.70:
                    if price > hh:
                        sl = price - a * self.stop_atr
                        open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                    elif price < ll:
                        sl = price + a * self.stop_atr
                        open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
                elif range_regime(ctx) and ctx.hurst < 0.48 and ctx.entropy > 0.78:
                    if z < -1.9:
                        sl = price - a * self.stop_atr
                        open_trade(ctx, "BUY", sl, e21, floor=self.floor_size)
                    elif z > 1.9:
                        sl = price + a * self.stop_atr
                        open_trade(ctx, "SELL", sl, e21, floor=self.floor_size)
        else:
            side = current_side(ctx)
            if side == "BUY":
                trail_to_level(ctx, e21, buffer=a * 0.06)
                if ctx.volatility_regime in ("high", "extreme"):
                    tighten_tp(ctx, price + a * 0.9)
                if not trend_regime(ctx) and not range_regime(ctx):
                    ctx.close("mode_unclear")
            elif side == "SELL":
                trail_to_level(ctx, e21, buffer=a * 0.06)
                if ctx.volatility_regime in ("high", "extreme"):
                    tighten_tp(ctx, price - a * 0.9)
                if not trend_regime(ctx) and not range_regime(ctx):
                    ctx.close("mode_unclear")


class DistributionShiftImpulse(Strategy):
    name = "U19 Distribution Shift Impulse"
    floor_size = 0.01
    stop_atr = 1.2
    tp_rr = 1.9

    def on_bar(self, ctx):
        if ctx.bar_index < 100:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        shift_prev = float(bars["distribution_shift_norm"].iloc[-2])
        hh = highest(bars["high"], 20, exclude_current=True)
        ll = lowest(bars["low"], 20, exclude_current=True)
        rr = range_ratio(bars, 20)
        br = body_ratio(ctx.bar)
        a = atr(bars, 14)
        if not enough(price, shift_prev, hh, ll, rr, br, a):
            return
        if not ctx.has_position:
            if spread_ok(ctx) and rr > 1.0 and br > 0.50 and 0.04 < ctx.jump_intensity < 0.22:
                if ctx.distribution_shift > shift_prev + 0.12 and price > hh:
                    sl = price - a * self.stop_atr
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
                elif ctx.distribution_shift > shift_prev + 0.12 and price < ll:
                    sl = price + a * self.stop_atr
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
        else:
            if ctx.distribution_shift < shift_prev * 0.90 or ctx.jump_intensity > 0.25:
                ctx.close("shift_decay")


class AdaptiveUniversalHybrid(Strategy):
    name = "U20 Adaptive Universal Hybrid"
    floor_size = 0.01
    stop_atr = 1.25
    tp_rr = 2.0

    def on_bar(self, ctx):
        if ctx.bar_index < 150:
            return
        bars = ctx.bars
        price = float(ctx.bar["close"])
        low = float(ctx.bar["low"])
        high = float(ctx.bar["high"])
        e21 = ema(bars["close"], 21)
        e55 = ema(bars["close"], 55)
        st_value, st_dir = supertrend_state(bars, 10, 3.0)
        upper, mid, lower = ctx.bbands(20, 2.0)
        z = zscore(bars["close"], 25)
        hh = highest(bars["high"], 20, exclude_current=True)
        ll = lowest(bars["low"], 20, exclude_current=True)
        a = atr(bars, 14)
        if not enough(price, low, high, e21, e55, st_value, st_dir, upper, mid, lower, z, hh, ll, a):
            return
        if not ctx.has_position and spread_ok(ctx):
            if trend_regime(ctx) and e21 > e55 and st_dir > 0 and low <= e21 + a * 0.10:
                sl = min(low, price - a * self.stop_atr)
                open_trade(ctx, "BUY", sl, risk_target(price, sl, self.tp_rr, "BUY"), floor=self.floor_size)
            elif trend_regime(ctx) and e21 < e55 and st_dir < 0 and high >= e21 - a * 0.10:
                sl = max(high, price + a * self.stop_atr)
                open_trade(ctx, "SELL", sl, risk_target(price, sl, self.tp_rr, "SELL"), floor=self.floor_size)
            elif range_regime(ctx) and ctx.jump_intensity < 0.18:
                if z < -2.0 and price < lower:
                    sl = price - a * 1.05
                    open_trade(ctx, "BUY", sl, mid, floor=self.floor_size)
                elif z > 2.0 and price > upper:
                    sl = price + a * 1.05
                    open_trade(ctx, "SELL", sl, mid, floor=self.floor_size)
            elif ctx.distribution_shift > 1.00 and ctx.jump_intensity < 0.22:
                if price > hh:
                    sl = price - a * 1.15
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, 1.8, "BUY"), floor=self.floor_size)
                elif price < ll:
                    sl = price + a * 1.15
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, 1.8, "SELL"), floor=self.floor_size)
        elif ctx.has_position:
            side = current_side(ctx)
            if side == "BUY":
                trail_level = max(e21, st_value if enough(st_value) else e21)
                trail_to_level(ctx, trail_level, buffer=a * 0.05)
                if ctx.volatility_regime in ("high", "extreme"):
                    tighten_tp(ctx, price + a * 0.8)
                if (range_regime(ctx) and price < e21) or ctx.jump_intensity > 0.24:
                    ctx.close("adaptive_exit")
            elif side == "SELL":
                trail_level = min(e21, st_value if enough(st_value) else e21)
                trail_to_level(ctx, trail_level, buffer=a * 0.05)
                if ctx.volatility_regime in ("high", "extreme"):
                    tighten_tp(ctx, price - a * 0.8)
                if (range_regime(ctx) and price > e21) or ctx.jump_intensity > 0.24:
                    ctx.close("adaptive_exit")
