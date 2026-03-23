from __future__ import annotations

from strategy_runtime import Strategy

from strategies.scalp_50._common import (
    ACTIVE_SESSIONS,
    adx_state,
    atr,
    atr_band_levels,
    body_ratio,
    breakout_ok,
    cci_value,
    cmf_value,
    current_side,
    ema,
    enough,
    highest,
    keltner_levels,
    lock_one_r,
    lowest,
    macd_state,
    manage_open,
    mfi_value,
    open_trade,
    range_ratio,
    reversion_ok,
    risk_distance_ok,
    risk_target,
    rsi,
    scalp_spread_ok,
    stoch_state,
    supertrend_state,
    trend_ok,
    trail_to_level,
    volume_ratio,
    zscore,
)


def _allow_long(strategy) -> bool:
    return getattr(strategy, "direction", "both") in ("both", "long")


def _allow_short(strategy) -> bool:
    return getattr(strategy, "direction", "both") in ("both", "short")


class BaseScalpStrategy(Strategy):
    name = "Base Scalp Strategy"
    style = "ema_pullback"
    direction = "both"
    floor_size = 0.01
    cap_size = 0.04
    tp_rr = 3.0
    fast = 8
    slow = 21
    anchor = 55
    channel = 10
    band_period = 20
    stop_atr = 1.0
    max_spread = 0.09
    pullback_mult = 0.14
    min_body = 0.45
    min_volume = 0.90
    min_shift = 0.45
    max_shift = 0.90
    min_hurst = 0.52
    max_entropy = 0.82
    max_jump = 0.25
    adx_floor = 18.0
    rsi_floor = 38.0
    rsi_ceil = 62.0
    osc_low = 20.0
    osc_high = 80.0
    cci_trigger = 80.0
    lock_rr = 0.10
    sessions = ACTIVE_SESSIONS

    def on_bar(self, ctx):
        if ctx.bar_index < max(self.anchor + 35, self.band_period + 35, self.channel + 60):
            return
        _STYLE_HANDLERS[self.style](self, ctx)


def _ema_pullback(strategy, ctx):
    bars = ctx.bars
    price = float(ctx.bar["close"])
    low = float(ctx.bar["low"])
    high = float(ctx.bar["high"])
    e_fast = ema(bars["close"], strategy.fast)
    e_slow = ema(bars["close"], strategy.slow)
    e_anchor = ema(bars["close"], strategy.anchor)
    atr_value = atr(bars, 14)
    vol = volume_ratio(bars, 20)
    rng = range_ratio(bars, strategy.channel + 4)
    if not enough(price, low, high, e_fast, e_slow, e_anchor, atr_value, vol, rng):
        return
    if not ctx.has_position:
        if (
            scalp_spread_ok(ctx, strategy.max_spread)
            and trend_ok(ctx, strategy.sessions, strategy.min_hurst, strategy.max_entropy, strategy.max_jump, strategy.min_shift)
            and vol >= strategy.min_volume
            and rng >= 0.78
        ):
            if _allow_long(strategy) and e_fast > e_slow > e_anchor and price > e_anchor and low <= e_fast + atr_value * strategy.pullback_mult:
                sl = min(lowest(bars["low"], strategy.channel, exclude_current=True), price - atr_value * strategy.stop_atr)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(
                        ctx,
                        "BUY",
                        sl,
                        risk_target(price, sl, strategy.tp_rr, "BUY"),
                        floor=strategy.floor_size,
                        cap=strategy.cap_size,
                    )
            elif _allow_short(strategy) and e_fast < e_slow < e_anchor and price < e_anchor and high >= e_fast - atr_value * strategy.pullback_mult:
                sl = max(highest(bars["high"], strategy.channel, exclude_current=True), price + atr_value * strategy.stop_atr)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(
                        ctx,
                        "SELL",
                        sl,
                        risk_target(price, sl, strategy.tp_rr, "SELL"),
                        floor=strategy.floor_size,
                        cap=strategy.cap_size,
                    )
    else:
        side = current_side(ctx)
        manage_open(ctx, e_slow, atr_value, invalidation=e_anchor, lock_rr=strategy.lock_rr)
        if side == "BUY" and (price < e_anchor or ctx.distribution_shift < strategy.min_shift * 0.60 or ctx.jump_intensity > strategy.max_jump * 1.15):
            ctx.close("trend_lost")
        elif side == "SELL" and (price > e_anchor or ctx.distribution_shift < strategy.min_shift * 0.60 or ctx.jump_intensity > strategy.max_jump * 1.15):
            ctx.close("trend_lost")


def _donchian_breakout(strategy, ctx):
    bars = ctx.bars
    price = float(ctx.bar["close"])
    atr_value = atr(bars, 14)
    high_break = highest(bars["high"], strategy.channel, exclude_current=True)
    low_break = lowest(bars["low"], strategy.channel, exclude_current=True)
    midpoint = (high_break + low_break) / 2.0 if enough(high_break, low_break) else float("nan")
    plus_di, minus_di, adx_now = adx_state(bars, 14)
    _, _, adx_prev = adx_state(bars, 14, back=1)
    vol = volume_ratio(bars, 20)
    body = body_ratio(ctx.bar)
    if not enough(price, atr_value, high_break, low_break, midpoint, plus_di, minus_di, adx_now, adx_prev, vol, body):
        return
    if not ctx.has_position:
        if (
            scalp_spread_ok(ctx, strategy.max_spread)
            and breakout_ok(ctx, strategy.sessions, max(strategy.min_shift, 0.60), strategy.max_jump)
            and adx_now >= max(strategy.adx_floor, 17.0)
            and adx_now >= adx_prev
            and body >= strategy.min_body
            and vol >= strategy.min_volume
        ):
            if _allow_long(strategy) and price > high_break and plus_di > minus_di:
                sl = price - atr_value * strategy.stop_atr
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, strategy.tp_rr, "BUY"), floor=strategy.floor_size, cap=strategy.cap_size)
            elif _allow_short(strategy) and price < low_break and minus_di > plus_di:
                sl = price + atr_value * strategy.stop_atr
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, strategy.tp_rr, "SELL"), floor=strategy.floor_size, cap=strategy.cap_size)
    else:
        side = current_side(ctx)
        manage_open(ctx, midpoint, atr_value, invalidation=midpoint, lock_rr=strategy.lock_rr)
        if side == "BUY" and (price < midpoint or adx_now < adx_prev):
            ctx.close("breakout_faded")
        elif side == "SELL" and (price > midpoint or adx_now < adx_prev):
            ctx.close("breakout_faded")


def _keltner_impulse(strategy, ctx):
    bars = ctx.bars
    price = float(ctx.bar["close"])
    upper, mid, lower = keltner_levels(bars, strategy.band_period, 14, 1.6)
    atr_value = atr(bars, 14)
    body = body_ratio(ctx.bar)
    vol = volume_ratio(bars, 20)
    if not enough(price, upper, mid, lower, atr_value, body, vol):
        return
    if not ctx.has_position:
        if (
            scalp_spread_ok(ctx, strategy.max_spread)
            and breakout_ok(ctx, strategy.sessions, max(strategy.min_shift, 0.70), strategy.max_jump)
            and body >= strategy.min_body
            and vol >= strategy.min_volume
            and ctx.entropy <= strategy.max_entropy + 0.06
        ):
            if _allow_long(strategy) and price > upper:
                sl = price - atr_value * strategy.stop_atr
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, strategy.tp_rr, "BUY"), floor=strategy.floor_size, cap=strategy.cap_size)
            elif _allow_short(strategy) and price < lower:
                sl = price + atr_value * strategy.stop_atr
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, strategy.tp_rr, "SELL"), floor=strategy.floor_size, cap=strategy.cap_size)
    else:
        side = current_side(ctx)
        manage_open(ctx, mid, atr_value, invalidation=mid, lock_rr=strategy.lock_rr)
        if side == "BUY" and price < mid:
            ctx.close("keltner_fail")
        elif side == "SELL" and price > mid:
            ctx.close("keltner_fail")


def _macd_reaccel(strategy, ctx):
    bars = ctx.bars
    price = float(ctx.bar["close"])
    low = float(ctx.bar["low"])
    high = float(ctx.bar["high"])
    e_fast = ema(bars["close"], strategy.fast)
    e_anchor = ema(bars["close"], strategy.anchor)
    atr_value = atr(bars, 14)
    _, _, hist_now = macd_state(bars["close"])
    _, _, hist_prev = macd_state(bars["close"], back=1)
    vol = volume_ratio(bars, 20)
    if not enough(price, low, high, e_fast, e_anchor, atr_value, hist_now, hist_prev, vol):
        return
    if not ctx.has_position:
        if (
            scalp_spread_ok(ctx, strategy.max_spread)
            and trend_ok(ctx, strategy.sessions, strategy.min_hurst, strategy.max_entropy, strategy.max_jump, strategy.min_shift)
            and vol >= strategy.min_volume
        ):
            if _allow_long(strategy) and e_fast > e_anchor and price > e_anchor and hist_now > 0 and hist_now > hist_prev and low <= e_fast + atr_value * strategy.pullback_mult:
                sl = min(low, price - atr_value * strategy.stop_atr)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, strategy.tp_rr, "BUY"), floor=strategy.floor_size, cap=strategy.cap_size)
            elif _allow_short(strategy) and e_fast < e_anchor and price < e_anchor and hist_now < 0 and hist_now < hist_prev and high >= e_fast - atr_value * strategy.pullback_mult:
                sl = max(high, price + atr_value * strategy.stop_atr)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, strategy.tp_rr, "SELL"), floor=strategy.floor_size, cap=strategy.cap_size)
    else:
        side = current_side(ctx)
        manage_open(ctx, e_fast, atr_value, invalidation=e_anchor, lock_rr=strategy.lock_rr)
        if side == "BUY" and (hist_now < 0 or price < e_anchor):
            ctx.close("macd_rollover")
        elif side == "SELL" and (hist_now > 0 or price > e_anchor):
            ctx.close("macd_rollover")


def _supertrend_pullback(strategy, ctx):
    bars = ctx.bars
    price = float(ctx.bar["close"])
    low = float(ctx.bar["low"])
    high = float(ctx.bar["high"])
    e_fast = ema(bars["close"], strategy.fast)
    e_anchor = ema(bars["close"], strategy.anchor)
    st_value, st_dir = supertrend_state(bars, 10, 2.6)
    atr_value = atr(bars, 14)
    vol = volume_ratio(bars, 20)
    if not enough(price, low, high, e_fast, e_anchor, st_value, st_dir, atr_value, vol):
        return
    if not ctx.has_position:
        if (
            scalp_spread_ok(ctx, strategy.max_spread)
            and trend_ok(ctx, strategy.sessions, strategy.min_hurst, strategy.max_entropy, strategy.max_jump, strategy.min_shift)
            and vol >= strategy.min_volume
        ):
            if _allow_long(strategy) and st_dir > 0 and e_fast > e_anchor and price > st_value and low <= e_fast + atr_value * strategy.pullback_mult:
                sl = min(low, price - atr_value * strategy.stop_atr)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, strategy.tp_rr, "BUY"), floor=strategy.floor_size, cap=strategy.cap_size)
            elif _allow_short(strategy) and st_dir < 0 and e_fast < e_anchor and price < st_value and high >= e_fast - atr_value * strategy.pullback_mult:
                sl = max(high, price + atr_value * strategy.stop_atr)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, strategy.tp_rr, "SELL"), floor=strategy.floor_size, cap=strategy.cap_size)
    else:
        side = current_side(ctx)
        manage_open(ctx, st_value, atr_value, invalidation=e_anchor, lock_rr=strategy.lock_rr)
        if side == "BUY" and (st_dir < 0 or price < e_anchor):
            ctx.close("supertrend_flip")
        elif side == "SELL" and (st_dir > 0 or price > e_anchor):
            ctx.close("supertrend_flip")


def _stoch_reversion(strategy, ctx):
    bars = ctx.bars
    price = float(ctx.bar["close"])
    low = float(ctx.bar["low"])
    high = float(ctx.bar["high"])
    upper, mid, lower = ctx.bbands(strategy.band_period, 2.0)
    stoch_k, stoch_d = stoch_state(bars["close"], 14, 14)
    rsi_now = rsi(bars["close"], 14)
    z_value = zscore(bars["close"], strategy.band_period)
    atr_value = atr(bars, 14)
    rng = range_ratio(bars, strategy.channel + 6)
    if not enough(price, low, high, upper, mid, lower, stoch_k, stoch_d, rsi_now, z_value, atr_value, rng):
        return
    if not ctx.has_position:
        if (
            scalp_spread_ok(ctx, strategy.max_spread)
            and reversion_ok(ctx, strategy.sessions, 0.70, 0.53, strategy.max_jump, strategy.max_shift)
            and rng < 1.45
        ):
            if _allow_long(strategy) and price <= lower and stoch_k < strategy.osc_low and stoch_d < strategy.osc_low + 8 and rsi_now < strategy.rsi_floor and z_value < -1.0:
                sl = min(lowest(bars["low"], strategy.channel, exclude_current=True), low - atr_value * 0.12)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, strategy.tp_rr, "BUY"), floor=strategy.floor_size, cap=strategy.cap_size)
            elif _allow_short(strategy) and price >= upper and stoch_k > strategy.osc_high and stoch_d > strategy.osc_high - 8 and rsi_now > strategy.rsi_ceil and z_value > 1.0:
                sl = max(highest(bars["high"], strategy.channel, exclude_current=True), high + atr_value * 0.12)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, strategy.tp_rr, "SELL"), floor=strategy.floor_size, cap=strategy.cap_size)
    else:
        side = current_side(ctx)
        lock_one_r(ctx, strategy.lock_rr)
        if side == "BUY":
            trail_to_level(ctx, mid, buffer=atr_value * 0.04)
            if price >= mid or z_value > -0.05:
                ctx.close("reversion_complete")
        elif side == "SELL":
            trail_to_level(ctx, mid, buffer=atr_value * 0.04)
            if price <= mid or z_value < 0.05:
                ctx.close("reversion_complete")


def _bollinger_fade(strategy, ctx):
    bars = ctx.bars
    price = float(ctx.bar["close"])
    low = float(ctx.bar["low"])
    high = float(ctx.bar["high"])
    bb_upper, bb_mid, bb_lower = ctx.bbands(strategy.band_period, 2.2)
    atr_upper, atr_mid, atr_lower = atr_band_levels(bars, 14, 2.1)
    rsi_now = rsi(bars["close"], 14)
    cci_now = cci_value(bars, 20)
    body = body_ratio(ctx.bar)
    atr_value = atr(bars, 14)
    if not enough(price, low, high, bb_upper, bb_mid, bb_lower, atr_upper, atr_mid, atr_lower, rsi_now, cci_now, body, atr_value):
        return
    if not ctx.has_position:
        if scalp_spread_ok(ctx, strategy.max_spread) and reversion_ok(ctx, strategy.sessions, 0.72, 0.54, strategy.max_jump, strategy.max_shift) and body <= strategy.min_body + 0.12:
            if _allow_long(strategy) and price <= min(bb_lower, atr_lower) and rsi_now < strategy.rsi_floor and cci_now < -strategy.cci_trigger:
                sl = min(lowest(bars["low"], strategy.channel, exclude_current=True), low - atr_value * 0.15)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, strategy.tp_rr, "BUY"), floor=strategy.floor_size, cap=strategy.cap_size)
            elif _allow_short(strategy) and price >= max(bb_upper, atr_upper) and rsi_now > strategy.rsi_ceil and cci_now > strategy.cci_trigger:
                sl = max(highest(bars["high"], strategy.channel, exclude_current=True), high + atr_value * 0.15)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, strategy.tp_rr, "SELL"), floor=strategy.floor_size, cap=strategy.cap_size)
    else:
        side = current_side(ctx)
        lock_one_r(ctx, strategy.lock_rr)
        if side == "BUY":
            trail_to_level(ctx, atr_mid, buffer=atr_value * 0.04)
            if price >= bb_mid or rsi_now > 54:
                ctx.close("band_filled")
        elif side == "SELL":
            trail_to_level(ctx, atr_mid, buffer=atr_value * 0.04)
            if price <= bb_mid or rsi_now < 46:
                ctx.close("band_filled")


def _compression_expansion(strategy, ctx):
    bars = ctx.bars
    price = float(ctx.bar["close"])
    atr_fast = atr(bars, 10)
    atr_slow = atr(bars, 40)
    atr_value = atr(bars, 14)
    high_break = highest(bars["high"], strategy.channel, exclude_current=True)
    low_break = lowest(bars["low"], strategy.channel, exclude_current=True)
    body = body_ratio(ctx.bar)
    vol = volume_ratio(bars, 20)
    plus_di, minus_di, adx_now = adx_state(bars, 14)
    if not enough(price, atr_fast, atr_slow, atr_value, high_break, low_break, body, vol, plus_di, minus_di, adx_now):
        return
    compressed = atr_fast < atr_slow * 0.84
    if not ctx.has_position:
        if (
            scalp_spread_ok(ctx, strategy.max_spread)
            and compressed
            and breakout_ok(ctx, strategy.sessions, max(strategy.min_shift, 0.62), strategy.max_jump)
            and body >= strategy.min_body
            and vol >= strategy.min_volume
            and adx_now >= strategy.adx_floor
        ):
            if _allow_long(strategy) and price > high_break and plus_di >= minus_di:
                sl = price - atr_value * strategy.stop_atr
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, strategy.tp_rr, "BUY"), floor=strategy.floor_size, cap=strategy.cap_size)
            elif _allow_short(strategy) and price < low_break and minus_di >= plus_di:
                sl = price + atr_value * strategy.stop_atr
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, strategy.tp_rr, "SELL"), floor=strategy.floor_size, cap=strategy.cap_size)
    else:
        midpoint = (high_break + low_break) / 2.0 if enough(high_break, low_break) else float("nan")
        side = current_side(ctx)
        manage_open(ctx, midpoint, atr_value, invalidation=midpoint, lock_rr=strategy.lock_rr)
        if side == "BUY" and price < high_break:
            ctx.close("compression_failed")
        elif side == "SELL" and price > low_break:
            ctx.close("compression_failed")


def _cci_trend(strategy, ctx):
    bars = ctx.bars
    price = float(ctx.bar["close"])
    low = float(ctx.bar["low"])
    high = float(ctx.bar["high"])
    e_fast = ema(bars["close"], strategy.fast)
    e_anchor = ema(bars["close"], strategy.anchor)
    cci_now = cci_value(bars, 20)
    cci_prev = cci_value(bars, 20, back=1)
    cmf_now = cmf_value(bars, 20)
    mfi_now = mfi_value(bars, 14)
    atr_value = atr(bars, 14)
    vol = volume_ratio(bars, 20)
    if not enough(price, low, high, e_fast, e_anchor, cci_now, cci_prev, cmf_now, mfi_now, atr_value, vol):
        return
    if not ctx.has_position:
        if (
            scalp_spread_ok(ctx, strategy.max_spread)
            and trend_ok(ctx, strategy.sessions, strategy.min_hurst, strategy.max_entropy, strategy.max_jump, strategy.min_shift)
            and vol >= strategy.min_volume
        ):
            if _allow_long(strategy) and e_fast > e_anchor and price > e_anchor and cci_now > strategy.cci_trigger and cci_now > cci_prev and cmf_now > 0 and mfi_now > 50:
                sl = min(low, price - atr_value * strategy.stop_atr)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "BUY", sl, risk_target(price, sl, strategy.tp_rr, "BUY"), floor=strategy.floor_size, cap=strategy.cap_size)
            elif _allow_short(strategy) and e_fast < e_anchor and price < e_anchor and cci_now < -strategy.cci_trigger and cci_now < cci_prev and cmf_now < 0 and mfi_now < 50:
                sl = max(high, price + atr_value * strategy.stop_atr)
                if risk_distance_ok(price, sl, atr_value):
                    open_trade(ctx, "SELL", sl, risk_target(price, sl, strategy.tp_rr, "SELL"), floor=strategy.floor_size, cap=strategy.cap_size)
    else:
        side = current_side(ctx)
        manage_open(ctx, e_fast, atr_value, invalidation=e_anchor, lock_rr=strategy.lock_rr)
        if side == "BUY" and (cci_now < 0 or cmf_now < 0):
            ctx.close("cci_fade")
        elif side == "SELL" and (cci_now > 0 or cmf_now > 0):
            ctx.close("cci_fade")


def _adaptive_hybrid(strategy, ctx):
    if ctx.market_regime in ("trend", "transition") and ctx.hurst >= strategy.min_hurst and ctx.entropy <= strategy.max_entropy:
        _ema_pullback(strategy, ctx)
        return
    if ctx.market_regime in ("range", "mean_revert") and ctx.entropy >= 0.72 and ctx.jump_intensity <= strategy.max_jump:
        _stoch_reversion(strategy, ctx)
        return
    _compression_expansion(strategy, ctx)


_STYLE_HANDLERS = {
    "ema_pullback": _ema_pullback,
    "donchian_breakout": _donchian_breakout,
    "keltner_impulse": _keltner_impulse,
    "macd_reaccel": _macd_reaccel,
    "supertrend_pullback": _supertrend_pullback,
    "stoch_reversion": _stoch_reversion,
    "bollinger_fade": _bollinger_fade,
    "compression_expansion": _compression_expansion,
    "cci_trend": _cci_trend,
    "adaptive_hybrid": _adaptive_hybrid,
}


def _make_def(class_name: str, display_name: str, style: str, **attrs):
    base = {
        "name": display_name,
        "style": style,
        "tp_rr": 3.0,
    }
    base.update(attrs)
    return class_name, base


STRATEGY_DEFS = [
    _make_def("S01LondonEmaPullback", "S01 London EMA Pullback 1:3", "ema_pullback", sessions=("london", "overlap"), fast=8, slow=21, anchor=55, channel=8, stop_atr=0.95, pullback_mult=0.12, min_shift=0.42, min_volume=0.88),
    _make_def("S02OverlapRibbonScalp", "S02 Overlap Ribbon Scalp 1:3", "ema_pullback", sessions=("overlap", "new_york"), fast=9, slow=20, anchor=50, channel=9, stop_atr=1.00, pullback_mult=0.15, min_shift=0.48, min_volume=0.92),
    _make_def("S03NewYorkHurstTrend", "S03 New York Hurst Trend 1:3", "ema_pullback", sessions=("new_york",), fast=13, slow=34, anchor=89, channel=10, stop_atr=1.05, pullback_mult=0.14, min_hurst=0.56, min_shift=0.50, min_volume=0.95),
    _make_def("S04FastEmaReentry", "S04 Fast EMA Reentry 1:3", "ema_pullback", sessions=ACTIVE_SESSIONS, fast=5, slow=13, anchor=34, channel=7, stop_atr=0.90, pullback_mult=0.10, max_entropy=0.80, min_shift=0.38),
    _make_def("S05ShiftedTrendScalp", "S05 Shifted Trend Scalp 1:3", "ema_pullback", sessions=ACTIVE_SESSIONS, fast=10, slow=21, anchor=55, channel=11, stop_atr=1.10, pullback_mult=0.18, min_shift=0.60, max_jump=0.22),
    _make_def("S06LondonDonchianBurst", "S06 London Donchian Burst 1:3", "donchian_breakout", sessions=("london", "overlap"), channel=10, stop_atr=0.95, min_body=0.48, min_shift=0.62, min_volume=1.00, adx_floor=18.0),
    _make_def("S07OverlapAdxBreakout", "S07 Overlap ADX Breakout 1:3", "donchian_breakout", sessions=("overlap",), channel=12, stop_atr=1.00, min_body=0.50, min_shift=0.68, min_volume=1.05, adx_floor=20.0),
    _make_def("S08NewYorkRangeRelease", "S08 New York Range Release 1:3", "donchian_breakout", sessions=("new_york",), channel=14, stop_atr=1.05, min_body=0.44, min_shift=0.65, min_volume=0.96, adx_floor=19.0),
    _make_def("S09ShiftBreakoutScalp", "S09 Shift Breakout Scalp 1:3", "donchian_breakout", sessions=ACTIVE_SESSIONS, channel=9, stop_atr=0.92, min_body=0.46, min_shift=0.72, max_jump=0.22, min_volume=1.02),
    _make_def("S10VolatilityBurstDrive", "S10 Volatility Burst Drive 1:3", "donchian_breakout", sessions=ACTIVE_SESSIONS, channel=15, stop_atr=1.10, min_body=0.52, min_shift=0.76, max_spread=0.10, adx_floor=22.0),
    _make_def("S11KeltnerImpulseLondon", "S11 Keltner Impulse London 1:3", "keltner_impulse", sessions=("london", "overlap"), band_period=18, stop_atr=0.92, min_body=0.48, min_shift=0.70, max_entropy=0.84),
    _make_def("S12KeltnerImpulseOverlap", "S12 Keltner Impulse Overlap 1:3", "keltner_impulse", sessions=("overlap",), band_period=20, stop_atr=0.98, min_body=0.52, min_shift=0.74, min_volume=1.00),
    _make_def("S13KeltnerTrendBurst", "S13 Keltner Trend Burst 1:3", "keltner_impulse", sessions=ACTIVE_SESSIONS, band_period=22, stop_atr=1.02, min_body=0.46, min_shift=0.66, min_volume=0.94, max_jump=0.22),
    _make_def("S14KeltnerShiftDrive", "S14 Keltner Shift Drive 1:3", "keltner_impulse", sessions=ACTIVE_SESSIONS, band_period=16, stop_atr=0.88, min_body=0.55, min_shift=0.82, max_entropy=0.78),
    _make_def("S15KeltnerJumpGuard", "S15 Keltner Jump Guard 1:3", "keltner_impulse", sessions=("new_york", "overlap"), band_period=24, stop_atr=1.05, min_body=0.44, min_shift=0.68, max_jump=0.18, min_volume=0.98),
    _make_def("S16MacdReaccelFast", "S16 MACD Reaccel Fast 1:3", "macd_reaccel", sessions=ACTIVE_SESSIONS, fast=8, anchor=34, stop_atr=0.95, pullback_mult=0.10, min_shift=0.42),
    _make_def("S17MacdReaccelOverlap", "S17 MACD Reaccel Overlap 1:3", "macd_reaccel", sessions=("overlap",), fast=13, anchor=55, stop_atr=1.00, pullback_mult=0.12, min_shift=0.48, min_volume=0.96),
    _make_def("S18MacdReaccelAnchor", "S18 MACD Anchor Reaccel 1:3", "macd_reaccel", sessions=("london", "new_york"), fast=9, anchor=50, stop_atr=1.02, pullback_mult=0.14, min_hurst=0.54, min_shift=0.52),
    _make_def("S19MacdDistributionDrive", "S19 MACD Distribution Drive 1:3", "macd_reaccel", sessions=ACTIVE_SESSIONS, fast=10, anchor=55, stop_atr=0.98, pullback_mult=0.16, min_shift=0.66, max_jump=0.20),
    _make_def("S20MacdHurstContinuation", "S20 MACD Hurst Continuation 1:3", "macd_reaccel", sessions=("overlap", "new_york"), fast=12, anchor=89, stop_atr=1.08, pullback_mult=0.18, min_hurst=0.58, min_shift=0.55),
    _make_def("S21SupertrendLondonPullback", "S21 Supertrend London Pullback 1:3", "supertrend_pullback", sessions=("london",), fast=8, anchor=34, stop_atr=0.92, pullback_mult=0.10, min_shift=0.42),
    _make_def("S22SupertrendOverlapSnap", "S22 Supertrend Overlap Snap 1:3", "supertrend_pullback", sessions=("overlap",), fast=9, anchor=50, stop_atr=0.98, pullback_mult=0.14, min_shift=0.48),
    _make_def("S23SupertrendFastReentry", "S23 Supertrend Fast Reentry 1:3", "supertrend_pullback", sessions=ACTIVE_SESSIONS, fast=5, anchor=34, stop_atr=0.88, pullback_mult=0.08, max_entropy=0.78),
    _make_def("S24SupertrendShiftScalp", "S24 Supertrend Shift Scalp 1:3", "supertrend_pullback", sessions=("overlap", "new_york"), fast=10, anchor=55, stop_atr=1.02, pullback_mult=0.15, min_shift=0.62, max_jump=0.20),
    _make_def("S25SupertrendHurstDrive", "S25 Supertrend Hurst Drive 1:3", "supertrend_pullback", sessions=ACTIVE_SESSIONS, fast=13, anchor=89, stop_atr=1.10, pullback_mult=0.18, min_hurst=0.58, min_shift=0.54),
    _make_def("S26StochRangeReclaim", "S26 Stoch Range Reclaim 1:3", "stoch_reversion", sessions=ACTIVE_SESSIONS, channel=8, band_period=18, stop_atr=0.90, max_shift=0.82, osc_low=18.0, osc_high=82.0, rsi_floor=36.0, rsi_ceil=64.0),
    _make_def("S27StochAtrSnapback", "S27 Stoch ATR Snapback 1:3", "stoch_reversion", sessions=("london", "new_york"), channel=9, band_period=20, stop_atr=0.94, max_shift=0.78, osc_low=16.0, osc_high=84.0, max_jump=0.18),
    _make_def("S28StochEntropyFade", "S28 Stoch Entropy Fade 1:3", "stoch_reversion", sessions=ACTIVE_SESSIONS, channel=10, band_period=22, stop_atr=1.00, max_shift=0.74, osc_low=20.0, osc_high=80.0, lock_rr=0.08),
    _make_def("S29StochBandRevert", "S29 Stoch Band Revert 1:3", "stoch_reversion", sessions=("overlap", "new_york"), channel=11, band_period=24, stop_atr=1.05, max_shift=0.86, osc_low=22.0, osc_high=78.0),
    _make_def("S30StochSessionFade", "S30 Stoch Session Fade 1:3", "stoch_reversion", sessions=("london", "overlap"), channel=8, band_period=18, stop_atr=0.92, max_shift=0.80, osc_low=17.0, osc_high=83.0, max_spread=0.08),
    _make_def("S31BollingerRsiCompressionFade", "S31 Bollinger RSI Compression Fade 1:3", "bollinger_fade", sessions=ACTIVE_SESSIONS, channel=8, band_period=18, stop_atr=0.95, max_shift=0.80, cci_trigger=90.0, min_body=0.36),
    _make_def("S32BollingerOverlapFade", "S32 Bollinger Overlap Fade 1:3", "bollinger_fade", sessions=("overlap",), channel=10, band_period=20, stop_atr=1.00, max_shift=0.76, cci_trigger=85.0, min_body=0.34),
    _make_def("S33BollingerZscoreScalp", "S33 Bollinger Z-Score Scalp 1:3", "bollinger_fade", sessions=("new_york", "overlap"), channel=9, band_period=22, stop_atr=1.04, max_shift=0.74, cci_trigger=100.0, rsi_floor=35.0, rsi_ceil=65.0),
    _make_def("S34BollingerEntropySnap", "S34 Bollinger Entropy Snap 1:3", "bollinger_fade", sessions=ACTIVE_SESSIONS, channel=11, band_period=24, stop_atr=1.08, max_shift=0.70, cci_trigger=95.0, max_jump=0.16),
    _make_def("S35BollingerShiftReject", "S35 Bollinger Shift Reject 1:3", "bollinger_fade", sessions=("london", "new_york"), channel=12, band_period=20, stop_atr=0.98, max_shift=0.72, cci_trigger=110.0, min_body=0.32),
    _make_def("S36CompressionLondonRelease", "S36 Compression London Release 1:3", "compression_expansion", sessions=("london",), channel=10, stop_atr=0.95, min_body=0.48, min_shift=0.66, min_volume=1.02, adx_floor=18.0),
    _make_def("S37CompressionOverlapRelease", "S37 Compression Overlap Release 1:3", "compression_expansion", sessions=("overlap",), channel=12, stop_atr=1.00, min_body=0.50, min_shift=0.70, min_volume=1.06, adx_floor=20.0),
    _make_def("S38CompressionShiftBreak", "S38 Compression Shift Break 1:3", "compression_expansion", sessions=("overlap", "new_york"), channel=9, stop_atr=0.90, min_body=0.54, min_shift=0.80, max_jump=0.20),
    _make_def("S39CompressionAdxExpansion", "S39 Compression ADX Expansion 1:3", "compression_expansion", sessions=ACTIVE_SESSIONS, channel=14, stop_atr=1.08, min_body=0.46, min_shift=0.68, adx_floor=22.0),
    _make_def("S40CompressionRangeFlip", "S40 Compression Range Flip 1:3", "compression_expansion", sessions=ACTIVE_SESSIONS, channel=11, stop_atr=0.98, min_body=0.44, min_shift=0.64, min_volume=0.94),
    _make_def("S41CciCmfTrendScalp", "S41 CCI CMF Trend Scalp 1:3", "cci_trend", sessions=("london", "overlap"), fast=8, anchor=34, stop_atr=0.94, cci_trigger=75.0, min_shift=0.40, min_volume=0.92),
    _make_def("S42CciMfiDrive", "S42 CCI MFI Drive 1:3", "cci_trend", sessions=("overlap",), fast=10, anchor=50, stop_atr=0.98, cci_trigger=90.0, min_shift=0.48, min_volume=0.96),
    _make_def("S43CciOverlapZeroCross", "S43 CCI Overlap Zero Cross 1:3", "cci_trend", sessions=("overlap", "new_york"), fast=13, anchor=55, stop_atr=1.02, cci_trigger=70.0, min_shift=0.52, max_entropy=0.80),
    _make_def("S44CciShiftContinuation", "S44 CCI Shift Continuation 1:3", "cci_trend", sessions=ACTIVE_SESSIONS, fast=9, anchor=34, stop_atr=0.90, cci_trigger=85.0, min_shift=0.64, max_jump=0.20),
    _make_def("S45CciHurstScalp", "S45 CCI Hurst Scalp 1:3", "cci_trend", sessions=("new_york",), fast=12, anchor=89, stop_atr=1.08, cci_trigger=95.0, min_hurst=0.58, min_shift=0.56),
    _make_def("S46AdaptiveRegimeBlender", "S46 Adaptive Regime Blender 1:3", "adaptive_hybrid", sessions=ACTIVE_SESSIONS, fast=8, slow=21, anchor=55, channel=10, band_period=20, stop_atr=0.98, min_shift=0.46, max_shift=0.82),
    _make_def("S47AdaptiveEntropySwitch", "S47 Adaptive Entropy Switch 1:3", "adaptive_hybrid", sessions=("london", "new_york"), fast=9, slow=21, anchor=50, channel=9, band_period=18, stop_atr=0.94, max_entropy=0.78, max_shift=0.78),
    _make_def("S48AdaptiveShiftRouter", "S48 Adaptive Shift Router 1:3", "adaptive_hybrid", sessions=("overlap",), fast=10, slow=34, anchor=55, channel=11, band_period=22, stop_atr=1.02, min_shift=0.62, max_shift=0.74, max_jump=0.18),
    _make_def("S49AdaptiveHurstRouter", "S49 Adaptive Hurst Router 1:3", "adaptive_hybrid", sessions=("overlap", "new_york"), fast=13, slow=34, anchor=89, channel=12, band_period=24, stop_atr=1.08, min_hurst=0.58, min_shift=0.54, max_shift=0.80),
    _make_def("S50AdaptiveSessionHybrid", "S50 Adaptive Session Hybrid 1:3", "adaptive_hybrid", sessions=ACTIVE_SESSIONS, fast=5, slow=13, anchor=34, channel=8, band_period=18, stop_atr=0.90, min_shift=0.40, max_shift=0.84, lock_rr=0.08),
]


__all__ = []
for class_name, attrs in STRATEGY_DEFS:
    globals()[class_name] = type(class_name, (BaseScalpStrategy,), attrs)
    __all__.append(class_name)
