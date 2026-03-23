from __future__ import annotations

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
    open_trade,
    range_ratio,
    risk_target,
    rsi,
    spread_ok,
    stoch_state,
    supertrend_state,
    trail_to_level,
    volume_ratio,
    zscore,
)

ACTIVE_SESSIONS = ("london", "overlap", "new_york")


def session_ok(ctx, sessions: tuple[str, ...] | None = None) -> bool:
    session = str(ctx.session or "")
    if not session:
        return True
    if sessions is None:
        sessions = ACTIVE_SESSIONS
    return session in sessions


def scalp_spread_ok(ctx, max_spread: float = 0.09) -> bool:
    return spread_ok(ctx, max_spread)


def trend_ok(
    ctx,
    sessions: tuple[str, ...] | None = None,
    min_hurst: float = 0.52,
    max_entropy: float = 0.82,
    max_jump: float = 0.26,
    min_shift: float = 0.45,
) -> bool:
    return (
        session_ok(ctx, sessions)
        and ctx.market_regime in ("trend", "transition")
        and ctx.hurst >= min_hurst
        and ctx.entropy <= max_entropy
        and ctx.jump_intensity <= max_jump
        and ctx.distribution_shift >= min_shift
        and ctx.volatility_regime in ("compressed", "normal", "high")
    )


def breakout_ok(
    ctx,
    sessions: tuple[str, ...] | None = None,
    min_shift: float = 0.60,
    max_jump: float = 0.30,
) -> bool:
    return (
        session_ok(ctx, sessions)
        and ctx.market_regime in ("trend", "transition", "stress")
        and ctx.distribution_shift >= min_shift
        and ctx.jump_intensity <= max_jump
        and ctx.volatility_regime in ("normal", "high", "extreme")
    )


def reversion_ok(
    ctx,
    sessions: tuple[str, ...] | None = None,
    min_entropy: float = 0.70,
    max_hurst: float = 0.52,
    max_jump: float = 0.20,
    max_shift: float = 0.90,
) -> bool:
    return (
        session_ok(ctx, sessions)
        and ctx.market_regime in ("range", "mean_revert", "transition")
        and ctx.entropy >= min_entropy
        and ctx.hurst <= max_hurst
        and ctx.jump_intensity <= max_jump
        and ctx.distribution_shift <= max_shift
        and ctx.volatility_regime in ("compressed", "normal", "high")
    )


def risk_distance_ok(
    price: float,
    sl: float,
    atr_value: float,
    min_atr: float = 0.28,
    max_atr: float = 2.50,
) -> bool:
    distance = abs(price - sl)
    return enough(distance, atr_value) and atr_value > 0 and atr_value * min_atr < distance < atr_value * max_atr


def lock_one_r(ctx, lock_rr: float = 0.10):
    if ctx.position is None:
        return
    trade = ctx.position.trade
    if trade.sl is None:
        return
    price = float(ctx.bar["close"])
    entry = float(trade.entry_price)
    sl = float(trade.sl)
    risk = abs(entry - sl)
    if not enough(price, entry, sl, risk) or risk <= 0:
        return
    side = current_side(ctx)
    if side == "BUY" and price >= entry + risk:
        candidate = min(price - risk * 0.15, entry + risk * lock_rr)
        if candidate > entry and candidate < price and (trade.sl is None or candidate > trade.sl):
            ctx.modify_sl(candidate)
    elif side == "SELL" and price <= entry - risk:
        candidate = max(price + risk * 0.15, entry - risk * lock_rr)
        if candidate < entry and candidate > price and (trade.sl is None or candidate < trade.sl):
            ctx.modify_sl(candidate)


def manage_open(ctx, anchor: float, atr_value: float, invalidation: float | None = None, lock_rr: float = 0.10):
    if ctx.position is None:
        return
    lock_one_r(ctx, lock_rr)
    if enough(anchor, atr_value):
        trail_to_level(ctx, anchor, buffer=atr_value * 0.08)
    if invalidation is None or not enough(invalidation):
        return
    price = float(ctx.bar["close"])
    side = current_side(ctx)
    if side == "BUY" and price < invalidation:
        ctx.close("scalp_invalidation")
    elif side == "SELL" and price > invalidation:
        ctx.close("scalp_invalidation")
