Universal 20 strategy pack
==========================

This pack is built to be usable on most timeframes.

Design rules:
- no session-specific entry windows
- no daily VWAP dependence
- no pending-order dependence
- no engine-config toggles required beyond default auto-enrichment

The strategies use the current engine surface:
- ctx.market_regime
- ctx.volatility_regime
- ctx.entropy
- ctx.hurst
- ctx.jump_intensity
- ctx.distribution_shift
- ctx.spread
- ctx.modify_sl / ctx.modify_tp / ctx.calc_size

They are intended as broad test candidates, not production-ready alpha.
