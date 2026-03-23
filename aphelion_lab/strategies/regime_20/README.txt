Regime 20 strategy pack
=======================

This pack is built for the current engine surface:
- ctx.market_regime
- ctx.volatility_regime
- ctx.entropy
- ctx.hurst
- ctx.jump_intensity
- ctx.distribution_shift
- ctx.session
- ctx.spread
- ctx.modify_sl / ctx.modify_tp / ctx.calc_size

The strategies are designed to remain loadable and runnable under the default
engine config. They rely on the engine's automatic data enrichment, so they
work best when BacktestConfig.auto_enrich_data is left enabled.
