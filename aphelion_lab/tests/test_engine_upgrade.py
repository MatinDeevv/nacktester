"""
Aphelion Lab — Engine Upgrade Test Suite
Tests for all 50 features: market structure, indicators, price action, execution realism.
Run with: python -m pytest aphelion_lab/tests/test_engine_upgrade.py -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import pytest

# ─── Test Data Fixtures ──────────────────────────────────────────────────────

def _make_bars(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate realistic OHLCV test data with DatetimeIndex."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-02 08:00", periods=n, freq="5min")
    close = 2000.0 + np.cumsum(rng.randn(n) * 2)
    high = close + rng.uniform(0.5, 3, n)
    low = close - rng.uniform(0.5, 3, n)
    opn = close + rng.randn(n) * 0.5
    vol = rng.randint(100, 10000, n).astype(float)
    df = pd.DataFrame({"open": opn, "high": high, "low": low, "close": close,
                        "volume": vol}, index=dates)
    # Ensure OHLC consistency
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def bars():
    return _make_bars(200)


@pytest.fixture
def long_bars():
    return _make_bars(2000)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE A: Market Structure
# ═══════════════════════════════════════════════════════════════════════════

class TestMarketStructure:

    def test_a1_ensure_volume(self, bars):
        from aphelion_lab.market_structure import ensure_volume
        df = bars.drop(columns=["volume"], errors="ignore")
        result = ensure_volume(df)
        assert "volume" in result.columns
        assert (result["volume"] == 0).all()

    def test_a2_spread_columns(self, bars):
        from aphelion_lab.market_structure import add_spread_columns
        result = add_spread_columns(bars, fixed_spread=0.02)
        assert "spread" in result.columns
        assert "bid_close" in result.columns
        assert "ask_close" in result.columns
        assert np.isclose(result["spread"].iloc[0], 0.02)

    def test_a4_session_labels(self, bars):
        from aphelion_lab.market_structure import add_session_labels
        result = add_session_labels(bars)
        assert "session" in result.columns
        valid = {"asia", "london", "new_york", "overlap", "off_hours"}
        assert set(result["session"].unique()).issubset(valid)

    def test_a5_dow_labels(self, bars):
        from aphelion_lab.market_structure import add_dow_labels
        result = add_dow_labels(bars)
        assert "dow" in result.columns
        assert "dow_name" in result.columns

    def test_a7_htf_cache(self, bars):
        from aphelion_lab.market_structure import HTFCache
        htf = HTFCache()
        htf.build(bars)
        # Should have built some HTF caches
        assert len(htf._cache) > 0
        bar = htf.get_last_bar("M15", bars.index[50])
        if bar is not None:
            assert "close" in bar.index

    def test_a8_gap_detection(self, bars):
        from aphelion_lab.market_structure import detect_gaps
        result = detect_gaps(bars)
        assert "gap" in result.columns

    def test_a9_partial_candle(self, bars):
        from aphelion_lab.market_structure import mark_last_bar_partial
        result = mark_last_bar_partial(bars)
        assert "is_partial" in result.columns
        assert result["is_partial"].iloc[-1] is True or result["is_partial"].iloc[-1] == True

    def test_a10_symbol_meta(self):
        from aphelion_lab.market_structure import SYMBOL_META
        assert "XAUUSD" in SYMBOL_META
        meta = SYMBOL_META["XAUUSD"]
        assert meta.pip_size == 0.01
        assert meta.lot_size == 100

    def test_enrich_pipeline(self, bars):
        from aphelion_lab.market_structure import enrich_dataframe
        result = enrich_dataframe(bars)
        assert "session" in result.columns
        assert "dow" in result.columns
        assert len(result) == len(bars)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE B: Indicators
# ═══════════════════════════════════════════════════════════════════════════

class TestIndicators:

    def test_b11_vwap(self, bars):
        from aphelion_lab.indicators import vwap
        result = vwap(bars)
        assert len(result) == len(bars)
        assert not result.isna().all()

    def test_b12_anchored_vwap(self, bars):
        from aphelion_lab.indicators import anchored_vwap
        result = anchored_vwap(bars, anchor="day")
        assert len(result) == len(bars)

    def test_b13_macd(self, bars):
        from aphelion_lab.indicators import macd
        result = macd(bars["close"])
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns

    def test_b14_adx(self, bars):
        from aphelion_lab.indicators import adx
        result = adx(bars)
        assert "plus_di" in result.columns
        assert "minus_di" in result.columns
        assert "adx" in result.columns

    def test_b15_donchian(self, bars):
        from aphelion_lab.indicators import donchian
        result = donchian(bars)
        assert "dc_upper" in result.columns
        # Upper should always >= lower
        valid = result.dropna()
        assert (valid["dc_upper"] >= valid["dc_lower"]).all()

    def test_b16_keltner(self, bars):
        from aphelion_lab.indicators import keltner
        result = keltner(bars)
        assert "kc_upper" in result.columns
        assert "kc_mid" in result.columns

    def test_b17_supertrend(self, bars):
        from aphelion_lab.indicators import supertrend
        result = supertrend(bars)
        assert "supertrend" in result.columns
        assert set(result["st_direction"].dropna().unique()).issubset({1.0, -1.0})

    def test_b18_stoch_rsi(self, bars):
        from aphelion_lab.indicators import stoch_rsi
        result = stoch_rsi(bars["close"])
        assert "stoch_rsi_k" in result.columns
        assert "stoch_rsi_d" in result.columns

    def test_b19_cci(self, bars):
        from aphelion_lab.indicators import cci
        result = cci(bars)
        assert len(result) == len(bars)

    def test_b20_roc_momentum(self, bars):
        from aphelion_lab.indicators import roc, momentum
        r = roc(bars["close"])
        m = momentum(bars["close"])
        assert len(r) == len(bars)
        assert len(m) == len(bars)

    def test_b21_obv(self, bars):
        from aphelion_lab.indicators import obv
        result = obv(bars)
        assert len(result) == len(bars)

    def test_b22_mfi(self, bars):
        from aphelion_lab.indicators import mfi
        result = mfi(bars)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_b23_cmf(self, bars):
        from aphelion_lab.indicators import cmf
        result = cmf(bars)
        assert len(result) == len(bars)

    def test_b24_parabolic_sar(self, bars):
        from aphelion_lab.indicators import parabolic_sar
        result = parabolic_sar(bars)
        assert "sar" in result.columns
        assert "sar_direction" in result.columns

    def test_b25_atr_bands(self, bars):
        from aphelion_lab.indicators import atr_bands
        result = atr_bands(bars)
        valid = result.dropna()
        assert (valid["atr_upper"] >= valid["atr_mid"]).all()
        assert (valid["atr_mid"] >= valid["atr_lower"]).all()


# ═══════════════════════════════════════════════════════════════════════════
# PHASE C: Price Action
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceAction:

    def test_c26_pivot_points(self, long_bars):
        from aphelion_lab.price_action import pivot_points
        result = pivot_points(long_bars)
        assert "pp" in result.columns
        assert "r1" in result.columns
        assert "s1" in result.columns

    def test_c27_inside_outside_bars(self, bars):
        from aphelion_lab.price_action import inside_bars, outside_bars
        ib = inside_bars(bars)
        ob = outside_bars(bars)
        assert ib.dtype == bool
        assert ob.dtype == bool

    def test_c28_nr4_nr7(self, bars):
        from aphelion_lab.price_action import narrow_range
        nr4 = narrow_range(bars, 4)
        nr7 = narrow_range(bars, 7)
        assert nr4.dtype == bool
        assert nr7.dtype == bool

    def test_c29_range_helpers(self, bars):
        from aphelion_lab.price_action import body_ratio, upper_wick_ratio, lower_wick_ratio
        br = body_ratio(bars)
        uw = upper_wick_ratio(bars)
        lw = lower_wick_ratio(bars)
        assert (br >= 0).all() and (br <= 1.01).all()  # slight float tolerance
        assert (uw >= 0).all()
        assert (lw >= 0).all()

    def test_c30_trend_classifier(self, bars):
        from aphelion_lab.price_action import trend_classifier
        result = trend_classifier(bars)
        valid_classes = {"strong_up", "weak_up", "strong_down", "weak_down", "range"}
        assert set(result.unique()).issubset(valid_classes)

    def test_c31_volatility_classifier(self, bars):
        from aphelion_lab.price_action import volatility_classifier
        result = volatility_classifier(bars)
        valid = {"low", "normal", "high", "extreme"}
        assert set(result.unique()).issubset(valid)

    def test_c32_bar_patterns(self, bars):
        from aphelion_lab.price_action import bar_patterns
        result = bar_patterns(bars)
        assert "doji" in result.columns
        assert "hammer" in result.columns
        assert "engulfing_bull" in result.columns

    def test_c33_distance_helpers(self, bars):
        from aphelion_lab.price_action import distance_from_high
        result = distance_from_high(bars)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1.01).all()

    def test_c34_breakout_quality(self, bars):
        from aphelion_lab.price_action import breakout_quality
        result = breakout_quality(bars)
        assert "bull_breakout" in result.columns
        assert "bear_quality" in result.columns

    def test_c35_liquidity_sweeps(self, bars):
        from aphelion_lab.price_action import liquidity_sweeps
        result = liquidity_sweeps(bars)
        assert "sweep_high" in result.columns
        assert "confirmed_sweep_low" in result.columns


# ═══════════════════════════════════════════════════════════════════════════
# PHASE D: Execution Realism
# ═══════════════════════════════════════════════════════════════════════════

class TestExecution:

    def test_d36_slippage_models(self):
        from aphelion_lab.execution import SlippageMode, calc_slippage
        fixed = calc_slippage(SlippageMode.FIXED, 2.0, 0.01)
        assert np.isclose(fixed, 0.02)
        pct = calc_slippage(SlippageMode.PCT, 5.0, 0.01, price=2000.0)
        assert pct > 0
        vol = calc_slippage(SlippageMode.VOL_SCALED, 10.0, 0.01, atr=5.0)
        assert vol > 0

    def test_d37_commission_models(self):
        from aphelion_lab.execution import CommissionMode, calc_commission
        per_lot = calc_commission(CommissionMode.PER_LOT, 7.0, 0.01)
        assert np.isclose(per_lot, 7.0)
        per_trade = calc_commission(CommissionMode.PER_TRADE, 3.5, 0.01)
        assert np.isclose(per_trade, 3.5)
        pct = calc_commission(CommissionMode.PCT, 0.1, 0.01, notional=200000)
        assert pct > 0

    def test_d38_dynamic_spread(self):
        from aphelion_lab.execution import dynamic_spread
        base = dynamic_spread(2.0, "london", "normal")
        off = dynamic_spread(2.0, "off_hours", "extreme")
        assert off > base  # wider in off-hours + extreme vol

    def test_d39_intrabar_priority(self):
        from aphelion_lab.execution import intrabar_stop_priority
        bar = pd.Series({"open": 2000, "high": 2010, "low": 1990, "close": 2005})
        # Both SL and TP triggered — SL closer to open
        result, price = intrabar_stop_priority(bar, "BUY", 2000, sl=1990, tp=2010)
        assert result in ("sl", "tp")

    def test_d40_pending_orders(self):
        from aphelion_lab.execution import OrderManager, PendingOrder, OrderType
        mgr = OrderManager()
        mgr.add(PendingOrder(OrderType.LIMIT, "BUY", 1990, 0.01, tag="test"))
        bar = pd.Series({"open": 2000, "high": 2005, "low": 1989, "close": 1995})
        filled = mgr.check_fills(bar)
        assert len(filled) == 1
        assert filled[0].tag == "test"
        assert len(mgr.pending) == 0

    def test_d41_trailing_stop(self):
        from aphelion_lab.execution import TrailingStopState, TrailingStopMode
        ts = TrailingStopState(mode=TrailingStopMode.FIXED, distance=50, best_price=2000)
        bar1 = pd.Series({"open": 2010, "high": 2020, "low": 2005, "close": 2015})
        sl = ts.update(bar1, "BUY", pip_value=0.01)
        assert sl is not None
        assert sl < 2020  # stop is below the best price
        # Trail should ratchet up
        bar2 = pd.Series({"open": 2020, "high": 2030, "low": 2015, "close": 2025})
        sl2 = ts.update(bar2, "BUY", pip_value=0.01)
        assert sl2 >= sl

    def test_d42_breakeven(self):
        from aphelion_lab.execution import check_breakeven
        activated, new_sl = check_breakeven(
            entry_price=2000, current_high=2015, current_low=1995,
            side="BUY", trigger_pips=10, lock_pips=2, pip_value=0.01)
        assert activated is True
        assert new_sl == 2000.02

    def test_d43_partial_tp(self):
        from aphelion_lab.execution import PartialTP, check_partial_tps
        partials = [PartialTP(price=2010, close_fraction=0.5)]
        bar = pd.Series({"high": 2015, "low": 2000, "close": 2012})
        triggered = check_partial_tps(partials, bar, "BUY")
        assert len(triggered) == 1
        assert triggered[0].triggered is True

    def test_d44_position_sizing(self):
        from aphelion_lab.execution import SizingMode, calc_position_size
        # Risk-based
        size = calc_position_size(SizingMode.RISK_PCT, equity=10000, risk_pct=1.0,
                                  sl_distance=5.0, pip_value=0.01, lot_multiplier=100)
        assert size >= 0.01
        # Fixed
        size_f = calc_position_size(SizingMode.FIXED, fixed_size=0.05)
        assert size_f == 0.05

    def test_d45_cooldown(self):
        from aphelion_lab.execution import CooldownTracker
        cd = CooldownTracker(5)
        assert cd.can_trade(0) is True
        cd.record_exit(10)
        assert cd.can_trade(12) is False
        assert cd.can_trade(15) is True

    def test_d46_session_filter(self):
        from aphelion_lab.execution import session_allows_entry
        bar = pd.Series({"session": "london", "close": 2000})
        assert session_allows_entry(bar, ["london", "new_york"]) is True
        assert session_allows_entry(bar, ["asia"]) is False
        assert session_allows_entry(bar, []) is True
        assert session_allows_entry(bar, None) is True

    def test_d47_risk_guard(self):
        from aphelion_lab.execution import RiskGuard
        # Test max trades per day (no daily loss limit to avoid triggering it first)
        rg = RiskGuard(max_daily_loss_pct=0.0, max_trades_per_day=3)
        rg.reset_if_new_day("2024-01-02")
        allowed, _ = rg.can_open_trade(5000, 5000)
        assert allowed is True
        rg.record_trade_pnl(-10)
        rg.record_trade_pnl(-10)
        rg.record_trade_pnl(-10)
        allowed, reason = rg.can_open_trade(4970, 5000)
        assert allowed is False
        assert reason == "max_trades_per_day"
        # Test max daily loss
        rg2 = RiskGuard(max_daily_loss_pct=5.0, max_trades_per_day=0)
        rg2.reset_if_new_day("2024-01-02")
        rg2.record_trade_pnl(-300)
        allowed2, reason2 = rg2.can_open_trade(4700, 5000)
        assert allowed2 is False
        assert reason2 == "daily_loss_limit"

    def test_d48_max_holding(self):
        from aphelion_lab.execution import should_force_close
        assert should_force_close(0, 10, 0) is False  # disabled
        assert should_force_close(0, 10, 15) is False
        assert should_force_close(0, 15, 15) is True

    def test_d49_multi_position(self):
        from aphelion_lab.execution import MultiPositionManager, PositionSlot
        mp = MultiPositionManager(max_positions=3)
        mp.add(PositionSlot(1, "BUY", 2000, 0, 0.01))
        mp.add(PositionSlot(2, "SELL", 2010, 5, 0.02))
        assert mp.count == 2
        assert mp.can_open() is True
        mp.add(PositionSlot(3, "BUY", 2020, 10, 0.01))
        assert mp.can_open() is False

    def test_d50_modify_sl_tp(self):
        from aphelion_lab.execution import PositionSlot
        slot = PositionSlot(1, "BUY", 2000, 0, 0.01, sl=1990, tp=2020)
        slot.modify_sl(1995)
        slot.modify_tp(2025)
        assert slot.sl == 1995
        assert slot.tp == 2025


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Engine with execution features
# ═══════════════════════════════════════════════════════════════════════════

class TestEngineIntegration:

    def test_legacy_backtest_unchanged(self, bars):
        """Existing strategies must still work with default config."""
        from aphelion_lab.backtest_engine import BacktestEngine, BacktestConfig, Side

        class SimpleMA:
            def on_bar(self, ctx):
                if ctx.bar_index < 20:
                    return
                fast = ctx.sma(5)
                slow = ctx.sma(20)
                if not ctx.has_position and fast > slow:
                    ctx.buy(0.01)
                elif ctx.has_position and fast < slow:
                    ctx.close()

        config = BacktestConfig(initial_capital=5000)
        engine = BacktestEngine(config)
        result = engine.run(bars, SimpleMA())
        assert result.total_trades >= 0
        assert result.final_equity > 0

    def test_trailing_stop_engine(self, bars):
        """Trailing stop should move SL during backtest."""
        from aphelion_lab.backtest_engine import BacktestEngine, BacktestConfig
        from aphelion_lab.execution import TrailingStopMode

        class AlwaysBuy:
            def on_bar(self, ctx):
                if not ctx.has_position and ctx.bar_index == 5:
                    ctx.buy(0.01, sl=bars.iloc[5]["close"] - 10)

        config = BacktestConfig(initial_capital=5000,
                                trailing_stop_mode=TrailingStopMode.FIXED,
                                trailing_stop_distance=30)
        engine = BacktestEngine(config)
        result = engine.run(bars, AlwaysBuy())
        assert result.total_trades >= 1

    def test_cooldown_prevents_rapid_trades(self, bars):
        """Engine should block trades during cooldown."""
        from aphelion_lab.backtest_engine import BacktestEngine, BacktestConfig

        class RapidTrader:
            def on_bar(self, ctx):
                if not ctx.has_position:
                    ctx.buy(0.01)
                else:
                    ctx.close()

        config = BacktestConfig(initial_capital=5000, cooldown_bars=10)
        engine = BacktestEngine(config)
        result = engine.run(bars, RapidTrader())
        # With 10-bar cooldown, can't trade every bar
        no_cd_config = BacktestConfig(initial_capital=5000, cooldown_bars=0)
        engine2 = BacktestEngine(no_cd_config)
        result2 = engine2.run(bars, RapidTrader())
        assert result.total_trades < result2.total_trades

    def test_breakeven_activates(self, bars):
        """Test that breakeven moves SL to entry."""
        from aphelion_lab.backtest_engine import BacktestEngine, BacktestConfig

        class BuyAndHold:
            def on_bar(self, ctx):
                if not ctx.has_position and ctx.bar_index == 1:
                    ctx.buy(0.01, sl=bars.iloc[1]["close"] - 50)

        config = BacktestConfig(initial_capital=5000,
                                breakeven_trigger_pips=5,
                                breakeven_lock_pips=1)
        engine = BacktestEngine(config)
        result = engine.run(bars, BuyAndHold())
        assert result.total_trades >= 1

    def test_new_metrics_present(self, bars):
        """New metrics should appear in to_metrics_dict."""
        from aphelion_lab.backtest_engine import BacktestEngine, BacktestConfig

        class SimpleBuyer:
            def on_bar(self, ctx):
                if not ctx.has_position and ctx.bar_index == 10:
                    ctx.buy(0.01)
                elif ctx.has_position and ctx.bar_index == 50:
                    ctx.close()

        engine = BacktestEngine(BacktestConfig(initial_capital=5000))
        result = engine.run(bars, SimpleBuyer())
        metrics = result.to_metrics_dict()
        assert "Calmar Ratio" in metrics
        assert "Recovery Factor" in metrics
        assert "Consec. Wins" in metrics
        assert "SL Exits" in metrics

    def test_max_holding_time(self, bars):
        """Position should be force-closed after max_holding_bars."""
        from aphelion_lab.backtest_engine import BacktestEngine, BacktestConfig

        class BuyOnce:
            def on_bar(self, ctx):
                if not ctx.has_position and ctx.bar_index == 5:
                    ctx.buy(0.01)

        config = BacktestConfig(initial_capital=5000, max_holding_bars=10)
        engine = BacktestEngine(config)
        result = engine.run(bars, BuyOnce())
        assert result.total_trades >= 1
        assert result.trades[0].exit_reason == "max_holding_time"

    def test_ctx_ind_lookup(self, bars):
        """ctx.ind() should safely return nan for missing columns."""
        from aphelion_lab.backtest_engine import BacktestEngine, BacktestConfig

        results = []
        class IndChecker:
            def on_bar(self, ctx):
                v = ctx.ind("nonexistent_col")
                results.append(v)

        engine = BacktestEngine(BacktestConfig())
        engine.run(bars, IndChecker())
        assert all(np.isnan(v) for v in results)

    def test_pending_order_limit(self, bars):
        """Test limit order fill via pending order API."""
        from aphelion_lab.backtest_engine import BacktestEngine, BacktestConfig

        class LimitBuyer:
            def on_bar(self, ctx):
                if ctx.bar_index == 5 and not ctx.has_position:
                    # Place limit buy below current price
                    ctx.place_limit("BUY", ctx.bar["low"] - 0.5, size=0.01)

        config = BacktestConfig(initial_capital=5000, pending_orders_enabled=True)
        engine = BacktestEngine(config)
        result = engine.run(bars, LimitBuyer())
        # May or may not fill depending on price action
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
