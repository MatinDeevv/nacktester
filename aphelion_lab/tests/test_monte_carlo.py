import numpy as np
import pandas as pd
import pytest

from aphelion_lab.backtest_engine import BacktestConfig, BacktestResult, Side, Trade
from aphelion_lab.monte_carlo import MonteCarloConfig, MonteCarloMode


def _make_result(trade_specs, initial_capital: float = 1000.0) -> BacktestResult:
    trades = []
    equity_curve = []
    drawdown_curve = []
    timestamps = []
    equity = initial_capital
    peak = initial_capital
    base = pd.Timestamp("2024-01-01 09:00:00")

    for idx, spec in enumerate(trade_specs, start=1):
        pnl, commission = spec
        entry_time = base + pd.Timedelta(hours=idx * 2)
        exit_time = entry_time + pd.Timedelta(hours=1)
        trades.append(
            Trade(
                id=idx,
                side=Side.BUY,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=100.0,
                exit_price=101.0,
                size=0.01,
                pnl=float(pnl),
                commission=float(commission),
                bars_held=1,
                exit_reason="signal",
            )
        )
        equity += float(pnl) - float(commission)
        peak = max(peak, equity)
        equity_curve.append(equity)
        drawdown_curve.append((equity - peak) / peak * 100.0 if peak > 0 else 0.0)
        timestamps.append(exit_time)

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        timestamps=timestamps,
        config=BacktestConfig(initial_capital=initial_capital),
        data=pd.DataFrame(),
    )


def test_monte_carlo_shuffle_preserves_final_equity_after_costs():
    result = _make_result([(100, 5), (-50, 5), (25, 5)], initial_capital=1000)

    mc = result.monte_carlo(
        MonteCarloConfig(
            iterations=128,
            method=MonteCarloMode.SHUFFLE,
            random_seed=7,
        )
    )

    expected_final_equity = 1000 + (100 - 5) + (-50 - 5) + (25 - 5)
    assert np.allclose(mc.final_equities, expected_final_equity)
    assert mc.final_equity_p05 == pytest.approx(expected_final_equity)
    assert mc.final_equity_p50 == pytest.approx(expected_final_equity)
    assert mc.final_equity_p95 == pytest.approx(expected_final_equity)


def test_monte_carlo_bootstrap_returns_ordered_path_bands_and_ruin_stats():
    result = _make_result([(-300, 0), (-250, 0), (350, 0), (-200, 0)], initial_capital=1000)

    mc = result.monte_carlo(
        MonteCarloConfig(
            iterations=300,
            method=MonteCarloMode.BOOTSTRAP,
            confidence_level=0.80,
            ruin_threshold_pct=40,
            random_seed=11,
        )
    )

    assert mc.iterations == 300
    assert mc.trades_per_path == result.total_trades
    assert len(mc.lower_path) == result.total_trades + 1
    assert np.all(mc.lower_path <= mc.median_path)
    assert np.all(mc.median_path <= mc.upper_path)
    assert mc.max_drawdown_p95 >= mc.max_drawdown_p50 >= 0
    assert mc.risk_of_ruin_pct > 0


def test_monte_carlo_handles_empty_trade_lists():
    result = _make_result([], initial_capital=2500)

    mc = result.monte_carlo(
        MonteCarloConfig(
            iterations=16,
            method=MonteCarloMode.BOOTSTRAP,
            random_seed=3,
        )
    )

    assert np.allclose(mc.final_equities, 2500.0)
    assert np.allclose(mc.lower_path, [2500.0])
    assert np.allclose(mc.median_path, [2500.0])
    assert np.allclose(mc.upper_path, [2500.0])
    assert mc.risk_of_ruin_pct == 0.0
