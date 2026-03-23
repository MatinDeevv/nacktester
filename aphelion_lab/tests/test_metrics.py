import numpy as np
import pandas as pd

from aphelion_lab.backtest_engine import BacktestConfig, BacktestResult, Side, Trade
from aphelion_lab.metrics import compute_sharpe, infer_periods_per_year_from_index


def _make_result(
    equity_curve,
    timestamps,
    trade_specs=None,
    initial_capital: float = 1000.0,
) -> BacktestResult:
    trades = []
    trade_specs = trade_specs or []
    for idx, spec in enumerate(trade_specs, start=1):
        pnl, commission = spec
        entry_time = timestamps[min(idx - 1, len(timestamps) - 1)]
        exit_time = timestamps[min(idx, len(timestamps) - 1)]
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

    peak = initial_capital
    drawdown_curve = []
    for equity in equity_curve:
        peak = max(peak, equity)
        drawdown_curve.append((equity - peak) / peak * 100.0 if peak > 0 else 0.0)

    data = pd.DataFrame(
        {"close": np.asarray(equity_curve, dtype=float)},
        index=pd.DatetimeIndex(timestamps),
    )

    return BacktestResult(
        trades=trades,
        equity_curve=[float(v) for v in equity_curve],
        drawdown_curve=drawdown_curve,
        timestamps=list(pd.DatetimeIndex(timestamps)),
        config=BacktestConfig(initial_capital=initial_capital),
        data=data,
    )


def test_compute_sharpe_sanity_for_rising_flat_and_falling_equity():
    periods_per_year = 252.0

    assert compute_sharpe([100, 101, 102, 103, 104, 105], periods_per_year) > 0
    assert compute_sharpe([100, 100, 100, 100, 100], periods_per_year) == 0.0
    assert compute_sharpe([100, 99, 98, 97, 96, 95], periods_per_year) < 0


def test_infer_periods_per_year_respects_data_density():
    business_daily = pd.bdate_range("2024-01-01", periods=252)
    inferred_daily = infer_periods_per_year_from_index(business_daily)
    assert 200 <= inferred_daily <= 300

    regular_session_index = []
    for day in pd.bdate_range("2024-01-02", periods=30):
        regular_session_index.extend(pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=78, freq="5min"))
    inferred_intraday = infer_periods_per_year_from_index(pd.DatetimeIndex(regular_session_index))
    assert 15000 <= inferred_intraday <= 30000


def test_backtest_result_sharpe_uses_frequency_inference_instead_of_hardcoded_hourly_factor():
    daily_index = pd.bdate_range("2024-01-01", periods=6)
    intraday_index = pd.date_range("2024-01-01 09:30", periods=6, freq="5min")
    equity = [1000, 1010, 1020, 1030, 1040, 1050]

    daily_result = _make_result(equity, daily_index, initial_capital=1000.0)
    intraday_result = _make_result(equity, intraday_index, initial_capital=1000.0)

    assert daily_result.sharpe_ratio > 0
    assert intraday_result.sharpe_ratio > daily_result.sharpe_ratio


def test_backtest_result_net_pnl_matches_final_equity_after_costs():
    timestamps = pd.date_range("2024-01-01 09:00", periods=3, freq="h")
    result = _make_result(
        equity_curve=[1095.0, 1040.0, 1060.0],
        timestamps=timestamps,
        trade_specs=[(100, 5), (-50, 5), (25, 5)],
        initial_capital=1000.0,
    )

    assert result.net_pnl == 60.0
    assert result.net_pnl_after_costs == 60.0
    assert result.final_equity == 1060.0


def test_metrics_dict_includes_equity_based_performance_fields():
    timestamps = pd.bdate_range("2024-01-01", periods=6)
    result = _make_result(
        equity_curve=[1000.0, 1010.0, 1005.0, 1020.0, 1030.0, 1040.0],
        timestamps=timestamps,
        initial_capital=1000.0,
    )

    metrics = result.to_metrics_dict()

    assert "Sharpe Ratio" in metrics
    assert "Sortino Ratio" in metrics
    assert "CAGR" in metrics
    assert "Annual Volatility" in metrics
    assert "Periods/Year" in metrics
    assert metrics["Max Drawdown"] == "0.50%"
