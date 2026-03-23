from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional, Sequence, Union
import math

import numpy as np
import pandas as pd

Number = Union[int, float, np.number]
ArrayLike = Union[Sequence[Number], np.ndarray, pd.Series]


@dataclass
class TradeStats:
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    expectancy: float


@dataclass
class PerformanceStats:
    start_equity: float
    end_equity: float
    net_profit: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    annual_volatility_pct: float
    sharpe: float
    sortino: float
    calmar: float
    bars: int
    periods_per_year: float
    trade_stats: Optional[TradeStats] = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.trade_stats is not None:
            payload["trade_stats"] = asdict(self.trade_stats)
        return payload


def _to_series(values: ArrayLike, name: str, allow_empty: bool = False) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.astype(float).copy()
    else:
        series = pd.Series(np.asarray(values, dtype=float), name=name)

    if series.empty and not allow_empty:
        raise ValueError(f"{name} is empty")
    if series.isna().any():
        raise ValueError(f"{name} contains NaN values")
    return series


def _supports_return_calculation(equity: pd.Series) -> bool:
    return len(equity) >= 2 and bool((equity > 0).all())


def _annual_rf_to_period_rf(rf_annual: float, periods_per_year: float) -> float:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    return (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0


def _equity_to_returns(
    equity: pd.Series,
    method: Literal["simple", "log"] = "simple",
) -> pd.Series:
    if not _supports_return_calculation(equity):
        return pd.Series(dtype=float)

    if method == "simple":
        returns = equity.pct_change().dropna()
    elif method == "log":
        returns = np.log(equity / equity.shift(1)).dropna()
    else:
        raise ValueError("method must be 'simple' or 'log'")

    returns = returns.astype(float)
    returns = returns[np.isfinite(returns)]
    return returns


def _safe_ratio(numerator: float, denominator: float) -> float:
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        if numerator > 0:
            return float("inf")
        if numerator < 0:
            return float("-inf")
        return 0.0
    return numerator / denominator


def compute_max_drawdown_pct(equity: ArrayLike) -> float:
    eq = _to_series(equity, "equity")
    if len(eq) < 2:
        return 0.0

    running_peak = eq.cummax().replace(0, np.nan)
    drawdown = eq / running_peak - 1.0
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan).dropna()
    if drawdown.empty:
        return 0.0
    return float(abs(drawdown.min()) * 100.0)


def compute_cagr_pct(equity: ArrayLike, periods_per_year: float) -> float:
    eq = _to_series(equity, "equity")
    if not _supports_return_calculation(eq):
        return 0.0

    n_periods = len(eq) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0

    start = float(eq.iloc[0])
    end = float(eq.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0

    log_growth = math.log(end / start)
    exponent = log_growth / years
    if exponent > 700:
        return float("inf")
    if exponent < -700:
        return -100.0

    cagr = math.exp(exponent) - 1.0
    return float(cagr * 100.0)


def compute_sharpe(
    equity: ArrayLike,
    periods_per_year: float,
    rf_annual: float = 0.0,
    return_method: Literal["simple", "log"] = "simple",
    ddof: int = 1,
) -> float:
    eq = _to_series(equity, "equity")
    returns = _equity_to_returns(eq, method=return_method)
    if len(returns) < 2:
        return 0.0

    rf_period = _annual_rf_to_period_rf(rf_annual, periods_per_year)
    excess = returns - rf_period
    mean_excess = float(excess.mean())
    std_excess = float(excess.std(ddof=ddof))

    return float(math.sqrt(periods_per_year) * _safe_ratio(mean_excess, std_excess))


def compute_sortino(
    equity: ArrayLike,
    periods_per_year: float,
    rf_annual: float = 0.0,
    return_method: Literal["simple", "log"] = "simple",
    ddof: int = 1,
) -> float:
    eq = _to_series(equity, "equity")
    returns = _equity_to_returns(eq, method=return_method)
    if len(returns) < 2:
        return 0.0

    rf_period = _annual_rf_to_period_rf(rf_annual, periods_per_year)
    excess = returns - rf_period
    mean_excess = float(excess.mean())
    downside = excess[excess < 0.0]

    if len(downside) == 0:
        return float("inf") if mean_excess > 0 else 0.0

    downside_std = float(downside.std(ddof=ddof))
    return float(math.sqrt(periods_per_year) * _safe_ratio(mean_excess, downside_std))


def compute_calmar(equity: ArrayLike, periods_per_year: float) -> float:
    cagr_pct = compute_cagr_pct(equity, periods_per_year)
    max_dd_pct = compute_max_drawdown_pct(equity)
    return float(_safe_ratio(cagr_pct, max_dd_pct))


def compute_annual_volatility_pct(
    equity: ArrayLike,
    periods_per_year: float,
    return_method: Literal["simple", "log"] = "simple",
    ddof: int = 1,
) -> float:
    eq = _to_series(equity, "equity")
    returns = _equity_to_returns(eq, method=return_method)
    if len(returns) < 2:
        return 0.0

    vol = float(returns.std(ddof=ddof) * math.sqrt(periods_per_year))
    return vol * 100.0


def compute_trade_stats(trade_pnls: ArrayLike) -> TradeStats:
    pnl = _to_series(trade_pnls, "trade_pnls", allow_empty=True)
    if pnl.empty:
        return TradeStats(
            trades=0,
            wins=0,
            losses=0,
            win_rate_pct=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=0.0,
            avg_trade=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            expectancy=0.0,
        )

    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]

    trades = int(len(pnl))
    win_count = int(len(wins))
    loss_count = int(len(losses))

    gross_profit = float(wins.sum())
    gross_loss_abs = float(abs(losses.sum()))

    win_rate = (win_count / trades * 100.0) if trades > 0 else 0.0
    profit_factor = _safe_ratio(gross_profit, gross_loss_abs)
    avg_trade = float(pnl.mean()) if trades > 0 else 0.0
    avg_win = float(wins.mean()) if win_count > 0 else 0.0
    avg_loss = float(losses.mean()) if loss_count > 0 else 0.0

    return TradeStats(
        trades=trades,
        wins=win_count,
        losses=loss_count,
        win_rate_pct=float(win_rate),
        gross_profit=gross_profit,
        gross_loss=-gross_loss_abs,
        profit_factor=float(profit_factor),
        avg_trade=avg_trade,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=avg_trade,
    )


def infer_periods_per_year_from_index(index: pd.Index, fallback: Optional[float] = None) -> float:
    if not isinstance(index, pd.DatetimeIndex):
        if fallback is None:
            raise ValueError("index is not a DatetimeIndex and no fallback was provided")
        return float(fallback)

    if len(index) < 3:
        if fallback is None:
            raise ValueError("need at least 3 timestamps to infer periods_per_year")
        return float(fallback)

    dt_index = pd.DatetimeIndex(index).sort_values()
    diffs = dt_index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    diffs = diffs[diffs > 0]

    seconds_per_year = 365.2425 * 24 * 60 * 60
    candidates = []

    if len(diffs) > 0:
        spacing_based = seconds_per_year / float(np.median(diffs))
        if np.isfinite(spacing_based) and spacing_based > 0:
            candidates.append(float(spacing_based))

    elapsed_seconds = float((dt_index[-1] - dt_index[0]).total_seconds())
    if elapsed_seconds > 0:
        density_based = ((len(dt_index) - 1) / elapsed_seconds) * seconds_per_year
        if np.isfinite(density_based) and density_based > 0:
            candidates.append(float(density_based))

    if candidates:
        return float(min(candidates))

    if fallback is None:
        raise ValueError("could not infer periods_per_year from index")
    return float(fallback)


def compute_performance_stats(
    equity: ArrayLike,
    *,
    initial_equity: Optional[float] = None,
    periods_per_year: Optional[float] = None,
    index: Optional[pd.Index] = None,
    rf_annual: float = 0.0,
    return_method: Literal["simple", "log"] = "simple",
    trade_pnls: Optional[ArrayLike] = None,
    ddof: int = 1,
) -> PerformanceStats:
    eq_raw = _to_series(equity, "equity", allow_empty=initial_equity is not None)

    if periods_per_year is None:
        periods_per_year = infer_periods_per_year_from_index(index, fallback=252.0) if index is not None else 252.0
    periods_per_year = float(periods_per_year)

    if eq_raw.empty:
        start_equity = float(initial_equity if initial_equity is not None else 0.0)
        trade_stats = compute_trade_stats(trade_pnls) if trade_pnls is not None else None
        return PerformanceStats(
            start_equity=start_equity,
            end_equity=start_equity,
            net_profit=0.0,
            total_return_pct=0.0,
            cagr_pct=0.0,
            max_drawdown_pct=0.0,
            annual_volatility_pct=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            bars=0,
            periods_per_year=periods_per_year,
            trade_stats=trade_stats,
        )

    if initial_equity is not None:
        start_equity = float(initial_equity)
        eq_for_metrics = pd.concat(
            [pd.Series([start_equity], dtype=float), eq_raw.reset_index(drop=True)],
            ignore_index=True,
        )
    else:
        start_equity = float(eq_raw.iloc[0])
        eq_for_metrics = eq_raw

    end_equity = float(eq_raw.iloc[-1])
    net_profit = end_equity - start_equity
    total_return_pct = _safe_ratio(end_equity - start_equity, start_equity) * 100.0 if start_equity != 0 else 0.0

    stats = PerformanceStats(
        start_equity=start_equity,
        end_equity=end_equity,
        net_profit=float(net_profit),
        total_return_pct=float(total_return_pct),
        cagr_pct=compute_cagr_pct(eq_for_metrics, periods_per_year),
        max_drawdown_pct=compute_max_drawdown_pct(eq_for_metrics),
        annual_volatility_pct=compute_annual_volatility_pct(eq_for_metrics, periods_per_year, return_method, ddof),
        sharpe=compute_sharpe(eq_for_metrics, periods_per_year, rf_annual, return_method, ddof),
        sortino=compute_sortino(eq_for_metrics, periods_per_year, rf_annual, return_method, ddof),
        calmar=compute_calmar(eq_for_metrics, periods_per_year),
        bars=int(len(eq_raw)),
        periods_per_year=periods_per_year,
        trade_stats=compute_trade_stats(trade_pnls) if trade_pnls is not None else None,
    )
    return stats


def compute_leaderboard_score(
    *,
    sharpe: float,
    total_return_pct: float,
    max_drawdown_pct: float,
    win_rate_pct: Optional[float] = None,
    profit_factor: Optional[float] = None,
) -> float:
    sharpe_value = 0.0
    if np.isfinite(sharpe):
        sharpe_value = max(-3.0, min(6.0, float(sharpe)))
    elif sharpe > 0:
        sharpe_value = 6.0
    elif sharpe < 0:
        sharpe_value = -3.0

    score = sharpe_value * 2.0

    safe_return = float(max(total_return_pct, -95.0))
    if safe_return >= 0:
        score += math.log1p(safe_return / 100.0) * 12.0
    else:
        score += safe_return * 0.10

    score -= float(max_drawdown_pct) * 0.75

    if win_rate_pct is not None and np.isfinite(win_rate_pct):
        score += (float(win_rate_pct) - 50.0) * 0.05

    if profit_factor is not None:
        if np.isfinite(profit_factor):
            score += max(0.0, min(float(profit_factor), 5.0) - 1.0) * 0.75
        elif profit_factor > 0:
            score += 3.0

    return float(score)
