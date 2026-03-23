"""
Monte Carlo analytics for trade-sequence backtests.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np


class MonteCarloMode(str, Enum):
    BOOTSTRAP = "bootstrap"
    SHUFFLE = "shuffle"


@dataclass(frozen=True)
class MonteCarloConfig:
    iterations: int = 2000
    method: MonteCarloMode = MonteCarloMode.BOOTSTRAP
    sample_size: Optional[int] = None
    confidence_level: float = 0.95
    ruin_threshold_pct: float = 30.0
    random_seed: Optional[int] = None


@dataclass
class MonteCarloResult:
    config: MonteCarloConfig
    initial_capital: float
    trades_per_path: int
    final_equities: np.ndarray
    max_drawdowns_pct: np.ndarray
    ruined: np.ndarray
    lower_path: np.ndarray
    median_path: np.ndarray
    upper_path: np.ndarray

    @property
    def iterations(self) -> int:
        return int(self.final_equities.size)

    @property
    def ruin_equity(self) -> float:
        if self.config.ruin_threshold_pct <= 0:
            return 0.0
        return self.initial_capital * (1.0 - self.config.ruin_threshold_pct / 100.0)

    @property
    def lower_percentile(self) -> float:
        tail = (1.0 - _normalize_confidence_level(self.config.confidence_level)) / 2.0
        return tail * 100.0

    @property
    def upper_percentile(self) -> float:
        return 100.0 - self.lower_percentile

    @property
    def mean_final_equity(self) -> float:
        return float(np.mean(self.final_equities))

    @property
    def median_final_equity(self) -> float:
        return float(np.median(self.final_equities))

    @property
    def final_equity_p05(self) -> float:
        return float(np.percentile(self.final_equities, 5))

    @property
    def final_equity_p50(self) -> float:
        return float(np.percentile(self.final_equities, 50))

    @property
    def final_equity_p95(self) -> float:
        return float(np.percentile(self.final_equities, 95))

    @property
    def mean_return_pct(self) -> float:
        return (self.mean_final_equity - self.initial_capital) / self.initial_capital * 100.0

    @property
    def median_return_pct(self) -> float:
        return (self.median_final_equity - self.initial_capital) / self.initial_capital * 100.0

    @property
    def loss_probability_pct(self) -> float:
        return float(np.mean(self.final_equities < self.initial_capital) * 100.0)

    @property
    def risk_of_ruin_pct(self) -> float:
        if self.config.ruin_threshold_pct <= 0:
            return 0.0
        return float(np.mean(self.ruined) * 100.0)

    @property
    def max_drawdown_p50(self) -> float:
        return float(np.percentile(self.max_drawdowns_pct, 50))

    @property
    def max_drawdown_p95(self) -> float:
        return float(np.percentile(self.max_drawdowns_pct, 95))

    def to_summary_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "mode": self.config.method.value,
            "trades_per_path": self.trades_per_path,
            "ruin_equity": self.ruin_equity,
            "mean_final_equity": self.mean_final_equity,
            "median_final_equity": self.median_final_equity,
            "final_equity_p05": self.final_equity_p05,
            "final_equity_p50": self.final_equity_p50,
            "final_equity_p95": self.final_equity_p95,
            "mean_return_pct": self.mean_return_pct,
            "median_return_pct": self.median_return_pct,
            "loss_probability_pct": self.loss_probability_pct,
            "risk_of_ruin_pct": self.risk_of_ruin_pct,
            "max_drawdown_p50": self.max_drawdown_p50,
            "max_drawdown_p95": self.max_drawdown_p95,
        }

    def to_metrics_dict(self) -> dict:
        return {
            "MC Mode": self.config.method.value,
            "MC Iterations": str(self.iterations),
            "MC Final Eq P5": f"${self.final_equity_p05:.2f}",
            "MC Final Eq P50": f"${self.final_equity_p50:.2f}",
            "MC Final Eq P95": f"${self.final_equity_p95:.2f}",
            "MC Mean Return": f"{self.mean_return_pct:.2f}%",
            "MC Loss Prob.": f"{self.loss_probability_pct:.1f}%",
            "MC Risk of Ruin": f"{self.risk_of_ruin_pct:.1f}%",
            "MC Max DD P50": f"{self.max_drawdown_p50:.2f}%",
            "MC Max DD P95": f"{self.max_drawdown_p95:.2f}%",
        }


def simulate_trade_sequence(
    trade_results: Sequence[float],
    initial_capital: float,
    config: Optional[MonteCarloConfig] = None,
) -> MonteCarloResult:
    cfg = config or MonteCarloConfig()
    method = _normalize_mode(cfg.method)
    confidence_level = _normalize_confidence_level(cfg.confidence_level)
    outcomes = np.asarray(list(trade_results), dtype=float)

    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if cfg.iterations <= 0:
        raise ValueError("iterations must be positive")
    if cfg.ruin_threshold_pct < 0 or cfg.ruin_threshold_pct >= 100:
        raise ValueError("ruin_threshold_pct must be in [0, 100)")

    trade_count = int(outcomes.size)
    if trade_count == 0:
        if cfg.sample_size not in (None, 0):
            raise ValueError("sample_size must be 0 or None when there are no trades")
        final_equities = np.full(cfg.iterations, initial_capital, dtype=float)
        flat_path = np.array([initial_capital], dtype=float)
        return MonteCarloResult(
            config=MonteCarloConfig(
                iterations=cfg.iterations,
                method=method,
                sample_size=0,
                confidence_level=confidence_level,
                ruin_threshold_pct=cfg.ruin_threshold_pct,
                random_seed=cfg.random_seed,
            ),
            initial_capital=initial_capital,
            trades_per_path=0,
            final_equities=final_equities,
            max_drawdowns_pct=np.zeros(cfg.iterations, dtype=float),
            ruined=np.zeros(cfg.iterations, dtype=bool),
            lower_path=flat_path.copy(),
            median_path=flat_path.copy(),
            upper_path=flat_path.copy(),
        )

    sample_size = trade_count if cfg.sample_size is None else int(cfg.sample_size)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if method == MonteCarloMode.SHUFFLE and sample_size > trade_count:
        raise ValueError("shuffle mode cannot sample more trades than are available")

    rng = np.random.default_rng(cfg.random_seed)
    sampled = np.empty((cfg.iterations, sample_size), dtype=float)

    for idx in range(cfg.iterations):
        if method == MonteCarloMode.BOOTSTRAP:
            picks = rng.integers(0, trade_count, size=sample_size)
            sampled[idx] = outcomes[picks]
        else:
            sampled[idx] = rng.permutation(outcomes)[:sample_size]

    equity_paths = np.empty((cfg.iterations, sample_size + 1), dtype=float)
    equity_paths[:, 0] = initial_capital
    equity_paths[:, 1:] = initial_capital + np.cumsum(sampled, axis=1)

    running_peaks = np.maximum.accumulate(equity_paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown_paths = np.where(
            running_peaks > 0,
            (equity_paths - running_peaks) / running_peaks * 100.0,
            0.0,
        )
    max_drawdowns_pct = -np.min(drawdown_paths, axis=1)

    if cfg.ruin_threshold_pct > 0:
        ruin_equity = initial_capital * (1.0 - cfg.ruin_threshold_pct / 100.0)
        ruined = np.any(equity_paths <= ruin_equity, axis=1)
    else:
        ruined = np.zeros(cfg.iterations, dtype=bool)

    lower_percentile = (1.0 - confidence_level) / 2.0 * 100.0
    upper_percentile = 100.0 - lower_percentile
    lower_path, median_path, upper_path = np.percentile(
        equity_paths,
        [lower_percentile, 50.0, upper_percentile],
        axis=0,
    )

    return MonteCarloResult(
        config=MonteCarloConfig(
            iterations=cfg.iterations,
            method=method,
            sample_size=sample_size,
            confidence_level=confidence_level,
            ruin_threshold_pct=cfg.ruin_threshold_pct,
            random_seed=cfg.random_seed,
        ),
        initial_capital=initial_capital,
        trades_per_path=sample_size,
        final_equities=equity_paths[:, -1],
        max_drawdowns_pct=max_drawdowns_pct,
        ruined=ruined,
        lower_path=lower_path,
        median_path=median_path,
        upper_path=upper_path,
    )


def _normalize_mode(mode: MonteCarloMode | str) -> MonteCarloMode:
    if isinstance(mode, MonteCarloMode):
        return mode
    return MonteCarloMode(str(mode).lower())


def _normalize_confidence_level(confidence_level: float) -> float:
    level = float(confidence_level)
    if 1.0 < level <= 100.0:
        level /= 100.0
    if level <= 0.0 or level >= 1.0:
        raise ValueError("confidence_level must be in (0, 1) or expressed as a percentage in (0, 100]")
    return level
