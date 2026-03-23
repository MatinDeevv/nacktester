"""
Aphelion Lab — GUI Application
PySide6 desktop app with dark theme, charts, metrics, and hot reload.
"""

import sys
import os
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock
import multiprocessing as mp

import numpy as np
import pandas as pd

# Try to import Numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Auto-detect optimal worker count
import psutil
CPU_COUNT = psutil.cpu_count(logical=False) or mp.cpu_count()  # Physical cores
OPTIMAL_WORKERS = max(4, min(CPU_COUNT - 1, 10))  # Leave 1 core free, max 10
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QToolBar, QComboBox, QPushButton, QLabel, QTextEdit, QFileDialog,
    QProgressBar, QStatusBar, QDockWidget, QGroupBox, QGridLayout,
    QDateEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QMessageBox,
    QFrame, QScrollArea, QSizePolicy, QCheckBox, QDialog, QDialogButtonBox,
    QRadioButton,
)
from PySide6.QtCore import Qt, QThread, Signal, QDate, QTimer, QSize
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QAction

from .data_manager import DataManager
from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from .strategy_runtime import StrategyLoader
from .execution import (
    SlippageMode, CommissionMode, TrailingStopMode, SizingMode,
)
from .metrics import compute_leaderboard_score

logger = logging.getLogger("aphelion.gui")

# Throttle noisy GUI signal emissions during large queue runs.
QUEUE_PROGRESS_EMIT_EVERY = 2
DOWNLOAD_LOG_EVERY_PCT = 5
REPLAY_DRAW_INTERVAL_SEC = 1.0 / 30.0

# Too-small batches often produce zero-trade runs because indicators need warmup.
QUEUE_STRATEGY_CONCURRENCY = 1

MIN_BATCH_BARS_BY_TF = {
    "M1": 1200,
    "M5": 600,
    "M15": 400,
    "M30": 300,
    "H1": 200,
    "H4": 120,
    "D1": 80,
}
DEFAULT_MIN_BATCH_BARS = 300


def recommended_batch_bars(tf: str) -> int:
    return int(MIN_BATCH_BARS_BY_TF.get((tf or "").upper(), DEFAULT_MIN_BATCH_BARS))

# ─── Performance Optimizations ───────────────────────────────────────────────

class PerformanceInfo:
    """System performance information and GPU detection."""
    
    @staticmethod
    def get_system_info():
        """Return system performance info for logging."""
        gpu_info = "Not available"
        if HAS_CUPY:
            try:
                mem = cp.cuda.MemoryPool().get_limit() / 1e9
                gpu_info = f"CUDA available ({mem:.1f}GB VRAM)"
            except:
                gpu_info = "CUDA not fully initialized"
        
        return {
            "cpu_cores": CPU_COUNT,
            "optimal_workers": OPTIMAL_WORKERS,
            "queue_concurrency": QUEUE_STRATEGY_CONCURRENCY,
            "has_numba": HAS_NUMBA,
            "has_cupy": HAS_CUPY,
            "gpu_info": gpu_info,
            "total_ram_gb": psutil.virtual_memory().total / 1e9,
        }

@jit(nopython=True)
def numba_calculate_returns(equity_curve):
    """Fast return calculation using Numba JIT."""
    if len(equity_curve) == 0:
        return 0.0
    initial = equity_curve[0]
    final = equity_curve[-1]
    if initial == 0:
        return 0.0
    return ((final - initial) / initial) * 100.0

@jit(nopython=True)
def numba_calculate_max_drawdown(equity_curve):
    """Fast max drawdown calculation using Numba JIT."""
    if len(equity_curve) == 0:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / max(peak, 1.0)
        if drawdown > max_dd:
            max_dd = drawdown
    
    return max_dd * 100.0

@jit(nopython=True)
def numba_calculate_sharpe(returns, risk_free_rate=0.02):
    """Fast Sharpe ratio calculation using Numba JIT."""
    if len(returns) == 0:
        return 0.0
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    if std_ret == 0:
        return 0.0
    
    # Annualized Sharpe (assuming daily data)
    annual_factor = np.sqrt(252.0)
    sharpe = ((mean_ret - (risk_free_rate / 252.0)) / std_ret) * annual_factor
    return sharpe

logger = logging.getLogger("aphelion.gui")

# ─── Colors ──────────────────────────────────────────────────────────────────

C = {
    "bg":       "#0f131a",
    "panel":    "#131722",
    "border":   "#2a2e39",
    "text":     "#d1d4dc",
    "dim":      "#787b86",
    "green":    "#26a69a",
    "red":      "#ef5350",
    "amber":    "#f5a623",
    "blue":     "#2962ff",
    "header":   "#1f2430",
    "row_alt":  "#171b26",
    "white":    "#ffffff",
}

DARK_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C['bg']};
    color: {C['text']};
    font-family: 'Segoe UI', 'Tahoma';
    font-size: 12px;
}}
QToolBar {{
    background: {C['panel']};
    border-bottom: 1px solid {C['border']};
    padding: 4px;
    spacing: 6px;
}}
QComboBox, QDateEdit, QSpinBox, QDoubleSpinBox {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    border-radius: 3px;
    padding: 4px 8px;
    color: {C['text']};
    min-width: 80px;
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background: {C['panel']};
    color: {C['text']};
    selection-background-color: {C['blue']};
}}
QPushButton {{
    background: {C['border']};
    border: 1px solid {C['dim']};
    border-radius: 3px;
    padding: 5px 14px;
    color: {C['text']};
    font-weight: bold;
}}
QPushButton:hover {{ background: {C['blue']}; color: white; }}
QPushButton:pressed {{ background: #1565c0; }}
QPushButton#run_btn {{ background: #1b5e20; border-color: {C['green']}; }}
QPushButton#run_btn:hover {{ background: {C['green']}; color: black; }}
QPushButton#refresh_btn {{ background: #e65100; border-color: {C['amber']}; }}
QPushButton#refresh_btn:hover {{ background: {C['amber']}; color: black; }}
QPushButton#download_btn {{ background: #0d47a1; border-color: {C['blue']}; }}
QPushButton#download_btn:hover {{ background: {C['blue']}; color: white; }}
QPushButton#queue_btn {{ background: #263238; border-color: {C['dim']}; }}
QPushButton#queue_btn:hover {{ background: {C['dim']}; color: white; }}
QPushButton#queue_run_btn {{ background: #004d40; border-color: {C['green']}; }}
QPushButton#queue_run_btn:hover {{ background: {C['green']}; color: black; }}
QPushButton#queue_settings_btn {{ background: #1a237e; border-color: #5c6bc0; }}
QPushButton#queue_settings_btn:hover {{ background: #3949ab; color: white; }}
QPushButton#tf_queue_run_btn {{ background: #006064; border-color: #00bcd4; }}
QPushButton#tf_queue_run_btn:hover {{ background: #00bcd4; color: black; }}
QTableWidget {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    gridline-color: {C['border']};
    color: {C['text']};
    font-family: 'Segoe UI', 'Consolas';
    font-size: 11px;
}}
QTableWidget::item {{ padding: 3px 6px; }}
QTableWidget::item:selected {{ background: {C['blue']}; }}
QHeaderView::section {{
    background: {C['panel']};
    color: {C['dim']};
    border: 1px solid {C['border']};
    padding: 4px;
    font-weight: bold;
    font-size: 10px;
}}
QTextEdit, QPlainTextEdit {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    color: {C['text']};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}}
QTabWidget::pane {{ border: 1px solid {C['border']}; }}
QTabBar::tab {{
    background: {C['bg']};
    color: {C['dim']};
    border: 1px solid {C['border']};
    padding: 6px 16px;
    font-weight: bold;
}}
QTabBar::tab:selected {{ background: {C['panel']}; color: {C['text']}; border-bottom: 2px solid {C['blue']}; }}
QProgressBar {{
    background: {C['bg']};
    border: 1px solid {C['border']};
    border-radius: 3px;
    text-align: center;
    color: {C['text']};
}}
QProgressBar::chunk {{ background: {C['blue']}; border-radius: 2px; }}
QLabel {{ color: {C['text']}; }}
QLabel#title {{ font-size: 14px; font-weight: bold; }}
QLabel#metric_value {{ font-family: 'Consolas'; font-size: 13px; font-weight: bold; }}
QLabel#metric_label {{ color: {C['dim']}; font-size: 10px; }}
QSplitter::handle {{ background: {C['border']}; }}
QStatusBar {{ background: {C['panel']}; color: {C['dim']}; border-top: 1px solid {C['border']}; }}
QScrollArea {{ border: none; }}
QGroupBox {{
    color: {C['text']};
    border: 1px solid {C['border']};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 12px;
    font-weight: bold;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
"""


# ─── Download Worker Thread ──────────────────────────────────────────────────

class DownloadWorker(QThread):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, dm: DataManager, symbols, timeframes):
        super().__init__()
        self.dm = dm
        self.symbols = symbols
        self.timeframes = timeframes

    def run(self):
        try:
            results = self.dm.download_all(
                symbols=self.symbols,
                timeframes=self.timeframes,
                workers=4,
                on_progress=lambda done, total, msg: self.progress.emit(done, total, msg),
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class BacktestWorker(QThread):
    """Run backtest in background thread with chart replay support"""
    progress = Signal(str)
    bar_update = Signal(int, int, object)  # bar_idx, total_bars, trades
    finished = Signal(object)  # BacktestResult
    error = Signal(str)

    def __init__(self, engine, data, strategy, replay_interval=50, replay_delay_ms=0):
        super().__init__()
        self.engine = engine
        self.data = data
        self.strategy = strategy
        self.replay_interval = replay_interval
        self.replay_delay_ms = replay_delay_ms

    def run(self):
        try:
            self.progress.emit("Running backtest...")

            def _on_progress(bar_idx, total, trades):
                self.bar_update.emit(bar_idx, total, trades)
                if self.replay_delay_ms > 0:
                    time.sleep(self.replay_delay_ms / 1000.0)

            result = self.engine.run(
                self.data, self.strategy,
                on_progress=_on_progress,
                progress_interval=self.replay_interval,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class StrategyQueueWorker(QThread):
    """Run multiple strategy files sequentially and score them on batch slices."""

    progress = Signal(int, int, str, int)  # done_steps, total_steps, message, run_number
    strategy_done = Signal(object)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, data: pd.DataFrame, strategy_paths: list[str], initial_capital: float,
                 batch_size: int, symbol: str, timeframe: str = ""):
        super().__init__()
        self.data = data
        self.strategy_paths = strategy_paths
        self.initial_capital = float(initial_capital)
        self.batch_size = int(batch_size)
        self.symbol = symbol
        self.timeframe = timeframe
        self.last_leaderboard = []
        self.last_error = None

    def _build_config(self) -> BacktestConfig:
        config = BacktestConfig(initial_capital=self.initial_capital)
        if "JPY" in self.symbol:
            config.pip_value = 0.01
        elif "XAU" in self.symbol or "GOLD" in self.symbol:
            config.pip_value = 0.01
            config.lot_multiplier = 100
        else:
            config.pip_value = 0.0001
            config.lot_multiplier = 100000
        return config

    @staticmethod
    def _score(total_return_pct: float, sharpe: float, win_rate: float, max_dd: float, total_trades: int) -> float:
        if total_trades == 0:
            return -1e9
        return compute_leaderboard_score(
            sharpe=sharpe,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_dd,
            win_rate_pct=win_rate,
        )

    def _run_strategy_worker(self, run_num, path, shared_state):
        """Process a single strategy in a thread. Updates shared_state with results."""
        loader = StrategyLoader()
        strategy = loader.load(path)
        strategy_name = loader.strategy_name or Path(path).stem
        effective_batch_size = max(int(self.batch_size), recommended_batch_bars(self.timeframe))
        min_batch = max(80, effective_batch_size // 3)
        engine = BacktestEngine(self._build_config())
        
        # Handle load error
        if strategy is None:
            row = {
                "strategy": strategy_name,
                "path": path,
                "tfs": self.timeframe,
                "best_tf": self.timeframe,
                "batches": 0, "trades": 0,
                "net_pnl": 0.0, "return_pct": 0.0,
                "profit_factor": 0.0, "avg_trade": 0.0,
                "win_rate": 0.0, "sharpe": 0.0,
                "calmar": 0.0, "max_dd": 0.0,
                "score": -1e9, "status": "load_error", "tf_breakdown": {},
            }
            self.strategy_done.emit(row)
            with shared_state["lock"]:
                shared_state["leaderboard"].append(row)
            return

        total_pnl = 0.0
        total_trades = 0
        wins = 0
        gross_profit = 0.0
        gross_loss = 0.0
        sharpe_vals = []
        dd_vals = []
        batches_run = 0

        for i in range(0, len(self.data), effective_batch_size):
            chunk = self.data.iloc[i:i + effective_batch_size]
            
            with shared_state["lock"]:
                shared_state["done_steps"] += 1
                done_steps = shared_state["done_steps"]
                total_steps = shared_state["total_steps"]
            
            if len(chunk) < min_batch:
                self.progress.emit(done_steps, total_steps, f"{strategy_name}: skipped tiny tail batch", run_num)
                continue

            result = engine.run(chunk, strategy)

            batches_run += 1
            total_pnl += float(result.net_pnl)
            total_trades += int(result.total_trades)
            wins += sum(1 for t in result.trades if t.pnl > 0)
            for _t in result.trades:
                if _t.pnl > 0:
                    gross_profit += _t.pnl
                else:
                    gross_loss += abs(_t.pnl)
            sharpe_vals.append(float(result.sharpe_ratio))
            dd_vals.append(abs(float(result.max_drawdown)))

            should_emit = (
                batches_run == 1
                or batches_run % QUEUE_PROGRESS_EMIT_EVERY == 0
                or i + effective_batch_size >= len(self.data)
            )
            if should_emit:
                warmup_note = ""
                if batches_run == 1 and effective_batch_size > self.batch_size:
                    warmup_note = f" | auto_batch={effective_batch_size}"
                self.progress.emit(
                    done_steps,
                    total_steps,
                    f"{strategy_name}: batch {batches_run} | pnl ${result.net_pnl:.2f}{warmup_note}",
                    run_num,
                )

        # Calculate final metrics
        win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
        return_pct = (total_pnl / self.initial_capital * 100.0) if self.initial_capital else 0.0
        sharpe = float(np.mean(sharpe_vals)) if sharpe_vals else 0.0
        max_dd = float(np.max(dd_vals)) if dd_vals else 0.0
        profit_factor = gross_profit / max(gross_loss, 0.01)
        avg_trade = total_pnl / max(total_trades, 1)
        calmar = return_pct / max(max_dd, 0.1) if total_trades else 0.0
        score = self._score(return_pct, sharpe, win_rate, max_dd, total_trades)

        row = {
            "strategy": strategy_name,
            "path": path,
            "tfs": self.timeframe,
            "best_tf": self.timeframe,
            "batches": batches_run,
            "trades": total_trades,
            "net_pnl": total_pnl,
            "return_pct": return_pct,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "calmar": calmar,
            "max_dd": max_dd,
            "score": score,
            "status": "ok",
            "tf_breakdown": {},
        }
        self.strategy_done.emit(row)
        with shared_state["lock"]:
            shared_state["leaderboard"].append(row)

    def run(self):
        try:
            if self.data is None or len(self.data) == 0:
                raise ValueError("No data available for queue run")
            if not self.strategy_paths:
                raise ValueError("No strategies queued")

            effective_batch_size = max(int(self.batch_size), recommended_batch_bars(self.timeframe))
            batches_per_strategy = max(1, (len(self.data) + effective_batch_size - 1) // effective_batch_size)
            total_steps = len(self.strategy_paths) * batches_per_strategy
            
            # Shared state for thread coordination
            shared_state = {
                "done_steps": 0,
                "total_steps": total_steps,
                "leaderboard": [],
                "lock": Lock(),
            }

            # Strict sequential mode: finish each strategy before starting the next.
            for run_num, path in enumerate(self.strategy_paths, start=1):
                self._run_strategy_worker(run_num, path, shared_state)

            leaderboard = shared_state["leaderboard"]
            leaderboard.sort(key=lambda x: x["score"], reverse=True)
            self.last_leaderboard = leaderboard
            self.finished.emit({"leaderboard": leaderboard})
        except Exception as e:
            self.last_error = f"{e}\n{traceback.format_exc()}"
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class MultiTFQueueWorker(QThread):
    """Run all queued strategies across multiple timeframes and score globally."""

    progress = Signal(int, int, str, int)  # done_steps, total_steps, message, run_number
    strategy_done = Signal(object)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, dm, symbol: str, timeframes: list, bars_per_tf: int,
                 num_batches: int, strategy_paths: list, initial_capital: float,
                 bars_per_batch: int = None):
        super().__init__()
        self.dm = dm
        self.symbol = symbol
        self.timeframes = timeframes
        self.bars_per_tf = bars_per_tf
        self.num_batches = num_batches
        self.strategy_paths = strategy_paths
        self.initial_capital = float(initial_capital)
        self.bars_per_batch = bars_per_batch

    def _build_config(self) -> BacktestConfig:
        config = BacktestConfig(initial_capital=self.initial_capital)
        if "JPY" in self.symbol:
            config.pip_value = 0.01
        elif "XAU" in self.symbol or "GOLD" in self.symbol:
            config.pip_value = 0.01
            config.lot_multiplier = 100
        else:
            config.pip_value = 0.0001
            config.lot_multiplier = 100000
        return config

    @staticmethod
    def _score(return_pct, sharpe, win_rate, max_dd, trades):
        if trades == 0:
            return -1e9
        return compute_leaderboard_score(
            sharpe=sharpe,
            total_return_pct=return_pct,
            max_drawdown_pct=max_dd,
            win_rate_pct=win_rate,
        )

    def _effective_batch_size(self, tf: str, data_len: int) -> int:
        if self.bars_per_batch and self.bars_per_batch > 0:
            base = int(self.bars_per_batch)
        else:
            num_batches = max(1, int(self.num_batches or 1))
            base = max(100, int(data_len) // num_batches)
        return max(base, recommended_batch_bars(tf))

    def _run_tf_strategy_worker(self, run_num, path, tf_data, shared_state):
        """Process a single strategy across all timeframes in a thread."""
        loader = StrategyLoader()
        strategy = loader.load(path)
        strategy_name = loader.strategy_name or Path(path).stem
        engine = BacktestEngine(self._build_config())
        
        # Handle load error
        if strategy is None:
            row = {
                "strategy": strategy_name, "path": path,
                "tfs": ",".join(tf_data.keys()), "best_tf": "-",
                "batches": 0, "trades": 0,
                "net_pnl": 0.0, "return_pct": 0.0,
                "profit_factor": 0.0, "avg_trade": 0.0,
                "win_rate": 0.0, "sharpe": 0.0,
                "calmar": 0.0, "max_dd": 0.0,
                "score": -1e9, "status": "load_error", "tf_breakdown": {},
            }
            self.strategy_done.emit(row)
            with shared_state["lock"]:
                shared_state["leaderboard"].append(row)
            return

        agg_pnl = 0.0
        agg_trades = 0
        agg_wins = 0
        agg_gross_profit = 0.0
        agg_gross_loss = 0.0
        agg_sharpe_vals = []
        agg_dd_vals = []
        agg_batches = 0
        tf_breakdown = {}
        tf_scores = {}

        for tf, data in tf_data.items():
            batch_size = self._effective_batch_size(tf, len(data))
            min_batch = max(80, min(len(data), batch_size // 3))
            tf_pnl = 0.0
            tf_trades = 0
            tf_wins = 0
            tf_gp = 0.0
            tf_gl = 0.0
            tf_sharpe_vals = []
            tf_dd_vals = []
            tf_batches = 0

            for i in range(0, len(data), batch_size):
                chunk = data.iloc[i:i + batch_size]
                
                with shared_state["lock"]:
                    shared_state["done_steps"] += 1
                    done_steps = shared_state["done_steps"]
                    total_steps = shared_state["total_steps"]
                
                if len(chunk) < min_batch:
                    self.progress.emit(done_steps, total_steps,
                                       f"{strategy_name}/{tf}: skip tail", run_num)
                    continue

                result = engine.run(chunk, strategy)
                tf_batches += 1
                tf_pnl += float(result.net_pnl)
                tf_trades += int(result.total_trades)
                tf_wins += sum(1 for t in result.trades if t.pnl > 0)
                for _t in result.trades:
                    if _t.pnl > 0:
                        tf_gp += _t.pnl
                    else:
                        tf_gl += abs(_t.pnl)
                tf_sharpe_vals.append(float(result.sharpe_ratio))
                tf_dd_vals.append(abs(float(result.max_drawdown)))
                should_emit = (
                    tf_batches == 1
                    or tf_batches % QUEUE_PROGRESS_EMIT_EVERY == 0
                    or i + batch_size >= len(data)
                )
                if should_emit:
                    warmup_note = ""
                    if tf_batches == 1:
                        req = int(self.bars_per_batch) if (self.bars_per_batch and self.bars_per_batch > 0) else max(100, len(data) // max(1, int(self.num_batches or 1)))
                        if batch_size > req:
                            warmup_note = f" | auto_batch={batch_size}"
                    self.progress.emit(
                        done_steps,
                        total_steps,
                        f"{strategy_name} | {tf} batch {tf_batches} | ${result.net_pnl:.2f}{warmup_note}",
                        run_num,
                    )

            # If chunked mode produced zero trades, retry once on full-window data.
            # This prevents false-negative "$0 everywhere" outcomes on strict strategies.
            if tf_trades == 0 and len(data) >= recommended_batch_bars(tf):
                fallback_result = engine.run(data, strategy)
                if int(fallback_result.total_trades) > 0:
                    tf_batches = max(tf_batches, 1)
                    tf_pnl = float(fallback_result.net_pnl)
                    tf_trades = int(fallback_result.total_trades)
                    tf_wins = sum(1 for t in fallback_result.trades if t.pnl > 0)
                    tf_gp = sum(float(t.pnl) for t in fallback_result.trades if t.pnl > 0)
                    tf_gl = sum(abs(float(t.pnl)) for t in fallback_result.trades if t.pnl <= 0)
                    tf_sharpe_vals = [float(fallback_result.sharpe_ratio)]
                    tf_dd_vals = [abs(float(fallback_result.max_drawdown))]
                    self.progress.emit(
                        shared_state["done_steps"],
                        shared_state["total_steps"],
                        f"{strategy_name} | {tf} fallback full-window | trades {tf_trades} | ${tf_pnl:.2f}",
                        run_num,
                    )

            tf_wr = tf_wins / tf_trades * 100.0 if tf_trades else 0.0
            tf_ret = tf_pnl / self.initial_capital * 100.0
            tf_sh = float(np.mean(tf_sharpe_vals)) if tf_sharpe_vals else 0.0
            tf_dd = float(np.max(tf_dd_vals)) if tf_dd_vals else 0.0
            tf_pf = tf_gp / max(tf_gl, 0.01)
            tf_score = self._score(tf_ret, tf_sh, tf_wr, tf_dd, tf_trades)
            tf_breakdown[tf] = {
                "batches": tf_batches, "trades": tf_trades,
                "pnl": tf_pnl, "return_pct": tf_ret,
                "win_rate": tf_wr, "sharpe": tf_sh,
                "max_dd": tf_dd, "profit_factor": tf_pf, "score": tf_score,
            }
            tf_scores[tf] = tf_score
            agg_pnl += tf_pnl
            agg_trades += tf_trades
            agg_wins += tf_wins
            agg_gross_profit += tf_gp
            agg_gross_loss += tf_gl
            agg_sharpe_vals.extend(tf_sharpe_vals)
            agg_dd_vals.extend(tf_dd_vals)
            agg_batches += tf_batches

        win_rate = agg_wins / agg_trades * 100.0 if agg_trades else 0.0
        return_pct = agg_pnl / self.initial_capital * 100.0
        sharpe = float(np.mean(agg_sharpe_vals)) if agg_sharpe_vals else 0.0
        max_dd = float(np.max(agg_dd_vals)) if agg_dd_vals else 0.0
        profit_factor = agg_gross_profit / max(agg_gross_loss, 0.01)
        avg_trade = agg_pnl / max(agg_trades, 1)
        calmar = return_pct / max(max_dd, 0.1) if agg_trades else 0.0
        score = self._score(return_pct, sharpe, win_rate, max_dd, agg_trades)
        best_tf = max(tf_scores, key=tf_scores.get) if tf_scores else "-"

        row = {
            "strategy": strategy_name, "path": path,
            "tfs": ",".join(tf_data.keys()), "best_tf": best_tf,
            "batches": agg_batches, "trades": agg_trades,
            "net_pnl": agg_pnl, "return_pct": return_pct,
            "profit_factor": profit_factor, "avg_trade": avg_trade,
            "win_rate": win_rate, "sharpe": sharpe,
            "calmar": calmar, "max_dd": max_dd,
            "score": score, "status": "ok" if agg_trades > 0 else "no_trades", "tf_breakdown": tf_breakdown,
        }
        self.strategy_done.emit(row)
        with shared_state["lock"]:
            shared_state["leaderboard"].append(row)

    def run(self):
        try:
            if not self.strategy_paths:
                raise ValueError("No strategies queued")

            # Load and prepare data for each TF
            tf_data = {}
            for tf in self.timeframes:
                df = self.dm.load(self.symbol, tf)
                if df is not None and len(df) > 0:
                    df = df.ffill().dropna(subset=["open", "high", "low", "close"])
                    if len(df) > self.bars_per_tf:
                        df = df.iloc[-self.bars_per_tf:]
                    tf_data[tf] = df

            if not tf_data:
                raise ValueError(
                    f"No cached data for {self.symbol} on any selected TF. "
                    f"Download first (selected: {', '.join(self.timeframes)})"
                )

            steps_per_tf = []
            for tf, data in tf_data.items():
                batch_size = self._effective_batch_size(tf, len(data))
                steps_per_tf.append(max(1, (len(data) + batch_size - 1) // batch_size))
            total_steps = len(self.strategy_paths) * sum(steps_per_tf)
            
            # Shared state for thread coordination
            shared_state = {
                "done_steps": 0,
                "total_steps": total_steps,
                "leaderboard": [],
                "lock": Lock(),
            }

            # Strict sequential mode: finish each strategy before starting the next.
            for run_num, path in enumerate(self.strategy_paths, start=1):
                self._run_tf_strategy_worker(run_num, path, tf_data, shared_state)

            leaderboard = shared_state["leaderboard"]
            leaderboard.sort(key=lambda x: x["score"], reverse=True)
            self.finished.emit({"leaderboard": leaderboard})
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class QueueSettingsDialog(QDialog):
    """Configure and launch a multi-timeframe strategy sweep."""

    def __init__(self, cached_symbols, parent=None):
        super().__init__(parent)
        self.setWindowTitle("\u2699 Queue Settings \u2014 Multi-TF Sweep")
        self.setMinimumWidth(460)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Symbol row
        sym_row = QHBoxLayout()
        sym_row.addWidget(QLabel("Symbol:"))
        self.sym_combo = QComboBox()
        syms = list(cached_symbols) if cached_symbols else ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
        self.sym_combo.addItems(syms)
        sym_row.addWidget(self.sym_combo, 1)
        layout.addLayout(sym_row)

        # Timeframes group
        tf_group = QGroupBox("Timeframes to Test")
        tf_lay = QHBoxLayout(tf_group)
        self._tf_checks = {}
        defaults = {"H1", "H4"}
        for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
            cb = QCheckBox(tf)
            cb.setChecked(tf in defaults)
            tf_lay.addWidget(cb)
            self._tf_checks[tf] = cb
        layout.addWidget(tf_group)

        # Settings grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(7)

        # Batch mode selector
        grid.addWidget(QLabel("Batch Mode:"), 0, 0)
        mode_row = QHBoxLayout()
        self.batch_mode_num = QRadioButton("Number of Batches")
        self.batch_mode_size = QRadioButton("Bars per Batch")
        self.batch_mode_num.setChecked(True)
        mode_row.addWidget(self.batch_mode_num)
        mode_row.addWidget(self.batch_mode_size)
        mode_row.addStretch()
        grid.addLayout(mode_row, 0, 1)

        # Batches spin (shown when mode = num)
        grid.addWidget(QLabel("Batches per TF:"), 1, 0)
        self.batches_spin = QSpinBox()
        self.batches_spin.setRange(1, 500)
        self.batches_spin.setValue(50)
        self.batches_spin.setToolTip("Number of evaluation windows.\n50 batches = thorough walk-forward test.")
        grid.addWidget(self.batches_spin, 1, 1)

        # Bars per Batch spin (shown when mode = size)
        grid.addWidget(QLabel("Bars/Batch:"), 2, 0)
        self.bars_per_batch_spin = QSpinBox()
        self.bars_per_batch_spin.setRange(50, 100000)
        self.bars_per_batch_spin.setValue(280)
        self.bars_per_batch_spin.setToolTip("Exact bars per batch window.\nE.g., 280 bars on M5 = ~1 trading day")
        self.bars_per_batch_spin.setVisible(False)
        grid.addWidget(self.bars_per_batch_spin, 2, 1)

        # Connect radio buttons to toggle spinbox visibility
        self.batch_mode_num.toggled.connect(lambda checked: self._sync_batch_mode_ui())

        grid.addWidget(QLabel("Bars per TF:"), 3, 0)
        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(500, 500000)
        self.bars_spin.setValue(5000)
        self.bars_spin.setSingleStep(1000)
        self.bars_spin.setToolTip("Maximum bars of data loaded for each timeframe.")
        grid.addWidget(self.bars_spin, 3, 1)

        grid.addWidget(QLabel("Capital per Run:"), 4, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 10_000_000)
        self.capital_spin.setValue(10000)
        self.capital_spin.setPrefix("$")
        grid.addWidget(self.capital_spin, 4, 1)
        layout.addLayout(grid)

        # Info label
        info = QLabel("Results are aggregated across all TFs. Best TF column shows\nwhere each strategy performed best.")
        info.setStyleSheet(f"color: {C['dim']}; font-size: 10px;")
        layout.addWidget(info)

        # OK / Cancel
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("\u2714 Apply & Run")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _sync_batch_mode_ui(self):
        """Show/hide batch mode controls based on selection."""
        is_num_mode = self.batch_mode_num.isChecked()
        self.batches_spin.setVisible(is_num_mode)
        self.bars_per_batch_spin.setVisible(not is_num_mode)

    def get_settings(self) -> dict:
        return {
            "symbol": self.sym_combo.currentText(),
            "timeframes": [tf for tf, cb in self._tf_checks.items() if cb.isChecked()],
            "batch_mode": "num_batches" if self.batch_mode_num.isChecked() else "bars_per_batch",
            "batches_per_tf": self.batches_spin.value(),
            "bars_per_batch": self.bars_per_batch_spin.value(),
            "bars_per_tf": self.bars_spin.value(),
            "capital": self.capital_spin.value(),
        }


# ─── Chart Widget ────────────────────────────────────────────────────────────

class CandlestickChart(FigureCanvasQTAgg):
    """Matplotlib candlestick chart with trade markers and live replay."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 5), facecolor=C["panel"])
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axis(self.ax)
        self.fig.tight_layout(pad=1)

        # Interactive toolbar for zoom/pan
        self.toolbar = NavigationToolbar2QT(self, parent)
        self.toolbar.setStyleSheet(f"background-color: {C['panel']}; color: {C['text']};")

        # Replay state
        self._all_dates = None
        self._playhead_line = None
        self._plotted_trade_ids = set()
        self._replay_active = False
        self._last_replay_draw_ts = 0.0

    def get_toolbar(self):
        """Return toolbar widget for layout"""
        return self.toolbar

    def _style_axis(self, ax):
        ax.set_facecolor(C["bg"])
        ax.tick_params(colors=C["dim"], labelsize=8)
        ax.tick_params(axis="y", which="both", right=True, left=False, labelright=True, labelleft=False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_color(C["border"])
        ax.spines["bottom"].set_color(C["border"])
        ax.grid(True, which="major", color=C["border"], alpha=0.45, linewidth=0.55)
        ax.grid(True, which="minor", color=C["border"], alpha=0.18, linewidth=0.4)
        ax.minorticks_on()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d %b"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))

    def plot(self, data: pd.DataFrame, trades: list = None, max_bars: int = 500,
             replay_mode: bool = False):
        self.ax.clear()
        self._style_axis(self.ax)
        self._replay_active = False
        self._all_dates = None
        self._plotted_trade_ids = set()
        self._playhead_line = None
        self._last_replay_draw_ts = 0.0

        if data is None or len(data) == 0:
            self.draw()
            return

        # For replay mode, use all data (capped at 5000); otherwise limit to max_bars
        if replay_mode:
            df = data.iloc[-min(5000, len(data)):]
        else:
            df = data.iloc[-max_bars:] if len(data) > max_bars else data

        dates = mdates.date2num(df.index.to_pydatetime())
        self._all_dates = dates
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Draw candles using TradingView-inspired filled bodies and right axis.
        width = 0.6 * (dates[1] - dates[0]) if len(dates) > 1 else 0.001
        for i in range(len(dates)):
            color = C["green"] if closes[i] >= opens[i] else C["red"]
            self.ax.plot([dates[i], dates[i]], [lows[i], highs[i]],
                color=color, linewidth=0.9, alpha=0.95)
            body_low = min(opens[i], closes[i])
            body_high = max(opens[i], closes[i])
            body_h = max(body_high - body_low, (highs[i] - lows[i]) * 0.01)
            rect = plt.Rectangle((dates[i] - width/2, body_low), width, body_h,
                    facecolor=color,
                    edgecolor=color, linewidth=0.8, alpha=0.85)
            self.ax.add_patch(rect)

        if replay_mode:
            # Start zoomed into the beginning with playhead
            window = min(200, len(dates))
            self.ax.set_xlim(dates[0], dates[min(window - 1, len(dates) - 1)])
            self._playhead_line = self.ax.axvline(
                x=dates[0], color=C["amber"], linewidth=1.5, alpha=0.8, linestyle="--")
            self._replay_active = True
        else:
            # Draw trade markers (normal mode)
            if trades:
                start_date = df.index[0]
                end_date = df.index[-1]
                for t in trades:
                    if t.entry_time and start_date <= t.entry_time <= end_date:
                        ed = mdates.date2num(t.entry_time.to_pydatetime())
                        marker = "^" if t.side.value == "BUY" else "v"
                        color = C["green"] if t.side.value == "BUY" else C["red"]
                        self.ax.scatter(ed, t.entry_price, marker=marker, color=color,
                                      s=60, zorder=5, edgecolors="white", linewidths=0.5)

                    if t.exit_time and start_date <= t.exit_time <= end_date:
                        xd = mdates.date2num(t.exit_time.to_pydatetime())
                        color = C["green"] if t.pnl > 0 else C["red"]
                        self.ax.scatter(xd, t.exit_price, marker="x", color=color,
                                      s=40, zorder=5, linewidths=1.5)

        self.ax.set_ylabel("Price", color=C["dim"], fontsize=9)
        self.ax.autoscale_view()  # Ensure axis limits reflect all plotted artists
        self.fig.autofmt_xdate()
        self.fig.tight_layout(pad=1)
        self.draw()

    def update_replay(self, bar_idx, trades):
        """Move playhead and add trade markers during live replay."""
        if not self._replay_active or self._all_dates is None:
            return
        if len(self._all_dates) == 0:
            return

        now = time.monotonic()
        if now - self._last_replay_draw_ts < REPLAY_DRAW_INTERVAL_SEC:
            return
        self._last_replay_draw_ts = now

        idx = min(bar_idx, len(self._all_dates) - 1)
        date_val = self._all_dates[idx]

        # Move playhead
        if self._playhead_line:
            self._playhead_line.set_xdata([date_val, date_val])

        # Auto-scroll: keep playhead at ~75% of visible window
        window = min(200, len(self._all_dates))
        ideal_start = idx - int(window * 0.75)
        start = max(0, ideal_start)
        end = min(len(self._all_dates) - 1, start + window)
        if start < end:
            self.ax.set_xlim(self._all_dates[start], self._all_dates[end])

        # Add trade markers for newly completed trades
        for t in trades:
            if t.id in self._plotted_trade_ids:
                continue
            self._plotted_trade_ids.add(t.id)
            try:
                if t.entry_time:
                    ed = mdates.date2num(t.entry_time.to_pydatetime())
                    m = "^" if t.side.value == "BUY" else "v"
                    c = C["green"] if t.side.value == "BUY" else C["red"]
                    self.ax.scatter(ed, t.entry_price, marker=m, color=c,
                                  s=60, zorder=5, edgecolors="white", linewidths=0.5)
                if t.exit_time:
                    xd = mdates.date2num(t.exit_time.to_pydatetime())
                    c = C["green"] if t.pnl > 0 else C["red"]
                    self.ax.scatter(xd, t.exit_price, marker="x", color=c,
                                  s=40, zorder=5, linewidths=1.5)
            except Exception:
                pass

        self.draw_idle()

    def end_replay(self):
        """Clean up replay mode."""
        if self._playhead_line:
            try:
                self._playhead_line.remove()
            except Exception:
                pass
            self._playhead_line = None
        self._replay_active = False


class EquityCurveChart(FigureCanvasQTAgg):
    """Equity curve + drawdown chart."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 2.5), facecolor=C["panel"])
        super().__init__(self.fig)
        self.ax_eq = self.fig.add_subplot(211)
        self.ax_dd = self.fig.add_subplot(212, sharex=self.ax_eq)
        self._style(self.ax_eq)
        self._style(self.ax_dd)
        self.fig.tight_layout(pad=1)
        
        # Add interactive toolbar
        self.toolbar = NavigationToolbar2QT(self, parent)
        self.toolbar.setStyleSheet(f"background-color: {C['panel']}; color: {C['text']};")

    def get_toolbar(self):
        """Return toolbar widget for layout"""
        return self.toolbar

    def _style(self, ax):
        ax.set_facecolor(C["bg"])
        ax.tick_params(colors=C["dim"], labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", which="both", right=True, left=False, labelright=True, labelleft=False)
        ax.spines["right"].set_color(C["border"])
        ax.spines["bottom"].set_color(C["border"])
        ax.grid(True, color=C["border"], alpha=0.35, linewidth=0.5)

    def plot(self, result: BacktestResult):
        self.ax_eq.clear()
        self.ax_dd.clear()
        self._style(self.ax_eq)
        self._style(self.ax_dd)

        if not result or not result.equity_curve:
            self.draw()
            return

        # Convert timestamps to matplotlib dates
        dates = mdates.date2num([ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts 
                                 for ts in result.timestamps])
        eq = result.equity_curve
        dd = result.drawdown_curve

        # Equity
        color = C["green"] if eq[-1] >= eq[0] else C["red"]
        self.ax_eq.plot(dates, eq, color=color, linewidth=1.2)
        self.ax_eq.axhline(y=result.config.initial_capital, color=C["dim"],
                          linewidth=0.5, linestyle="--", alpha=0.5)
        self.ax_eq.fill_between(dates, result.config.initial_capital, eq,
                               where=[e >= result.config.initial_capital for e in eq],
                               alpha=0.1, color=C["green"])
        self.ax_eq.fill_between(dates, result.config.initial_capital, eq,
                               where=[e < result.config.initial_capital for e in eq],
                               alpha=0.1, color=C["red"])
        self.ax_eq.set_ylabel("Equity", color=C["dim"], fontsize=8)

        # Drawdown
        self.ax_dd.fill_between(dates, 0, dd, color=C["red"], alpha=0.3)
        self.ax_dd.plot(dates, dd, color=C["red"], linewidth=0.8)
        self.ax_dd.set_ylabel("DD %", color=C["dim"], fontsize=8)
        
        # Format dates on x-axis
        self.ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        self.ax_eq.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.fig.autofmt_xdate()

        self.fig.tight_layout(pad=1)
        self.draw()


# ─── Metrics Panel ───────────────────────────────────────────────────────────

class MetricsPanel(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.container = QWidget()
        self.layout = QGridLayout(self.container)
        self.layout.setSpacing(2)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.setWidget(self.container)
        self._labels = {}

    def update_metrics(self, result: BacktestResult):
        # Clear
        for w in self.container.findChildren(QWidget):
            w.deleteLater()
        self._labels = {}

        if not result: return
        metrics = result.to_metrics_dict()
        row = 0
        for key, val in metrics.items():
            lbl = QLabel(key)
            lbl.setStyleSheet(f"color: {C['dim']}; font-size: 10px;")
            val_lbl = QLabel(val)
            val_lbl.setAlignment(Qt.AlignRight)

            # Color code P&L values
            style = f"font-family: Consolas; font-size: 12px; font-weight: bold;"
            if "$" in val or "%" in val:
                try:
                    num = float(val.replace("$", "").replace("%", "").replace(",", ""))
                    if num > 0: style += f" color: {C['green']};"
                    elif num < 0: style += f" color: {C['red']};"
                    else: style += f" color: {C['text']};"
                except: style += f" color: {C['text']};"
            else:
                style += f" color: {C['text']};"
            val_lbl.setStyleSheet(style)

            self.layout.addWidget(lbl, row, 0)
            self.layout.addWidget(val_lbl, row, 1)
            self._labels[key] = val_lbl
            row += 1

        self.layout.setRowStretch(row, 1)


# ─── Trade List Panel ────────────────────────────────────────────────────────

class TradeListPanel(QTableWidget):
    def __init__(self):
        super().__init__()
        self.setColumnCount(0)
        self.horizontalHeader().setStretchLastSection(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setStyleSheet(f"""
            QTableWidget {{ alternate-background-color: {C['row_alt']}; }}
        """)

    def load_trades(self, result: BacktestResult):
        if not result or not result.trades:
            self.setRowCount(0)
            self.setColumnCount(0)
            return

        df = result.trades_df()
        self.setRowCount(len(df))
        self.setColumnCount(len(df.columns))
        self.setHorizontalHeaderLabels(list(df.columns))

        for r in range(len(df)):
            for c in range(len(df.columns)):
                val = str(df.iloc[r, c])
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)

                # Color P&L column
                col_name = df.columns[c]
                if col_name == "P&L":
                    try:
                        num = float(val)
                        item.setForeground(QColor(C["green"] if num > 0 else C["red"]))
                    except: pass
                elif col_name == "Side":
                    item.setForeground(QColor(C["green"] if val == "BUY" else C["red"]))

                self.setItem(r, c, item)

        self.resizeColumnsToContents()


# ─── Main Window ─────────────────────────────────────────────────────────────

class AphelionLab(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("APHELION LAB — Visual Backtesting")
        self.setMinimumSize(1400, 900)

        self._strategies_dir = Path(__file__).resolve().parent / "strategies"
        self.dm = DataManager()
        self.engine = BacktestEngine()
        self.loader = StrategyLoader()
        self.result: BacktestResult = None
        self._strategy_path = None
        self._strategy_queue: list[dict] = []
        self._queue_results: list[dict] = []
        self._queue_settings: dict = None
        self._tf_queue_worker = None
        self._last_progress_value = -1
        self._last_status_text = ""
        self._last_download_log_pct = -DOWNLOAD_LOG_EVERY_PCT

        self._setup_logging()
        self._build_toolbar()
        self._build_ui()
        self._build_statusbar()

        # Log system performance info
        perf_info = PerformanceInfo.get_system_info()
        self._log("="*60, "INFO")
        self._log(f"🚀 APHELION LAB — Performance Mode Enabled", "INFO")
        self._log(f"  CPU Cores: {perf_info['cpu_cores']} (queue concurrency: {perf_info['queue_concurrency']})", "INFO")
        self._log(f"  RAM: {perf_info['total_ram_gb']:.1f} GB available", "INFO")
        self._log(f"  Numba JIT: {'✓ Enabled' if perf_info['has_numba'] else '✗ Not installed'}", "INFO")
        self._log(f"  GPU Support: {perf_info['gpu_info']}", "INFO")
        self._log("="*60, "INFO")

        # Wire symbol/TF combos to auto-refresh chart from cache
        self.sym_combo.currentTextChanged.connect(self._try_load_chart)
        self.tf_combo.currentTextChanged.connect(self._try_load_chart)

        # Load example strategy
        ex = self._strategies_dir / "st_01_sma_crossover.py"
        if ex.exists():
            self._strategy_path = str(ex.resolve())
            self.loader.load(self._strategy_path)
            self._log(f"Loaded strategy: {self.loader.strategy_name}")
        else:
            self._log("No default strategy loaded", "WARN")

        # Auto-show chart if cache already has data
        self._try_load_chart()

    def _setup_logging(self):
        self._log_buffer = []
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.INFO, handlers=[handler])

    def _log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] [{level}] {msg}"
        self._log_buffer.append(entry)
        if hasattr(self, "log_panel"):
            self.log_panel.appendPlainText(entry)

    def _set_progress_value(self, value: int):
        """Avoid repaint churn by only updating progress when value changes."""
        value = int(max(0, min(100, value)))
        if value == self._last_progress_value:
            return
        self._last_progress_value = value
        self.progress.setValue(value)

    def _set_status_text(self, text: str):
        """Avoid redundant status bar paints from repeated identical messages."""
        if text == self._last_status_text:
            return
        self._last_status_text = text
        self.status_label.setText(text)

    # ─── Toolbar ─────────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("Controls")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        # Symbol
        tb.addWidget(QLabel(" Symbol: "))
        self.sym_combo = QComboBox()
        self.sym_combo.addItems(["XAUUSD","XAGUSD","EURUSD","GBPUSD","USDJPY","US500","BTCUSD"])
        self.sym_combo.setCurrentText("XAUUSD")
        tb.addWidget(self.sym_combo)

        # Timeframe
        tb.addWidget(QLabel("  TF: "))
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["M1","M5","M15","M30","H1","H4","D1"])
        self.tf_combo.setCurrentText("H1")
        tb.addWidget(self.tf_combo)
        
        # Download timeframes (multi-select hint)
        tb.addWidget(QLabel("  Download TF:"))
        self.download_tf_combo = QComboBox()
        self.download_tf_combo.addItems([
            "M1 only", "M1-M5", "M1-M15", "M1-M30", "M1-H1", 
            "M1-H4", "M1-D1 (ALL)", "H1-D1", "H4-D1"
        ])
        self.download_tf_combo.setCurrentText("H1-D1")
        tb.addWidget(self.download_tf_combo)

        # Max Bars
        tb.addWidget(QLabel("  Max Bars: "))
        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(100, 500000)
        self.bars_spin.setValue(5000)
        self.bars_spin.setSingleStep(1000)
        self.bars_spin.setToolTip("Maximum bars of data to use for backtest")
        tb.addWidget(self.bars_spin)

        tb.addWidget(QLabel("  Batches: "))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1000)
        self.batch_spin.setValue(10)
        self.batch_spin.setSingleStep(5)
        self.batch_spin.setToolTip("Number of evaluation chunks in queue mode. Batch size = total bars / batches.")
        tb.addWidget(self.batch_spin)

        # Capital
        tb.addWidget(QLabel("  Capital: "))
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 10000000)
        self.capital_spin.setValue(5000)
        self.capital_spin.setPrefix("$")
        tb.addWidget(self.capital_spin)

        tb.addSeparator()

        # Download
        dl_btn = QPushButton("⬇ Download Data")
        dl_btn.setObjectName("download_btn")
        dl_btn.clicked.connect(self._on_download)
        tb.addWidget(dl_btn)

        # Load Strategy
        load_btn = QPushButton("📂 Load Strategy")
        load_btn.clicked.connect(self._on_load_strategy)
        tb.addWidget(load_btn)

        queue_btn = QPushButton("🗂 Queue Strategies")
        queue_btn.setObjectName("queue_btn")
        queue_btn.clicked.connect(self._on_queue_strategies)
        tb.addWidget(queue_btn)

        # Run
        run_btn = QPushButton("▶ Run Backtest")
        run_btn.setObjectName("run_btn")
        run_btn.clicked.connect(self._on_run)
        tb.addWidget(run_btn)

        run_queue_btn = QPushButton("⚡ Run Queue")
        run_queue_btn.setObjectName("queue_run_btn")
        run_queue_btn.clicked.connect(self._on_run_queue)
        tb.addWidget(run_queue_btn)
        self.run_queue_btn = run_queue_btn

        qs_btn = QPushButton("⚙ Queue Settings")
        qs_btn.setObjectName("queue_settings_btn")
        qs_btn.clicked.connect(self._on_queue_settings)
        tb.addWidget(qs_btn)

        run_tf_btn = QPushButton("🌐 Multi-TF")
        run_tf_btn.setObjectName("tf_queue_run_btn")
        run_tf_btn.clicked.connect(self._on_run_tf_queue)
        tb.addWidget(run_tf_btn)
        self.run_tf_queue_btn = run_tf_btn

        self.turbo_chk = QCheckBox("Turbo")
        self.turbo_chk.setChecked(True)
        self.turbo_chk.setToolTip("Keep UI updates minimal during queue run for max throughput")
        tb.addWidget(self.turbo_chk)

        tb.addWidget(QLabel("  Replay: "))
        self.replay_speed = QComboBox()
        self.replay_speed.addItems(["Off", "Fast", "Medium", "Slow"])
        self.replay_speed.setCurrentText("Medium")
        self.replay_speed.setToolTip("Live chart replay speed during backtest")
        self.replay_speed.setFixedWidth(80)
        tb.addWidget(self.replay_speed)

        # Refresh (hot reload)
        ref_btn = QPushButton("🔄 Refresh")
        ref_btn.setObjectName("refresh_btn")
        ref_btn.clicked.connect(self._on_refresh)
        tb.addWidget(ref_btn)
        self.refresh_btn = ref_btn

        # Strategy label
        tb.addSeparator()
        self.strat_label = QLabel("  Strategy: None")
        self.strat_label.setStyleSheet(f"color: {C['amber']}; font-weight: bold;")
        tb.addWidget(self.strat_label)

    # ─── UI Layout ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Main splitter: chart area (left) + metrics (right)
        h_split = QSplitter(Qt.Horizontal)

        # Left side: charts stacked
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        # Candlestick chart with toolbar
        self.chart = CandlestickChart(self)
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)
        chart_layout.addWidget(self.chart.get_toolbar())
        chart_layout.addWidget(self.chart, stretch=1)
        chart_container.setMinimumHeight(250)

        # Equity curve with toolbar
        self.equity_chart = EquityCurveChart(self)
        equity_container = QWidget()
        equity_layout = QVBoxLayout(equity_container)
        equity_layout.setContentsMargins(0, 0, 0, 0)
        equity_layout.setSpacing(0)
        equity_layout.addWidget(self.equity_chart.get_toolbar())
        equity_layout.addWidget(self.equity_chart, stretch=1)
        equity_container.setMinimumHeight(90)

        # Vertical splitter so user can drag to resize chart vs equity panel
        v_split = QSplitter(Qt.Vertical)
        v_split.addWidget(chart_container)
        v_split.addWidget(equity_container)
        v_split.setSizes([420, 160])
        v_split.setHandleWidth(5)
        left_layout.addWidget(v_split)

        h_split.addWidget(left)

        # Right side: tabs for metrics, trades, logs
        right_tabs = QTabWidget()

        # Metrics tab
        self.metrics_panel = MetricsPanel()
        right_tabs.addTab(self.metrics_panel, "Metrics")

        # Trades tab
        self.trades_panel = TradeListPanel()
        right_tabs.addTab(self.trades_panel, "Trades")

        # Queue tab
        self.queue_panel = QTableWidget()
        self.queue_panel.setColumnCount(3)
        self.queue_panel.setHorizontalHeaderLabels(["Strategy", "Status", "Path"])
        self.queue_panel.horizontalHeader().setStretchLastSection(True)
        self.queue_panel.verticalHeader().setVisible(False)
        right_tabs.addTab(self.queue_panel, "Queue")

        # Leaderboard tab
        self.leaderboard_panel = QTableWidget()
        self.leaderboard_panel.setColumnCount(14)
        self.leaderboard_panel.setHorizontalHeaderLabels([
            "Strategy", "TFs", "Best TF", "Batches", "Trades",
            "Net P&L", "Return %", "PF", "Avg Trade",
            "Win Rate %", "Sharpe", "Calmar", "Max DD %", "Score"
        ])
        lb_hdr = self.leaderboard_panel.horizontalHeader()
        lb_hdr.setSectionResizeMode(QHeaderView.Interactive)
        lb_hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        lb_hdr.setMinimumSectionSize(50)
        self.leaderboard_panel.verticalHeader().setVisible(False)
        self.leaderboard_panel.setAlternatingRowColors(True)
        self.leaderboard_panel.setSelectionBehavior(QTableWidget.SelectRows)
        right_tabs.addTab(self.leaderboard_panel, "Leaderboard")

        # Execution Config tab
        self._build_exec_config_tab(right_tabs)

        # Cache tab
        self.cache_panel = QTableWidget()
        self.cache_panel.setColumnCount(5)
        self.cache_panel.setHorizontalHeaderLabels(["Symbol", "TF", "Bars", "Start", "End"])
        self.cache_panel.horizontalHeader().setStretchLastSection(True)
        self.cache_panel.verticalHeader().setVisible(False)
        right_tabs.addTab(self.cache_panel, "Data Cache")

        # Logs tab
        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumBlockCount(2000)
        right_tabs.addTab(self.log_panel, "Logs")

        right_tabs.setMinimumWidth(350)
        h_split.addWidget(right_tabs)

        h_split.setSizes([900, 580])
        h_split.setHandleWidth(6)
        h_split.setStretchFactor(0, 2)
        h_split.setStretchFactor(1, 1)
        main_layout.addWidget(h_split)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(18)
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)

    def _build_exec_config_tab(self, tabs):
        """Build the Execution Config tab with all D36-D50 controls."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        lay = QGridLayout(container)
        lay.setSpacing(4)
        lay.setContentsMargins(6, 6, 6, 6)
        row = 0

        def add_label(text, r):
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {C['dim']}; font-size: 10px;")
            lay.addWidget(lbl, r, 0)

        # Slippage model
        add_label("Slippage Model:", row)
        self.exec_slippage_mode = QComboBox()
        self.exec_slippage_mode.addItems(["fixed", "pct", "vol"])
        lay.addWidget(self.exec_slippage_mode, row, 1)
        row += 1

        # Commission model
        add_label("Commission Model:", row)
        self.exec_commission_mode = QComboBox()
        self.exec_commission_mode.addItems(["per_lot", "per_trade", "pct"])
        lay.addWidget(self.exec_commission_mode, row, 1)
        row += 1

        # Dynamic spread
        add_label("Dynamic Spread:", row)
        self.exec_dynamic_spread = QCheckBox()
        lay.addWidget(self.exec_dynamic_spread, row, 1)
        row += 1

        # Intrabar stop priority
        add_label("Intrabar SL/TP:", row)
        self.exec_intrabar_priority = QCheckBox()
        lay.addWidget(self.exec_intrabar_priority, row, 1)
        row += 1

        # Pending orders
        add_label("Pending Orders:", row)
        self.exec_pending_orders = QCheckBox()
        lay.addWidget(self.exec_pending_orders, row, 1)
        row += 1

        # Trailing stop
        add_label("Trailing Stop:", row)
        self.exec_trailing_mode = QComboBox()
        self.exec_trailing_mode.addItems(["none", "fixed", "atr", "percent"])
        lay.addWidget(self.exec_trailing_mode, row, 1)
        row += 1

        add_label("Trail Distance:", row)
        self.exec_trail_dist = QDoubleSpinBox()
        self.exec_trail_dist.setRange(0, 1000)
        self.exec_trail_dist.setValue(0)
        self.exec_trail_dist.setDecimals(1)
        lay.addWidget(self.exec_trail_dist, row, 1)
        row += 1

        # Breakeven
        add_label("BE Trigger (pips):", row)
        self.exec_be_trigger = QDoubleSpinBox()
        self.exec_be_trigger.setRange(0, 1000)
        self.exec_be_trigger.setValue(0)
        self.exec_be_trigger.setDecimals(1)
        lay.addWidget(self.exec_be_trigger, row, 1)
        row += 1

        add_label("BE Lock (pips):", row)
        self.exec_be_lock = QDoubleSpinBox()
        self.exec_be_lock.setRange(0, 1000)
        self.exec_be_lock.setValue(0)
        self.exec_be_lock.setDecimals(1)
        lay.addWidget(self.exec_be_lock, row, 1)
        row += 1

        # Sizing mode
        add_label("Sizing Mode:", row)
        self.exec_sizing_mode = QComboBox()
        self.exec_sizing_mode.addItems(["fixed", "risk_pct", "kelly", "vol_adj"])
        lay.addWidget(self.exec_sizing_mode, row, 1)
        row += 1

        add_label("Risk %:", row)
        self.exec_risk_pct = QDoubleSpinBox()
        self.exec_risk_pct.setRange(0.1, 50)
        self.exec_risk_pct.setValue(1.0)
        self.exec_risk_pct.setDecimals(1)
        lay.addWidget(self.exec_risk_pct, row, 1)
        row += 1

        # Cooldown
        add_label("Cooldown (bars):", row)
        self.exec_cooldown = QSpinBox()
        self.exec_cooldown.setRange(0, 1000)
        self.exec_cooldown.setValue(0)
        lay.addWidget(self.exec_cooldown, row, 1)
        row += 1

        # Session filter
        add_label("Allowed Sessions:", row)
        self.exec_session_filter = QComboBox()
        self.exec_session_filter.addItems([
            "All", "london,new_york", "london", "new_york", "asia",
            "london,new_york,overlap",
        ])
        lay.addWidget(self.exec_session_filter, row, 1)
        row += 1

        # Max daily loss
        add_label("Max Daily Loss %:", row)
        self.exec_max_daily_loss = QDoubleSpinBox()
        self.exec_max_daily_loss.setRange(0, 100)
        self.exec_max_daily_loss.setValue(0)
        self.exec_max_daily_loss.setDecimals(1)
        lay.addWidget(self.exec_max_daily_loss, row, 1)
        row += 1

        # Max trades per day
        add_label("Max Trades/Day:", row)
        self.exec_max_trades_day = QSpinBox()
        self.exec_max_trades_day.setRange(0, 100)
        self.exec_max_trades_day.setValue(0)
        lay.addWidget(self.exec_max_trades_day, row, 1)
        row += 1

        # Max holding bars
        add_label("Max Hold (bars):", row)
        self.exec_max_hold = QSpinBox()
        self.exec_max_hold.setRange(0, 10000)
        self.exec_max_hold.setValue(0)
        lay.addWidget(self.exec_max_hold, row, 1)
        row += 1

        # Partial TP
        add_label("Partial TP:", row)
        self.exec_partial_tp = QCheckBox()
        lay.addWidget(self.exec_partial_tp, row, 1)
        row += 1

        lay.setRowStretch(row, 1)
        scroll.setWidget(container)
        tabs.addTab(scroll, "Exec Config")

    def _apply_exec_config(self, config: BacktestConfig):
        """Read GUI controls and apply to config."""
        config.slippage_mode = SlippageMode(self.exec_slippage_mode.currentText())
        config.commission_mode = CommissionMode(self.exec_commission_mode.currentText())
        config.dynamic_spread_enabled = self.exec_dynamic_spread.isChecked()
        config.intrabar_stop_priority = self.exec_intrabar_priority.isChecked()
        config.pending_orders_enabled = self.exec_pending_orders.isChecked()
        config.trailing_stop_mode = TrailingStopMode(self.exec_trailing_mode.currentText())
        config.trailing_stop_distance = self.exec_trail_dist.value()
        config.breakeven_trigger_pips = self.exec_be_trigger.value()
        config.breakeven_lock_pips = self.exec_be_lock.value()
        config.sizing_mode = SizingMode(self.exec_sizing_mode.currentText())
        config.risk_pct = self.exec_risk_pct.value()
        config.cooldown_bars = self.exec_cooldown.value()
        sess = self.exec_session_filter.currentText()
        config.allowed_sessions = [] if sess == "All" else [s.strip() for s in sess.split(",")]
        config.max_daily_loss_pct = self.exec_max_daily_loss.value()
        config.max_trades_per_day = self.exec_max_trades_day.value()
        config.max_holding_bars = self.exec_max_hold.value()
        config.partial_tp_enabled = self.exec_partial_tp.isChecked()

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_label = QLabel("Ready")
        sb.addWidget(self.status_label)
        self.cache_label = QLabel("")
        sb.addPermanentWidget(self.cache_label)
        self._update_cache_status()

    def _try_load_chart(self, *_):
        """Load chart from cache for currently selected symbol/TF (silently if not available)."""
        try:
            sym = self.sym_combo.currentText()
            tf = self.tf_combo.currentText()
            data = self.dm.load(sym, tf)
            if data is not None and len(data) > 0:
                max_bars = self.bars_spin.value()
                if len(data) > max_bars:
                    data = data.iloc[-max_bars:]
                self.chart.plot(data, max_bars=min(max_bars, 500))
        except Exception as e:
            logger.debug(f"_try_load_chart: {e}")

    def _update_cache_status(self):
        entries = self.dm.get_cached()
        size = self.dm.cache_size_mb()
        self.cache_label.setText(f"Cache: {len(entries)} datasets | {size:.1f} MB")

        # Update cache tab
        self.cache_panel.setRowCount(len(entries))
        for i, e in enumerate(entries):
            for j, val in enumerate([e.symbol, e.timeframe, str(e.bars), e.start, e.end]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.cache_panel.setItem(i, j, item)

    # ─── Actions ─────────────────────────────────────────────────────────

    def _on_download(self):
        self._log("Starting MT5 data download...")
        self.progress.setVisible(True)
        self._last_download_log_pct = -DOWNLOAD_LOG_EVERY_PCT
        self._set_progress_value(0)

        sym = self.sym_combo.currentText()
        
        # Parse timeframe selection
        tf_selection = self.download_tf_combo.currentText()
        all_tfs = ["M1","M5","M15","M30","H1","H4","D1"]
        
        tf_map = {
            "M1 only": ["M1"],
            "M1-M5": ["M1", "M5"],
            "M1-M15": ["M1", "M5", "M15"],
            "M1-M30": ["M1", "M5", "M15", "M30"],
            "M1-H1": ["M1", "M5", "M15", "M30", "H1"],
            "M1-H4": ["M1", "M5", "M15", "M30", "H1", "H4"],
            "M1-D1 (ALL)": ["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
            "H1-D1": ["H1", "H4", "D1"],
            "H4-D1": ["H4", "D1"],
        }
        
        timeframes_to_download = tf_map.get(tf_selection, ["H1", "H4", "D1"])
        
        self._log(f"Downloading {sym} for timeframes: {', '.join(timeframes_to_download)}")
        self._worker = DownloadWorker(self.dm, [sym], timeframes_to_download)
        self._worker.progress.connect(self._on_download_progress)
        self._worker.finished.connect(self._on_download_done)
        self._worker.error.connect(self._on_download_error)
        self._worker.start()

    def _on_download_progress(self, done, total, msg):
        pct = int(done / total * 100) if total > 0 else 0
        self._set_progress_value(pct)
        self._set_status_text(msg)
        if pct - self._last_download_log_pct >= DOWNLOAD_LOG_EVERY_PCT or pct >= 100:
            self._last_download_log_pct = pct
            self._log(msg)

    def _on_download_done(self, results):
        self.progress.setVisible(False)
        ok = sum(1 for v in results.values() if v > 0)
        self._log(f"Download complete: {ok}/{len(results)} successful", "INFO")
        self._set_status_text(f"Download complete: {ok} datasets")
        self._update_cache_status()

        # Display chart with downloaded data
        sym = self.sym_combo.currentText()
        tf = self.tf_combo.currentText()
        data = self.dm.load(sym, tf)
        if data is not None and len(data) > 0:
            max_bars = self.bars_spin.value()
            if len(data) > max_bars:
                data = data.iloc[-max_bars:]
            self.chart.plot(data, max_bars=min(max_bars, 500))
            self._log(f"Chart loaded: {len(data)} bars of {sym} {tf}")

    def _on_download_error(self, err):
        self.progress.setVisible(False)
        self._log(f"Download error: {err}", "ERROR")
        self._set_status_text("Download failed")

    def _on_load_strategy(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Strategy", str(self._strategies_dir), "Python Files (*.py)"
        )
        if path:
            self._strategy_path = path
            strat = self.loader.load(path)
            if strat:
                self.strat_label.setText(f"  Strategy: {self.loader.strategy_name}")
                self._log(f"Loaded: {self.loader.strategy_name} from {path}")
            else:
                self._log(f"Load error: {self.loader.error}", "ERROR")

    def _on_queue_strategies(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Queue Strategy Files", str(self._strategies_dir), "Python Files (*.py)"
        )
        if not paths:
            return

        existing = {s["path"] for s in self._strategy_queue}
        added = 0
        temp_loader = StrategyLoader()
        for p in paths:
            if p in existing:
                continue
            strategy = temp_loader.load(p)
            name = temp_loader.strategy_name if strategy is not None else Path(p).stem
            status = "queued" if strategy is not None else "load_error"
            self._strategy_queue.append({"name": name, "path": p, "status": status})
            added += 1

        self._refresh_queue_table(force_resize=True)
        self._log(f"Queue updated: +{added} strategies (total {len(self._strategy_queue)})")

    def _refresh_queue_table(self, force_resize: bool = False):
        self.queue_panel.setRowCount(len(self._strategy_queue))
        for i, row in enumerate(self._strategy_queue):
            values = [row["name"], row["status"], row["path"]]
            for j, val in enumerate(values):
                item = QTableWidgetItem(str(val))
                if j == 1:
                    if row["status"] == "ok":
                        item.setForeground(QColor(C["green"]))
                    elif row["status"] in ("queued", "running", "no_trades"):
                        item.setForeground(QColor(C["amber"]))
                    elif row["status"] == "load_error":
                        item.setForeground(QColor(C["red"]))
                self.queue_panel.setItem(i, j, item)
        if force_resize:
            self.queue_panel.resizeColumnsToContents()

    def _set_queue_status(self, strategy_name: str, status: str):
        for row in self._strategy_queue:
            if row["name"] == strategy_name:
                row["status"] = status
        self._refresh_queue_table(force_resize=False)

    def _render_leaderboard(self, force_resize: bool = False):
        rows = self._queue_results
        self.leaderboard_panel.setSortingEnabled(False)
        self.leaderboard_panel.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [
                r.get("strategy", ""),
                r.get("tfs", ""),
                r.get("best_tf", ""),
                str(r.get("batches", 0)),
                str(r.get("trades", 0)),
                f"{r.get('net_pnl', 0):.2f}",
                f"{r.get('return_pct', 0):.2f}",
                f"{r.get('profit_factor', 0):.2f}",
                f"{r.get('avg_trade', 0):.2f}",
                f"{r.get('win_rate', 0):.2f}",
                f"{r.get('sharpe', 0):.3f}",
                f"{r.get('calmar', 0):.2f}",
                f"{r.get('max_dd', 0):.2f}",
                f"{r.get('score', 0):.3f}",
            ]
            is_best = (i == 0 and r.get("score", -1e9) > -1e8)
            for j, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                # Green/red colouring for profit-related columns
                if j in (5, 6, 8, 11, 13):
                    try:
                        n = float(val)
                        if n > 0:
                            item.setForeground(QColor(C["green"]))
                        elif n < 0:
                            item.setForeground(QColor(C["red"]))
                    except Exception:
                        pass
                # Gold highlight for top row
                if is_best:
                    item.setBackground(QColor("#0d1f10"))
                    if j == 0:
                        item.setForeground(QColor(C["amber"]))
                self.leaderboard_panel.setItem(i, j, item)
        self.leaderboard_panel.setSortingEnabled(True)
        if force_resize:
            self.leaderboard_panel.resizeColumnsToContents()

    def _on_run_queue(self):
        if not self._strategy_queue:
            self._log("No strategies in queue. Use Queue Strategies first.", "WARN")
            return

        sym = self.sym_combo.currentText()
        tf = self.tf_combo.currentText()
        max_bars = self.bars_spin.value()

        data = self.dm.load(sym, tf)
        if data is None:
            self._log(f"No cached data for {sym} {tf}. Download first.", "ERROR")
            return

        if len(data) > max_bars:
            data = data.iloc[-max_bars:]

        strategy_paths = [s["path"] for s in self._strategy_queue if s["status"] != "load_error"]
        if not strategy_paths:
            self._log("Queue has no valid strategies to run.", "ERROR")
            return

        for row in self._strategy_queue:
            if row["status"] != "load_error":
                row["status"] = "queued"
        self._refresh_queue_table(force_resize=False)

        self._queue_results = []
        self._render_leaderboard(force_resize=True)
        num_batches = max(1, self.batch_spin.value())
        batch_size = max(200, len(data) // num_batches)
        actual_batches = max(1, (len(data) + batch_size - 1) // batch_size)

        self._log(
            f"Queue run started: {len(strategy_paths)} strategies | "
            f"{actual_batches} batches ({batch_size} bars each) | data={len(data)} bars"
        )

        self.refresh_btn.setEnabled(False)
        self.run_queue_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setMaximum(100)
        self._set_progress_value(0)

        self._queue_data = data  # Keep reference for chart update after queue
        self._queue_worker = StrategyQueueWorker(
            data=data,
            strategy_paths=strategy_paths,
            initial_capital=self.capital_spin.value(),
            batch_size=batch_size,
            symbol=sym,
            timeframe=tf,
        )
        self._queue_worker.progress.connect(self._on_queue_progress)
        self._queue_worker.strategy_done.connect(self._on_queue_strategy_done)
        self._queue_worker.finished.connect(self._on_queue_finished)
        self._queue_worker.error.connect(self._on_queue_error)
        self._queue_worker.start()

    def _on_queue_progress(self, done, total, msg, run_num):
        pct = int(done / total * 100) if total else 0
        self._set_progress_value(pct)
        # Display run number in the status label
        run_info = f" (Run {run_num})" if run_num else ""
        self._set_status_text(f"{msg}{run_info}")
        strategy_name = msg.split(":", 1)[0].strip() if ":" in msg else ""
        if strategy_name:
            for row in self._strategy_queue:
                if row["name"] == strategy_name and row["status"] == "queued":
                    row["status"] = "running"
                    self._refresh_queue_table(force_resize=False)
                    break
        if not self.turbo_chk.isChecked():
            self._log(f"{msg}{run_info}")

    def _on_queue_strategy_done(self, row):
        self._queue_results = [r for r in self._queue_results if r["strategy"] != row["strategy"]]
        self._queue_results.append(row)
        self._queue_results.sort(key=lambda x: x["score"], reverse=True)
        self._set_queue_status(row["strategy"], "ok" if row["status"] == "ok" else row["status"])
        self._render_leaderboard(force_resize=False)
        if row["status"] == "ok":
            self._log(
                f"Queue result | {row['strategy']} | score={row['score']:.3f} | "
                f"pnl=${row['net_pnl']:.2f} | wr={row['win_rate']:.1f}%"
            )
        elif row["status"] == "no_trades":
            self._log(
                f"Queue result | {row['strategy']} | no trades produced on selected data/settings",
                "WARN",
            )
        else:
            self._log(f"Queue result | {row['strategy']} | status={row['status']}", "WARN")

    def _on_queue_finished(self, payload):
        leaderboard = payload.get("leaderboard", [])
        self._queue_results = leaderboard
        self._render_leaderboard(force_resize=True)

        self._set_progress_value(100)
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.run_queue_btn.setEnabled(True)

        # Show chart with the data used by the queue
        if hasattr(self, "_queue_data") and self._queue_data is not None:
            self.chart.plot(self._queue_data, max_bars=500)

        if leaderboard:
            best = leaderboard[0]
            self._log(
                f"Leaderboard complete | BEST: {best['strategy']} | "
                f"score={best['score']:.3f} | pnl=${best['net_pnl']:.2f} | "
                f"return={best['return_pct']:.2f}%"
            )
            self._set_status_text(f"Best: {best['strategy']} | score {best['score']:.2f}")
        else:
            self._log("Queue finished with no valid strategy results", "WARN")
            self._set_status_text("Queue finished")

    def _on_queue_error(self, error_msg):
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.run_queue_btn.setEnabled(True)
        self._log(f"Queue error:\n{error_msg}", "ERROR")
        self._set_status_text("Queue failed")

    # ─── Queue Settings / Multi-TF Queue ─────────────────────────────────

    def _on_queue_settings(self):
        """Open Queue Settings dialog and, on confirm, immediately run Multi-TF queue."""
        cached_syms = self.dm.get_symbols()
        if not cached_syms:
            cached_syms = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
        dlg = QueueSettingsDialog(cached_syms, parent=self)
        # Pre-fill with last saved settings
        if self._queue_settings:
            s = self._queue_settings
            if s.get("symbol") in cached_syms:
                dlg.sym_combo.setCurrentText(s["symbol"])
            if s.get("batch_mode", "num_batches") == "bars_per_batch":
                dlg.batch_mode_size.setChecked(True)
            else:
                dlg.batch_mode_num.setChecked(True)
            dlg.batches_spin.setValue(s.get("batches_per_tf", 50))
            dlg.bars_per_batch_spin.setValue(s.get("bars_per_batch", 280))
            dlg.bars_spin.setValue(s.get("bars_per_tf", 5000))
            dlg.capital_spin.setValue(s.get("capital", 10000))
            for tf, cb in dlg._tf_checks.items():
                cb.setChecked(tf in s.get("timeframes", ["H1", "H4"]))
            dlg._sync_batch_mode_ui()
        if dlg.exec() == QDialog.Accepted:
            self._queue_settings = dlg.get_settings()
            tfs = self._queue_settings["timeframes"]
            batch_mode = self._queue_settings.get("batch_mode", "num_batches")
            mode_text = (
                f"batches={self._queue_settings['batches_per_tf']}/TF"
                if batch_mode == "num_batches"
                else f"bars/batch={self._queue_settings['bars_per_batch']}"
            )
            self._log(
                f"Queue Settings: {self._queue_settings['symbol']} | "
                f"TFs={','.join(tfs) if tfs else 'none'} | "
                f"{mode_text} | "
                f"bars={self._queue_settings['bars_per_tf']}/TF"
            )
            self._on_run_tf_queue()

    def _on_run_tf_queue(self):
        """Run all queued strategies across the TFs configured in Queue Settings."""
        if not self._strategy_queue:
            self._log("No strategies in queue. Use 'Queue Strategies' first.", "WARN")
            return

        if not self._queue_settings:
            self._on_queue_settings()
            return

        tfs = self._queue_settings.get("timeframes", [])
        if not tfs:
            self._log("No timeframes selected. Open Queue Settings and check at least one TF.", "WARN")
            return

        strategy_paths = [s["path"] for s in self._strategy_queue if s["status"] != "load_error"]
        if not strategy_paths:
            self._log("Queue has no valid strategies to run.", "ERROR")
            return

        sym = self._queue_settings["symbol"]
        batch_mode = self._queue_settings.get("batch_mode", "num_batches")
        bars_per_tf = self._queue_settings["bars_per_tf"]
        capital = self._queue_settings["capital"]

        # Determine batch configuration
        if batch_mode == "num_batches":
            num_batches = self._queue_settings["batches_per_tf"]
            bars_per_batch = None
            mode_desc = f"{num_batches} batches/TF"
            if num_batches > 0:
                tiny_tfs = []
                for tf in tfs:
                    est_batch = max(100, bars_per_tf // num_batches)
                    rec = recommended_batch_bars(tf)
                    if est_batch < rec:
                        tiny_tfs.append(f"{tf}:{est_batch}->{rec}")
                if tiny_tfs:
                    self._log(
                        "Queue warning: batch windows are too small for indicator warmup; "
                        f"auto-adjusting ({', '.join(tiny_tfs)})",
                        "WARN",
                    )
        else:
            num_batches = None
            bars_per_batch = self._queue_settings["bars_per_batch"]
            mode_desc = f"{bars_per_batch} bars/batch"

        for row in self._strategy_queue:
            if row["status"] != "load_error":
                row["status"] = "queued"
        self._refresh_queue_table(force_resize=False)
        self._queue_results = []
        self._render_leaderboard(force_resize=True)

        self._log(
            f"Multi-TF Queue: {len(strategy_paths)} strategies | "
            f"{sym} | TFs: {', '.join(tfs)} | "
            f"{mode_desc} | {bars_per_tf} bars/TF | ${capital:,.0f} capital"
        )

        # Explicit data visibility before run so users can verify bars are loaded.
        tf_bars = []
        for tf in tfs:
            df = self.dm.load(sym, tf)
            tf_bars.append(f"{tf}:{0 if df is None else len(df)}")
        self._log(f"Multi-TF input bars | {' | '.join(tf_bars)}")

        self.refresh_btn.setEnabled(False)
        self.run_queue_btn.setEnabled(False)
        self.run_tf_queue_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setMaximum(100)
        self._set_progress_value(0)

        self._tf_queue_worker = MultiTFQueueWorker(
            dm=self.dm,
            symbol=sym,
            timeframes=tfs,
            bars_per_tf=bars_per_tf,
            num_batches=num_batches,
            bars_per_batch=bars_per_batch,
            strategy_paths=strategy_paths,
            initial_capital=capital,
        )
        self._tf_queue_worker.progress.connect(self._on_queue_progress)
        self._tf_queue_worker.strategy_done.connect(self._on_queue_strategy_done)
        self._tf_queue_worker.finished.connect(self._on_mtf_queue_finished)
        self._tf_queue_worker.error.connect(self._on_mtf_queue_error)
        self._tf_queue_worker.start()

    def _on_mtf_queue_finished(self, payload):
        """Handle multi-TF queue completion."""
        leaderboard = payload.get("leaderboard", [])
        self._queue_results = leaderboard
        self._render_leaderboard(force_resize=True)

        self._set_progress_value(100)
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.run_queue_btn.setEnabled(True)
        self.run_tf_queue_btn.setEnabled(True)

        if leaderboard:
            best = leaderboard[0]
            best_tf = best.get("best_tf", "")
            sym = self._queue_settings.get("symbol", self.sym_combo.currentText())

            # Load and show chart for best strategy's best TF
            if best_tf and best_tf != "-":
                data = self.dm.load(sym, best_tf)
                if data is not None:
                    bars = self._queue_settings.get("bars_per_tf", 5000)
                    if len(data) > bars:
                        data = data.iloc[-bars:]
                    self.chart.plot(data, max_bars=500)
                    # Sync toolbar dropdowns
                    self.sym_combo.blockSignals(True)
                    self.tf_combo.blockSignals(True)
                    self.sym_combo.setCurrentText(sym)
                    tf_items = [self.tf_combo.itemText(i) for i in range(self.tf_combo.count())]
                    if best_tf in tf_items:
                        self.tf_combo.setCurrentText(best_tf)
                    self.sym_combo.blockSignals(False)
                    self.tf_combo.blockSignals(False)

            self._log(
                f"\u2713 Multi-TF complete | BEST: {best['strategy']} | "
                f"best_tf={best_tf} | tfs={best.get('tfs','')} | "
                f"score={best['score']:.3f} | pnl=${best['net_pnl']:.2f} | "
                f"return={best['return_pct']:.2f}% | trades={best['trades']}"
            )
            self._set_status_text(
                f"Best: {best['strategy']} | {best_tf} | score {best['score']:.2f}"
            )
        else:
            self._log("Multi-TF Queue finished — no valid results", "WARN")
            self._set_status_text("Multi-TF Queue finished")

    def _on_mtf_queue_error(self, error_msg):
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.run_queue_btn.setEnabled(True)
        self.run_tf_queue_btn.setEnabled(True)
        self._log(f"Multi-TF Queue error:\n{error_msg}", "ERROR")
        self._set_status_text("Multi-TF Queue failed")

    def _on_run(self):
        self._run_backtest()

    def _on_refresh(self):
        """Hot reload: reload strategy + rerun backtest."""
        self._log("🔄 Hot reloading strategy...")
        if self._strategy_path:
            strat = self.loader.reload()
            if strat:
                self.strat_label.setText(f"  Strategy: {self.loader.strategy_name}")
                self._log(f"Reloaded: {self.loader.strategy_name}")
                self._run_backtest()
            else:
                self._log(f"Reload error:\n{self.loader.error}", "ERROR")
        else:
            self._log("No strategy loaded", "WARN")

    def _run_backtest(self):
        strat = self.loader.strategy
        if strat is None:
            self._log("No strategy loaded!", "ERROR")
            return

        sym = self.sym_combo.currentText()
        tf = self.tf_combo.currentText()
        max_bars = self.bars_spin.value()

        self._log(f"Running backtest: {sym} {tf} | {self.loader.strategy_name}")

        # Load data
        data = self.dm.load(sym, tf)
        if data is None:
            self._log(f"No cached data for {sym} {tf}. Download first.", "ERROR")
            return

        if len(data) > max_bars:
            data = data.iloc[-max_bars:]

        self._log(f"Data: {len(data)} bars [{data.index[0]} → {data.index[-1]}]")

        # Configure engine
        config = BacktestConfig(initial_capital=self.capital_spin.value())
        if "JPY" in sym:
            config.pip_value = 0.01
        elif "XAU" in sym or "GOLD" in sym:
            config.pip_value = 0.01
            config.lot_multiplier = 100
        else:
            config.pip_value = 0.0001
            config.lot_multiplier = 100000

        self._apply_exec_config(config)
        engine = BacktestEngine(config)

        # Replay settings
        speed_text = self.replay_speed.currentText()
        replay_on = speed_text != "Off"
        delay_map = {"Off": 0, "Fast": 5, "Medium": 30, "Slow": 100}
        replay_delay = delay_map.get(speed_text, 0)
        replay_interval = max(1, len(data) // 150) if replay_on else max(1, len(data) // 20)

        # Pre-plot chart for replay
        if replay_on:
            self.chart.plot(data, replay_mode=True)
            self._log(f"Replay: {speed_text} | ~{min(150, len(data))} chart updates")

        # UI state
        self.refresh_btn.setEnabled(False)
        self.progress.setVisible(True)
        self._set_progress_value(0)
        self.progress.setMaximum(100)

        # Store data reference for done handler
        self._backtest_data = data

        # Run backtest in background thread with replay
        self._backtest_worker = BacktestWorker(
            engine, data, strat,
            replay_interval=replay_interval,
            replay_delay_ms=replay_delay,
        )
        self._backtest_worker.bar_update.connect(self._on_bar_update)
        self._backtest_worker.finished.connect(self._on_backtest_done)
        self._backtest_worker.error.connect(self._on_backtest_error)
        self._backtest_worker.start()

    def _on_bar_update(self, bar_idx, total_bars, trades):
        """Handle incremental bar update during backtest replay."""
        pct = int(bar_idx / max(total_bars, 1) * 100)
        self._set_progress_value(pct)
        self._set_status_text(f"Bar {bar_idx}/{total_bars} | {len(trades)} trades")
        self.chart.update_replay(bar_idx, trades)

    def _on_backtest_done(self, result):
        """Handle backtest completion"""
        self._set_progress_value(100)
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        
        if result is None:
            return
        
        self.result = result
        r = self.result
        sym = self.sym_combo.currentText()
        tf = self.tf_combo.currentText()
        max_bars = self.bars_spin.value()
        
        self._log(
            f"✓ Backtest complete | "
            f"{r.total_trades} trades | "
            f"P&L: ${r.net_pnl:.2f} ({r.total_return_pct:.2f}%) | "
            f"WR: {r.win_rate:.1f}% | "
            f"Sharpe: {r.sharpe_ratio:.3f}"
        )

        # End replay and show final chart with all trades
        self.chart.end_replay()
        data = self.result.data
        self.chart.plot(data, r.trades, max_bars=min(max_bars, 500))
        self.equity_chart.plot(r)
        self.metrics_panel.update_metrics(r)
        self.trades_panel.load_trades(r)
        self._set_status_text(
            f"{sym} {tf} | {r.total_trades} trades | ${r.net_pnl:.2f}"
        )

    def _on_backtest_error(self, error_msg):
        """Handle backtest error"""
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self._log(f"❌ Backtest error:\n{error_msg}", "ERROR")
        self._set_status_text("Backtest failed")


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run_app():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(C["bg"]))
    palette.setColor(QPalette.WindowText, QColor(C["text"]))
    palette.setColor(QPalette.Base, QColor(C["panel"]))
    palette.setColor(QPalette.AlternateBase, QColor(C["row_alt"]))
    palette.setColor(QPalette.ToolTipBase, QColor(C["panel"]))
    palette.setColor(QPalette.ToolTipText, QColor(C["text"]))
    palette.setColor(QPalette.Text, QColor(C["text"]))
    palette.setColor(QPalette.Button, QColor(C["border"]))
    palette.setColor(QPalette.ButtonText, QColor(C["text"]))
    palette.setColor(QPalette.Highlight, QColor(C["blue"]))
    palette.setColor(QPalette.HighlightedText, QColor(C["white"]))
    app.setPalette(palette)

    window = AphelionLab()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
