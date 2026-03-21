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

import numpy as np
import pandas as pd
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
    QFrame, QScrollArea, QSizePolicy, QCheckBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QDate, QTimer, QSize
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QAction

from .data_manager import DataManager
from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from .strategy_runtime import StrategyLoader

logger = logging.getLogger("aphelion.gui")

# ─── Colors ──────────────────────────────────────────────────────────────────

C = {
    "bg":       "#0a0e17",
    "panel":    "#0f1520",
    "border":   "#1a2332",
    "text":     "#c8d6e5",
    "dim":      "#4a6a8a",
    "green":    "#00e676",
    "red":      "#ff5252",
    "amber":    "#ff9500",
    "blue":     "#2196f3",
    "header":   "#1a237e",
    "row_alt":  "#111b27",
    "white":    "#ffffff",
}

DARK_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C['bg']};
    color: {C['text']};
    font-family: 'Segoe UI', 'Consolas';
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
QTableWidget {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    gridline-color: {C['border']};
    color: {C['text']};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}}
QTableWidget::item {{ padding: 3px 6px; }}
QTableWidget::item:selected {{ background: {C['blue']}; }}
QHeaderView::section {{
    background: {C['bg']};
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
    """Run backtest in background thread"""
    progress = Signal(str)
    finished = Signal(object)  # BacktestResult
    error = Signal(str)

    def __init__(self, engine, data, strategy):
        super().__init__()
        self.engine = engine
        self.data = data
        self.strategy = strategy

    def run(self):
        try:
            self.progress.emit("Running backtest...")
            result = self.engine.run(self.data, self.strategy)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class StrategyQueueWorker(QThread):
    """Run multiple strategy files sequentially and score them on batch slices."""

    progress = Signal(int, int, str)
    strategy_done = Signal(object)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, data: pd.DataFrame, strategy_paths: list[str], initial_capital: float,
                 batch_size: int, symbol: str):
        super().__init__()
        self.data = data
        self.strategy_paths = strategy_paths
        self.initial_capital = float(initial_capital)
        self.batch_size = int(batch_size)
        self.symbol = symbol
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
    def _score(total_return_pct: float, sharpe: float, win_rate: float, max_dd: float) -> float:
        # Balance profitability and consistency.
        return (total_return_pct * 0.55) + (sharpe * 12.0) + (win_rate * 0.20) - (max_dd * 0.25)

    def run(self):
        try:
            if self.data is None or len(self.data) == 0:
                raise ValueError("No data available for queue run")
            if not self.strategy_paths:
                raise ValueError("No strategies queued")

            loader = StrategyLoader()
            batches_per_strategy = max(1, (len(self.data) + self.batch_size - 1) // self.batch_size)
            total_steps = len(self.strategy_paths) * batches_per_strategy
            done_steps = 0
            leaderboard = []
            min_batch = max(200, min(self.batch_size, 1000))

            for path in self.strategy_paths:
                strategy = loader.load(path)
                strategy_name = loader.strategy_name or Path(path).stem
                if strategy is None:
                    row = {
                        "strategy": strategy_name,
                        "path": path,
                        "batches": 0,
                        "trades": 0,
                        "net_pnl": 0.0,
                        "return_pct": 0.0,
                        "win_rate": 0.0,
                        "sharpe": 0.0,
                        "max_dd": 0.0,
                        "score": -1e9,
                        "status": "load_error",
                    }
                    self.strategy_done.emit(row)
                    leaderboard.append(row)
                    continue

                total_pnl = 0.0
                total_trades = 0
                wins = 0
                sharpe_vals = []
                dd_vals = []
                batches_run = 0

                for i in range(0, len(self.data), self.batch_size):
                    chunk = self.data.iloc[i:i + self.batch_size]
                    done_steps += 1
                    if len(chunk) < min_batch:
                        self.progress.emit(done_steps, total_steps, f"{strategy_name}: skipped tiny tail batch")
                        continue

                    engine = BacktestEngine(self._build_config())
                    result = engine.run(chunk, strategy)

                    batches_run += 1
                    total_pnl += float(result.net_pnl)
                    total_trades += int(result.total_trades)
                    wins += sum(1 for t in result.trades if t.pnl > 0)
                    sharpe_vals.append(float(result.sharpe_ratio))
                    dd_vals.append(abs(float(result.max_drawdown)))

                    self.progress.emit(
                        done_steps,
                        total_steps,
                        f"{strategy_name}: batch {batches_run} | pnl ${result.net_pnl:.2f}"
                    )

                win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
                return_pct = (total_pnl / self.initial_capital * 100.0) if self.initial_capital else 0.0
                sharpe = float(np.mean(sharpe_vals)) if sharpe_vals else 0.0
                max_dd = float(np.max(dd_vals)) if dd_vals else 0.0
                score = self._score(return_pct, sharpe, win_rate, max_dd)

                row = {
                    "strategy": strategy_name,
                    "path": path,
                    "batches": batches_run,
                    "trades": total_trades,
                    "net_pnl": total_pnl,
                    "return_pct": return_pct,
                    "win_rate": win_rate,
                    "sharpe": sharpe,
                    "max_dd": max_dd,
                    "score": score,
                    "status": "ok",
                }
                self.strategy_done.emit(row)
                leaderboard.append(row)

            leaderboard.sort(key=lambda x: x["score"], reverse=True)
            self.last_leaderboard = leaderboard
            self.finished.emit({"leaderboard": leaderboard})
        except Exception as e:
            self.last_error = f"{e}\n{traceback.format_exc()}"
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ─── Chart Widget ────────────────────────────────────────────────────────────

class CandlestickChart(FigureCanvasQTAgg):
    """Matplotlib candlestick chart with trade markers."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 5), facecolor=C["panel"])
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axis(self.ax)
        self.fig.tight_layout(pad=1)
        
        # Add interactive toolbar for zoom/pan
        self.toolbar = NavigationToolbar2QT(self, parent)
        self.toolbar.setStyleSheet(f"background-color: {C['panel']}; color: {C['text']};")

    def get_toolbar(self):
        """Return toolbar widget for layout"""
        return self.toolbar

    def _style_axis(self, ax):
        ax.set_facecolor(C["bg"])
        ax.tick_params(colors=C["dim"], labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(C["border"])
        ax.spines["left"].set_color(C["border"])
        ax.grid(True, color=C["border"], alpha=0.3, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    def plot(self, data: pd.DataFrame, trades: list = None, max_bars: int = 500):
        self.ax.clear()
        self._style_axis(self.ax)

        if data is None or len(data) == 0:
            self.draw()
            return

        # Limit bars for performance
        df = data.iloc[-max_bars:] if len(data) > max_bars else data
        dates = mdates.date2num(df.index.to_pydatetime())
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Draw candles
        width = 0.6 * (dates[1] - dates[0]) if len(dates) > 1 else 0.001
        for i in range(len(dates)):
            color = C["green"] if closes[i] >= opens[i] else C["red"]
            # Wick
            self.ax.plot([dates[i], dates[i]], [lows[i], highs[i]],
                        color=color, linewidth=0.7)
            # Body
            body_low = min(opens[i], closes[i])
            body_high = max(opens[i], closes[i])
            body_h = max(body_high - body_low, (highs[i] - lows[i]) * 0.01)
            rect = plt.Rectangle((dates[i] - width/2, body_low), width, body_h,
                                facecolor=color if closes[i] < opens[i] else C["bg"],
                                edgecolor=color, linewidth=0.8)
            self.ax.add_patch(rect)

        # Trade markers
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
        self.fig.autofmt_xdate()
        self.fig.tight_layout(pad=1)
        self.draw()


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
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(C["border"])
        ax.spines["left"].set_color(C["border"])
        ax.grid(True, color=C["border"], alpha=0.3, linewidth=0.5)

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

        self.dm = DataManager()
        self.engine = BacktestEngine()
        self.loader = StrategyLoader()
        self.result: BacktestResult = None
        self._strategy_path = None
        self._strategy_queue: list[dict] = []
        self._queue_results: list[dict] = []

        self._setup_logging()
        self._build_toolbar()
        self._build_ui()
        self._build_statusbar()

        # Load example strategy
        ex = Path("aphelion_lab/strategies/st_01_sma_crossover.py")
        if ex.exists():
            self._strategy_path = str(ex)
            self.loader.load(str(ex))
            self._log(f"Loaded strategy: {self.loader.strategy_name}")
        else:
            self._log("No default strategy loaded", "WARN")

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

        # Bars
        tb.addWidget(QLabel("  Bars: "))
        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(100, 500000)
        self.bars_spin.setValue(5000)
        self.bars_spin.setSingleStep(1000)
        tb.addWidget(self.bars_spin)

        tb.addWidget(QLabel("  Batch: "))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(500, 50000)
        self.batch_spin.setValue(5000)
        self.batch_spin.setSingleStep(500)
        self.batch_spin.setToolTip("Queue mode chunk size per backtest batch")
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

        self.turbo_chk = QCheckBox("Turbo")
        self.turbo_chk.setChecked(True)
        self.turbo_chk.setToolTip("Keep UI updates minimal during queue run for max throughput")
        tb.addWidget(self.turbo_chk)

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
        chart_container.setMinimumHeight(350)
        left_layout.addWidget(chart_container, stretch=3)

        # Equity curve with toolbar
        self.equity_chart = EquityCurveChart(self)
        equity_container = QWidget()
        equity_layout = QVBoxLayout(equity_container)
        equity_layout.setContentsMargins(0, 0, 0, 0)
        equity_layout.setSpacing(0)
        equity_layout.addWidget(self.equity_chart.get_toolbar())
        equity_layout.addWidget(self.equity_chart, stretch=1)
        equity_container.setMinimumHeight(150)
        left_layout.addWidget(equity_container, stretch=1)

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
        self.leaderboard_panel.setColumnCount(9)
        self.leaderboard_panel.setHorizontalHeaderLabels([
            "Strategy", "Batches", "Trades", "Net P&L", "Return %",
            "Win Rate %", "Sharpe", "Max DD %", "Score"
        ])
        self.leaderboard_panel.horizontalHeader().setStretchLastSection(True)
        self.leaderboard_panel.verticalHeader().setVisible(False)
        right_tabs.addTab(self.leaderboard_panel, "Leaderboard")

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

        right_tabs.setMinimumWidth(320)
        right_tabs.setMaximumWidth(500)
        h_split.addWidget(right_tabs)

        h_split.setSizes([1000, 380])
        main_layout.addWidget(h_split)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(18)
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_label = QLabel("Ready")
        sb.addWidget(self.status_label)
        self.cache_label = QLabel("")
        sb.addPermanentWidget(self.cache_label)
        self._update_cache_status()

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
        self.progress.setValue(0)

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
        self.progress.setValue(pct)
        self.status_label.setText(msg)
        self._log(msg)

    def _on_download_done(self, results):
        self.progress.setVisible(False)
        ok = sum(1 for v in results.values() if v > 0)
        self._log(f"Download complete: {ok}/{len(results)} successful", "INFO")
        self.status_label.setText(f"Download complete: {ok} datasets")
        self._update_cache_status()

    def _on_download_error(self, err):
        self.progress.setVisible(False)
        self._log(f"Download error: {err}", "ERROR")
        self.status_label.setText("Download failed")

    def _on_load_strategy(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Strategy", "aphelion_lab/strategies/", "Python Files (*.py)"
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
            self, "Queue Strategy Files", "aphelion_lab/strategies/", "Python Files (*.py)"
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

        self._refresh_queue_table()
        self._log(f"Queue updated: +{added} strategies (total {len(self._strategy_queue)})")

    def _refresh_queue_table(self):
        self.queue_panel.setRowCount(len(self._strategy_queue))
        for i, row in enumerate(self._strategy_queue):
            values = [row["name"], row["status"], row["path"]]
            for j, val in enumerate(values):
                item = QTableWidgetItem(str(val))
                if j == 1:
                    if row["status"] == "ok":
                        item.setForeground(QColor(C["green"]))
                    elif row["status"] in ("queued", "running"):
                        item.setForeground(QColor(C["amber"]))
                    elif row["status"] == "load_error":
                        item.setForeground(QColor(C["red"]))
                self.queue_panel.setItem(i, j, item)
        self.queue_panel.resizeColumnsToContents()

    def _set_queue_status(self, strategy_name: str, status: str):
        for row in self._strategy_queue:
            if row["name"] == strategy_name:
                row["status"] = status
        self._refresh_queue_table()

    def _render_leaderboard(self):
        rows = self._queue_results
        self.leaderboard_panel.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [
                r["strategy"],
                str(r["batches"]),
                str(r["trades"]),
                f"{r['net_pnl']:.2f}",
                f"{r['return_pct']:.2f}",
                f"{r['win_rate']:.2f}",
                f"{r['sharpe']:.3f}",
                f"{r['max_dd']:.2f}",
                f"{r['score']:.3f}",
            ]
            for j, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if j in (3, 4, 8):
                    try:
                        n = float(val)
                        if n > 0:
                            item.setForeground(QColor(C["green"]))
                        elif n < 0:
                            item.setForeground(QColor(C["red"]))
                    except:
                        pass
                self.leaderboard_panel.setItem(i, j, item)
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

        data = data.ffill().dropna()
        if len(data) > max_bars:
            data = data.iloc[-max_bars:]

        strategy_paths = [s["path"] for s in self._strategy_queue if s["status"] != "load_error"]
        if not strategy_paths:
            self._log("Queue has no valid strategies to run.", "ERROR")
            return

        for row in self._strategy_queue:
            if row["status"] != "load_error":
                row["status"] = "queued"
        self._refresh_queue_table()

        self._queue_results = []
        self._render_leaderboard()
        self._log(
            f"Queue run started: {len(strategy_paths)} strategies | "
            f"batch={self.batch_spin.value()} bars | data={len(data)} bars"
        )

        self.refresh_btn.setEnabled(False)
        self.run_queue_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setMaximum(100)
        self.progress.setValue(0)

        self._queue_worker = StrategyQueueWorker(
            data=data,
            strategy_paths=strategy_paths,
            initial_capital=self.capital_spin.value(),
            batch_size=self.batch_spin.value(),
            symbol=sym,
        )
        self._queue_worker.progress.connect(self._on_queue_progress)
        self._queue_worker.strategy_done.connect(self._on_queue_strategy_done)
        self._queue_worker.finished.connect(self._on_queue_finished)
        self._queue_worker.error.connect(self._on_queue_error)
        self._queue_worker.start()

    def _on_queue_progress(self, done, total, msg):
        pct = int(done / total * 100) if total else 0
        self.progress.setValue(pct)
        self.status_label.setText(msg)
        strategy_name = msg.split(":", 1)[0].strip() if ":" in msg else ""
        if strategy_name:
            for row in self._strategy_queue:
                if row["name"] == strategy_name and row["status"] == "queued":
                    row["status"] = "running"
                    self._refresh_queue_table()
                    break
        if not self.turbo_chk.isChecked():
            self._log(msg)

    def _on_queue_strategy_done(self, row):
        self._queue_results = [r for r in self._queue_results if r["strategy"] != row["strategy"]]
        self._queue_results.append(row)
        self._queue_results.sort(key=lambda x: x["score"], reverse=True)
        self._set_queue_status(row["strategy"], "ok" if row["status"] == "ok" else row["status"])
        self._render_leaderboard()
        if row["status"] == "ok":
            self._log(
                f"Queue result | {row['strategy']} | score={row['score']:.3f} | "
                f"pnl=${row['net_pnl']:.2f} | wr={row['win_rate']:.1f}%"
            )
        else:
            self._log(f"Queue result | {row['strategy']} | status={row['status']}", "WARN")

    def _on_queue_finished(self, payload):
        leaderboard = payload.get("leaderboard", [])
        self._queue_results = leaderboard
        self._render_leaderboard()

        self.progress.setValue(100)
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.run_queue_btn.setEnabled(True)

        if leaderboard:
            best = leaderboard[0]
            self._log(
                f"Leaderboard complete | BEST: {best['strategy']} | "
                f"score={best['score']:.3f} | pnl=${best['net_pnl']:.2f} | "
                f"return={best['return_pct']:.2f}%"
            )
            self.status_label.setText(f"Best: {best['strategy']} | score {best['score']:.2f}")
        else:
            self._log("Queue finished with no valid strategy results", "WARN")
            self.status_label.setText("Queue finished")

    def _on_queue_error(self, error_msg):
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self.run_queue_btn.setEnabled(True)
        self._log(f"Queue error:\n{error_msg}", "ERROR")
        self.status_label.setText("Queue failed")

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

        # Fill data gaps (market closures, weekends) by forward-filling
        original_shape = len(data)
        data = data.ffill()  # Forward fill any NaN values
        data = data.dropna()  # Then drop remaining NaNs
        
        if len(data) < original_shape:
            self._log(f"Data has {original_shape - len(data)} gaps (market closure). Filled.")

        if len(data) > max_bars:
            data = data.iloc[-max_bars:]

        self._log(f"Data: {len(data)} bars [{data.index[0]} → {data.index[-1]}]")

        # Configure engine
        config = BacktestConfig(initial_capital=self.capital_spin.value())

        # Adjust pip_value for different instruments
        if "JPY" in sym:
            config.pip_value = 0.01
        elif "XAU" in sym or "GOLD" in sym:
            config.pip_value = 0.01
            config.lot_multiplier = 100
        else:
            config.pip_value = 0.0001
            config.lot_multiplier = 100000

        engine = BacktestEngine(config)

        # Disable buttons during backtest
        self.refresh_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.progress.setMaximum(0)  # Indeterminate progress

        # Run backtest in background thread
        self._backtest_worker = BacktestWorker(engine, data, strat)
        self._backtest_worker.finished.connect(self._on_backtest_done)
        self._backtest_worker.error.connect(self._on_backtest_error)
        self._backtest_worker.start()

    def _on_backtest_done(self, result):
        """Handle backtest completion"""
        self.progress.setMaximum(100)
        self.progress.setValue(100)
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        
        if result is None:
            return
        
        self.result = result
        sym = self.sym_combo.currentText()
        tf = self.tf_combo.currentText()
        max_bars = self.bars_spin.value()
        
        r = self.result
        elapsed = 0  # We don't track this in background thread, estimate
        
        self._log(
            f"✓ Backtest complete | "
            f"{r.total_trades} trades | "
            f"P&L: ${r.net_pnl:.2f} ({r.total_return_pct:.2f}%) | "
            f"WR: {r.win_rate:.1f}% | "
            f"Sharpe: {r.sharpe_ratio:.3f}"
        )

        # Update all panels
        data = self.result.data
        self.chart.plot(data, r.trades, max_bars=min(max_bars, 500))
        self.equity_chart.plot(r)
        self.metrics_panel.update_metrics(r)
        self.trades_panel.load_trades(r)
        self.status_label.setText(
            f"{sym} {tf} | {r.total_trades} trades | ${r.net_pnl:.2f}"
        )

    def _on_backtest_error(self, error_msg):
        """Handle backtest error"""
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        self._log(f"❌ Backtest error:\n{error_msg}", "ERROR")
        self.status_label.setText("Backtest failed")


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
