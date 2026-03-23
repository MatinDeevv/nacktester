from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

from aphelion_lab.data_manager import DataManager
from aphelion_lab.strategy_runtime import StrategyLoader

ROOT = Path(__file__).resolve().parents[2]


def _run_script(script_name: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    return subprocess.run(
        [sys.executable, str(ROOT / script_name)],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_test_system_runs_from_parent_dir_with_cp1252_stdout():
    result = _run_script("test_system.py")
    assert result.returncode == 0, result.stderr
    assert "[OK] Backtest ran successfully" in result.stdout
    assert "All systems operational!" in result.stdout


def test_verify_strategies_runs_from_parent_dir_with_cp1252_stdout():
    result = _run_script("verify_strategies.py")
    assert result.returncode == 0, result.stderr
    assert "RESULT: 10/10 strategies loaded successfully" in result.stdout
    assert "[OK] ALL SYSTEMS OPERATIONAL" in result.stdout


def test_strategy_loader_supports_sibling_helper_imports(tmp_path: Path):
    helper_module = "loader_helper_mod"
    (tmp_path / f"{helper_module}.py").write_text("VALUE = 'Helper-backed strategy'\n", encoding="utf-8")
    (tmp_path / "custom_strategy.py").write_text(
        textwrap.dedent(
            f"""
            from strategy_runtime import Strategy
            from {helper_module} import VALUE

            class CustomStrategy(Strategy):
                name = VALUE

                def on_bar(self, ctx):
                    return None
            """
        ),
        encoding="utf-8",
    )

    loader = StrategyLoader()
    strategy = loader.load(str(tmp_path / "custom_strategy.py"))

    assert strategy is not None
    assert loader.strategy_name == "Helper-backed strategy"


def test_strategy_loader_clears_stale_strategy_after_failed_reload(tmp_path: Path):
    good_strategy = tmp_path / "good_strategy.py"
    bad_strategy = tmp_path / "bad_strategy.py"

    good_strategy.write_text(
        textwrap.dedent(
            """
            from strategy_runtime import Strategy

            class GoodStrategy(Strategy):
                name = "Good"

                def on_bar(self, ctx):
                    return None
            """
        ),
        encoding="utf-8",
    )
    bad_strategy.write_text(
        textwrap.dedent(
            """
            from strategy_runtime import Strategy

            class BadStrategy(Strategy):
                def on_bar(self, ctx)
                    return None
            """
        ),
        encoding="utf-8",
    )

    loader = StrategyLoader()
    assert loader.load(str(good_strategy)) is not None
    assert loader.strategy_name == "Good"

    failed = loader.load(str(bad_strategy))

    assert failed is None
    assert loader.strategy is None
    assert loader.strategy_name == "None"
    assert loader.error is not None
    assert "Error loading strategy" in loader.error


def test_data_manager_defaults_to_repo_cache_dir(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    manager = DataManager()

    assert manager.cache_dir == (ROOT / "cache").resolve()
