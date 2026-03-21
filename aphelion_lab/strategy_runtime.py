"""
Aphelion Lab — Strategy Runtime
Base class for strategies + hot reload mechanism.
"""

import importlib
import importlib.util
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger("aphelion.strategy")


class Strategy:
    """Base class for all strategies."""
    name = "BaseStrategy"

    def on_init(self, ctx):
        """Called once before backtest starts."""
        pass

    def on_bar(self, ctx):
        """Called on each bar. Override this."""
        pass


class StrategyLoader:
    """Loads and hot-reloads strategy Python files."""

    def __init__(self):
        self._current_path: Optional[str] = None
        self._current_strategy: Optional[Strategy] = None
        self._module = None
        self._error: Optional[str] = None

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def strategy(self) -> Optional[Strategy]:
        return self._current_strategy

    @property
    def strategy_name(self) -> str:
        if self._current_strategy:
            return getattr(self._current_strategy, "name", self._current_strategy.__class__.__name__)
        return "None"

    def load(self, filepath: str) -> Optional[Strategy]:
        """Load a strategy from a Python file."""
        self._error = None
        self._current_path = filepath
        path = Path(filepath)

        if not path.exists():
            self._error = f"File not found: {filepath}"
            logger.error(self._error)
            return None

        try:
            module_name = f"_strategy_{path.stem}"

            # Remove old module if exists
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Add the aphelion_lab directory to sys.path so strategies can import modules
            aphelion_lab_dir = str(Path(__file__).parent)
            if aphelion_lab_dir not in sys.path:
                sys.path.insert(0, aphelion_lab_dir)

            spec = importlib.util.spec_from_file_location(module_name, str(path))
            module = importlib.util.module_from_spec(spec)
            
            # Inject Strategy class into module globals
            module.__dict__["Strategy"] = Strategy
            
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self._module = module

            # Find the Strategy subclass
            strategy_cls = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, Strategy) and
                    attr is not Strategy):
                    strategy_cls = attr
                    break

            # Fallback: look for any class with on_bar method
            if strategy_cls is None:
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and hasattr(attr, "on_bar") and attr_name != "Strategy":
                        strategy_cls = attr
                        break

            if strategy_cls is None:
                self._error = f"No Strategy class found in {filepath}"
                logger.error(self._error)
                return None

            self._current_strategy = strategy_cls()
            logger.info(f"Loaded strategy: {self.strategy_name} from {filepath}")
            return self._current_strategy

        except Exception as e:
            self._error = f"Error loading strategy:\n{traceback.format_exc()}"
            logger.error(self._error)
            return None

    def reload(self) -> Optional[Strategy]:
        """Reload the current strategy (hot reload)."""
        if self._current_path:
            logger.info(f"Hot-reloading: {self._current_path}")
            return self.load(self._current_path)
        return None

    def load_from_code(self, code: str, name: str = "InlineStrategy") -> Optional[Strategy]:
        """Load strategy from raw Python code string."""
        self._error = None
        try:
            module_name = f"_strategy_inline_{name}"
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Create a module from code
            import types
            module = types.ModuleType(module_name)
            module.__dict__["Strategy"] = Strategy  # Make base class available
            exec(code, module.__dict__)
            sys.modules[module_name] = module
            self._module = module

            # Find strategy class
            strategy_cls = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and hasattr(attr, "on_bar") and
                    attr_name != "Strategy" and attr_name[0].isupper()):
                    strategy_cls = attr
                    break

            if strategy_cls is None:
                self._error = "No Strategy class found in code"
                return None

            self._current_strategy = strategy_cls()
            self._current_path = None
            return self._current_strategy

        except Exception as e:
            self._error = f"Error in strategy code:\n{traceback.format_exc()}"
            return None
