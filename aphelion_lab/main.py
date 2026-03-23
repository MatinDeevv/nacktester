"""
APHELION LAB - Visual Backtesting Laboratory
Run: python main.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .gui_app import run_app
except ImportError:
    from gui_app import run_app


if __name__ == "__main__":
    run_app()
