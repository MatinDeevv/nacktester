"""
APHELION LAB — Visual Backtesting Laboratory
Run: python main.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui_app import run_app

if __name__ == "__main__":
    run_app()
