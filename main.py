#!/usr/bin/env python3
"""
APHELION LAB — Visual Backtesting Laboratory
Entry point for the unified application.

Usage:
    python main.py

Requirements installed via:
    pip install -r requirements.txt

Note: MetaTrader5 only available on Windows. On other platforms,
the app generates synthetic data for development/testing.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the GUI app
from aphelion_lab.gui_app import run_app

if __name__ == "__main__":
    run_app()
