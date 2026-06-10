#!/usr/bin/env python3
"""Refresh CoinGecko data (last 365d) and merge new settled bars into *_full.csv,
then re-run backtests and regenerate all result pages."""
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import refresh_full_data

if __name__ == "__main__":
    new_rows = refresh_full_data()
    print(f"\nTotal new rows merged: {new_rows}")
    if new_rows and "--no-backtest" not in sys.argv:
        subprocess.run([sys.executable, str(PROJECT / "run_backtest.py")], check=True)
