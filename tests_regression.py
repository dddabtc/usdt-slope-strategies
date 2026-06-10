#!/usr/bin/env python3
"""Regression: with costs disabled, the long side must reproduce canonical V27
from dddabtc/usdt-slope-research exactly (+320.6%, 27 trades, same dates).

If a local clone of the original repo is present, every trade is diffed
field-by-field against its frozen artifact; otherwise the embedded expected
values are used.
"""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from src.data import load_full_data
from src.engine import BENCHMARK_TEST_END, TRAIN_END, LONG_ONLY, StrategyConfig, backtest

REF = Path.home() / "usdt-slope-research/experiments/v27-more-trades/trades_frozen_plus_one_bar_executable.json"

EXPECTED = {
    "n_trades": 27,
    "total_return": 3.206,
    "daily_sharpe": 0.97,
    "max_drawdown": -0.273,
    "first_buy": "2023-01-26",
    "last_sell": "2025-10-10",
}

cfg = StrategyConfig(
    mode=LONG_ONLY, leverage=2.5,
    fee_bps_per_side=0.0, long_funding_bps_per_day=0.0, short_funding_bps_per_day=0.0,
)
data = load_full_data()
result = backtest(data, cfg, TRAIN_END, BENCHMARK_TEST_END)
m = result["metrics"]
trades = result["trades"]

print(f"new engine: {m['n_trades']} trades, total {m['total_return']*100:.1f}%, "
      f"sharpe {m['daily_sharpe']:.2f}, dd {m['max_drawdown']*100:.1f}%, wr {m['win_rate']*100:.0f}%")

assert m["n_trades"] == EXPECTED["n_trades"], f"trade count {m['n_trades']}"
assert abs(m["total_return"] - EXPECTED["total_return"]) < 0.01, f"total_return {m['total_return']}"
assert abs(m["daily_sharpe"] - EXPECTED["daily_sharpe"]) < 0.02, f"sharpe {m['daily_sharpe']}"
assert abs(m["max_drawdown"] - EXPECTED["max_drawdown"]) < 0.005, f"dd {m['max_drawdown']}"
assert trades[0]["buy_date"] == EXPECTED["first_buy"], trades[0]["buy_date"]
assert trades[-1]["sell_date"] == EXPECTED["last_sell"], trades[-1]["sell_date"]

if REF.exists():
    ref = json.loads(REF.read_text())
    assert len(trades) == len(ref), f"trade count {len(trades)} != ref {len(ref)}"
    for i, (a, b) in enumerate(zip(trades, ref)):
        for k in ["entry_signal_date", "buy_date", "sell_date"]:
            assert a[k] == b[k], f"trade {i} {k}: {a[k]} != {b[k]}"
        assert abs(a["buy_price"] - b["buy_price"]) < 0.01, f"trade {i} buy_price"
        assert abs(a["sell_price"] - b["sell_price"]) < 0.01, f"trade {i} sell_price"
        assert abs(a["return_pct"] - b["return_pct"]) < 0.01, f"trade {i} return_pct"
    print(f"✅ REGRESSION PASS — all {len(ref)} trades identical to the canonical V27 artifact")
else:
    print("✅ REGRESSION PASS — headline metrics match canonical V27 (reference clone not present)")
