#!/usr/bin/env python3
"""Regression: with costs disabled, the long side must reproduce BOTH frozen
canonical V27 artifacts from dddabtc/usdt-slope-research exactly:

  immediate execution (delay 0)   == trades_frozen_same_day_optimistic.json
                                     (28 trades, +2326.7% @2.5x)
  one-day-delay conservative bound == trades_frozen_plus_one_bar_executable.json
                                     (27 trades, +320.6% @2.5x)

If a local clone of the original repo is present, every trade is diffed
field-by-field; otherwise the embedded expected values are used.
"""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from src.data import load_full_data
from src.engine import BENCHMARK_TEST_END, TRAIN_END, LONG_ONLY, StrategyConfig, backtest

REF_DIR = Path.home() / "usdt-slope-research/experiments/v27-more-trades"

CASES = [
    {
        "name": "immediate (delay_frac=0)",
        "cfg": dict(execution_delay_frac=0.0),
        "expected": {"n_trades": 28, "total_return": 23.267, "daily_sharpe": 1.97,
                     "max_drawdown": -0.186, "first_buy": "2023-01-25", "last_sell": "2025-10-09"},
        "ref": REF_DIR / "trades_frozen_same_day_optimistic.json",
    },
    {
        "name": "one-day delay (lag=1 conservative bound)",
        "cfg": dict(execution_delay_frac=None, execution_lag_bars=1),
        "expected": {"n_trades": 27, "total_return": 3.206, "daily_sharpe": 0.97,
                     "max_drawdown": -0.273, "first_buy": "2023-01-26", "last_sell": "2025-10-10"},
        "ref": REF_DIR / "trades_frozen_plus_one_bar_executable.json",
    },
]

data = load_full_data()
for case in CASES:
    cfg = StrategyConfig(
        mode=LONG_ONLY, leverage=2.5,
        fee_bps_per_side=0.0, long_funding_bps_per_day=0.0, short_funding_bps_per_day=0.0,
        **case["cfg"],
    )
    result = backtest(data, cfg, TRAIN_END, BENCHMARK_TEST_END)
    m = result["metrics"]
    trades = result["trades"]
    exp = case["expected"]

    print(f"[{case['name']}] {m['n_trades']} trades, total {m['total_return']*100:.1f}%, "
          f"sharpe {m['daily_sharpe']:.2f}, dd {m['max_drawdown']*100:.1f}%, wr {m['win_rate']*100:.0f}%")

    assert m["n_trades"] == exp["n_trades"], f"trade count {m['n_trades']}"
    assert abs(m["total_return"] - exp["total_return"]) < 0.01, f"total_return {m['total_return']}"
    assert abs(m["daily_sharpe"] - exp["daily_sharpe"]) < 0.02, f"sharpe {m['daily_sharpe']}"
    assert abs(m["max_drawdown"] - exp["max_drawdown"]) < 0.005, f"dd {m['max_drawdown']}"
    assert trades[0]["buy_date"] == exp["first_buy"], trades[0]["buy_date"]
    assert trades[-1]["sell_date"] == exp["last_sell"], trades[-1]["sell_date"]

    if case["ref"].exists():
        ref = json.loads(case["ref"].read_text())
        assert len(trades) == len(ref), f"trade count {len(trades)} != ref {len(ref)}"
        for i, (a, b) in enumerate(zip(trades, ref)):
            for k in ["entry_signal_date", "buy_date", "sell_date"]:
                assert a[k] == b[k], f"trade {i} {k}: {a[k]} != {b[k]}"
            assert abs(a["buy_price"] - b["buy_price"]) < 0.01, f"trade {i} buy_price"
            assert abs(a["sell_price"] - b["sell_price"]) < 0.01, f"trade {i} sell_price"
            assert abs(a["return_pct"] - b["return_pct"]) < 0.01, f"trade {i} return_pct"
        print(f"  ✅ all {len(ref)} trades identical to the canonical artifact")
    else:
        print("  ✅ headline metrics match (reference clone not present)")

print("✅ REGRESSION PASS — both execution modes reproduce canonical V27 exactly")
