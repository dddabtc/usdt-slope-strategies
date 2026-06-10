#!/usr/bin/env python3
"""EXPLORATORY (post-hoc) — pct band filter test.

The A2 dose-response found an inverted-U: events in the 0.93-0.97 percentile
band outperform the extreme 0.97-1.00 tail. This script runs the same nested
walk-forward with the pct filter replaced by a band [0.93, 0.97).

This hypothesis was formed AFTER seeing A2, so the result is an exploratory
estimate only — it must not be shipped or claimed as validated until it
survives data that did not exist when the hypothesis was written.
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "experiments" / "open-search"))

import search as osearch  # noqa: E402
from src.data import load_full_data  # noqa: E402

BAND_LO, BAND_HI = 0.93, 0.97


def band_configs():
    out = []
    for entry in osearch.F3_CORE_ENTRIES:
        for exit_name in osearch.F3_EXITS:
            out.append({
                "family": "F3",  # reuse F3 engine config (long-only)
                "config_id": f"M6_{entry}_band{BAND_LO:.2f}-{BAND_HI:.2f}_{exit_name}",
                "entry_kind": entry,
                "pct_filter": None,   # handled by the band below
                "exit": exit_name,
                "confirm": True,
                "band": True,
            })
    return out


def apply_band(cache, params):
    df = osearch.apply_config_signals(cache, params)
    pct = df["slope_pct"].values
    band = (pct >= BAND_LO) & (pct < BAND_HI)
    df["signal"] = (df["signal"].values.astype(bool) & band).astype(int)
    return df


def backtest(cache, params, start, end):
    config = osearch.engine_config(params)
    signals = apply_band(cache, params)
    trades, merged = osearch.collect_custom_trades(signals, params, config, start, end)
    from src.engine import daily_equity_curve, equity_metrics, trade_metrics
    equity = daily_equity_curve(merged, trades, config)
    metrics = {**equity_metrics(equity, config.initial_capital), **trade_metrics(trades)}
    return {"metrics": metrics, "equity": equity, "trades": trades}


def main():
    data = load_full_data()
    btc, usdt, actual_end = osearch.common_data(data)
    cache = osearch.build_signal_cache(btc, usdt)
    folds = osearch.make_folds(actual_end)
    configs = band_configs()

    baseline = json.loads((PROJECT / "experiments/open-search/results/leaderboard.json").read_text())
    base_folds = {f["fold"]: f["oos_metrics"]["total_return"]
                  for f in baseline["families"]["F3"]["folds"]}

    returns, trades_all, rows = [], [], []
    wins = 0
    for fold in folds:
        best, best_score = None, None
        for p in configs:
            r = backtest(cache, p, fold.train_start, fold.train_end)
            s = osearch.selection_score(r["metrics"])
            if best_score is None or s > best_score:
                best_score, best = s, p
        oos = backtest(cache, best, fold.test_start, fold.test_end)
        m = oos["metrics"]
        rows.append({"fold": fold.fold, "selected": best["config_id"],
                     "oos_return": m["total_return"], "n_trades": m["n_trades"],
                     "baseline_oos": base_folds.get(fold.fold)})
        if m["total_return"] > base_folds.get(fold.fold, -9):
            wins += 1
        returns.extend(osearch.daily_return_records(oos["equity"]))
        trades_all.extend(oos["trades"])
        print(f"fold {fold.fold}: {best['config_id'][:46]:46} oos {m['total_return']*100:+6.1f}% "
              f"(baseline {base_folds.get(fold.fold, 0)*100:+6.1f}%) trades {m['n_trades']}")

    agg = osearch.aggregate_from_returns(returns, trades_all)
    out = {
        "label": "EXPLORATORY_POST_HOC — do not ship without fresh-data confirmation",
        "band": [BAND_LO, BAND_HI],
        "aggregate": agg,
        "folds": rows,
        "folds_beating_baseline": wins,
        "baseline_f3_total_return": baseline["families"]["F3"]["aggregate"]["total_return"],
    }
    (BASE / "results" / "exploratory_band.json").write_text(json.dumps(out, indent=2))
    print(f"\nband aggregate: ret {agg['total_return']*100:+.1f}%  sharpe {agg['daily_sharpe']:+.2f} "
          f"dd {agg['max_drawdown']*100:.1f}%  trades {agg['n_trades']}  beats baseline {wins}/8 folds")
    print(f"baseline F3:    ret {out['baseline_f3_total_return']*100:+.1f}%")


if __name__ == "__main__":
    main()
