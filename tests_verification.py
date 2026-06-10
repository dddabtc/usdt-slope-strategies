#!/usr/bin/env python3
"""Independent effectiveness checks behind README section 1.

Runs on the frozen benchmark window with costs disabled (matching the
original engine) so the numbers are directly comparable:
  - buy & hold comparison
  - per-trade significance (t-stat, bootstrap)
  - random-entry placebo (same exit rule, same trade count, bull regime pool)
  - cost sensitivity
  - close-based liquidation scan up to 5x
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from src.data import load_full_data
from src.engine import (
    BENCHMARK_TEST_END, LONG_ONLY, RESEARCH_START, TRAIN_END, StrategyConfig,
    _exit_decision, backtest, generate_signals, slice_data_window,
)

rng = np.random.default_rng(42)

cfg = StrategyConfig(mode=LONG_ONLY, leverage=2.5,
                     fee_bps_per_side=0.0, long_funding_bps_per_day=0.0)
data = load_full_data()
result = backtest(data, cfg, TRAIN_END, BENCHMARK_TEST_END)
trades, merged = result["trades"], result["merged"]
tdf = pd.DataFrame(trades)
raw = tdf["raw_return_pct"].values / 100.0

print("=" * 70)
print("1) Buy & hold comparison (frozen window, no costs)")
bh = merged["price"].iloc[-1] / merged["price"].iloc[0] - 1
strat_1x = np.prod(1 + raw) - 1
print(f"   window {merged['date'].iloc[0].date()} -> {merged['date'].iloc[-1].date()}")
print(f"   BTC buy&hold {bh*100:+.1f}%   strategy@1x {strat_1x*100:+.1f}%   "
      f"time in market {tdf['hold_days'].sum()}/{len(merged)} days")

print("=" * 70)
print("2) Per-trade significance")
mean, sd = raw.mean(), raw.std(ddof=1)
t = mean / (sd / np.sqrt(len(raw)))
boots = np.array([np.prod(1 + rng.choice(raw, len(raw), replace=True)) - 1 for _ in range(10000)])
print(f"   mean {mean*100:+.2f}%/trade  t-stat {t:.2f}  n={len(raw)}  "
      f"bootstrap P(total<=0) {(boots <= 0).mean()*100:.1f}%")

print("=" * 70)
print("3) Random-entry placebo (same exit rule)")
bdata = slice_data_window(data, RESEARCH_START, BENCHMARK_TEST_END)
signals = generate_signals(bdata, cfg)
sig = signals[(signals["date"] >= TRAIN_END) & (signals["date"] <= BENCHMARK_TEST_END)]
df = sig.merge(bdata["btc"][["date", "price"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
elig = df.index[df["regime"] == 1].values
elig = elig[elig + 1 < len(df)]
finals = []
for _ in range(2000):
    entries = np.sort(rng.choice(elig, size=len(raw), replace=False))
    cap, last_exit = 1.0, -1
    for e in entries:
        if e <= last_exit:
            continue
        buy_idx = e + 1
        exit_idx, _ = _exit_decision(df, buy_idx, 1, cfg)
        sell_idx = min(exit_idx + 1, len(df) - 1)
        cap *= 1 + (df.iloc[sell_idx]["price"] - df.iloc[buy_idx]["price"]) / df.iloc[buy_idx]["price"]
        last_exit = sell_idx
    finals.append(cap - 1)
finals = np.array(finals)
print(f"   placebo median {np.median(finals)*100:+.1f}%  actual {strat_1x*100:+.1f}%  "
      f"-> beats {(finals < strat_1x).mean()*100:.1f}% of random entry sets")

print("=" * 70)
print("4) Cost sensitivity @2.5x")
for bps in [0, 5, 10, 20, 30]:
    net = np.maximum(raw * 2.5 - 2 * (bps / 10000.0) * 2.5, -0.99)
    print(f"   {bps:>2} bps/side: total {(np.prod(1+net)-1)*100:+8.1f}%")

print("=" * 70)
print("5) Close-based liquidation scan (0.5% maintenance)")
prices = df["price"].values
idx = {str(d)[:10]: i for i, d in enumerate(df["date"].astype(str))}
mae = np.array([prices[idx[t["buy_date"]]:idx[t["sell_date"]] + 1].min()
                / prices[idx[t["buy_date"]]] - 1 for t in trades])
for L in [2.0, 2.5, 3.0, 4.0, 5.0]:
    n_liq = int((1 + L * mae <= L * 0.005).sum())
    print(f"   {L:.1f}x: {n_liq} liquidations (worst excursion {mae.min()*100:.1f}%)")
