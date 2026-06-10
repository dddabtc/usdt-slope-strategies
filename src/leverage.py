"""Leverage analysis: liquidation-aware grid sweep + growth-optimal (Kelly).

Methodology
-----------
1. Run the full backtest once per leverage point (costs and close-based
   isolated-margin liquidation included) -> realized return / max drawdown.
2. Per-trade net returns at each leverage -> empirical expected log-growth
   E[log(1 + r_net)]; the leverage maximizing it is the growth-optimal
   (full Kelly) point for the observed trade distribution.
3. Bootstrap the trade sequence (10k resamples) -> dispersion of total
   return and of max drawdown at each leverage; estimation error with only
   ~30 trades is large, which is exactly why full Kelly is not the
   recommendation.
4. A wick-stress column re-checks liquidation assuming an intraday spike
   beyond the daily closes (default 10%): daily data cannot see wicks, so
   any leverage that only survives wick-free closes is flagged.

Recommended leverage = the largest grid point that simultaneously:
  - survives the wick stress with zero liquidations,
  - keeps bootstrap-median max drawdown <= 35%,
  - keeps bootstrap-P5 total return > 0 (95% of resamples profitable),
  - stays at or below half of the growth-optimal point (half-Kelly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine import (
    BENCHMARK_TEST_END, RESEARCH_START, TRAIN_END, StrategyConfig, backtest,
    collect_trades, daily_equity_curve, equity_metrics, generate_signals,
    replace, slice_data_window, trade_metrics,
)

LEVERAGE_GRID = [round(0.5 + 0.25 * i, 2) for i in range(23)]  # 0.5 .. 6.0


def _trade_arrays(result: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """raw returns, sides, hold days, and worst close-based excursion per trade."""
    trades = [t for t in result["trades"] if t.get("status") != "OPEN"]
    df = result["merged"]
    idx_by_date = {str(d)[:10]: i for i, d in enumerate(df["date"].astype(str))}
    raw, sides, holds, mae = [], [], [], []
    for t in trades:
        bi, si = idx_by_date[t["buy_date"]], idx_by_date[t["sell_date"]]
        seg = df["price"].values[bi:si + 1]
        side = 1 if t["side"] == "LONG" else -1
        raw.append(t["raw_return_pct"] / 100.0)
        sides.append(side)
        holds.append(t["hold_days"])
        if side > 0:
            mae.append(seg.min() / seg[0] - 1.0)   # worst drop while long
        else:
            mae.append(seg.max() / seg[0] - 1.0)   # worst rally while short
    return np.array(raw), np.array(sides), np.array(holds), np.array(mae)


def _net_returns(raw, sides, holds, mae, cfg: StrategyConfig, wick_stress: float = 0.0):
    """Vectorized per-trade net returns at cfg.leverage, with liquidation."""
    L = cfg.leverage
    fees = 2.0 * (cfg.fee_bps_per_side / 10_000.0) * L
    fund_long = (cfg.long_funding_bps_per_day / 10_000.0) * L
    fund_short = (cfg.short_funding_bps_per_day / 10_000.0) * L
    funding = np.where(sides > 0, fund_long, fund_short) * holds
    net = np.maximum(sides * raw * L - fees - funding, -0.99)
    if L > 1.0:
        adverse = np.where(sides > 0, (1.0 - wick_stress) * (1 + mae) - 1.0,
                           (1.0 + wick_stress) * (1 + mae) - 1.0)
        equity_ratio = 1.0 + sides * adverse * L
        liq = equity_ratio <= L * cfg.maintenance_margin
        net = np.where(liq, -0.99, net)
    else:
        liq = np.zeros(len(net), dtype=bool)
    return net, liq


def sweep(data: dict, base_cfg: StrategyConfig, grid=None, n_boot: int = 10_000,
          wick_stress: float = 0.10, seed: int = 42,
          eval_end: pd.Timestamp = BENCHMARK_TEST_END) -> pd.DataFrame:
    grid = grid or LEVERAGE_GRID
    rng = np.random.default_rng(seed)
    # signals are leverage-independent: build once, reuse across the grid
    bdata = slice_data_window(data, RESEARCH_START, eval_end)
    signals = generate_signals(bdata, base_cfg)
    base = backtest(data, replace(base_cfg, leverage=1.0), TRAIN_END, eval_end)
    raw, sides, holds, mae = _trade_arrays(base)
    n = len(raw)
    boot_idx = rng.integers(0, n, size=(n_boot, n))

    rows = []
    for L in grid:
        cfg = replace(base_cfg, leverage=L)
        trades, merged = collect_trades(signals, bdata["btc"], cfg, TRAIN_END, eval_end)
        equity = daily_equity_curve(merged, trades, cfg)
        m = {**equity_metrics(equity, cfg.initial_capital), **trade_metrics(trades)}

        net, liq = _net_returns(raw, sides, holds, mae, cfg)
        net_stress, liq_stress = _net_returns(raw, sides, holds, mae, cfg, wick_stress)
        growth = float(np.mean(np.log1p(net)))

        b = net[boot_idx]
        totals = np.prod(1 + b, axis=1) - 1
        eq = np.cumprod(1 + b, axis=1)
        peaks = np.maximum.accumulate(eq, axis=1)
        dds = ((eq / peaks) - 1).min(axis=1)

        rows.append({
            "leverage": L,
            "total_return": m["total_return"],
            "ann_return": m["ann_return"],
            "daily_sharpe": m["daily_sharpe"],
            "max_drawdown": m["max_drawdown"],
            "n_liquidated": int(liq.sum()),
            "n_liquidated_wick10": int(liq_stress.sum()),
            "exp_log_growth": round(growth, 5),
            "boot_p5_return": round(float(np.percentile(totals, 5)), 4),
            "boot_median_return": round(float(np.percentile(totals, 50)), 4),
            "boot_p95_return": round(float(np.percentile(totals, 95)), 4),
            "boot_median_maxdd": round(float(np.percentile(dds, 50)), 4),
            "boot_p95_maxdd": round(float(np.percentile(dds, 5)), 4),  # 5th pct = worse tail
            "p_loss": round(float((totals <= 0).mean()), 4),
        })
    return pd.DataFrame(rows)


def recommend(sweep_df: pd.DataFrame, dd_cap: float = -0.35) -> dict:
    """Pick recommended leverage per the predeclared rule in the module doc."""
    df = sweep_df.copy()
    kelly_row = df.loc[df["exp_log_growth"].idxmax()]
    kelly = float(kelly_row["leverage"])
    ok = df[
        (df["n_liquidated_wick10"] == 0)
        & (df["boot_median_maxdd"] >= dd_cap)
        & (df["boot_p5_return"] > 0)
        & (df["leverage"] <= kelly / 2 + 1e-9)
    ]
    rec = float(ok["leverage"].max()) if not ok.empty else 1.0
    return {
        "growth_optimal_leverage": kelly,
        "recommended_leverage": rec,
        "rule": "max leverage with zero wick-stress liquidations, bootstrap-median "
                f"maxDD >= {dd_cap:.0%}, bootstrap-P5 return > 0, and <= half of growth-optimal",
    }
