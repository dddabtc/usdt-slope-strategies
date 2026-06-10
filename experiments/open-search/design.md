# Open Strategy Search — design

Experiment ID: open-search-001 · Started: 2026-06-10 · Status: see current_status.md

## Question

What is the *best* strategy on this data, selected honestly? Not "does frozen
V27 hold up" (already verified) but an open competition across strategy
families, with the selection machinery itself audited for luck.

## Methodology

Follows practical-autoresearch discipline (baseline-first, telescope
smoke→full, durable ledger/log, resumable, 5-question analysis) plus two
additions the methodology does not cover, both required for honest *strategy
selection* on financial data:

1. **Nested walk-forward selection.** Any parameter or config choice happens
   only inside a rolling 730d training window; the chosen config is then
   evaluated exactly once on the following 180d out-of-sample window. The
   chained OOS equity across all folds is the family's honest score. Nothing
   is ever selected using full-sample results.
2. **Surrogate-null audit (multiple-testing control).** The full search is
   re-run on 12 surrogate USDT series (circular shifts ≥180d, preserving the
   series' own autocorrelation/trend but destroying any real USDT→BTC link;
   BTC stays real). The distribution of "best honest OOS score" under the
   null prices the selection luck; the real result must clear it.

## Fixed evaluation protocol (identical for every candidate)

- Data: CoinGecko daily 00:00 UTC snapshots, signals warm up from 2020-09-27.
- Evaluation range: 2021-09-27 → 2026-06-09 (all available, incl. the 2026 bear).
- Folds: train 730d / test 180d, sliding 180d → 9 folds; fold-1 test starts 2022-09-27.
- Execution: immediate (fill at the signal snapshot), single position,
  costs 5 bps/side + 3 bps/day long funding.
- Leverage fixed at 1x for all selection and family comparison (leverage is a
  separate post-hoc scaling decision; it multiplies but does not reorder).
- Train selection score: total return, tiebreak daily Sharpe (predeclared,
  same rule the original repo used).

## Families and grids (predeclared)

| Family | Idea | Grid size |
|---|---|---|
| F0 buy_hold | hold BTC | 1 (reference, no selection) |
| F1 regime_switch | hold BTC while USDT slope(60d-ish) > 0 | 3 (window 30/60/90) |
| F2 btc_momentum | BTC-only: N-day breakout + 3d momentum (no USDT at all — ablation) | 12 (breakout 10/20/55 × exits fixed3/5/7/trail10) |
| F3 usdt_long | USDT event entries: bb / zscore / regime / or_bb_regime / accel / consec ×
pct filter {none,0.90,0.93} × exits {fixed3,fixed5,slope_peak,trail10,kalman75} + pct-event entries + 2 no-confirm ablations | ~115 |
| F4 long_short | F3 core entries + predeclared mirrored shorts (bear-regime-gated) | 30 |

Exits: fixed h∈{3,5,7}; slope_peak (min3/max14/20% drop, conflict no-op);
trailing 10% on closes (max14); kalman_vel decay to 75% of entry (min3/max14)
— the V30 claim, now under honest selection. Shorts mirror each rule.

## Verdict rules (predeclared)

- Family leaderboard = chained OOS metrics. Primary: total OOS return;
  context: daily Sharpe, maxDD, n_trades, per-fold consistency.
- F3 (and F4 if it wins) must beat the surrogate-null 95th percentile of the
  same searched score to claim the USDT signal is real (p < 0.05).
- n_trades < 30 on the chained OOS ⇒ flag small-sample, temper claims.
- "Best strategy" = best family per the above + the config pattern the folds
  actually kept selecting (selection stability matters, not one golden config).
