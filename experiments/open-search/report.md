# Open Strategy Search — report

Experiment: open-search-001 · Completed: 2026-06-10 · Verdict: **ACCEPT**

## Setup recap

5 predeclared families, ~160 configs, nested walk-forward (730d train /
180d test, 8 folds, 2022-09-27 → 2026-06-09), immediate execution, 1x,
fees 5 bps/side + 3 bps/day long funding. Selection only ever sees its
training sub-window. 12 surrogate-USDT re-runs of the full F3 search as the
multiple-testing null. Implementation: codex (gpt-5.5) from written spec,
reviewed line-by-line; one F0 bug found and fixed before the full run
(`position` column, smoke had only covered F3).

## Leaderboard (chained OOS across all 8 folds)

| Family | Total return | Daily Sharpe | Max DD | Trades | Win rate |
|---|---|---|---|---|---|
| F0 buy & hold | **+189.1%** | 0.84 | −52.4% | — | — |
| **F3 USDT long events** | +174.8% | **1.68** | **−8.1%** | 32 | 68.8% |
| F4 long/short mirror | +135.3% | 1.44 | −8.1% | 28 | 71.4% |
| F1 regime switch | +110.9% | 0.83 | −46.5% | 5 | 60% |
| F2 BTC-only momentum | +92.5% | 0.85 | −22.8% | 56 | 51.8% |

F3 per-fold OOS: +26.7, +8.4, +16.8, +10.6, +28.4, +10.6, +4.5, +4.3 —
**8 of 8 positive**, including folds 7–8 (2025-09 → 2026-06 bear, BTC
≈ −45% peak-to-trough, where buy & hold took its −52% drawdown).

Surrogate null (best honest score per fake-USDT search):
−45.4, −40.1, −39.8, −33.4, −7.0, +0.1, +9.8, +32.3, +40.6, +52.9, +59.9,
+60.0 (%). Real F3 +174.8% beats **all 12** (≈3× the best null);
p = 1/13 ≈ 0.077, the floor attainable with n=12.

## 5-question analysis

1. **Did the primary metric improve?** The question was selection-honest
   ranking, not improvement. Answer: F3 is the best strategy family —
   highest Sharpe (1.68, double everything else), 1/6 of buy&hold's
   drawdown, ~92% of its raw return, ~11% time in market.
2. **Which slices?** F3 is the only family positive in every fold. B&H wins
   only the raw-return slice and only because of the two bull legs; it gave
   back −52% in the bear slice where F3 stayed positive. F4's single
   negative fold (7) is the only fold where its shorts actually traded
   (2 shorts, both net losers).
3. **What changed in the failure distribution?** The naive-mirror short
   thesis weakens further: under honest per-fold selection the short book
   almost never activates (bear gate requires USDT contraction, which only
   began 2026-02), and when it did it lost. Long/short ≠ wrong, but on this
   data shorts are insurance that has not yet paid net of costs.
4. **How strong is the signal?** Strong by three independent tests:
   beats all 12 surrogates (limited-n p=0.077 but ~3× margin over the best
   null); beats the BTC-only ablation by +82pp total return at double the
   Sharpe (the liquidity layer is not repackaged price momentum); and
   per-fold consistency 8/8 vs the null searches' typical 3-5/8.
5. **Best next move?** (a) more surrogates (≥19) if a p<0.05 stamp is ever
   needed; (b) keep shorts as optional insurance, default off; (c) the
   deployable config stays the frozen V27-pattern long-only — the per-fold
   winners rotate within one pattern family (USDT slope event + 90-93pct
   filter + BTC confirm + decay exit), and chasing the per-fold winner adds
   selection variance for no demonstrated OOS gain.

## Verdict

**ACCEPT: the best strategy is the long-only USDT liquidity-event family
(F3) — entry on a USDT slope event (regime-acceleration / bollinger break /
top-decile percentile) in the top ~7-10% of its 1-year distribution with
BTC 3d confirmation, exit on momentum decay (slope-peak / kalman-velocity /
trail), no shorts.** It is the only family that is simultaneously:
positive in all 8 OOS folds, double the Sharpe of every alternative, ~6×
less drawdown than buy & hold, ahead of its price-only ablation, and ahead
of every surrogate-null search.

Caveats: 32 chained OOS trades is still a small sample; the surrogate p is
floored at 0.077 by n=12; fold selection rotates configs within the pattern
(do not over-trust any single parameterization, including frozen V27's).
