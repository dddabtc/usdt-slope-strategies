# Open Strategy Search — plan

Written before the run (2026-06-10). Verdict rules predeclared in design.md.

## Hypotheses (stated before seeing results)

- H1: F3 (USDT long events) beats F0 buy&hold on chained OOS return AND beats
  the surrogate-null 95th percentile — i.e. the liquidity signal is real, not
  search luck. Prior: likely true (the frozen-V27 verification already showed
  placebo-beating behavior, but that was config-level, not search-level).
- H2: F2 (BTC-only momentum, no USDT) captures a meaningful share of F3's
  OOS return. If F2 ≈ F3, the "USDT liquidity" framing is mostly repackaged
  BTC momentum; if F3 >> F2, the liquidity layer adds real timing value.
  Prior: genuinely uncertain — this is the most informative comparison.
- H3: F4 (long/short) ends within ±15% of F3's OOS return (shorts are rare,
  small net add). Prior: likely true.
- H4: F1 (regime switching) underperforms F3 on Sharpe but may rival raw
  return with far higher exposure. Prior: likely true.
- H5: Within F3, folds will NOT keep selecting one golden config; selection
  will be unstable across folds (small-sample reality). The honest "best
  strategy" claim must therefore be at the family/pattern level.

## Success criteria

- Search completes all families + ≥10 surrogates with resumable ledger.
- Every family has ≥6 evaluable folds.
- A leaderboard with chained OOS metrics, per-fold selections, and a
  surrogate p-value for F3.

## Expected cost

~160 configs × 8 folds train evals + 12 surrogate × 115-config searches;
single process, target < 60 min wall clock.

## Deliverables

ledger.jsonl, results/leaderboard.json, results/per_fold.json, report.md
(5-question analysis), README section + docs page, push.
