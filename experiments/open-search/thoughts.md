# Open Strategy Search — thoughts (decision log)

Written as work progressed.

## Why nested WF + surrogate null instead of plain methodology

practical-autoresearch covers process discipline (baseline-first, telescope,
ledger, resumability) but not financial selection bias. With ~160 candidates
the full-sample winner is partly luck by construction. Two additions:
nested walk-forward (selection only ever sees its training sub-window) and a
surrogate-null audit (the same search run on 12 fake USDT series prices how
much return the machinery extracts when the liquidity signal is fake by
construction). Both were predeclared in design.md before any result existed.

## Division of labor

Design/plan/verdict rules: Claude. Implementation + validations: codex
(gpt-5.5, xhigh) from a written spec; reviewed line-by-line before the full
run. codex validations: regression intact, smoke pass, resume pass, surrogate
sanity, fold-1 full-grid spot check (115 configs).

## Mid-run notes

- F0 path crashed on first full launch (`position` column missing in the
  buy&hold equity frame) — smoke only covered F3, exactly the blind spot
  §15 incremental validation warns about. One-line fix, relaunched clean.
- Surrogate seed 1 scored +52.9% chained OOS on a FAKE USDT series — strong
  reminder that "searched +X%" is not evidence by itself: BTC confirmation +
  two bull legs carry a lot. The real F3 must clear this distribution, and
  the F2 (BTC-only) family is the second, structural control for the same
  effect.
- accel entry uses diff-of-slope z-score (codex) instead of slope-of-slope —
  accepted: it is a predeclared family member, not a replication of the
  original `acceleration` algorithm.
- Boundary trades at fold edges get force-closed at the last bar with fees —
  identical treatment across families, so comparisons stay fair.

## 5-question analysis

(in report.md after results)
