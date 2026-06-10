# Mechanism Round — report

Experiment: mech-002 · Completed: 2026-06-10 · Verdict: **NO RULE-SHAPE ROOM
(all five mechanism variants rejected); mechanism structure confirmed and
explains why the existing rules win; remaining room is in data, not rules.**

## Part A — what the mechanism actually looks like (descriptive, full history)

- **A1 event study** (79 non-overlapping pct≥0.93 burst starts, no confirm):
  excess over all-days baseline +0.69% by day 1, +1.2% by day 3, +1.3-1.7%
  days 5-7, +2.1-2.7% days 8-14 (90% CIs above zero almost everywhere).
  The response is *immediate AND persistent* — half the 5-day effect arrives
  in the first two days, but drift continues for ~2 weeks.
- **A2 dose-response**: inverted-U. 5-day forward returns by percentile
  bucket: 0.90-0.93 → +1.58% (n=35); **0.93-0.97 → +2.91%** (n=49, CI
  [+1.6,+4.7]); 0.97-1.00 → +1.67% (n=40, CI includes 0). The most extreme
  mint bursts are NOT the best — consistent with euphoria-top/treasury-op
  contamination at the tail.
- **A3 hold-day alpha** (31 V27-entry trades held 14d): mean cumulative
  +1.3% d1 → +3.2% d5 → +3.9% d10 → +6.3% d14; the curve never flattens
  inside 14 days. Cutting early loses money; decay exits that ride trends
  are the right family.
- **A4 response lag**: corr(USDT growth_t, BTC ret_{t+k}) peaks at k=0
  (+0.06 full sample, +0.12 on high-percentile days) with a secondary echo
  at k≈4 (+0.09 full, +0.23 high-pct) — initial impact plus a deployment
  wave a few days later, all inside the typical 5-9d hold.

## Part B — five mechanism-motivated rule changes vs the F3 baseline

Same nested walk-forward as open-search-001 (8 folds, immediate execution,
costs, 1x). Baseline F3: +174.8%, Sharpe 1.68, 8/8 folds positive.

| Variant | Mechanism story | OOS return | Sharpe | Wins vs baseline | Verdict |
|---|---|---|---|---|---|
| M2_norm (relative slope) | impact ∝ flow/depth | +44.4% | 0.54 | 2/8 | REJECT |
| M3_persist (2-bar confirm) | sustained issuance = real demand | +81.6% | 0.93 | 3/8 | NOTED* |
| M4_catchup (buy the lag) | repricing-lag hypothesis | +47.7% | 0.53 | 2/8 | NOTED* |
| M5_priceexit (exit on price stall) | absorption failure | +102.8% | 0.92 | 2/8 | REJECT |
| M6_band (cap extreme tail, **post-hoc**) | A2 inverted-U | +77.4% | 0.94 | 2/8 | REJECT (exploratory) |

*NOTED = the mapped Part-A direction exists but the effect is not
exploitable net of frequency loss and costs.

M6 was formed after seeing A2 and is labeled exploratory; it failed anyway,
so no forking-paths debt is carried forward.

## Why the existing rules win (the synthesis)

The current stack — dollar-slope event + 1y percentile filter + BTC 3d
confirm + flow-decay exits (slope-peak / kalman) with min 3 / max 14 holds —
already matches every measured mechanism property:

1. Immediate response → immediate execution at the signal snapshot (the
   biggest single lever, fixed earlier; 1h delay costs ~4%, 24h costs ~70%).
2. Persistent multi-day drift with a k≈4 echo → decay exits that hold while
   the flow lasts beat both fast fixed exits and price-side exits.
3. The percentile filter already does the depth normalization that explicit
   relative-slope signals attempt — and keeps dollar-magnitude information
   that normalization destroys.
4. Weak-tail events are handled by the confirm+exit stack better than by
   hard entry caps (M6) or persistence requirements (M3) that halve trade
   counts.

## 5-question analysis

1. Primary metric improved? No variant beat baseline — the round's value is
   evidence of *local optimality*, not a new config.
2. Slices? Variants lose broadly (2-3 wins of 8 folds each), not in one
   regime — the baseline's edge is not fold-specific.
3. Failure distribution? M3/M6 fail by frequency starvation (16-39 trades
   vs 32 with higher selectivity but fewer compounding events); M2/M4 fail
   by signal destruction; M5 fails by exiting late relative to flow decay.
4. Signal strength? Part A structure is solid (CIs clear zero); Part B
   margins are decisively negative (-72 to -130pp) — no ambiguity.
5. Best next move? The remaining interpretable room is **data, not rules**:
   exchange-deposit flows (mint→exchange is the actual deployment trigger),
   funding/OI absorption, order-book depth. On-chain mint feeds would also
   move execution earlier than the 00:00 snapshot. All require new data
   sources; predeclare before testing.
