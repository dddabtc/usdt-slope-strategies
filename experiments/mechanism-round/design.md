# Mechanism Round — design

Experiment ID: mech-002 · Started: 2026-06-10 · Status: see current_status.md

## Question

Is there *mechanism-supported, interpretable* optimization room left in the
entries and exits — beyond what the open search (open-search-001) already
priced? Every candidate below is motivated by an explicit causal story about
how minted USDT becomes BTC price impact; nothing is a free-form parameter
sweep. Verdicts are per-hypothesis.

## Causal chain under test

fiat → USDT mint (on-chain, t≈0) → exchange deposit → spot deployment over
hours/days → price impact ∝ flow / market depth → wave exhausts → drift ends.

## Part A — descriptive mechanism studies (no selection, full history,

labeled as explanatory)

- A1 **Event study**: non-overlapping issuance-event starts (slope_pct ≥ 0.93,
  first day of a burst, no BTC confirm) → mean cumulative BTC forward return
  days +0..+14 with bootstrap CI, vs all-days baseline. Shape answers: how
  big, how fast, how long the response is.
- A2 **Dose-response**: bucket events by slope_pct (0.90–0.93 / 0.93–0.97 /
  0.97–1.00) → mean forward 5d return per bucket. Mechanism predicts a
  monotone curve (impact ∝ flow size).
- A3 **Hold-day alpha profile**: across baseline F3 trades, average
  cumulative trade return by day-in-trade 1..14. Where the curve flattens is
  the deployment-wave length; current min-hold 3 / max-hold 14 should bracket it.
- A4 **Response lag**: corr(Δlog USDT mcap_t, BTC return_{t+k}) for k=0..10.
  Mass at small k justifies immediate execution; mass at k≥2 would suggest
  delayed entries — directly testable against M3/M4.

## Part B — predeclared mechanism variants (nested walk-forward, same

protocol as open-search-001: 730d/180d × 8 folds, immediate, costs, 1x)

Each variant is a *family*: the F3 baseline grid transformed by exactly one
mechanism change, selection per fold on train only, chained OOS vs F3
baseline (+174.8%, Sharpe 1.68, 8/8 folds positive).

| Family | Change (one variable) | Mechanism story |
|---|---|---|
| M2_norm | all slope signals built on relative slope = 3d OLS slope of log(mcap) (growth rate), percentile/bb/zscore/regime on that series | impact ∝ flow relative to depth; dollar slope is non-stationary across eras |
| M3_persist | entries require the event condition true on 2 consecutive bars | sustained issuance = real demand wave; single-day spikes contain treasury noise |
| M4_catchup | BTC confirm inverted: enter only if BTC 3d return ≤ 0 | repricing-lag hypothesis: buy the not-yet-moved, not the already-moved |
| M5_priceexit | exits replaced by price-side rule: exit when BTC 3d return < 0 (min hold 3, max 14) | exit on absorption failure (price stops responding) instead of flow decay |

Grid per family ≈ the F3 core entries × pct filters × exits (M5 replaces the
exit dimension), ~30–90 configs each — same order as F3 baseline so the
comparison is selection-fair.

## Verdict rules (predeclared)

- A variant is **ACCEPT (real optimization room)** only if ALL of:
  chained OOS beats F3 baseline by > +15pp total return OR > +0.20 daily
  Sharpe; per-fold OOS wins ≥ 5/8 vs baseline; and the Part-A study for its
  mechanism is directionally consistent.
- **NOTED (structure, not exploitable)** if Part A shows clear structure but
  Part B fails the margin (costs/variance eat it).
- **REJECT (no room)** otherwise.
- Four variants = four extra comparisons; the +15pp/+0.2 margin exists
  precisely to price that multiple-comparison layer.

## Out of scope (no data in repo)

Exchange-deposit flows, funding/OI absorption, order-book depth — listed as
future mechanisms; stubs only.
