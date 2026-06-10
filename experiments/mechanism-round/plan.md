# Mechanism Round — plan

Written before results (2026-06-10). Verdict rules in design.md.

## Hypotheses (priors stated first)

- H-A1: the event-study profile rises for the first ~3-7 days then flattens;
  most of the response is in days 0-2 (consistent with the execution-delay
  decay table). Prior: likely.
- H-A2 (dose-response): higher slope percentile buckets show larger forward
  returns, monotonically. Prior: probable but noisy at the top bucket (few events).
- H-A3: hold-day alpha flattens between day 5 and day 10 → current min3/max14
  brackets the wave; no obvious room from re-timing exits. Prior: likely.
- H-A4: response concentrated at k=0..1 → no exploitable delayed-entry edge,
  catch-up entries (M4) unlikely to win. Prior: likely.
- H-M2 (normalization): relative-slope signals beat dollar-slope baseline —
  the strongest mechanism candidate (depth normalization), but the 1y
  percentile filter already absorbs much of the non-stationarity, so the
  margin may not clear +15pp. Prior: genuinely uncertain — the round's most
  interesting test.
- H-M3 (persistence): fewer, better trades but lower total return (frequency
  loss dominates at these trade counts). Prior: REJECT expected.
- H-M4 (catch-up): worse — fights the documented same-day momentum carry.
  Prior: REJECT expected.
- H-M5 (price-side exit): roughly ties flow-side exits (both proxy the same
  wave exhaustion); will not clear the margin. Prior: NOTED/REJECT.

If all four variants fail the margin while Part A shows the predicted
structure, the honest conclusion is: the mechanism is real, already mostly
harvested by the existing rules, and the remaining interpretable room is in
DATA (exchange flows, funding absorption) rather than rule shape.

## Success criteria

Part A four studies with CIs; Part B four families × 8 folds with verdicts
per design rules; report with per-hypothesis answers; page + push.
