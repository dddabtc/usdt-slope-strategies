# USDT Slope Strategies — Long-Only & Long/Short, Leverage-Tunable

**Language / 语言：** [English](README.md) · [简体中文](README.zh-CN.md)

Two deployable BTC strategies built on the USDT-liquidity signal from
[dddabtc/usdt-slope-research](https://github.com/dddabtc/usdt-slope-research):
the canonical V27 long side — **verified trade-for-trade here, then made honest
with real trading costs and a correct execution-timing model** — plus a
predeclared mirrored short side. Leverage is a single config parameter; an
optimal value is derived from a liquidation-aware Kelly/bootstrap sweep.

**Interactive results (same pages as the original repo):**

- [Dashboard](https://dddabtc.github.io/usdt-slope-strategies/)
- [Long-Only ⭐](https://dddabtc.github.io/usdt-slope-strategies/visualization_long_only.html)
- [Long/Short](https://dddabtc.github.io/usdt-slope-strategies/visualization_long_short.html)
- [Leverage Sweep → optimal leverage](https://dddabtc.github.io/usdt-slope-strategies/visualization_leverage.html)
- [Open Strategy Search → which strategy is honestly best](https://dddabtc.github.io/usdt-slope-strategies/visualization_open_search.html)
- [Mechanism Round → is there interpretable room left in the rules](https://dddabtc.github.io/usdt-slope-strategies/visualization_mechanism.html)

---

## 1. Verification of the original strategy

Canonical V27 was independently re-validated before anything was built.
All checks below deliberately use the **worst-case execution bound** (wait a
full day after each signal) — if the edge survives there, prompt execution
only improves it:

| Check | Result |
|-------|--------|
| Reproduce both frozen artifacts | ✅ identical: immediate mode 28 trades +2326.7%, one-day-delay mode 27 trades +320.6% (@2.5x, no costs) |
| Trade-by-trade diff vs original artifacts | ✅ all trades match to the cent in both modes (`tests_regression.py`) |
| Beats BTC buy & hold | ✅ +90.9% @1x worst-case bound vs +65.7% B&H, with only 10.5% time in market |
| Entry-timing value (placebo test) | ✅ beats **96.5%** of 2,000 random bull-regime entry sets using the same exit rule |
| Statistical significance | ✅ per-trade mean +2.60%, t-stat 2.18; bootstrap P(total ≤ 0) = 1.3% |
| Cost robustness | ✅ at 10 bps/side the worst-case-bound return is still +269.8% |
| Liquidation risk | ✅ zero close-based liquidations up to 5x in-sample |

## 2. Execution timing — why immediate execution is the right model

Every daily bar in this dataset is a **00:00 UTC snapshot**, not an
end-of-day aggregate: the bar labeled day T is fully known seconds after
00:00 UTC on day T. The underlying event — USDT issuance — is observable
on-chain in real time (Tether mints are public the moment they confirm),
and BTC trades 24/7 with no open/close gate. So a signal computed from the
day-T snapshot is executable within minutes of 00:00 UTC on day T.

The original repo treated "+1 day" as the only honest mode; that is the
right *worst-case bound* but the wrong default — it models an operator who
sees a liquidity surge and then waits 24 hours. Fills here are modeled at
`execution_delay_frac` of the way to the next snapshot. How the edge decays
with reaction time (long-only @1x, costs included):

| Reaction time | 0h (headline) | 1h | 3h | 6h | 12h | 24h (worst case) |
|---|---|---|---|---|---|---|
| Total return | +282.1% | +270.3% | +247.7% | +216.4% | +162.2% | +80.4% |

Reacting within the hour keeps ~96% of the immediate-execution result; most
of the alpha sits in the first hours after the snapshot — exactly what a
liquidity-inflow signal should look like. A 6-hourly cron still keeps ~77%.
Both bounds are pinned to the original repo's frozen artifacts by
`tests_regression.py`.

## 3. What was optimized / changed

1. **Execution model**: continuous-delay fills (`execution_delay_frac`),
   headline = immediate; the 24h bound remains available and tested.
2. **Real costs**: 5 bps/side fee+slippage and 3 bps/day long funding,
   charged on leveraged notional (configurable). The original engine traded free.
3. **Liquidation modeling**: close-based isolated-margin check
   (0.5% maintenance) plus a 10% intraday-wick stress column in the sweep.
4. **Short side** (long/short strategy): every long rule mirrored with **zero
   new fitted parameters** — lower Bollinger-band break / contraction regime,
   BTC 3d down-confirmation, slope percentile ≤ 7%, slope-trough exit — gated
   to bear regime (60d USDT slope < 0), the original research's V23 regime
   rule mirrored. Without the gate, mirrored shorts fire on bull-market USDT
   dips and lose to BTC drift (verified: ungated shorts ≈ net zero with
   double the drawdown).
5. **Leverage as a first-class parameter** with a derived optimum (below).

## 4. Headline results (immediate execution + costs)

Evaluation window 2021-09-27 → latest settled bar (refreshed daily by the
cron; parameters are frozen so extending the window adds data, not fitting).
As of 2026-06-10: BTC buy & hold +46.2%. Fees and funding included. Daily
Sharpe, daily max drawdown.

**Long-Only** (frozen V27 entry/exit):

| Leverage | Return | Ann. | Sharpe | Max DD | Trades |
|----------|--------|------|--------|--------|--------|
| 1.0x | +294% | +33.8% | 1.86 | −8.1% | 30 |
| 2.0x | +1,196% | +72.3% | 1.86 | −15.8% | 30 |
| **3.0x (recommended)** | **+3,606%** | **+115.4%** | **1.86** | **−23.2%** | **30 (WR 67%)** |

**Long/Short** (V27 longs + mirrored regime-gated shorts):

| Leverage | Return | Ann. | Sharpe | Max DD | Trades |
|----------|--------|------|--------|--------|--------|
| 1.0x | +315% | +35.3% | 1.72 | −9.6% | 36 (30L/6S) |
| 2.0x | +1,323% | +75.8% | 1.73 | −19.2% | 36 |
| **3.0x (recommended)** | **+4,114%** | **+121.4%** | **1.73** | **−28.7%** | **36 (WR 67%)** |

Frozen-window reference (→2026-03-19, comparable to the original repo):
long-only 3x +3,299%, long/short 3x +3,576%. Worst-case execution bound
(wait 24h, frozen window): long-only +76.8% @1x — still above buy & hold.
Note the protocol difference on shorts: with frozen params over the full
window the short book is net positive (the 2026 contraction trades), while
under per-fold re-selection (open search) shorts added nothing — both
statements hold in their own protocol.

The short book is small by construction — USDT market cap rarely contracts
(2022 was the only sustained episode). Its value is regime coverage: the
June 2022 capitulation short made +20.9% at 3x and the February 2026
contraction +1.7%, while the long side sat flat.

## 4b. Open strategy search — is this actually the best strategy?

An open competition (`experiments/open-search/`): 5 predeclared families,
~160 configs, **nested walk-forward** (configs selected only inside each
rolling 730d training window, evaluated once on the next unseen 180d window,
8 folds chained), plus a **surrogate-null audit** — the identical search
re-run on 12 fake USDT series to price how much return the search machinery
extracts from nothing.

| Family | OOS return | Sharpe | Max DD | Trades |
|---|---|---|---|---|
| Buy & hold BTC | **+189.1%** | 0.84 | −52.4% | — |
| **USDT long events (winner)** | +174.8% | **1.68** | **−8.1%** | 32 |
| Long/short mirror | +135.3% | 1.44 | −8.1% | 28 |
| Regime switch | +110.9% | 0.83 | −46.5% | 5 |
| BTC-only momentum (ablation) | +92.5% | 0.85 | −22.8% | 56 |

The long-only USDT liquidity-event family wins: **positive in all 8 unseen
windows** (including the 2025-26 bear where buy&hold drew down −52%), double
the Sharpe of every alternative, +82pp over the BTC-only ablation (the
liquidity layer is not repackaged momentum), and **ahead of all 12
surrogate-null searches** (≈3× the best null; p = 1/13 ≈ 0.077, the floor at
n=12). Shorts added nothing net of costs under honest selection. The frozen
V27 long-only config shipped here is one member of the validated pattern
(USDT slope event + top-decile percentile filter + BTC confirm + decay exit).

## 4c. Mechanism round — any interpretable room left in the rules?

Five mechanism-motivated rule changes (`experiments/mechanism-round/`), each
with an explicit causal story, tested under the same nested walk-forward:
depth normalization (−130pp vs baseline), 2-bar persistence (−93pp),
buy-the-lag (−127pp), price-side exits (−72pp), and capping the extreme
percentile tail (post-hoc, −97pp). **All five lose — the incumbent rules sit
at a local optimum.** The descriptive studies explain why: the issuance
response is immediate AND persistent (~2 weeks, with a k≈4 deployment echo),
dose-response is inverted-U (93-97th percentile is the sweet spot), and
per-trade alpha keeps accruing through day 14 — exactly the shape that
immediate entry + flow-decay exits already harvest. Remaining interpretable
room is in **data** (exchange-deposit flows, funding absorption, real-time
mint feeds), not rule geometry.

## 5. Optimal leverage

Method: per-trade expected log-growth (Kelly), 10,000-resample bootstrap of
the trade sequence, close-based liquidation plus 10% wick stress, full grid
0.5–6.0x re-backtested. See the [sweep page](https://dddabtc.github.io/usdt-slope-strategies/visualization_leverage.html).

| | Long-Only | Long/Short |
|---|---|---|
| Growth-optimal (full Kelly) | ≥6x (grid edge — n≈30 sample, do not trust) | ≥6x (same caveat) |
| **Recommended** | **3.0x** | **3.0x** |
| Conservative (DD ≤ ~20%) | 2.0x | 2.0x |
| Under the 24h worst-case bound | 3.0x | 2.5x |

Recommendation rule (predeclared): largest leverage with zero wick-stress
liquidations, bootstrap-median max DD ≥ −35%, bootstrap-5th-percentile total
return > 0, and ≤ half of growth-optimal. Under immediate execution the
in-sample stats are so strong that only the half-Kelly cap binds — which is
precisely why it exists: with ~30 trades and daily-close-only data, the
constraint protecting you is estimation-error discipline, not the backtest.

## 6. How to run

```bash
pip install -r requirements.txt

python tests_regression.py        # both execution modes == canonical V27 artifacts
python tests_verification.py      # placebo / significance / cost checks
python run_backtest.py            # both strategies at recommended leverage
python run_backtest.py --leverage 2.0   # any leverage you want
python run_leverage_sweep.py      # full grid + optimal-leverage page
python scripts/refresh_data.py    # pull latest CoinGecko data + regenerate pages
```

Every strategy parameter (leverage, execution delay, fees, funding,
maintenance margin, signal constants) lives in `StrategyConfig`
(`src/engine.py`).

## 7. Project structure

```
├── src/
│   ├── engine.py        # signals (long + mirrored short), exits, execution
│   │                    # model, costs, liquidation, daily MTM equity
│   ├── leverage.py      # Kelly / bootstrap / liquidation-aware sweep
│   ├── data.py          # CoinGecko fetch + *_full.csv history merge
│   └── visualize.py     # interactive result pages (original repo template)
├── run_backtest.py      # headline backtests + pages
├── run_leverage_sweep.py
├── tests_regression.py  # both modes must equal canonical V27 artifacts
├── tests_verification.py
├── scripts/refresh_data.py
├── data/                # daily CSVs (BTC price, USDT/USDC/DAI mcap)
├── experiments/         # artifacts: trades.json, equity.csv, ledger.jsonl
└── docs/                # GitHub Pages output
```

## 8. Caveats

- The headline assumes you execute within minutes of the 00:00 UTC snapshot
  (on-chain mint watching or a tight polling loop). The decay table in §2 is
  the honest price of being slower; the 24h bound is the floor.
- ~30 closed trades is a small sample; every number carries wide confidence
  bands (quantified by the bootstrap columns on the sweep page).
- Benchmark metrics are frozen at 2026-03-19; later trades are shown as a
  Post-benchmark overlay and never enter headline numbers.
- Daily closes cannot see intraday wicks; the wick-stress column is an
  approximation. Liquidation at high leverage can be worse in practice.
- Funding is modeled flat (3 bps/day longs, 0 shorts); real funding varies
  and occasionally inverts.
- Not financial advice.
