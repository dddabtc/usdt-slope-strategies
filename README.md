# USDT Slope Strategies — Long-Only & Long/Short, Leverage-Tunable

**Language / 语言：** [English](README.md) · [简体中文](README.zh-CN.md)

Two deployable BTC strategies built on the USDT-liquidity signal from
[dddabtc/usdt-slope-research](https://github.com/dddabtc/usdt-slope-research):
the canonical V27 long side — **verified bit-for-bit here, then made honest with
real trading costs** — plus a predeclared mirrored short side. Leverage is a
single config parameter; an optimal value is derived from a liquidation-aware
Kelly/bootstrap sweep.

**Interactive results (same pages as the original repo):**

- [Dashboard](https://dddabtc.github.io/usdt-slope-strategies/)
- [Long-Only ⭐](https://dddabtc.github.io/usdt-slope-strategies/visualization_long_only.html)
- [Long/Short](https://dddabtc.github.io/usdt-slope-strategies/visualization_long_short.html)
- [Leverage Sweep → optimal leverage](https://dddabtc.github.io/usdt-slope-strategies/visualization_leverage.html)

---

## 1. Verification of the original strategy

Before building anything, canonical V27 was independently re-validated
(frozen data, frozen parameters, honest +1-day execution lag):

| Check | Result |
|-------|--------|
| Reproduce frozen benchmark | ✅ identical: 27 trades, +320.6% @2.5x no-cost, Sharpe 0.97, DD −27.3% |
| Trade-by-trade diff vs original artifact | ✅ all 27 trades match to the cent (`tests_regression.py`) |
| Beats BTC buy & hold | ✅ +90.9% @1x (no costs) vs +65.7% B&H, with only 10.5% time in market |
| Entry-timing value (placebo test) | ✅ beats **96.5%** of 2,000 random bull-regime entry sets using the same exit rule |
| Statistical significance | ✅ per-trade mean +2.60%, t-stat 2.18; bootstrap P(total ≤ 0) = 1.3% |
| Cost robustness | ✅ at 10 bps/side the 2.5x return is still +269.8% (vs +320.9% free) |
| Liquidation risk | ✅ zero close-based liquidations up to 5x in-sample |

Important context: the original README headline (+2152%, Sharpe 5.57) is the
**same-day-execution legacy mode** — the original repo itself froze the honest
number at +320.6% (`execution_lag_bars=1`). Everything in this repo uses the
honest mode only, and adds costs on top.

## 2. What was optimized / changed

1. **Real costs**: 5 bps/side fee+slippage and 3 bps/day long funding, charged
   on leveraged notional (both configurable). The original engine traded free.
2. **Liquidation modeling**: close-based isolated-margin check
   (0.5% maintenance) plus a 10% intraday-wick stress column in the sweep —
   daily data cannot see wicks, so leverage that only survives wick-free
   closes is flagged as unsafe.
3. **Short side** (long/short strategy): every long rule mirrored with **zero
   new fitted parameters** — lower Bollinger-band break / contraction regime,
   BTC 3d down-confirmation, slope percentile ≤ 7%, slope-trough exit — gated
   to bear regime (60d USDT slope < 0), which is the original research's V23
   regime rule mirrored. Without the gate, mirrored shorts fire on bull-market
   USDT dips and lose to BTC drift (verified: 15 ungated shorts ≈ net zero
   with double the drawdown; gated: 4 shorts, positive).
4. **Leverage as a first-class parameter** with a derived optimum (below).

## 3. Headline results (honest execution + costs)

Frozen benchmark window 2021-09-27 → 2026-03-19, BTC buy & hold +65.7%.
Signals execute one day after they appear. Fees and funding included.

**Long-Only** (frozen V27 entry/exit):

| Leverage | Return | Ann. | Sharpe | Max DD | Win Rate | Trades |
|----------|--------|------|--------|--------|----------|--------|
| 1.0x | +76.8% | +13.6% | 0.83 | −12.5% | 59% | 27 |
| 2.0x | +183.8% | +26.2% | 0.85 | −24.2% | 59% | 27 |
| **3.0x (recommended)** | **+319.2%** | **+37.7%** | **0.88** | **−34.8%** | **59%** | **27** |

**Long/Short** (V27 longs + mirrored regime-gated shorts):

| Leverage | Return | Ann. | Sharpe | Max DD | Win Rate | Trades |
|----------|--------|------|--------|--------|----------|--------|
| 1.0x | +79.6% | +14.0% | 0.78 | −13.4% | 58% | 31 (27L/4S) |
| 2.0x | +189.7% | +26.8% | 0.81 | −25.8% | 58% | 31 |
| **2.5x (recommended)** | **+254.9%** | **+32.7%** | **0.83** | **−31.6%** | **58%** | **31** |

The short book is small by construction — USDT market cap rarely contracts
(2022 was the only sustained episode). Its value is regime coverage: in a
2022-style liquidity contraction the short side activates (June 2022
capitulation short +8.8% at 1x; February 2026 +2.0%) while the long side
stays flat.

## 4. Optimal leverage

Method: per-trade expected log-growth (Kelly), 10,000-resample bootstrap of
the trade sequence, close-based liquidation plus 10% wick stress, full grid
0.5–6.0x re-backtested. See the [sweep page](https://dddabtc.github.io/usdt-slope-strategies/visualization_leverage.html).

| | Long-Only | Long/Short |
|---|---|---|
| Growth-optimal (full Kelly) | >6x (grid edge — n=27 sample, do not trust) | >6x (same caveat) |
| **Recommended** | **3.0x** | **2.5x** |
| Conservative (DD ≤ ~25%) | 2.0x | 2.0x |

Recommendation rule (predeclared): largest leverage with zero wick-stress
liquidations, bootstrap-median max DD ≥ −35%, bootstrap-5th-percentile total
return > 0, and ≤ half of growth-optimal. The binding constraints in-sample:
long-only breaks the DD cap at 3.25x and goes P5-negative at 3.5x; long/short
goes P5-negative at 2.75x.

Why not full Kelly: with ~30 trades the Kelly estimate is dominated by
estimation error, daily closes hide intraday wicks, and the worst observed
trade (−13% raw) would mean −39% at 3x. Half-Kelly-style discounting is the
standard answer to both.

## 5. How to run

```bash
pip install -r requirements.txt

python tests_regression.py        # verify long side == canonical V27
python run_backtest.py            # both strategies at recommended leverage
python run_backtest.py --leverage 2.0   # any leverage you want
python run_leverage_sweep.py      # full grid + optimal-leverage page
python scripts/refresh_data.py    # pull latest CoinGecko data + regenerate pages
```

Every strategy parameter (leverage, fees, funding, maintenance margin, signal
constants) lives in `StrategyConfig` (`src/engine.py`).

## 6. Project structure

```
├── src/
│   ├── engine.py        # signals (long + mirrored short), exits, costs,
│   │                    # liquidation, trade collection, daily MTM equity
│   ├── leverage.py      # Kelly / bootstrap / liquidation-aware sweep
│   ├── data.py          # CoinGecko fetch + *_full.csv history merge
│   └── visualize.py     # interactive result pages (original repo template)
├── run_backtest.py      # headline backtests + pages
├── run_leverage_sweep.py
├── tests_regression.py  # long side must equal canonical V27 exactly
├── scripts/refresh_data.py
├── data/                # daily CSVs (BTC price, USDT/USDC/DAI mcap)
├── experiments/         # artifacts: trades.json, equity.csv, ledger.jsonl
└── docs/                # GitHub Pages output
```

## 7. Caveats

- 27–31 closed trades is a small sample; every number above carries wide
  confidence bands (the bootstrap columns on the sweep page quantify them).
- Benchmark metrics are frozen at 2026-03-19; later trades are shown as a
  Post-benchmark overlay and never enter headline numbers.
- Daily CoinGecko closes cannot see intraday wicks; the wick-stress column is
  an approximation, not a guarantee. Liquidation at high leverage can be
  worse in practice.
- Funding is modeled as a flat 3 bps/day for longs and 0 for shorts;
  real funding varies and occasionally inverts.
- Not financial advice.
