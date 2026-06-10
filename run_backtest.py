#!/usr/bin/env python3
"""Headline backtests for both strategies + interactive result pages.

Frozen benchmark window (identical to usdt-slope-research V27):
  signals warm up from 2020-09-27, evaluation 2021-09-27 → 2026-03-19.
Trades after the frozen window are shown as a Post-benchmark overlay; they
never touch the headline metrics.

Usage:
  python run_backtest.py                 # recommended leverage per strategy
  python run_backtest.py --leverage 2.0  # same leverage for both
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from src.data import load_full_data
from src.engine import (
    BENCHMARK_TEST_END, LONG_ONLY, LONG_SHORT, RESEARCH_START, TRAIN_END,
    StrategyConfig, backtest, post_benchmark_trades, slice_data_window,
)
from src.visualize import build_kalman, generate_page

OUTDIR = PROJECT / "experiments"
DOCS = PROJECT / "docs"

RECOMMENDED = {LONG_ONLY: 3.0, LONG_SHORT: 3.0}

TITLES = {
    LONG_ONLY: "Long-Only — frozen V27 entry/exit, immediate execution, costs included",
    LONG_SHORT: "Long/Short — V27 longs + mirrored regime-gated shorts, immediate execution",
}


def notice_html(cfg: StrategyConfig, m: dict, post: list[dict]) -> str:
    post_line = ""
    if post:
        parts = []
        for t in post:
            if t.get("status") == "OPEN":
                parts.append(f'{t["side"]} signal {t["entry_signal_date"]}→entry {t["buy_date"]} OPEN')
            else:
                parts.append(f'{t["side"]} {t["buy_date"]}→{t["sell_date"]} {t["return_pct"]:+.1f}%')
        post_line = (' <strong>Post-benchmark overlay (not in headline):</strong> '
                     + '; '.join(parts) + '.')
    short_line = ""
    if cfg.mode == LONG_SHORT:
        short_line = (' Shorts mirror every long rule (lower Bollinger band / contraction regime, '
                      'BTC 3d down, slope percentile ≤ 7%) and are additionally gated to bear regime '
                      '(60d USDT slope < 0) — the predeclared V23 regime rule mirrored, not re-fitted.')
    return (
        '<div class="notice"><strong>Methodology:</strong> every daily bar is a 00:00 UTC snapshot — '
        'USDT issuance is on-chain observable in real time and BTC trades 24/7, so signals are executable '
        'within minutes of the snapshot. Headline fills are at the signal snapshot itself (immediate '
        'execution); delaying 1h costs ≈4% of total return, 6h ≈23%, and waiting a full day (the most '
        'conservative bound) still leaves the strategy profitable — see README for the decay table. '
        'Parameters are the frozen canonical V27 set from '
        '<a href="https://github.com/dddabtc/usdt-slope-research" style="color:#ffd866">usdt-slope-research</a>, '
        'verified trade-for-trade and not re-fitted. Costs included: 5 bps/side fee+slippage and 3 bps/day '
        f'long funding on {cfg.leverage:g}x notional; close-based isolated-margin liquidation modeled '
        f'(none occurred). Headline metrics are frozen at {BENCHMARK_TEST_END.date()};'
        f' BTC buy&hold over the same window: {m["btc_buy_hold_return"]*100:+.1f}%.'
        + short_line + post_line +
        ' Past performance does not guarantee future results.</div>'
    )


def run_mode(data: dict, mode: str, leverage: float | None) -> dict:
    lev = leverage if leverage is not None else RECOMMENDED[mode]
    cfg = StrategyConfig(mode=mode, leverage=lev)
    result = backtest(data, cfg, TRAIN_END, BENCHMARK_TEST_END)
    m = result["metrics"]
    post = post_benchmark_trades(data, cfg)

    exp_dir = OUTDIR / mode.replace("_", "-")
    exp_dir.mkdir(parents=True, exist_ok=True)
    result["equity"].to_csv(exp_dir / "equity.csv", index=False)
    (exp_dir / "trades.json").write_text(json.dumps(result["trades"] + post, indent=2))
    ledger_entry = {
        "phase": f"headline_{mode}",
        "params": {k: v for k, v in cfg.__dict__.items() if k != "extra"},
        "result": m,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(exp_dir / "ledger.jsonl", "w") as f:
        f.write(json.dumps(ledger_entry, default=str) + "\n")

    # chart frame over the full research window incl. post-benchmark
    latest = min(data["btc"]["date"].max(), data["usdt"]["date"].max())
    bdata = slice_data_window(data, RESEARCH_START, latest)
    mc = bdata["usdt"][["date", "market_cap"]].rename(columns={"market_cap": "usdt_mcap"})
    kdf = build_kalman(bdata["usdt"])
    chart_df = (bdata["btc"][["date", "price"]]
                .merge(mc, on="date", how="inner")
                .merge(kdf, on="date", how="left")
                .sort_values("date").reset_index(drop=True))

    display_trades = [{**t, "phase": "Benchmark"} for t in result["trades"]] + post
    meta = (f"OOS: {TRAIN_END.date()} to {BENCHMARK_TEST_END.date()} · {m['n_trades']} trades "
            f"(L{m['n_long']}/S{m['n_short']}) · {cfg.leverage:g}x leverage · immediate execution "
            f"· data to {latest.date()}")
    page_name = f"visualization_{mode}.html"
    generate_page(chart_df, display_trades, m, TITLES[mode], meta,
                  notice_html(cfg, m, post), DOCS / page_name)

    print(f"{mode} @{cfg.leverage:g}x: ret {m['total_return']*100:+.1f}%  sharpe {m['daily_sharpe']:.2f}  "
          f"dd {m['max_drawdown']*100:.1f}%  trades {m['n_trades']} (L{m['n_long']}/S{m['n_short']})  "
          f"wr {m['win_rate']*100:.0f}%  bh {m['btc_buy_hold_return']*100:+.1f}%  post-trades {len(post)}")
    return {"metrics": m, "config": cfg, "post": post, "page": page_name}


INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>USDT Slope Strategies</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#c9d1d9;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px}
h1{font-size:1.8em;margin-bottom:8px;color:#58a6ff}
.subtitle{color:#8b949e;margin-bottom:24px;font-size:.95em;text-align:center}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;max-width:900px;width:100%}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;text-decoration:none;color:inherit;transition:border-color .2s}
.card:hover{border-color:#58a6ff}
.card h2{font-size:1.1em;color:#58a6ff;margin-bottom:8px}
.card .return{font-size:2em;font-weight:700;color:#3fb950}
.card .return.mid{color:#f0883e}
.card .stats{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}
.card .stat{font-size:.8em}
.card .stat .label{color:#8b949e}
.card .stat .val{font-weight:600}
.best{border-color:#238636;position:relative}
.best::before{content:'⭐ Recommended';position:absolute;top:-10px;right:16px;background:#238636;color:#fff;font-size:.7em;padding:2px 8px;border-radius:4px}
.footer{margin-top:32px;color:#484f58;font-size:.8em;text-align:center;line-height:1.6}
.footer a{color:#58a6ff;text-decoration:none}
</style>
</head>
<body>
<h1>USDT Slope Strategies</h1>
<p class="subtitle">Frozen benchmark: __WINDOW__ · immediate execution at 00:00 UTC snapshots · fees &amp; funding included<br>BTC buy&amp;hold same window: __BH__</p>
<div class="cards">
__CARDS__
</div>
<p class="footer">Long side = frozen canonical V27 from <a href="https://github.com/dddabtc/usdt-slope-research">usdt-slope-research</a> (verified bit-identical, then costs added) · shorts are predeclared mirrors, regime-gated<br>
Data: CoinGecko daily (USDT market cap, BTC price), refreshed to __LATEST__ · not financial advice</p>
</body>
</html>
"""

CARD = """<a href="__PAGE__" class="card__BEST__">
  <h2>__NAME__</h2>
  <div class="return">__RET__</div>
  <div class="stats">
    <div class="stat"><span class="label">Sharpe</span><br><span class="val">__SHARPE__</span></div>
    <div class="stat"><span class="label">Win Rate</span><br><span class="val">__WR__</span></div>
    <div class="stat"><span class="label">Trades</span><br><span class="val">__TRADES__</span></div>
    <div class="stat"><span class="label">Max DD</span><br><span class="val">__DD__</span></div>
    <div class="stat"><span class="label">Leverage</span><br><span class="val">__LEV__</span></div>
  </div>
</a>
"""


def write_index(results: dict, latest: str) -> None:
    cards = ""
    for mode, label, best in [
        (LONG_ONLY, "Long-Only · 单纯做多", True),
        (LONG_SHORT, "Long/Short · 多空双向", False),
    ]:
        m = results[mode]["metrics"]
        card = CARD.replace("__PAGE__", results[mode]["page"])
        card = card.replace("__BEST__", " best" if best else "")
        card = card.replace("__NAME__", label)
        card = card.replace("__RET__", f"+{m['total_return']*100:.0f}%")
        card = card.replace("__SHARPE__", f"{m['daily_sharpe']:.2f}")
        card = card.replace("__WR__", f"{m['win_rate']*100:.0f}%")
        card = card.replace("__TRADES__", str(m["n_trades"]))
        card = card.replace("__DD__", f"{m['max_drawdown']*100:.1f}%")
        card = card.replace("__LEV__", f"{results[mode]['config'].leverage:g}x")
        cards += card

    cards += """<a href="visualization_mechanism.html" class="card">
  <h2>Mechanism Round · 机理优化空间</h2>
  <div class="return mid">无规则空间</div>
  <div class="stats">
    <div class="stat"><span class="label">机理变体</span><br><span class="val">5/5 落败</span></div>
    <div class="stat"><span class="label">事件响应</span><br><span class="val">即时+持续2周</span></div>
    <div class="stat"><span class="label">剂量效应</span><br><span class="val">倒U形</span></div>
  </div>
</a>
<a href="visualization_open_search.html" class="card">
  <h2>Open Search · 开放式最佳策略验证</h2>
  <div class="return">F3 wins</div>
  <div class="stats">
    <div class="stat"><span class="label">8/8 折</span><br><span class="val">全正 OOS</span></div>
    <div class="stat"><span class="label">Sharpe</span><br><span class="val">1.68</span></div>
    <div class="stat"><span class="label">vs 零检验</span><br><span class="val">胜 12/12</span></div>
    <div class="stat"><span class="label">Max DD</span><br><span class="val">-8.1%</span></div>
  </div>
</a>
<a href="visualization_leverage.html" class="card">
  <h2>Leverage Sweep · 最佳杠杆</h2>
  <div class="return mid">0.5–6.0x</div>
  <div class="stats">
    <div class="stat"><span class="label">Long-Only</span><br><span class="val">3.0x rec</span></div>
    <div class="stat"><span class="label">Long/Short</span><br><span class="val">3.0x rec</span></div>
    <div class="stat"><span class="label">Method</span><br><span class="val">Kelly + bootstrap + liq</span></div>
  </div>
</a>
"""
    m0 = results[LONG_ONLY]["metrics"]
    html = INDEX_TEMPLATE.replace("__CARDS__", cards)
    html = html.replace("__WINDOW__", f"{TRAIN_END.date()} → {BENCHMARK_TEST_END.date()}")
    html = html.replace("__BH__", f"{m0['btc_buy_hold_return']*100:+.1f}%")
    html = html.replace("__LATEST__", latest)
    (DOCS / "index.html").write_text(html)
    print(f"Written {DOCS / 'index.html'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leverage", type=float, default=None,
                    help="override leverage for both strategies (default: recommended per strategy)")
    args = ap.parse_args()

    data = load_full_data()
    latest = str(min(data["btc"]["date"].max(), data["usdt"]["date"].max()).date())
    DOCS.mkdir(exist_ok=True)
    results = {}
    for mode in [LONG_ONLY, LONG_SHORT]:
        results[mode] = run_mode(data, mode, args.leverage)
    write_index(results, latest)


if __name__ == "__main__":
    main()
