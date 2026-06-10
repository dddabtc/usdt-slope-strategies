#!/usr/bin/env python3
"""Generate docs/visualization_open_search.html from open-search results."""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / "experiments" / "open-search" / "results" / "leaderboard.json"
OUT = PROJECT / "docs" / "visualization_open_search.html"

FAMILY_LABELS = {
    "F0": "F0 · Buy & hold BTC",
    "F1": "F1 · Regime switch (hold while USDT slope > 0)",
    "F2": "F2 · BTC-only momentum (no USDT — ablation)",
    "F3": "F3 · USDT long events ⭐",
    "F4": "F4 · Long/short mirror",
}

PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>USDT Slope Strategies — Open Strategy Search</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 12px; }
h1 { font-size: 17px; margin-bottom: 6px; color: #58a6ff; line-height: 1.3; }
h2 { font-size: 14px; margin: 18px 0 8px; color: #8b949e; }
.back { display: inline-block; font-size: 13px; color: #8b949e; text-decoration: none; margin-bottom: 10px; padding: 4px 10px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; }
.back:active, .back:hover { color: #58a6ff; border-color: #58a6ff; }
.meta { font-size: 12px; color: #8b949e; margin-bottom: 12px; }
.notice { background: #211a0d; border: 1px solid #9e6a03; color: #f0c674; padding: 10px 12px; border-radius: 8px; margin: 10px 0 12px; font-size: 12px; line-height: 1.45; }
.notice strong { color: #ffd866; }
.rec { background: #0f2418; border: 1px solid #238636; color: #7ee2a8; padding: 10px 12px; border-radius: 8px; margin: 10px 0 12px; font-size: 13px; line-height: 1.55; }
.rec strong { color: #3fb950; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 16px; font-size: 12px; }
th { background: #161b22; color: #8b949e; text-align: right; padding: 6px 8px; border-bottom: 1px solid #30363d; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
td { padding: 5px 8px; border-bottom: 1px solid #21262d; text-align: right; }
tr.best { background: #0f2418; }
.pos { color: #3fb950; } .neg { color: #f85149; }
@media (min-width: 768px) { body { padding: 20px; max-width: 1100px; margin: 0 auto; } h1 { font-size: 22px; } }
</style>
</head>
<body>
<a class="back" href="index.html">← 返回总览 · All Strategies</a>
<h1>Open Strategy Search — what is honestly the best strategy?</h1>
<div class="meta">__META__</div>
<div class="notice"><strong>Method:</strong> 5 predeclared families (~160 configs) compete under nested walk-forward:
configs are selected only inside each rolling 730-day training window, then evaluated once on the following
180-day unseen window; the chained unseen-window equity is the family's score. Immediate execution, 1x leverage,
5 bps/side + 3 bps/day funding. <strong>Selection-luck audit:</strong> the identical F3 search was re-run on 12 surrogate
USDT series (circularly shifted log-changes — same statistics, no real USDT→BTC link); the real result must clear
that null distribution. Full artifacts in <code>experiments/open-search/</code>.</div>
__REC__
<h2>Leaderboard — chained out-of-sample, 8 folds (2022-09-27 → 2026-06-09)</h2>
<table>
<thead><tr><th>Family</th><th>Total return</th><th>Daily Sharpe</th><th>Max DD</th><th>Trades</th><th>Win rate</th></tr></thead>
<tbody>__LEADERBOARD__</tbody>
</table>
<h2>F3 per fold — selected config and unseen-window result (8/8 positive)</h2>
<table>
<thead><tr><th>Fold</th><th>Test window</th><th>Selected on train only</th><th>OOS return</th><th>Trades</th></tr></thead>
<tbody>__FOLDS__</tbody>
</table>
<h2>Surrogate null — best score each fake-USDT search achieved</h2>
<table>
<thead><tr><th>Seed</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th><th>10</th><th>11</th><th>12</th><th>REAL</th></tr></thead>
<tbody><tr><td>Total return</td>__NULLS__<td class="pos"><strong>__REAL__</strong></td></tr></tbody>
</table>
<div class="notice">Real F3 beats all 12 surrogates (≈3× the best null); p = 1/13 ≈ 0.077 — the smallest value n=12 allows.
Caveats: 32 chained OOS trades is a small sample; per-fold winners rotate within the pattern, so trust the pattern,
not any single parameterization. Past performance ≠ future results.</div>
</body>
</html>
"""


def main():
    data = json.loads(RES.read_text())
    run = data["run"]
    meta = (f"Nested walk-forward · {run['fold_count']} folds · eval {run['eval_start']} → {run['actual_end']} · "
            f"{run['surrogates_requested']} surrogate-null searches · immediate execution · 1x · costs included")

    rows = ""
    for r in data["leaderboard"]:
        cls = ' class="best"' if r["family"] == "F3" else ""
        wr = f"{r['win_rate']*100:.0f}%" if r["n_trades"] else "—"
        n = r["n_trades"] if r["n_trades"] else "—"
        rows += (f'<tr{cls}><td>{FAMILY_LABELS.get(r["family"], r["family"])}</td>'
                 f'<td class="pos">{r["total_return"]*100:+.1f}%</td>'
                 f'<td>{r["daily_sharpe"]:.2f}</td>'
                 f'<td class="neg">{r["max_drawdown"]*100:.1f}%</td>'
                 f'<td>{n}</td><td>{wr}</td></tr>\n')

    folds = ""
    for f in data["families"]["F3"]["folds"]:
        om = f["oos_metrics"]
        cls = "pos" if om["total_return"] >= 0 else "neg"
        folds += (f'<tr><td>{f["fold"]}</td><td>{om["eval_start"]} → {om["eval_end"]}</td>'
                  f'<td style="text-align:left">{f["selected_config"]["config_id"]}</td>'
                  f'<td class="{cls}">{om["total_return"]*100:+.1f}%</td>'
                  f'<td>{om["n_trades"]}</td></tr>\n')

    sn = data["surrogate_null"]
    nulls = ""
    for d in sorted(sn["distribution"], key=lambda x: x["seed"]):
        cls = "pos" if d["total_return"] >= 0 else "neg"
        nulls += f'<td class="{cls}">{d["total_return"]*100:+.0f}%</td>'

    rec = ('<div class="rec"><strong>Verdict: the best strategy is the long-only USDT liquidity-event family (F3).</strong><br>'
           'Entry = USDT slope event (regime acceleration / Bollinger break / top-decile percentile) in the top ~7–10% of its '
           '1-year distribution, confirmed by BTC 3-day momentum; exit = momentum decay (slope-peak / Kalman-velocity / trail); no shorts. '
           'It is the only family positive in all 8 unseen windows, has double the Sharpe of every alternative (1.68), one-sixth of '
           'buy&hold\'s drawdown (−8.1% vs −52.4%) at ~92% of its return, beats the BTC-only ablation by +82pp, and beats every '
           'surrogate-null search. The deployed frozen V27 long-only config is one member of this validated pattern.</div>')

    html = PAGE.replace("__META__", meta).replace("__REC__", rec)
    html = html.replace("__LEADERBOARD__", rows).replace("__FOLDS__", folds)
    html = html.replace("__NULLS__", nulls)
    html = html.replace("__REAL__", f"{sn['real_f3_total_return']*100:+.1f}%")
    OUT.write_text(html)
    print(f"Written {OUT}")


if __name__ == "__main__":
    main()
