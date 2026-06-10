#!/usr/bin/env python3
"""Leverage grid sweep for both strategies -> optimal leverage + results page.

Writes:
  experiments/leverage-sweep/sweep_<mode>.csv
  experiments/leverage-sweep/recommendation.json
  docs/visualization_leverage.html
"""

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from src.data import load_full_data
from src.engine import LONG_ONLY, LONG_SHORT, StrategyConfig, BENCHMARK_TEST_END, TRAIN_END
from src.leverage import recommend, sweep

OUTDIR = PROJECT / "experiments" / "leverage-sweep"
DOCS = PROJECT / "docs"

PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>USDT Slope Strategies — Leverage Sweep</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 12px; }
h1 { font-size: 17px; margin-bottom: 6px; color: #58a6ff; line-height: 1.3; }
.back { display: inline-block; font-size: 13px; color: #8b949e; text-decoration: none; margin-bottom: 10px; padding: 4px 10px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; }
.back:active, .back:hover { color: #58a6ff; border-color: #58a6ff; }
h2 { font-size: 14px; margin: 16px 0 8px; color: #8b949e; }
.meta { font-size: 12px; color: #8b949e; margin-bottom: 12px; }
.notice { background: #211a0d; border: 1px solid #9e6a03; color: #f0c674; padding: 10px 12px; border-radius: 8px; margin: 10px 0 12px; font-size: 12px; line-height: 1.45; }
.notice strong { color: #ffd866; }
.rec { background: #0f2418; border: 1px solid #238636; color: #7ee2a8; padding: 10px 12px; border-radius: 8px; margin: 10px 0 12px; font-size: 13px; line-height: 1.5; }
.rec strong { color: #3fb950; font-size: 15px; }
.chart { width: 100%; height: 360px; }
.zoombox { fill: rgba(1, 4, 9, 0.62) !important; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 20px; font-size: 12px; }
th { background: #161b22; color: #8b949e; text-align: right; padding: 6px 8px; border-bottom: 1px solid #30363d; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
td { padding: 5px 8px; border-bottom: 1px solid #21262d; text-align: right; }
tr.rec-row { background: #0f2418; }
.pos { color: #3fb950; } .neg { color: #f85149; } .warn { color: #d29922; }
@media (min-width: 768px) { body { padding: 20px; max-width: 1200px; margin: 0 auto; } h1 { font-size: 22px; } .chart { height: 420px; } }
</style>
</head>
<body>
<a class="back" href="index.html">← 返回总览 · All Strategies</a>
<h1>Leverage Sweep — Long-Only vs Long/Short</h1>
<div class="meta">__META__</div>
__REC__
<div class="notice"><strong>Method:</strong> every leverage point re-runs the full backtest with fees (5 bps/side),
long funding (3 bps/day) and close-based isolated-margin liquidation (0.5% maintenance).
<em>Growth</em> = mean log-growth per trade (full-Kelly optimum is its peak). Bootstrap columns resample the
trade sequence 10,000×. <em>Wick liq</em> counts liquidations if an intraday wick pierced 10% beyond the worst daily close —
daily data cannot see wicks, so any leverage failing this is unsafe regardless of its return.
Past performance ≠ future results; ~30 trades is a small sample.</div>
<h2>Total return &amp; max drawdown vs leverage</h2>
<div id="chart-return" class="chart"></div>
<h2>Expected log-growth per trade vs leverage (peak = full Kelly)</h2>
<div id="chart-growth" class="chart"></div>
__TABLES__
<script>
var sweeps = __SWEEPS__;
var colors = {long_only: '#58a6ff', long_short: '#f0883e'};
var retTraces = [], gTraces = [];
Object.keys(sweeps).forEach(function(mode) {
  var s = sweeps[mode];
  retTraces.push({x: s.leverage, y: s.total_return.map(function(v){return v*100;}), name: mode + ' return %', line: {color: colors[mode], width: 2}, yaxis: 'y'});
  retTraces.push({x: s.leverage, y: s.max_drawdown.map(function(v){return v*100;}), name: mode + ' maxDD %', line: {color: colors[mode], width: 1.5, dash: 'dot'}, yaxis: 'y'});
  gTraces.push({x: s.leverage, y: s.exp_log_growth, name: mode + ' E[log growth]/trade', line: {color: colors[mode], width: 2}});
});
function render() {
  var narrow = window.innerWidth < 768;
  // 4 legend items wrap to 2 rows on phones; reserve top margin accordingly
  var base = {paper_bgcolor: '#0d1117', plot_bgcolor: '#0d1117', font: {color: '#c9d1d9', size: 11},
    margin: {t: narrow ? 78 : 34, b: 40, l: 56, r: 16}, hovermode: 'x unified',
    legend: {bgcolor: 'rgba(0,0,0,0)', x: 0, y: 1.0, yanchor: 'bottom', orientation: 'h', font: {size: 11}},
    xaxis: {title: 'Leverage', gridcolor: '#21262d', dtick: narrow ? 1 : 0.5}};
  var cfg = {responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toImage']};
  Plotly.react('chart-return', retTraces, Object.assign({}, base, {yaxis: {title: '%', gridcolor: '#21262d'}}), cfg);
  Plotly.react('chart-growth', gTraces, Object.assign({}, base, {yaxis: {title: 'E[log(1+r)]', gridcolor: '#21262d'}}), cfg);
}
render();
window.addEventListener('resize', render);
</script>
</body>
</html>
"""


def table_html(df, rec_lev, title):
    rows = ""
    for _, r in df.iterrows():
        cls = ' class="rec-row"' if abs(r["leverage"] - rec_lev) < 1e-9 else ""
        liq_cls = "neg" if r["n_liquidated_wick10"] > 0 else "pos"
        rows += (
            f'<tr{cls}><td>{r["leverage"]:.2f}x</td>'
            f'<td class="pos">{r["total_return"]*100:+.0f}%</td>'
            f'<td>{r["ann_return"]*100:+.1f}%</td>'
            f'<td>{r["daily_sharpe"]:.2f}</td>'
            f'<td class="neg">{r["max_drawdown"]*100:.1f}%</td>'
            f'<td>{r["exp_log_growth"]:+.4f}</td>'
            f'<td>{r["boot_p5_return"]*100:+.0f}%</td>'
            f'<td class="neg">{r["boot_median_maxdd"]*100:.1f}%</td>'
            f'<td>{r["p_loss"]*100:.1f}%</td>'
            f'<td class="{liq_cls}">{int(r["n_liquidated"])} / {int(r["n_liquidated_wick10"])}</td></tr>\n'
        )
    return (
        f"<h2>{title}</h2>"
        '<table><thead><tr><th>Lev</th><th>Return</th><th>Ann.</th><th>Sharpe</th><th>MaxDD</th>'
        '<th>Growth</th><th>Boot P5</th><th>Boot med DD</th><th>P(loss)</th><th>Liq close/wick10%</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_full_data()
    latest = min(data["btc"]["date"].max(), data["usdt"]["date"].max())
    results, recs, tables = {}, {}, ""
    for mode in [LONG_ONLY, LONG_SHORT]:
        print(f"\n=== sweep {mode} (to {latest.date()}) ===")
        cfg = StrategyConfig(mode=mode)
        df = sweep(data, cfg, eval_end=latest)
        rec = recommend(df)
        results[mode] = df
        recs[mode] = rec
        df.to_csv(OUTDIR / f"sweep_{mode}.csv", index=False)
        print(df.to_string(index=False))
        print(f"--> growth-optimal {rec['growth_optimal_leverage']}x, recommended {rec['recommended_leverage']}x")
        tables += table_html(df, rec["recommended_leverage"], f"{mode} (green row = recommended)")

    (OUTDIR / "recommendation.json").write_text(json.dumps(recs, indent=2))

    sweeps_js = {
        mode: {
            "leverage": df["leverage"].tolist(),
            "total_return": df["total_return"].tolist(),
            "max_drawdown": df["max_drawdown"].tolist(),
            "exp_log_growth": df["exp_log_growth"].tolist(),
        }
        for mode, df in results.items()
    }
    max_grid = max(results[LONG_ONLY]["leverage"])

    def kelly_str(mode):
        k = recs[mode]["growth_optimal_leverage"]
        return f"≥{k:.1f}x (grid edge — do not trust at n≈30)" if k >= max_grid else f"{k:.2f}x"

    rec_html = (
        '<div class="rec"><strong>Recommended leverage: '
        f'long-only {recs[LONG_ONLY]["recommended_leverage"]:.2f}x · '
        f'long/short {recs[LONG_SHORT]["recommended_leverage"]:.2f}x</strong><br>'
        f'Growth-optimal (full Kelly): long-only {kelly_str(LONG_ONLY)}, '
        f'long/short {kelly_str(LONG_SHORT)}. '
        f'Rule: {recs[LONG_ONLY]["rule"]}.</div>'
    )
    page = PAGE.replace("__META__", f"Evaluation: {TRAIN_END.date()} → {latest.date()} (frozen params) · immediate execution · grid 0.5–6.0x · 10,000 bootstrap resamples")
    page = page.replace("__REC__", rec_html)
    page = page.replace("__TABLES__", tables)
    page = page.replace("__SWEEPS__", json.dumps(sweeps_js))
    DOCS.mkdir(exist_ok=True)
    (DOCS / "visualization_leverage.html").write_text(page)
    print(f"\nWritten {DOCS / 'visualization_leverage.html'}")


if __name__ == "__main__":
    main()
