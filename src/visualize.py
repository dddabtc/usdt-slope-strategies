"""Interactive result pages — same template as dddabtc/usdt-slope-research.

Dark GitHub theme, Plotly chart with buy/sell triangles, USDT Kalman
一阶导/二阶导 views, summary stat cards, trade table (desktop) + trade cards
(mobile), methodology notice box.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.engine import _rolling_slope

DOCS_DIR = Path(__file__).parent.parent / "docs"


class KalmanSlope:
    def __init__(self, dt=1.0, process_var=1e5, measure_var=1e9):
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = process_var * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        self.R = np.array([[measure_var]])
        self.P = np.eye(2) * 1e10
        self.x = np.array([[0.0], [0.0]])
        self.ok = False

    def update(self, z):
        if not self.ok:
            self.x = np.array([[z], [0.0]])
            self.P = np.eye(2) * 1e8
            self.ok = True
            return
        xp = self.F @ self.x
        Pp = self.F @ self.P @ self.F.T + self.Q
        y = z - (self.H @ xp)[0, 0]
        S = self.H @ Pp @ self.H.T + self.R
        K = Pp @ self.H.T @ np.linalg.inv(S)
        self.x = xp + K * y
        self.P = (np.eye(2) - K @ self.H) @ Pp


def build_kalman(mc_df: pd.DataFrame, sw: int = 3) -> pd.DataFrame:
    mc = mc_df.sort_values("date").reset_index(drop=True)
    slopes = _rolling_slope(mc["market_cap"].values.astype(float), sw)
    kf = KalmanSlope()
    n = len(slopes)
    ks = np.full(n, np.nan)
    kv = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(slopes[i]):
            continue
        kf.update(slopes[i])
        ks[i] = kf.x[0, 0]
        kv[i] = kf.x[1, 0]
    return pd.DataFrame({"date": mc["date"], "kalman_slope": ks, "kalman_vel": kv})


def build_kalman_btc(btc_df: pd.DataFrame) -> pd.DataFrame:
    btc = btc_df.sort_values("date").reset_index(drop=True)
    prices = btc["price"].values.astype(float)
    price_vel = np.full(len(prices), np.nan)
    price_vel[1:] = prices[1:] - prices[:-1]
    kf = KalmanSlope(process_var=1e3, measure_var=1e6)
    n = len(price_vel)
    ks = np.full(n, np.nan)
    kv = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(price_vel[i]):
            continue
        kf.update(price_vel[i])
        ks[i] = kf.x[0, 0]
        kv[i] = kf.x[1, 0]
    return pd.DataFrame({"date": btc["date"], "btc_vel": ks, "btc_accel": kv})


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>USDT Slope Strategies — __TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 12px; }
h1 { font-size: 17px; margin-bottom: 6px; color: #58a6ff; line-height: 1.3; }
.meta { font-size: 12px; color: #8b949e; margin-bottom: 12px; }
#chart { width: 100%; height: 320px; min-height: 250px; }
#controls { margin: 10px 0; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
#controls label { font-size: 12px; color: #8b949e; }
#controls select { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 6px 12px; border-radius: 6px; font-size: 13px; -webkit-appearance: none; min-height: 36px; }
#trade-table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }
#trade-table th { background: #161b22; color: #8b949e; text-align: left; padding: 8px; border-bottom: 1px solid #30363d; position: sticky; top: 0; z-index: 2; white-space: nowrap; }
#trade-table td { padding: 8px; border-bottom: 1px solid #21262d; cursor: pointer; }
#trade-table tr:active { background: #1c2333; }
#trade-table tr.highlight { background: #1c2333; }
.win { color: #3fb950; font-weight: 600; }
.loss { color: #f85149; font-weight: 600; }
.open { color: #d29922; font-weight: 600; }
.short-tag { color: #f0883e; font-weight: 600; }
.rangeslider-mask-min, .rangeslider-mask-max { fill-opacity: 0 !important; }
.rangeslider-slidebox { fill: #8b949e !important; fill-opacity: 0.22 !important; stroke: #8b949e !important; stroke-width: 1px !important; }
#summary { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 12px 0; }
.notice { background: #211a0d; border: 1px solid #9e6a03; color: #f0c674; padding: 10px 12px; border-radius: 8px; margin: 10px 0 12px; font-size: 12px; line-height: 1.45; }
.notice strong { color: #ffd866; }
.stat { background: #161b22; padding: 10px 12px; border-radius: 8px; }
.stat .label { font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
.stat .value { font-size: 18px; font-weight: 600; margin-top: 2px; }
.trade-cards { display: none; }
.trade-card { background: #161b22; border-radius: 8px; padding: 12px; margin-bottom: 8px; border-left: 3px solid #30363d; cursor: pointer; }
.trade-card.win { border-left-color: #3fb950; }
.trade-card.loss { border-left-color: #f85149; }
.trade-card .row { display: flex; justify-content: space-between; margin: 4px 0; font-size: 13px; }
.trade-card .label { color: #8b949e; }
@media (min-width: 768px) {
  body { padding: 20px; max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 22px; }
  #chart { height: 500px; }
  #summary { grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .stat .value { font-size: 22px; }
  .trade-cards { display: none !important; }
  #trade-table { display: table !important; }
}
@media (max-width: 767px) {
  #trade-table { display: none; }
  .trade-cards { display: block; }
}
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="meta">__META__</div>
__NOTICE__
<div id="summary">
  <div class="stat"><div class="label">Return</div><div class="value" style="color:#3fb950">__RETURN__%</div></div>
  <div class="stat"><div class="label">Sharpe</div><div class="value">__SHARPE__</div></div>
  <div class="stat"><div class="label">Win Rate</div><div class="value">__WIN_RATE__%</div></div>
  <div class="stat"><div class="label">Max DD</div><div class="value" style="color:#f85149">__MAX_DD__%</div></div>
</div>
<div id="chart"></div>
<div id="chart-deriv1" style="display:none; margin-top:8px; height:400px;"></div>
<div id="chart-deriv2" style="display:none; margin-top:8px; height:400px;"></div>
<div id="controls">
  <label>Scale:</label>
  <select id="scale-select" onchange="render()">
    <option value="percent">Percent Change</option>
    <option value="absolute">Absolute Price</option>
    <option value="log">Log Scale</option>
  </select>
  <label>Show:</label>
  <select id="show-select" onchange="render()">
    <option value="all">BTC + USDT</option>
    <option value="usdt">USDT Only</option>
    <option value="deriv1">USDT 一阶导</option>
    <option value="deriv2">USDT 二阶导</option>
  </select>
</div>
<table id="trade-table">
<thead><tr><th>#</th><th>Phase</th><th>Side</th><th>Signal</th><th>Entry</th><th>Exit</th><th>Entry $</th><th>Exit $</th><th>Return</th><th>Days</th></tr></thead>
<tbody>__TABLE_ROWS__</tbody>
</table>
<div class="trade-cards">__CARD_ROWS__</div>

<script>
var btcDates = __BTC_DATES__;
var btcPrices = __BTC_PRICES__;
var usdtMcap = __USDT_MCAP__;
var kalmanSlope = __KALMAN_SLOPE__;
var kalmanVel = __KALMAN_VEL__;
var buyDates = __BUY_DATES__;
var buyY = __BUY_Y__;
var buySides = __BUY_SIDES__;
var sellDates = __SELL_DATES__;
var plotSellDates = __PLOT_SELL_DATES__;
var plotSellY = __PLOT_SELL_Y__;

function entryColors() { return buySides.map(function(s){ return s === 'SHORT' ? '#f0883e' : '#3fb950'; }); }

function buildTraces(scale) {
  var traces = [];
  var fp = btcPrices[0];
  var fm = usdtMcap[0];
  var show = document.getElementById('show-select').value;

  if (show === 'usdt') {
    traces.push({x: btcDates, y: scale === 'percent' ? usdtMcap.map(function(m){return ((m-fm)/fm)*100;}) : usdtMcap, name: 'USDT Mcap', line: {color: '#f0883e', width: 1.5}, yaxis: 'y'});
    return traces;
  }

  if (scale === 'percent') {
    traces.push({x: btcDates, y: btcPrices.map(function(p){return ((p-fp)/fp)*100;}), name: 'BTC %', line: {color: '#58a6ff', width: 1.5}, yaxis: 'y'});
    traces.push({x: btcDates, y: usdtMcap.map(function(m){return ((m-fm)/fm)*100;}), name: 'USDT Mcap %', line: {color: '#f0883e', width: 1, dash: 'dot'}, yaxis: 'y2'});
  } else {
    traces.push({x: btcDates, y: btcPrices, name: 'BTC $', line: {color: '#58a6ff', width: 1.5}, yaxis: 'y'});
    traces.push({x: btcDates, y: usdtMcap, name: 'USDT Mcap', line: {color: '#f0883e', width: 1, dash: 'dot'}, yaxis: 'y2'});
  }

  traces.push({x: buyDates, y: scale === 'percent' ? buyY.map(function(y){return ((y-fp)/fp)*100;}) : buyY,
    mode: 'markers', name: 'Entry',
    marker: {color: entryColors(), size: 12, symbol: 'triangle-up', line: {color: '#238636', width: 1}},
    hovertemplate: 'Entry<br>%{x}<br>$%{customdata:,.0f}<extra></extra>',
    customdata: buyY, yaxis: 'y'});
  traces.push({x: plotSellDates, y: scale === 'percent' ? plotSellY.map(function(y){return ((y-fp)/fp)*100;}) : plotSellY,
    mode: 'markers', name: 'Exit',
    marker: {color: '#ff4444', size: 12, symbol: 'triangle-down', line: {color: '#da3633', width: 1}},
    hovertemplate: 'Exit<br>%{x}<br>$%{customdata:,.0f}<extra></extra>',
    customdata: plotSellY, yaxis: 'y'});

  return traces;
}

function derivLayout(titleText, yTitle, color) {
  return {
    paper_bgcolor: '#0d1117', plot_bgcolor: '#0d1117',
    font: {color: '#c9d1d9', size: 11},
    margin: {t: 50, b: 80, l: 60, r: 60},
    title: {text: titleText, font: {size: 13, color: color}, y: 0.98},
    xaxis: {gridcolor: '#21262d', type: 'date',
      rangeselector: {buttons: [
        {count: 3, label: '3m', step: 'month', stepmode: 'backward'},
        {count: 1, label: '1y', step: 'year', stepmode: 'backward'},
        {step: 'all'}
      ], bgcolor: '#161b22', font: {color: '#c9d1d9'}},
      rangeslider: {visible: true, bgcolor: '#161b22'}},
    yaxis: {title: yTitle, gridcolor: '#21262d', side: 'left'},
    yaxis2: {title: 'BTC %', overlaying: 'y', side: 'right', gridcolor: '#161b22'},
    legend: {bgcolor: 'rgba(0,0,0,0)', x: 0, y: 1.12, orientation: 'h'},
    hovermode: 'x unified'
  };
}

function render() {
  var scale = document.getElementById('scale-select').value;
  var show = document.getElementById('show-select').value;
  var mainChart = document.getElementById('chart');
  var deriv1 = document.getElementById('chart-deriv1');
  var deriv2 = document.getElementById('chart-deriv2');

  if (show === 'deriv1' || show === 'deriv2') {
    var vals = show === 'deriv1' ? kalmanSlope : kalmanVel;
    var el = show === 'deriv1' ? deriv1 : deriv2;
    mainChart.style.display = 'none';
    deriv1.style.display = show === 'deriv1' ? 'block' : 'none';
    deriv2.style.display = show === 'deriv2' ? 'block' : 'none';
    var fp = btcPrices[0];
    var d = [
      {x: btcDates, y: vals, name: show === 'deriv1' ? 'USDT 一阶导 (slope $/d)' : 'USDT 二阶导 (velocity $/d²)', line: {color: show === 'deriv1' ? '#f0883e' : '#a371f7', width: 1.5}, yaxis: 'y'},
      {x: btcDates, y: btcDates.map(function(){return 0;}), name: '', line: {color: '#484f58', width: 1, dash: 'dot'}, yaxis: 'y', showlegend: false},
      {x: btcDates, y: btcPrices.map(function(p){return ((p-fp)/fp)*100;}), name: 'BTC %', line: {color: '#58a6ff', width: 1}, yaxis: 'y2'},
      {x: buyDates, y: buyDates.map(function(dd){var i=btcDates.indexOf(dd);return i>=0?vals[i]:0;}),
        mode: 'markers', name: 'Entry', marker: {color: entryColors(), size: 10, symbol: 'triangle-up'}, yaxis: 'y'},
      {x: plotSellDates, y: plotSellDates.map(function(dd){var i=btcDates.indexOf(dd);return i>=0?vals[i]:0;}),
        mode: 'markers', name: 'Exit', marker: {color: '#ff4444', size: 10, symbol: 'triangle-down'}, yaxis: 'y'}
    ];
    Plotly.react(el, d, derivLayout(
      show === 'deriv1' ? 'USDT 一阶导 (slope) + BTC' : 'USDT 二阶导 (velocity) + BTC',
      show === 'deriv1' ? 'USDT slope ($/d)' : 'USDT velocity ($/d²)',
      show === 'deriv1' ? '#f0883e' : '#a371f7'
    ), {responsive: true, scrollZoom: true});
    return;
  }

  mainChart.style.display = 'block';
  deriv1.style.display = 'none';
  deriv2.style.display = 'none';

  var traces = buildTraces(scale);
  var yType = scale === 'log' ? 'log' : 'linear';
  var yTitle = scale === 'percent' ? 'Change %' : (scale === 'log' ? 'Log Scale' : 'Price ($)');
  var layout = {
    paper_bgcolor: '#0d1117', plot_bgcolor: '#0d1117',
    font: {color: '#c9d1d9', size: 11},
    margin: {t: 30, b: 40, l: 60, r: 60},
    xaxis: {gridcolor: '#21262d',
      rangeselector: {buttons: [
        {count: 3, label: '3m', step: 'month', stepmode: 'backward'},
        {count: 1, label: '1y', step: 'year', stepmode: 'backward'},
        {step: 'all'}
      ], bgcolor: '#161b22', font: {color: '#c9d1d9'}},
      rangeslider: {visible: true, bgcolor: '#161b22'}, type: 'date'},
    yaxis: {title: yTitle, gridcolor: '#21262d', side: 'left', type: yType},
    yaxis2: {title: 'USDT Mcap', overlaying: 'y', side: 'right', gridcolor: '#161b22', type: yType},
    legend: {bgcolor: 'rgba(0,0,0,0)', x: 0, y: 1.1, orientation: 'h'},
    hovermode: 'x unified',
    dragmode: 'zoom'
  };
  Plotly.newPlot('chart', traces, layout, {responsive: true, scrollZoom: true});
}

function highlightTrade(idx) {
  var rows = document.querySelectorAll('#trade-table tr');
  for (var i = 0; i < rows.length; i++) rows[i].classList.remove('highlight');
  var cards = document.querySelectorAll('.trade-card');
  for (var i = 0; i < cards.length; i++) cards[i].style.borderColor = '#30363d';
  var row = document.getElementById('trade-' + idx);
  var card = document.getElementById('card-' + idx);
  if (row) row.classList.add('highlight');
  if (card) card.style.borderColor = '#58a6ff';
  var buyD = buyDates[idx];
  var sellD = sellDates[idx];
  var mid = btcDates.indexOf(buyD);
  if (mid < 0) return;
  var sellIdx = btcDates.indexOf(sellD);
  if (sellIdx < 0) sellIdx = mid;
  var start = Math.max(0, mid - 50);
  var end = Math.min(btcDates.length - 1, sellIdx + 50);
  if (end < start) end = Math.min(btcDates.length - 1, start + 100);
  Plotly.relayout('chart', {'xaxis.range': [btcDates[start], btcDates[end]]});
}

window.addEventListener('resize', function() { Plotly.Plots.resize('chart'); });

function initChart() {
  render();
  document.getElementById('chart').on('plotly_click', function(data) {
    if (data.points && data.points[0]) {
      var clickDate = data.points[0].x;
      var closest = 0;
      var minDist = Infinity;
      buyDates.forEach(function(d, i) {
        var dist = Math.abs(new Date(d) - new Date(clickDate));
        if (dist < minDist) { minDist = dist; closest = i; }
      });
      highlightTrade(closest);
    }
  });
}

if (document.readyState === 'complete') {
  initChart();
} else {
  window.addEventListener('load', initChart);
}
</script>
</body>
</html>
"""


def build_trade_rows(trades: list[dict]):
    table_rows = ""
    card_rows = ""
    indexed = list(enumerate(trades))

    def action_date(item):
        i, t = item
        return (t.get("sell_date") or t.get("buy_date") or t.get("entry_signal_date") or "", i)

    indexed.sort(key=action_date, reverse=True)
    for i, t in indexed:
        is_open = not t.get("sell_date") or t.get("status") == "OPEN" or t.get("return_pct") is None
        if is_open:
            win_cls, ret_txt, sell_txt, sp_txt = "open", "OPEN", "OPEN", "—"
        else:
            win_cls = "win" if t["return_pct"] >= 0 else "loss"
            ret_txt = f'{t["return_pct"]:+.1f}%'
            sell_txt = t["sell_date"]
            sp_txt = f'${t["sell_price"]:,.0f}'
        phase = t.get("phase", "Benchmark")
        side = t.get("side", "LONG")
        side_cls = "short-tag" if side == "SHORT" else ""
        signal = t.get("entry_signal_date", "") or "—"
        bp_txt = f'${t["buy_price"]:,.0f}'
        hold = t.get("hold_days", 0)
        table_rows += (
            f'<tr id="trade-{i}" onclick="highlightTrade({i})">'
            f'<td>{i+1}</td><td>{phase}</td><td class="{side_cls}">{side}</td><td>{signal}</td>'
            f'<td>{t["buy_date"]}</td><td class="{win_cls}">{sell_txt}</td>'
            f'<td>{bp_txt}</td><td>{sp_txt}</td>'
            f'<td class="{win_cls}">{ret_txt}</td>'
            f'<td>{hold}d</td></tr>\n'
        )
        card_rows += (
            f'<div class="trade-card {win_cls}" id="card-{i}" onclick="highlightTrade({i})">'
            f'<div class="row"><span class="label">#{i+1} · {phase} · <span class="{side_cls}">{side}</span></span><span class="{win_cls}">{ret_txt}</span></div>'
            f'<div class="row"><span class="label">Signal</span><span>{signal}</span></div>'
            f'<div class="row"><span class="label">Entry</span><span>{t["buy_date"]} @ {bp_txt}</span></div>'
            f'<div class="row"><span class="label">Exit</span><span>{sell_txt} @ {sp_txt}</span></div>'
            f'<div class="row"><span class="label">Hold</span><span>{hold}d</span></div></div>\n'
        )
    return table_rows, card_rows


def generate_page(chart_df: pd.DataFrame, trades: list[dict], metrics: dict,
                  title: str, meta_line: str, notice_html: str, output_path: Path) -> None:
    dates = chart_df["date"].dt.strftime("%Y-%m-%d").tolist()
    prices = [round(float(v), 2) for v in chart_df["price"]]
    usdt_mcap = [round(float(v), 2) for v in chart_df["usdt_mcap"]]
    ks = [round(float(v), 2) if not np.isnan(float(v)) else 0 for v in chart_df["kalman_slope"]]
    kv = [round(float(v), 2) if not np.isnan(float(v)) else 0 for v in chart_df["kalman_vel"]]

    buy_dates = [t["buy_date"] for t in trades]
    buy_y = [prices[dates.index(t["buy_date"])] if t["buy_date"] in dates else 0 for t in trades]
    buy_sides = [t.get("side", "LONG") for t in trades]
    sell_dates = [t.get("sell_date") or "" for t in trades]
    plot_sell_dates, plot_sell_y = [], []
    for d in sell_dates:
        if d and d in dates:
            plot_sell_dates.append(d)
            plot_sell_y.append(prices[dates.index(d)])

    table_rows, card_rows = build_trade_rows(trades)

    html = HTML_TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__META__", meta_line)
    html = html.replace("__NOTICE__", notice_html)
    html = html.replace("__RETURN__", f"{metrics['total_return']*100:.1f}")
    html = html.replace("__SHARPE__", f"{metrics['daily_sharpe']:.2f}")
    html = html.replace("__WIN_RATE__", f"{metrics['win_rate']*100:.0f}")
    html = html.replace("__MAX_DD__", f"{metrics['max_drawdown']*100:.1f}")
    html = html.replace("__TABLE_ROWS__", table_rows)
    html = html.replace("__CARD_ROWS__", card_rows)
    html = html.replace("__BTC_DATES__", json.dumps(dates))
    html = html.replace("__BTC_PRICES__", json.dumps(prices))
    html = html.replace("__USDT_MCAP__", json.dumps(usdt_mcap))
    html = html.replace("__KALMAN_SLOPE__", json.dumps(ks))
    html = html.replace("__KALMAN_VEL__", json.dumps(kv))
    html = html.replace("__BUY_DATES__", json.dumps(buy_dates))
    html = html.replace("__BUY_Y__", json.dumps(buy_y))
    html = html.replace("__BUY_SIDES__", json.dumps(buy_sides))
    html = html.replace("__SELL_DATES__", json.dumps(sell_dates))
    html = html.replace("__PLOT_SELL_DATES__", json.dumps(plot_sell_dates))
    html = html.replace("__PLOT_SELL_Y__", json.dumps(plot_sell_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Written {output_path}")
