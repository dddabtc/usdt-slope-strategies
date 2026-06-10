#!/usr/bin/env python3
"""Daily live signal check for both strategies — cron entry point.

Designed to run shortly after 00:00 UTC (the daily snapshot time):
  1. refresh CoinGecko data, waiting up to MONITOR_WAIT_MIN for today's
     settled bar to appear;
  2. replay both strategies (immediate execution, recommended leverage)
     over the post-warmup window to derive the current position and any
     action that fired on the latest bar;
  3. regenerate all result pages (run_backtest.py);
  4. commit/push data + pages (MONITOR_COMMIT / MONITOR_PUSH, default on);
  5. print a JSON report as the last line for the cron agent to relay.

Signals execute immediately: an entry/exit on today's bar means act now,
not tomorrow.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import load_full_data, refresh_full_data
from src.engine import (
    LONG_ONLY, LONG_SHORT, RESEARCH_START, TRAIN_END, StrategyConfig,
    collect_trades, generate_signals, slice_data_window,
)

STATE_PATH = PROJECT / "data" / "monitor_state.json"
RECOMMENDED = {LONG_ONLY: 3.0, LONG_SHORT: 3.0}


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "no", "")


def wait_for_today_bar(max_wait_min: int) -> pd.Timestamp:
    """Refresh until the bar stamped today (00:00 UTC snapshot) is settled."""
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    deadline = time.time() + max_wait_min * 60
    while True:
        refresh_full_data()
        data = load_full_data()
        latest = min(data["btc"]["date"].max(), data["usdt"]["date"].max())
        if latest >= today or time.time() >= deadline:
            return latest
        print(f"  bar for {today.date()} not settled yet (have {latest.date()}); retry in 5 min")
        time.sleep(300)


def strategy_status(data: dict, mode: str, latest: pd.Timestamp) -> dict:
    cfg = StrategyConfig(mode=mode, leverage=RECOMMENDED[mode])
    bdata = slice_data_window(data, RESEARCH_START, latest)
    signals = generate_signals(bdata, cfg)
    trades, merged = collect_trades(signals, bdata["btc"], cfg, TRAIN_END, latest,
                                    right_censor_open=True)
    last_bar = str(latest.date())
    last_price = float(merged["price"].iloc[-1])
    row = signals[signals["date"] == latest]
    slope_pct = float(row["slope_pct"].iloc[0]) if not row.empty and pd.notna(row["slope_pct"].iloc[0]) else None

    open_trade = trades[-1] if trades and trades[-1].get("status") == "OPEN" else None
    closed_today = [t for t in trades
                    if t.get("status") == "CLOSED" and t.get("sell_date") == last_bar]

    if open_trade and open_trade["buy_date"] == last_bar:
        action = f"ENTER {open_trade['side']} NOW @ ~${open_trade['buy_price']:,.0f}"
    elif closed_today:
        t = closed_today[-1]
        action = (f"EXIT {t['side']} NOW ({t['exit_reason']}) @ ~${t['sell_price']:,.0f}, "
                  f"trade {t['return_pct']:+.1f}% net @{cfg.leverage:g}x")
    elif open_trade:
        side = 1 if open_trade["side"] == "LONG" else -1
        raw = side * (last_price / open_trade["buy_price"] - 1.0)
        action = (f"HOLD {open_trade['side']} since {open_trade['buy_date']} "
                  f"(raw {raw*100:+.1f}%, {raw*cfg.leverage*100:+.1f}% @{cfg.leverage:g}x)")
    else:
        action = "FLAT — no signal"

    return {
        "mode": mode,
        "leverage": cfg.leverage,
        "position": (open_trade["side"] if open_trade else "FLAT"),
        "entry_date": open_trade["buy_date"] if open_trade else None,
        "entry_price": open_trade["buy_price"] if open_trade else None,
        "action_today": action,
        "actionable": bool((open_trade and open_trade["buy_date"] == last_bar) or closed_today),
        "slope_pct": round(slope_pct, 4) if slope_pct is not None else None,
    }


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(PROJECT), *args],
                          capture_output=True, text=True)


def main() -> int:
    wait_min = int(os.environ.get("MONITOR_WAIT_MIN", "45"))
    do_commit = _env_flag("MONITOR_COMMIT")
    do_push = _env_flag("MONITOR_PUSH")

    print(f"=== USDT slope strategies daily check === {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")
    latest = wait_for_today_bar(wait_min)
    data = load_full_data()
    latest = min(data["btc"]["date"].max(), data["usdt"]["date"].max())
    today = pd.Timestamp(datetime.now(timezone.utc).date())

    btc_price = float(data["btc"]["price"].iloc[-1])
    usdt = data["usdt"]["market_cap"]
    mcap_chg_1d = float(usdt.iloc[-1] / usdt.iloc[-2] - 1.0) if len(usdt) > 1 else 0.0

    statuses = {mode: strategy_status(data, mode, latest) for mode in [LONG_ONLY, LONG_SHORT]}

    prev = {}
    if STATE_PATH.exists():
        try:
            prev = json.loads(STATE_PATH.read_text())
        except Exception:
            prev = {}
    already_alerted = (prev.get("last_bar") == str(latest.date())
                       and all(prev.get("strategies", {}).get(m, {}).get("action_today")
                               == statuses[m]["action_today"] for m in statuses))
    state = {
        "last_bar": str(latest.date()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "strategies": statuses,
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))

    print("--- regenerating pages/artifacts ---")
    r = subprocess.run([sys.executable, str(PROJECT / "run_backtest.py")],
                       capture_output=True, text=True)
    pages_ok = r.returncode == 0
    if not pages_ok:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])

    committed = pushed = False
    if do_commit:
        git("add", "data", "docs", "experiments")
        if git("diff", "--cached", "--quiet").returncode != 0:
            msg = f"signal update {datetime.now(timezone.utc):%Y-%m-%d %H:%M}"
            committed = git("commit", "-m", msg).returncode == 0
            if committed and do_push:
                p = git("push", "origin", "main")
                if p.returncode != 0:
                    git("pull", "--rebase", "--autostash")
                    p = git("push", "origin", "main")
                pushed = p.returncode == 0

    report = {
        "date": str(today.date()),
        "latest_bar": str(latest.date()),
        "stale_data": bool(latest < today),
        "btc_price": round(btc_price, 0),
        "usdt_mcap_change_1d_pct": round(mcap_chg_1d * 100, 3),
        "strategies": {m: {k: s[k] for k in
                           ["leverage", "position", "entry_date", "action_today", "actionable", "slope_pct"]}
                       for m, s in statuses.items()},
        "notify": any(s["actionable"] for s in statuses.values()) and not already_alerted,
        "already_alerted": already_alerted,
        "pages_regenerated": pages_ok,
        "committed": committed,
        "pushed": pushed,
        "pages": "https://dddabtc.github.io/usdt-slope-strategies/",
    }
    print("REPORT_JSON " + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
