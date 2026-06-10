#!/usr/bin/env python3
"""Nested walk-forward open strategy search.

Implements experiments/open-search/design.md:
  - five predeclared strategy families
  - 730d train / 180d test nested walk-forward selection
  - chained OOS daily returns per family
  - surrogate-null audit for the full F3 search

The BTC buy-and-hold reference deliberately ignores fold-boundary entry fees:
it is a raw BTC price-ratio benchmark, as predeclared in the task prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.data import load_full_data
from src.engine import (  # noqa: E402
    LONG_ONLY,
    LONG_SHORT,
    StrategyConfig,
    _exit_decision,
    _liquidation_idx,
    _net_trade_return,
    _rolling_slope,
    daily_equity_curve,
    equity_metrics,
    trade_metrics,
)
from src.visualize import build_kalman  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
LEDGER_PATH = BASE_DIR / "ledger.jsonl"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "run.log"

WARMUP_START = pd.Timestamp("2020-09-27")
EVAL_START = pd.Timestamp("2021-09-27")
PREDECLARED_END = pd.Timestamp("2026-06-09")
FIRST_TEST_START = pd.Timestamp("2022-09-27")
TRAIN_DAYS = 730
TEST_DAYS = 180

FAMILIES = ["F0", "F1", "F2", "F3", "F4"]
F3_EXITS = ["fixed3", "fixed5", "slope_peak", "trail10", "kalman75"]
F2_EXITS = ["fixed3", "fixed5", "fixed7", "trail10"]
F3_CORE_ENTRIES = ["bb", "zscore", "regime", "or_bb_regime", "accel", "consec"]
F4_ENTRIES = ["bb", "or_bb_regime", "zscore"]
PCT_FILTERS = [None, 0.90, 0.93]


@dataclass(frozen=True)
class Fold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._fh.close()

    def log(self, msg: str) -> None:
        line = f"{_utc_now()} {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()


class Progress:
    def __init__(self, total: int):
        self.total = max(int(total), 1)
        self.done = 0
        self.started = time.time()

    def tick(self) -> str:
        self.done += 1
        elapsed = max(time.time() - self.started, 1e-9)
        rate = elapsed / max(self.done, 1)
        remain = max(self.total - self.done, 0) * rate
        return f"{self.done}/{self.total}, eta {_fmt_duration(remain)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _date_str(ts: pd.Timestamp) -> str:
    return str(pd.Timestamp(ts).date())


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.normalize()


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return _date_str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_ledger(record: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "ts": _utc_now()}
    with LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonify(record), sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_ledger(scope: str) -> tuple[dict[tuple[str, int], dict], dict[str, dict], dict[int, dict]]:
    folds: dict[tuple[str, int], dict] = {}
    aggregates: dict[str, dict] = {}
    surrogates: dict[int, dict] = {}
    if not LEDGER_PATH.exists():
        return folds, aggregates, surrogates
    with LEDGER_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("scope") != scope:
                continue
            kind = rec.get("kind")
            if kind == "family_fold":
                folds[(rec["family"], int(rec["fold"]))] = rec
            elif kind == "family_aggregate":
                aggregates[rec["family"]] = rec
            elif kind == "surrogate_aggregate":
                surrogates[int(rec["seed"])] = rec
    return folds, aggregates, surrogates


def parse_families(value: str | None) -> list[str]:
    if not value:
        return FAMILIES.copy()
    families = [x.strip().upper() for x in value.split(",") if x.strip()]
    unknown = [x for x in families if x not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown families: {','.join(unknown)}")
    return families


def common_data(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    btc = data["btc"][["date", "price"]].copy()
    usdt = data["usdt"][["date", "market_cap"]].copy()
    btc["date"] = _norm_date(btc["date"])
    usdt["date"] = _norm_date(usdt["date"])
    latest = min(btc["date"].max(), usdt["date"].max(), PREDECLARED_END)
    btc = btc[(btc["date"] >= WARMUP_START) & (btc["date"] <= latest)].sort_values("date").reset_index(drop=True)
    usdt = usdt[(usdt["date"] >= WARMUP_START) & (usdt["date"] <= latest)].sort_values("date").reset_index(drop=True)
    common = btc[["date"]].merge(usdt[["date"]], on="date", how="inner")
    btc = common.merge(btc, on="date", how="left")
    usdt = common.merge(usdt, on="date", how="left")
    return btc, usdt, latest


def make_folds(actual_end: pd.Timestamp, smoke: bool = False) -> list[Fold]:
    folds = []
    test_start = FIRST_TEST_START
    while test_start <= actual_end:
        train_start = test_start - pd.Timedelta(days=TRAIN_DAYS)
        train_end = test_start - pd.Timedelta(days=1)
        test_end = min(test_start + pd.Timedelta(days=TEST_DAYS - 1), actual_end)
        folds.append(Fold(len(folds) + 1, train_start, train_end, test_start, test_end))
        test_start = test_start + pd.Timedelta(days=TEST_DAYS)
    return folds[:2] if smoke else folds


def rolling_percentile(values: np.ndarray, lookback: int = 365, min_valid: int = 30) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        window = values[max(0, i - lookback): i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < min_valid:
            continue
        out[i] = sp_stats.percentileofscore(valid, value) / 100.0
    return out


def zscore_signal(values: np.ndarray, window: int = 30) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(values)
    mid = s.rolling(window).mean().values
    std = s.rolling(window).std().values
    valid = ~np.isnan(values) & ~np.isnan(std) & (std > 0)
    z = np.full(len(values), np.nan)
    z[valid] = (values[valid] - mid[valid]) / std[valid]
    return z > 2.0, z < -2.0


def build_signal_cache(btc: pd.DataFrame, usdt: pd.DataFrame) -> pd.DataFrame:
    table = btc.merge(usdt, on="date", how="inner").sort_values("date").reset_index(drop=True)
    prices = table["price"].values.astype(float)
    mcap = table["market_cap"].values.astype(float)

    slopes = _rolling_slope(mcap, 3)
    table["slope"] = slopes
    table["slope_pct"] = rolling_percentile(slopes)
    table["regime_slope"] = _rolling_slope(mcap, 60)

    slope_s = pd.Series(slopes)
    mid = slope_s.rolling(30).mean().values
    std = slope_s.rolling(30).std().values
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    valid_bb = ~np.isnan(slopes) & ~np.isnan(std) & (std > 0)
    table["sig_bb"] = valid_bb & (slopes > upper)
    table["sig_bb_short"] = valid_bb & (slopes < lower)
    table["sig_zscore"], table["sig_zscore_short"] = zscore_signal(slopes)

    sos = np.diff(slopes, prepend=np.nan)
    sos_smooth = pd.Series(sos).rolling(3).mean().values
    avg_slope = slope_s.rolling(30).mean().values
    valid_rg = ~np.isnan(slopes) & ~np.isnan(sos_smooth) & ~np.isnan(avg_slope)
    accel = valid_rg & (slopes > 0) & (sos_smooth > 0) & (avg_slope > 0)
    ratio_l = np.where(accel & (avg_slope > 0), slopes / avg_slope, 0.0)
    regime_long = accel & (ratio_l > 1.0)
    decel = valid_rg & (slopes < 0) & (sos_smooth < 0) & (avg_slope < 0)
    ratio_s = np.where(decel & (avg_slope < 0), slopes / avg_slope, 0.0)
    regime_short = decel & (ratio_s > 1.0)
    table["sig_regime"] = regime_long
    table["sig_regime_short"] = regime_short
    table["sig_or_bb_regime"] = table[["sig_bb", "sig_regime"]].max(axis=1).astype(bool)
    table["sig_or_bb_regime_short"] = table[["sig_bb_short", "sig_regime_short"]].max(axis=1).astype(bool)

    second = np.diff(slopes, prepend=np.nan)
    accel_long, _accel_short = zscore_signal(second)
    table["sig_accel"] = accel_long

    up_days = pd.Series(mcap).pct_change() > 0
    table["sig_consec"] = (
        up_days & up_days.shift(1, fill_value=False) & up_days.shift(2, fill_value=False)
        & (pd.Series(mcap).pct_change(3) > 0.005)
    ).values

    btc_ret3 = pd.Series(prices).pct_change(3)
    table["btc_up"] = (btc_ret3 > 0).values
    table["btc_down"] = (btc_ret3 < 0).values
    for n in [10, 20, 55]:
        roll_max = pd.Series(prices).rolling(n).max().values
        table[f"sig_btc_breakout_{n}"] = (prices == roll_max) & table["btc_up"].values

    for window in [30, 60, 90]:
        slope = _rolling_slope(mcap, window)
        hold = slope > 0
        table[f"f1_slope_{window}"] = slope
        table[f"f1_hold_{window}"] = hold
        table[f"f1_entry_{window}"] = hold & ~pd.Series(hold).shift(1, fill_value=False).values

    kalman = build_kalman(usdt)
    kalman["date"] = _norm_date(kalman["date"])
    table = table.merge(kalman[["date", "kalman_vel"]], on="date", how="left")

    bool_cols = [c for c in table.columns if c.startswith("sig_") or c.startswith("f1_hold_") or c.startswith("f1_entry_")]
    for col in bool_cols + ["btc_up", "btc_down"]:
        table[col] = table[col].fillna(False).astype(bool)
    return table


def surrogate_usdt(usdt: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = usdt.copy()
    mcap = out["market_cap"].values.astype(float)
    d = np.diff(np.log(mcap))
    if len(d) <= 360:
        raise ValueError("USDT series is too short for the surrogate shift rule")
    k = int(rng.integers(180, len(d) - 180))
    shifted = np.roll(d, k)
    level = np.empty(len(mcap), dtype=float)
    level[0] = mcap[0]
    level[1:] = mcap[0] * np.exp(np.cumsum(shifted))
    out["market_cap"] = level
    return out


def surrogate_sanity() -> dict:
    data = load_full_data()
    btc, usdt, _actual_end = common_data(data)
    s = surrogate_usdt(usdt, 1)
    real = usdt["market_cap"].values.astype(float)
    sur = s["market_cap"].values.astype(float)
    assert len(sur) == len(real), "surrogate length mismatch"
    assert not np.isnan(sur).any(), "surrogate contains NaN"
    assert sur[0] == real[0], "surrogate first level changed"
    assert not np.allclose(sur, real), "surrogate equals real series"
    return {
        "length": int(len(sur)),
        "first_level": float(sur[0]),
        "real_first_level": float(real[0]),
        "differs": bool(not np.allclose(sur, real)),
    }


def pct_label(value: float | None) -> str:
    return "none" if value is None else f"{value:.2f}"


def family_configs(family: str, smoke: bool = False) -> list[dict]:
    if family == "F0":
        return [{"family": "F0", "config_id": "F0_buy_hold", "kind": "buy_hold"}]
    if family == "F1":
        return [
            {"family": "F1", "config_id": f"F1_W{window}", "window": window, "exit": "regime_flip"}
            for window in [30, 60, 90]
        ]
    if family == "F2":
        out = []
        for n in [10, 20, 55]:
            for exit_name in F2_EXITS:
                out.append({
                    "family": "F2",
                    "config_id": f"F2_N{n}_{exit_name}",
                    "breakout_n": n,
                    "exit": exit_name,
                })
        return out
    if family == "F3" and smoke:
        out = []
        for entry in ["bb", "or_bb_regime"]:
            for pct in [None, 0.93]:
                for exit_name in ["fixed3", "slope_peak"]:
                    out.append({
                        "family": "F3",
                        "config_id": f"F3_{entry}_pct{pct_label(pct)}_{exit_name}_confirm",
                        "entry_kind": entry,
                        "pct_filter": pct,
                        "exit": exit_name,
                        "confirm": True,
                    })
        return out
    if family == "F3":
        out = []
        for entry in F3_CORE_ENTRIES:
            for pct in PCT_FILTERS:
                for exit_name in F3_EXITS:
                    out.append({
                        "family": "F3",
                        "config_id": f"F3_{entry}_pct{pct_label(pct)}_{exit_name}_confirm",
                        "entry_kind": entry,
                        "pct_filter": pct,
                        "exit": exit_name,
                        "confirm": True,
                    })
        for threshold in [0.90, 0.93, 0.95]:
            for exit_name in F3_EXITS:
                out.append({
                    "family": "F3",
                    "config_id": f"F3_pct_event_{threshold:.2f}_{exit_name}_confirm",
                    "entry_kind": "pct_event",
                    "entry_threshold": threshold,
                    "pct_filter": None,
                    "exit": exit_name,
                    "confirm": True,
                })
        for entry in ["bb", "or_bb_regime"]:
            for exit_name in F3_EXITS:
                out.append({
                    "family": "F3",
                    "config_id": f"F3_{entry}_pctnone_{exit_name}_noconfirm",
                    "entry_kind": entry,
                    "pct_filter": None,
                    "exit": exit_name,
                    "confirm": False,
                })
        return out
    if family == "F4":
        out = []
        for entry in F4_ENTRIES:
            for pct in [None, 0.93]:
                for exit_name in F3_EXITS:
                    out.append({
                        "family": "F4",
                        "config_id": f"F4_{entry}_pct{pct_label(pct)}_{exit_name}",
                        "entry_kind": entry,
                        "pct_filter": pct,
                        "exit": exit_name,
                        "confirm": True,
                    })
        return out
    raise ValueError(f"unknown family {family}")


def engine_config(params: dict) -> StrategyConfig:
    mode = LONG_SHORT if params.get("family") == "F4" else LONG_ONLY
    return StrategyConfig(
        mode=mode,
        leverage=1.0,
        execution_delay_frac=0.0,
        fee_bps_per_side=5.0,
        long_funding_bps_per_day=3.0,
        short_funding_bps_per_day=0.0,
    )


def apply_config_signals(cache: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = cache.copy()
    family = params["family"]
    signal = np.zeros(len(df), dtype=bool)
    signal_short = np.zeros(len(df), dtype=bool)

    if family == "F1":
        window = int(params["window"])
        signal = df[f"f1_entry_{window}"].values.astype(bool)
        df["regime_hold"] = df[f"f1_hold_{window}"].values.astype(bool)
    elif family == "F2":
        signal = df[f"sig_btc_breakout_{int(params['breakout_n'])}"].values.astype(bool)
    elif family == "F3":
        entry = params["entry_kind"]
        if entry == "pct_event":
            signal = df["slope_pct"].values >= float(params["entry_threshold"])
        else:
            signal = df[f"sig_{entry}"].values.astype(bool)
        if params.get("confirm", True):
            signal &= df["btc_up"].values.astype(bool)
        pct = params.get("pct_filter")
        if pct is not None:
            signal &= df["slope_pct"].values >= float(pct)
    elif family == "F4":
        entry = params["entry_kind"]
        signal = df[f"sig_{entry}"].values.astype(bool)
        signal &= df["btc_up"].values.astype(bool)
        signal_short = df[f"sig_{entry}_short"].values.astype(bool)
        signal_short &= df["btc_down"].values.astype(bool)
        pct = params.get("pct_filter")
        if pct is not None:
            pct = float(pct)
            signal &= df["slope_pct"].values >= pct
            signal_short &= df["slope_pct"].values <= 1.0 - pct
        signal_short &= df["regime_slope"].values < 0
    else:
        raise ValueError(f"signals are not defined for {family}")

    df["signal"] = signal.astype(int)
    df["signal_short"] = signal_short.astype(int)
    return df


def fixed_exit_idx(df: pd.DataFrame, entry_idx: int, hold_days: int) -> tuple[int, str]:
    entry_date = pd.Timestamp(df.iloc[entry_idx]["date"])
    for j in range(entry_idx + 1, len(df)):
        days_held = (pd.Timestamp(df.iloc[j]["date"]) - entry_date).days
        if days_held >= hold_days:
            return j, f"fixed{hold_days}"
    return len(df) - 1, "open"


def trail_exit_idx(df: pd.DataFrame, entry_idx: int, side: int) -> tuple[int, str]:
    entry_date = pd.Timestamp(df.iloc[entry_idx]["date"])
    if side > 0:
        peak = float(df.iloc[entry_idx]["price"])
        for j in range(entry_idx + 1, len(df)):
            price = float(df.iloc[j]["price"])
            peak = max(peak, price)
            days_held = (pd.Timestamp(df.iloc[j]["date"]) - entry_date).days
            if price < peak * 0.90:
                return j, "trail10"
            if days_held >= 14:
                return j, "max_hold"
    else:
        trough = float(df.iloc[entry_idx]["price"])
        for j in range(entry_idx + 1, len(df)):
            price = float(df.iloc[j]["price"])
            trough = min(trough, price)
            days_held = (pd.Timestamp(df.iloc[j]["date"]) - entry_date).days
            if price > trough * 1.10:
                return j, "trail10"
            if days_held >= 14:
                return j, "max_hold"
    return len(df) - 1, "open"


def kalman_exit_idx(df: pd.DataFrame, entry_idx: int, side: int) -> tuple[int, str]:
    entry_date = pd.Timestamp(df.iloc[entry_idx]["date"])
    kv_e = df.iloc[entry_idx].get("kalman_vel", np.nan)
    kv_e = float(kv_e) if not pd.isna(kv_e) else np.nan
    for j in range(entry_idx + 1, len(df)):
        days_held = (pd.Timestamp(df.iloc[j]["date"]) - entry_date).days
        if days_held >= 14:
            return j, "max_hold"
        if days_held < 3:
            continue
        kv_j = df.iloc[j].get("kalman_vel", np.nan)
        kv_j = float(kv_j) if not pd.isna(kv_j) else np.nan
        slope_j = df.iloc[j].get("slope", np.nan)
        slope_j = float(slope_j) if not pd.isna(slope_j) else np.nan
        if side > 0:
            if not np.isnan(kv_e) and kv_e > 0:
                if not np.isnan(kv_j) and kv_j < 0.75 * kv_e:
                    return j, "kalman75"
            elif not np.isnan(slope_j) and slope_j < 0:
                return j, "slope_negative"
        else:
            if not np.isnan(kv_e) and kv_e < 0:
                if not np.isnan(kv_j) and kv_j > 0.75 * kv_e:
                    return j, "kalman75"
            elif not np.isnan(slope_j) and slope_j > 0:
                return j, "slope_positive"
    return len(df) - 1, "open"


def regime_flip_exit_idx(df: pd.DataFrame, entry_idx: int) -> tuple[int, str]:
    for j in range(entry_idx + 1, len(df)):
        if not bool(df.iloc[j].get("regime_hold", False)):
            return j, "regime_flip_negative"
    return len(df) - 1, "open"


def exit_idx(df: pd.DataFrame, entry_idx: int, side: int, params: dict, config: StrategyConfig) -> tuple[int, str]:
    exit_name = params.get("exit", "")
    if exit_name == "regime_flip":
        return regime_flip_exit_idx(df, entry_idx)
    if exit_name.startswith("fixed"):
        return fixed_exit_idx(df, entry_idx, int(exit_name.replace("fixed", "")))
    if exit_name == "slope_peak":
        return _exit_decision(df, entry_idx, side, config)
    if exit_name == "trail10":
        return trail_exit_idx(df, entry_idx, side)
    if exit_name == "kalman75":
        return kalman_exit_idx(df, entry_idx, side)
    raise ValueError(f"unknown exit {exit_name}")


def collect_custom_trades(signal_df: pd.DataFrame, params: dict, config: StrategyConfig,
                          eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> tuple[list[dict], pd.DataFrame]:
    df = signal_df[(signal_df["date"] >= eval_start) & (signal_df["date"] <= eval_end)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        return [], df

    allow_short = config.mode == LONG_SHORT
    trades: list[dict] = []
    capital = float(config.initial_capital)
    i = 0
    while i < len(df):
        row = df.iloc[i]
        if int(row.get("signal", 0)) == 1:
            side = 1
        elif allow_short and int(row.get("signal_short", 0)) == 1:
            side = -1
        else:
            i += 1
            continue

        buy_idx = i
        sell_idx, reason = exit_idx(df, buy_idx, side, params, config)
        sell_idx = min(sell_idx, len(df) - 1)
        buy = df.iloc[buy_idx]
        sell = df.iloc[sell_idx]
        buy_price = float(buy["price"])

        liq_idx = _liquidation_idx(df, buy_idx, sell_idx, side, config, entry_price=buy_price)
        liquidated = liq_idx is not None
        if liquidated:
            sell_idx = int(liq_idx)
            sell = df.iloc[sell_idx]
            reason = "liquidated"

        sell_price = float(sell["price"])
        raw_ret = (sell_price - buy_price) / buy_price
        hold_days = int((pd.Timestamp(sell["date"]) - pd.Timestamp(buy["date"])).days)
        net = -0.99 if liquidated else _net_trade_return(raw_ret, hold_days, side, config)
        capital *= 1.0 + net

        trades.append({
            "side": "LONG" if side > 0 else "SHORT",
            "entry_signal_date": _date_str(buy["date"]),
            "buy_date": _date_str(buy["date"]),
            "exit_signal_date": _date_str(sell["date"]),
            "sell_date": _date_str(sell["date"]),
            "buy_price": round(buy_price, 2),
            "sell_price": round(sell_price, 2),
            "raw_return_pct": round(raw_ret * 100, 2),
            "return_pct": round(net * 100, 2),
            "hold_days": hold_days,
            "exit_reason": reason,
            "status": "CLOSED",
            "execution_delay_frac": 0.0,
            "execution_lag_bars": None,
            "capital_after": round(capital, 2),
        })
        i = max(sell_idx + 1, buy_idx + 1)
    return trades, df


def buy_hold_backtest(cache: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> dict:
    df = cache[(cache["date"] >= eval_start) & (cache["date"] <= eval_end)][["date", "price"]].copy()
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        equity = pd.DataFrame(columns=["date", "equity", "daily_return", "drawdown"])
        metrics = aggregate_from_returns([], [])
        metrics.update({"eval_start": _date_str(eval_start), "eval_end": _date_str(eval_end)})
        return {"metrics": metrics, "equity": equity, "trades": [], "merged": df}
    rets = df["price"].pct_change().fillna(0.0)
    equity = pd.DataFrame({"date": df["date"], "daily_return": rets})
    equity["equity"] = (1.0 + equity["daily_return"]).cumprod() * 10_000.0
    equity["position"] = 1
    equity["peak_equity"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["peak_equity"] - 1.0
    metrics = {**equity_metrics(equity, 10_000.0), **trade_metrics([])}
    metrics.update({
        "eval_start": _date_str(eval_start),
        "eval_end": _date_str(eval_end),
        "btc_buy_hold_entry_fee_ignored": True,
    })
    return {"metrics": metrics, "equity": equity, "trades": [], "merged": df}


def backtest_params(cache: pd.DataFrame, params: dict, eval_start: pd.Timestamp,
                    eval_end: pd.Timestamp) -> dict:
    if params["family"] == "F0":
        return buy_hold_backtest(cache, eval_start, eval_end)
    config = engine_config(params)
    signals = apply_config_signals(cache, params)
    trades, merged = collect_custom_trades(signals, params, config, eval_start, eval_end)
    equity = daily_equity_curve(merged, trades, config)
    metrics = {**equity_metrics(equity, config.initial_capital), **trade_metrics(trades)}
    metrics.update({
        "eval_start": _date_str(eval_start),
        "eval_end": _date_str(eval_end),
        "mode": config.mode,
        "leverage": config.leverage,
        "fee_bps_per_side": config.fee_bps_per_side,
        "long_funding_bps_per_day": config.long_funding_bps_per_day,
        "short_funding_bps_per_day": config.short_funding_bps_per_day,
        "execution_delay_frac": config.execution_delay_frac,
        "execution_lag_bars": None,
    })
    return {"metrics": metrics, "equity": equity, "trades": trades, "merged": merged}


def selection_score(metrics: dict) -> tuple[int, float, float, int]:
    n_trades = int(metrics.get("n_trades", 0))
    return (
        1 if n_trades > 0 else 0,
        float(metrics.get("total_return", 0.0)),
        float(metrics.get("daily_sharpe", 0.0)),
        n_trades,
    )


def daily_return_records(equity: pd.DataFrame) -> list[dict]:
    if equity.empty:
        return []
    return [
        {"date": _date_str(row["date"]), "daily_return": float(row["daily_return"])}
        for _, row in equity[["date", "daily_return"]].iterrows()
    ]


def run_fold(family: str, configs: list[dict], cache: pd.DataFrame, fold: Fold) -> dict:
    best_params = None
    best_train = None
    best_score = None
    for params in configs:
        result = backtest_params(cache, params, fold.train_start, fold.train_end)
        score = selection_score(result["metrics"])
        if best_score is None or score > best_score:
            best_score = score
            best_params = params
            best_train = result["metrics"]
    assert best_params is not None
    assert best_train is not None

    oos = backtest_params(cache, best_params, fold.test_start, fold.test_end)
    return {
        "kind": "family_fold",
        "family": family,
        "fold": fold.fold,
        "train_start": _date_str(fold.train_start),
        "train_end": _date_str(fold.train_end),
        "test_start": _date_str(fold.test_start),
        "test_end": _date_str(fold.test_end),
        "selected_config": best_params,
        "selection_score": list(best_score or ()),
        "train_metrics": best_train,
        "oos_metrics": oos["metrics"],
        "oos_trades": oos["trades"],
        "oos_daily_returns": daily_return_records(oos["equity"]),
    }


def aggregate_from_returns(return_records: list[dict], trades: list[dict]) -> dict:
    if return_records:
        returns = np.array([float(r["daily_return"]) for r in return_records], dtype=float)
        equity = np.cumprod(1.0 + returns)
        total = float(equity[-1] - 1.0)
        peaks = np.maximum.accumulate(equity)
        max_dd = float((equity / peaks - 1.0).min())
        std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe = 0.0 if std == 0.0 else float(np.mean(returns) / std * np.sqrt(365))
    else:
        total = 0.0
        sharpe = 0.0
        max_dd = 0.0

    closed = [t for t in trades if t.get("status") != "OPEN" and t.get("return_pct") is not None]
    if closed:
        returns_t = np.array([float(t["return_pct"]) / 100.0 for t in closed])
        win_rate = float((returns_t > 0).mean())
        avg_return = float(returns_t.mean())
        avg_hold = float(np.mean([float(t["hold_days"]) for t in closed]))
    else:
        win_rate = 0.0
        avg_return = 0.0
        avg_hold = 0.0
    return {
        "total_return": round(total, 6),
        "daily_sharpe": round(sharpe, 6),
        "max_drawdown": round(max_dd, 6),
        "n_trades": int(len(closed)),
        "n_long": int(sum(1 for t in closed if t.get("side") == "LONG")),
        "n_short": int(sum(1 for t in closed if t.get("side") == "SHORT")),
        "win_rate": round(win_rate, 6),
        "avg_return_per_trade": round(avg_return, 6),
        "avg_hold_days": round(avg_hold, 2),
    }


def aggregate_family(family: str, fold_records: list[dict]) -> dict:
    fold_records = sorted(fold_records, key=lambda r: int(r["fold"]))
    returns: list[dict] = []
    trades: list[dict] = []
    selected = []
    for rec in fold_records:
        returns.extend(rec.get("oos_daily_returns", []))
        trades.extend(rec.get("oos_trades", []))
        selected.append({
            "fold": rec["fold"],
            "selected_config": rec["selected_config"],
            "train_metrics": rec["train_metrics"],
            "oos_metrics": rec["oos_metrics"],
        })
    metrics = aggregate_from_returns(returns, trades)
    if family == "F0":
        metrics["btc_buy_hold_entry_fee_ignored"] = True
    return {
        "kind": "family_aggregate",
        "family": family,
        "aggregate": metrics,
        "folds": selected,
    }


def compute_family(family: str, cache: pd.DataFrame, folds: list[Fold],
                   smoke: bool = False) -> tuple[list[dict], dict]:
    configs = family_configs(family, smoke=smoke)
    fold_records = [run_fold(family, configs, cache, fold) for fold in folds]
    return fold_records, aggregate_family(family, fold_records)


def run_family_with_ledger(family: str, cache: pd.DataFrame, folds: list[Fold], scope: str,
                           completed_folds: dict[tuple[str, int], dict],
                           completed_aggs: dict[str, dict], progress: Progress,
                           logger: Logger, smoke: bool = False) -> tuple[list[dict], dict]:
    configs = family_configs(family, smoke=smoke)
    fold_records = []
    for fold in folds:
        key = (family, fold.fold)
        if key in completed_folds:
            rec = completed_folds[key]
            fold_records.append(rec)
            logger.log(f"skip {scope} {family} fold {fold.fold}: selected {rec['selected_config']['config_id']} ({progress.tick()})")
            continue
        rec = run_fold(family, configs, cache, fold)
        rec = {**rec, "scope": scope}
        _append_ledger(rec)
        fold_records.append(rec)
        m = rec["oos_metrics"]
        logger.log(
            f"done {scope} {family} fold {fold.fold}: selected {rec['selected_config']['config_id']} "
            f"train_ret {rec['train_metrics']['total_return']:+.4f} oos_ret {m['total_return']:+.4f} "
            f"oos_trades {m['n_trades']} ({progress.tick()})"
        )

    if family in completed_aggs:
        agg = completed_aggs[family]
        logger.log(f"skip {scope} {family} aggregate: ret {agg['aggregate']['total_return']:+.4f} ({progress.tick()})")
        return fold_records, agg

    agg = aggregate_family(family, fold_records)
    agg = {**agg, "scope": scope}
    _append_ledger(agg)
    logger.log(
        f"done {scope} {family} aggregate: ret {agg['aggregate']['total_return']:+.4f} "
        f"sharpe {agg['aggregate']['daily_sharpe']:+.3f} trades {agg['aggregate']['n_trades']} ({progress.tick()})"
    )
    return fold_records, agg


def run_surrogates(n: int, real_f3: dict | None, btc: pd.DataFrame, usdt: pd.DataFrame,
                   folds: list[Fold], scope: str, completed: dict[int, dict],
                   progress: Progress, logger: Logger) -> list[dict]:
    rows = []
    for seed in range(1, n + 1):
        if seed in completed:
            rec = completed[seed]
            rows.append(rec)
            logger.log(f"skip {scope} surrogate seed {seed}: ret {rec['aggregate']['total_return']:+.4f} ({progress.tick()})")
            continue
        shifted = surrogate_usdt(usdt, seed)
        cache = build_signal_cache(btc, shifted)
        fold_records, agg = compute_family("F3", cache, folds, smoke=False)
        rec = {
            "kind": "surrogate_aggregate",
            "scope": scope,
            "seed": seed,
            "aggregate": agg["aggregate"],
            "folds": [
                {
                    "fold": r["fold"],
                    "selected_config": r["selected_config"],
                    "train_metrics": r["train_metrics"],
                    "oos_metrics": r["oos_metrics"],
                }
                for r in fold_records
            ],
        }
        _append_ledger(rec)
        rows.append(rec)
        logger.log(
            f"done {scope} surrogate seed {seed}: ret {rec['aggregate']['total_return']:+.4f} "
            f"sharpe {rec['aggregate']['daily_sharpe']:+.3f} ({progress.tick()})"
        )
    if real_f3 is not None and rows:
        real = float(real_f3["aggregate"]["total_return"])
        null = np.array([float(r["aggregate"]["total_return"]) for r in rows], dtype=float)
        pct = float((null <= real).mean())
        p = float((1 + (null >= real).sum()) / (len(null) + 1))
        logger.log(f"surrogate null: real F3 percentile {pct:.3f}, p_value {p:.3f}")
    return rows


def surrogate_summary(real_f3: dict | None, rows: list[dict]) -> dict:
    distribution = [
        {
            "seed": int(r["seed"]),
            "total_return": r["aggregate"]["total_return"],
            "daily_sharpe": r["aggregate"]["daily_sharpe"],
        }
        for r in sorted(rows, key=lambda x: int(x["seed"]))
    ]
    if real_f3 is None or not distribution:
        return {
            "n": len(distribution),
            "distribution": distribution,
            "real_f3_total_return": None if real_f3 is None else real_f3["aggregate"]["total_return"],
            "total_return_percentile": None,
            "p_value": None,
        }
    real = float(real_f3["aggregate"]["total_return"])
    null = np.array([float(r["total_return"]) for r in distribution], dtype=float)
    return {
        "n": len(distribution),
        "distribution": distribution,
        "real_f3_total_return": real_f3["aggregate"]["total_return"],
        "real_f3_daily_sharpe": real_f3["aggregate"]["daily_sharpe"],
        "total_return_percentile": round(float((null <= real).mean()), 6),
        "p_value": round(float((1 + (null >= real).sum()) / (len(null) + 1)), 6),
    }


def write_results(scope: str, actual_end: pd.Timestamp, folds: list[Fold], families: list[str],
                  family_records: dict[str, list[dict]], family_aggs: dict[str, dict],
                  surrogate_rows: list[dict], surrogates_requested: int) -> dict:
    leaderboard = sorted(
        [
            {
                "family": family,
                **agg["aggregate"],
            }
            for family, agg in family_aggs.items()
        ],
        key=lambda r: float(r["total_return"]),
        reverse=True,
    )
    real_f3 = family_aggs.get("F3")
    payload = {
        "run": {
            "scope": scope,
            "warmup_start": _date_str(WARMUP_START),
            "eval_start": _date_str(EVAL_START),
            "actual_end": _date_str(actual_end),
            "fold_count": len(folds),
            "families": families,
            "surrogates_requested": surrogates_requested,
            "buy_hold_note": "F0 ignores fold-boundary entry fees and uses raw BTC price ratios.",
        },
        "leaderboard": leaderboard,
        "families": {
            family: {
                "aggregate": family_aggs[family]["aggregate"],
                "folds": family_aggs[family]["folds"],
            }
            for family in family_aggs
        },
        "surrogate_null": surrogate_summary(real_f3, surrogate_rows),
    }

    per_fold = []
    for family in families:
        for rec in sorted(family_records.get(family, []), key=lambda r: int(r["fold"])):
            per_fold.append({
                "family": family,
                "fold": rec["fold"],
                "train_start": rec["train_start"],
                "train_end": rec["train_end"],
                "test_start": rec["test_start"],
                "test_end": rec["test_end"],
                "selected_config": rec["selected_config"],
                "selection_score": rec["selection_score"],
                "train_metrics": rec["train_metrics"],
                "oos_metrics": rec["oos_metrics"],
            })

    _atomic_write_json(RESULTS_DIR / "leaderboard.json", payload)
    _atomic_write_json(RESULTS_DIR / "per_fold.json", per_fold)
    return payload


def print_leaderboard(payload: dict) -> None:
    print("\nLeaderboard", flush=True)
    print("family  total_return  sharpe  max_dd  trades  win_rate", flush=True)
    for row in payload["leaderboard"]:
        print(
            f"{row['family']:>6}  {row['total_return']*100:+11.2f}%  "
            f"{row['daily_sharpe']:+6.2f}  {row['max_drawdown']*100:+6.2f}%  "
            f"{row['n_trades']:>6}  {row['win_rate']*100:>7.1f}%",
            flush=True,
        )
    null = payload["surrogate_null"]
    if null["n"]:
        print(
            f"surrogate_null n={null['n']} real_f3={null['real_f3_total_return']*100:+.2f}% "
            f"pct={null['total_return_percentile']:.3f} p={null['p_value']:.3f}",
            flush=True,
        )


def run(args: argparse.Namespace) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scope = "smoke" if args.smoke else "full"
    families = ["F3"] if args.smoke else parse_families(args.families)
    surrogates = 0 if args.smoke else int(args.surrogates)
    if surrogates > 0 and "F3" not in families:
        families.append("F3")

    logger = Logger(LOG_PATH)
    try:
        logger.log(
            f"startup scope={scope} families={','.join(families)} surrogates={surrogates} "
            f"ledger={LEDGER_PATH}"
        )
        data = load_full_data()
        btc, usdt, actual_end = common_data(data)
        folds = make_folds(actual_end, smoke=args.smoke)
        cache = build_signal_cache(btc, usdt)
        logger.log(
            f"data warmup={_date_str(WARMUP_START)} actual_end={_date_str(actual_end)} "
            f"rows={len(cache)} folds={len(folds)}"
        )

        completed_folds, completed_aggs, completed_surrogates = load_ledger(scope)
        total_units = len(families) * (len(folds) + 1) + surrogates
        progress = Progress(total_units)

        family_records: dict[str, list[dict]] = {}
        family_aggs: dict[str, dict] = {}
        for family in families:
            records, agg = run_family_with_ledger(
                family, cache, folds, scope, completed_folds, completed_aggs,
                progress, logger, smoke=args.smoke,
            )
            family_records[family] = records
            family_aggs[family] = agg

        surrogate_rows = run_surrogates(
            surrogates, family_aggs.get("F3"), btc, usdt, folds, scope,
            completed_surrogates, progress, logger,
        )
        payload = write_results(
            scope, actual_end, folds, families, family_records, family_aggs,
            surrogate_rows, surrogates,
        )
        logger.log(f"results written {RESULTS_DIR / 'leaderboard.json'} and {RESULTS_DIR / 'per_fold.json'}")
        print_leaderboard(payload)
        logger.log("final summary complete")
        return payload
    finally:
        logger.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", help="comma-separated subset: F0,F1,F2,F3,F4")
    parser.add_argument("--surrogates", type=int, default=12, help="surrogate seeds 1..N (default: 12)")
    parser.add_argument("--smoke", action="store_true", help="F3 first 2 folds on the reduced smoke grid; no surrogates")
    parser.add_argument("--surrogate-sanity", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.surrogate_sanity:
        result = surrogate_sanity()
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    run(args)


if __name__ == "__main__":
    main()
