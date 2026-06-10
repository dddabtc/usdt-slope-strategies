#!/usr/bin/env python3
"""Mechanism-round studies and walk-forward variants.

Implements experiments/mechanism-round/design.md using the open-search
walk-forward machinery for folds, scoring, costs, metrics, logging, and ledger
style.  The descriptive studies are explanatory only; variant selection is
train-only and chained OOS, matching open-search.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "experiments" / "open-search"))
import search as osearch  # noqa: E402

sys.path.insert(0, str(PROJECT))
from src.data import load_full_data  # noqa: E402
from src.engine import LONG_ONLY, StrategyConfig  # noqa: E402
from src.visualize import KalmanSlope  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
LEDGER_PATH = BASE_DIR / "ledger.jsonl"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "run.log"
DESCRIPTIVE_PATH = RESULTS_DIR / "descriptive.json"
VARIANTS_PATH = RESULTS_DIR / "variants.json"

SCOPE = "mech"
FAMILIES = ["M2_NORM", "M3_PERSIST", "M4_CATCHUP", "M5_PRICEEXIT"]
DISPLAY_FAMILY = {
    "M2_NORM": "M2_norm",
    "M3_PERSIST": "M3_persist",
    "M4_CATCHUP": "M4_catchup",
    "M5_PRICEEXIT": "M5_priceexit",
}
FAMILY_ALIASES = {v.upper(): k for k, v in DISPLAY_FAMILY.items()}
FAMILY_ALIASES.update({k: k for k in FAMILIES})
FAMILY_ALIASES.update({
    "M2": "M2_NORM",
    "M3": "M3_PERSIST",
    "M4": "M4_CATCHUP",
    "M5": "M5_PRICEEXIT",
})

CORE_ENTRIES = osearch.F3_CORE_ENTRIES
PCT_FILTERS = osearch.PCT_FILTERS
F3_EXITS = osearch.F3_EXITS
PCT_EVENT_THRESHOLDS = [0.90, 0.93, 0.95]
BOOTSTRAP_SEED = 20260610
RANDOM_MATCH_SEED = 20260611

ORIG_OSEARCH_EXIT_IDX = osearch.exit_idx


def _jsonify(obj: Any) -> Any:
    return osearch._jsonify(obj)


def _date_str(ts: pd.Timestamp) -> str:
    return osearch._date_str(ts)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_ledger(record: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "ts": osearch._utc_now()}
    with LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonify(record), sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_mech_ledger() -> tuple[dict[tuple[str, int], dict], dict[str, dict], dict[str, dict]]:
    folds: dict[tuple[str, int], dict] = {}
    aggregates: dict[str, dict] = {}
    descriptive: dict[str, dict] = {}
    if not LEDGER_PATH.exists():
        return folds, aggregates, descriptive
    with LEDGER_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("scope") != SCOPE:
                continue
            kind = rec.get("kind")
            if kind == "family_fold":
                folds[(rec["family"], int(rec["fold"]))] = rec
            elif kind == "family_aggregate":
                aggregates[rec["family"]] = rec
            elif kind == "descriptive":
                descriptive[str(rec.get("part", "all"))] = rec
    return folds, aggregates, descriptive


def parse_families(value: str | None, smoke: bool = False) -> list[str]:
    if smoke:
        return ["M5_PRICEEXIT"]
    if not value:
        return FAMILIES.copy()
    out = []
    unknown = []
    for raw in value.split(","):
        key = raw.strip().upper()
        if not key:
            continue
        fam = FAMILY_ALIASES.get(key)
        if fam is None:
            unknown.append(raw.strip())
        else:
            out.append(fam)
    if unknown:
        raise SystemExit(f"unknown mechanism families: {','.join(unknown)}")
    return out


def add_btc_momentum_cols(cache: pd.DataFrame) -> pd.DataFrame:
    df = cache.copy()
    ret3 = pd.Series(df["price"].values.astype(float)).pct_change(3)
    df["btc_ret3"] = ret3.values
    df["btc_down_or_flat"] = (ret3 <= 0).fillna(False).values
    return df


def kalman_on_slope(dates: pd.Series, slopes: np.ndarray) -> pd.DataFrame:
    kf = KalmanSlope()
    ks = np.full(len(slopes), np.nan)
    kv = np.full(len(slopes), np.nan)
    for i, value in enumerate(slopes):
        if np.isnan(value):
            continue
        kf.update(float(value))
        ks[i] = kf.x[0, 0]
        kv[i] = kf.x[1, 0]
    return pd.DataFrame({"date": dates, "kalman_slope": ks, "kalman_vel": kv})


def build_relative_signal_cache(btc: pd.DataFrame, usdt: pd.DataFrame) -> pd.DataFrame:
    table = btc.merge(usdt, on="date", how="inner").sort_values("date").reset_index(drop=True)
    prices = table["price"].values.astype(float)
    mcap = table["market_cap"].values.astype(float)
    log_mcap = np.log(mcap)

    slopes = osearch._rolling_slope(log_mcap, 3)
    table["slope"] = slopes
    table["slope_pct"] = osearch.rolling_percentile(slopes)
    table["regime_slope"] = osearch._rolling_slope(log_mcap, 60)

    slope_s = pd.Series(slopes)
    mid = slope_s.rolling(30).mean().values
    std = slope_s.rolling(30).std().values
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    valid_bb = ~np.isnan(slopes) & ~np.isnan(std) & (std > 0)
    table["sig_bb"] = valid_bb & (slopes > upper)
    table["sig_bb_short"] = valid_bb & (slopes < lower)
    table["sig_zscore"], table["sig_zscore_short"] = osearch.zscore_signal(slopes)

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
    accel_long, _accel_short = osearch.zscore_signal(second)
    table["sig_accel"] = accel_long

    rel = pd.Series(slopes)
    table["sig_consec"] = (
        (rel > 0)
        & (rel.shift(1, fill_value=np.nan) > 0)
        & (rel.shift(2, fill_value=np.nan) > 0)
        & (rel.rolling(3).sum() > 0.005)
    ).values

    btc_ret3 = pd.Series(prices).pct_change(3)
    table["btc_up"] = (btc_ret3 > 0).fillna(False).values
    table["btc_down"] = (btc_ret3 < 0).fillna(False).values
    table["btc_ret3"] = btc_ret3.values
    table["btc_down_or_flat"] = (btc_ret3 <= 0).fillna(False).values

    for n in [10, 20, 55]:
        roll_max = pd.Series(prices).rolling(n).max().values
        table[f"sig_btc_breakout_{n}"] = (prices == roll_max) & table["btc_up"].values

    for window in [30, 60, 90]:
        slope = osearch._rolling_slope(log_mcap, window)
        hold = slope > 0
        table[f"f1_slope_{window}"] = slope
        table[f"f1_hold_{window}"] = hold
        table[f"f1_entry_{window}"] = hold & ~pd.Series(hold).shift(1, fill_value=False).values

    kalman = kalman_on_slope(table["date"], slopes)
    kalman["date"] = osearch._norm_date(kalman["date"])
    table = table.merge(kalman[["date", "kalman_vel"]], on="date", how="left")

    bool_cols = [
        c for c in table.columns
        if c.startswith("sig_") or c.startswith("f1_hold_") or c.startswith("f1_entry_")
    ]
    for col in bool_cols + ["btc_up", "btc_down", "btc_down_or_flat"]:
        table[col] = table[col].fillna(False).astype(bool)
    return table


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.05, 0.95])
    return float(lo), float(hi)


def forward_return_matrix(cache: pd.DataFrame, indices: np.ndarray, max_horizon: int) -> np.ndarray:
    prices = cache["price"].values.astype(float)
    out = np.full((len(indices), max_horizon + 1), np.nan)
    for row, idx in enumerate(indices):
        max_k = min(max_horizon, len(prices) - int(idx) - 1)
        if max_k < 0:
            continue
        base = prices[int(idx)]
        out[row, : max_k + 1] = prices[int(idx): int(idx) + max_k + 1] / base - 1.0
    return out


def crossing_indices(values: np.ndarray, threshold: float) -> np.ndarray:
    prev = np.roll(values, 1)
    prev[0] = np.nan
    mask = (values >= threshold) & ((prev < threshold) | np.isnan(prev))
    return np.flatnonzero(mask & np.isfinite(values))


def a1_event_study(cache: pd.DataFrame, n_boot: int, random_draws: int) -> dict:
    slope_pct = cache["slope_pct"].values.astype(float)
    event_idx = crossing_indices(slope_pct, 0.93)
    event_mat = forward_return_matrix(cache, event_idx, 14)

    horizons = []
    for k in range(15):
        vals = event_mat[:, k]
        vals = vals[np.isfinite(vals)]
        ci_low, ci_high = bootstrap_mean_ci(vals, n_boot, BOOTSTRAP_SEED + k)
        horizons.append({
            "k": k,
            "mean": None if len(vals) == 0 else float(vals.mean()),
            "median": None if len(vals) == 0 else float(np.median(vals)),
            "ci90_low": ci_low,
            "ci90_high": ci_high,
            "n_events": int(len(vals)),
        })

    all_indices = np.arange(len(cache), dtype=int)
    all_mat = forward_return_matrix(cache, all_indices, 14)
    all_days_baseline = [
        {
            "k": k,
            "mean": None if np.isfinite(all_mat[:, k]).sum() == 0 else float(np.nanmean(all_mat[:, k])),
            "n_days": int(np.isfinite(all_mat[:, k]).sum()),
        }
        for k in range(15)
    ]

    event_set = set(int(i) for i in event_idx)
    non_event = np.array(
        [i for i in range(len(cache) - 5) if i not in event_set and np.isfinite(cache["price"].iloc[i + 5])],
        dtype=int,
    )
    rng = np.random.default_rng(RANDOM_MATCH_SEED)
    sample_means = []
    if len(event_idx) and len(non_event):
        for _ in range(random_draws):
            replace = len(non_event) < len(event_idx)
            picks = rng.choice(non_event, size=len(event_idx), replace=replace)
            rets = cache["price"].values[picks + 5] / cache["price"].values[picks] - 1.0
            sample_means.append(float(np.mean(rets)))
    if sample_means:
        band = np.quantile(np.array(sample_means, dtype=float), [0.05, 0.95])
        random_matched = {
            "horizon": 5,
            "draws": int(random_draws),
            "sample_size": int(len(event_idx)),
            "mean": float(np.mean(sample_means)),
            "ci90_low": float(band[0]),
            "ci90_high": float(band[1]),
        }
    else:
        random_matched = {
            "horizon": 5,
            "draws": int(random_draws),
            "sample_size": int(len(event_idx)),
            "mean": None,
            "ci90_low": None,
            "ci90_high": None,
        }

    return {
        "event_definition": "slope_pct crosses upward through 0.93; no BTC confirmation",
        "n_events": int(len(event_idx)),
        "events": [_date_str(cache.iloc[int(i)]["date"]) for i in event_idx],
        "horizons": horizons,
        "all_days_baseline": all_days_baseline,
        "random_matched_non_event_5d": random_matched,
    }


def a2_dose_response(cache: pd.DataFrame, n_boot: int) -> dict:
    slope_pct = cache["slope_pct"].values.astype(float)
    buckets = [
        ("0.90_0.93", 0.90, 0.93),
        ("0.93_0.97", 0.93, 0.97),
        ("0.97_1.00", 0.97, 1.0000001),
    ]
    rows = []
    used_dates: set[str] = set()
    for pos, (name, lower, upper) in enumerate(buckets):
        idx = crossing_indices(slope_pct, lower)
        vals_idx = []
        for i in idx:
            value = slope_pct[int(i)]
            date = _date_str(cache.iloc[int(i)]["date"])
            if date in used_dates:
                continue
            if lower <= value < upper and int(i) + 5 < len(cache):
                vals_idx.append(int(i))
                used_dates.add(date)
        rets = np.array(
            [
                cache["price"].iloc[i + 5] / cache["price"].iloc[i] - 1.0
                for i in vals_idx
            ],
            dtype=float,
        )
        ci_low, ci_high = bootstrap_mean_ci(rets, n_boot, BOOTSTRAP_SEED + 100 + pos)
        rows.append({
            "bucket": name,
            "lower": lower,
            "upper": 1.0 if upper > 1.0 else upper,
            "n": int(len(rets)),
            "mean_5d": None if len(rets) == 0 else float(rets.mean()),
            "median_5d": None if len(rets) == 0 else float(np.median(rets)),
            "ci90_low": ci_low,
            "ci90_high": ci_high,
        })
    return {
        "event_definition": "upward crossing of each bucket lower bound; assigned by slope_pct at crossing date",
        "buckets": rows,
    }


def a3_hold_day_alpha(cache: pd.DataFrame, n_boot: int) -> dict:
    df = cache.sort_values("date").reset_index(drop=True)
    signal = (
        df["sig_or_bb_regime"].values.astype(bool)
        & df["btc_up"].values.astype(bool)
        & (df["slope_pct"].values.astype(float) >= 0.93)
    )
    prices = df["price"].values.astype(float)
    paths = []
    dates = []
    i = 0
    while i < len(df):
        if not signal[i]:
            i += 1
            continue
        if i + 14 < len(df):
            paths.append(prices[i + 1: i + 15] / prices[i] - 1.0)
            dates.append(_date_str(df.iloc[i]["date"]))
        i += 15

    mat = np.array(paths, dtype=float) if paths else np.empty((0, 14))
    profile = []
    for day in range(1, 15):
        vals = mat[:, day - 1] if len(mat) else np.array([], dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(vals, n_boot, BOOTSTRAP_SEED + 200 + day)
        profile.append({
            "day": day,
            "mean": None if len(vals) == 0 else float(vals.mean()),
            "median": None if len(vals) == 0 else float(np.median(vals)),
            "ci90_low": ci_low,
            "ci90_high": ci_high,
            "n_trades": int(len(vals)),
        })
    return {
        "entry_definition": "sig_or_bb_regime AND btc_up AND slope_pct >= 0.93; fixed 14d sequencing",
        "n_trades": int(len(mat)),
        "entry_dates": dates,
        "profile": profile,
    }


def corr_pair(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    xs = x[mask]
    ys = y[mask]
    if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def a4_response_lag(cache: pd.DataFrame) -> dict:
    mcap = cache["market_cap"].values.astype(float)
    prices = cache["price"].values.astype(float)
    dlog = np.diff(np.log(mcap), prepend=np.nan)
    btc_ret = pd.Series(prices).pct_change().values.astype(float)
    restricted_mask = cache["slope_pct"].values.astype(float) >= 0.90

    full = []
    restricted = []
    for k in range(11):
        y = np.full(len(btc_ret), np.nan)
        if k == 0:
            y[:] = btc_ret
        else:
            y[:-k] = btc_ret[k:]
        mask = np.isfinite(dlog) & np.isfinite(y)
        full.append({
            "k": k,
            "corr": corr_pair(dlog, y),
            "n": int(mask.sum()),
        })
        rmask = mask & restricted_mask
        restricted.append({
            "k": k,
            "corr": corr_pair(dlog[rmask], y[rmask]),
            "n": int(rmask.sum()),
        })
    return {
        "full": full,
        "slope_pct_ge_0.90": restricted,
    }


def part_a_checks(descriptive: dict) -> dict:
    a1_h = descriptive["a1_event_study"]["horizons"]
    day1_3 = [float(r["mean"]) for r in a1_h if 1 <= int(r["k"]) <= 3 and r["mean"] is not None]
    a1_day1_3_mean = None if not day1_3 else float(np.mean(day1_3))
    a1_positive = bool(a1_day1_3_mean is not None and a1_day1_3_mean > 0)

    bucket_means = [
        r["mean_5d"]
        for r in descriptive["a2_dose_response"]["buckets"]
        if r["mean_5d"] is not None and int(r["n"]) > 0
    ]
    a2_monotone = bool(len(bucket_means) >= 2 and all(bucket_means[i] <= bucket_means[i + 1] for i in range(len(bucket_means) - 1)))

    profile = descriptive["a3_hold_day_alpha"]["profile"]
    means = [r["mean"] for r in profile if r["mean"] is not None]
    if means:
        day_of_max = int(np.nanargmax(np.array(means, dtype=float)) + 1)
        mean10 = means[9] if len(means) >= 10 else means[-1]
        mean14 = means[13] if len(means) >= 14 else means[-1]
        early_gain = abs(mean10 - means[0]) if len(means) >= 10 else abs(means[-1] - means[0])
        late_gain = mean14 - mean10
        a3_flattens = bool(day_of_max <= 10 or late_gain <= max(0.002, 0.25 * early_gain))
    else:
        day_of_max = None
        late_gain = None
        a3_flattens = False

    a4_full = descriptive["a4_response_lag"]["full"]
    corr_rows = [r for r in a4_full if r["corr"] is not None]
    if corr_rows:
        peak_row = max(corr_rows, key=lambda r: float(r["corr"]))
        peak_k = int(peak_row["k"])
        peak_corr = float(peak_row["corr"])
    else:
        peak_k = None
        peak_corr = None

    return {
        "a1_day1_3_mean_positive": a1_positive,
        "a1_day1_3_mean": a1_day1_3_mean,
        "a2_monotone_bucket_means": a2_monotone,
        "a3_curve_flattens_before_day14": a3_flattens,
        "a3_day_of_max_mean": day_of_max,
        "a3_day10_to_day14_gain": late_gain,
        "a4_peak_corr_k_le_1": bool(peak_k is not None and peak_k <= 1),
        "a4_peak_corr_k": peak_k,
        "a4_peak_corr": peak_corr,
        "family_direction_checks": {
            "M2_norm": bool(a2_monotone and peak_k is not None and peak_k <= 1),
            "M3_persist": a1_positive,
            "M4_catchup": bool(a1_positive and peak_k is not None and peak_k > 1),
            "M5_priceexit": a3_flattens,
        },
    }


def compute_part_a(cache: pd.DataFrame, actual_end: pd.Timestamp, smoke: bool, logger: osearch.Logger | None = None) -> dict:
    n_boot = 200 if smoke else 10_000
    random_draws = 200 if smoke else 2_000
    if logger is not None:
        logger.log(
            f"Part A recompute start smoke={smoke} bootstraps={n_boot} random_draws={random_draws}; "
            "descriptive resume is by atomic overwrite"
        )
    descriptive = {
        "run": {
            "scope": SCOPE,
            "smoke": smoke,
            "warmup_start": _date_str(osearch.WARMUP_START),
            "actual_end": _date_str(actual_end),
            "bootstraps": n_boot,
            "random_matched_draws": random_draws,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "a1_event_study": a1_event_study(cache, n_boot, random_draws),
        "a2_dose_response": a2_dose_response(cache, n_boot),
        "a3_hold_day_alpha": a3_hold_day_alpha(cache, n_boot),
        "a4_response_lag": a4_response_lag(cache),
    }
    descriptive["checks"] = part_a_checks(descriptive)
    _atomic_write_json(DESCRIPTIVE_PATH, descriptive)
    _append_ledger({
        "kind": "descriptive",
        "scope": SCOPE,
        "part": "A",
        "smoke": smoke,
        "path": str(DESCRIPTIVE_PATH.relative_to(PROJECT)),
        "bootstraps": n_boot,
    })
    if logger is not None:
        logger.log(f"Part A wrote {DESCRIPTIVE_PATH}")
    return descriptive


def print_part_a_tables(descriptive: dict) -> None:
    print("\nA1 event study: event forward returns", flush=True)
    print("k  n_events  mean    median  ci90_low  ci90_high  all_days_mean", flush=True)
    all_days = {int(r["k"]): r for r in descriptive["a1_event_study"]["all_days_baseline"]}
    for row in descriptive["a1_event_study"]["horizons"]:
        base = all_days[int(row["k"])]
        print(
            f"{int(row['k']):2d} {int(row['n_events']):8d} "
            f"{fmt_pct(row['mean']):>7} {fmt_pct(row['median']):>7} "
            f"{fmt_pct(row['ci90_low']):>9} {fmt_pct(row['ci90_high']):>10} "
            f"{fmt_pct(base['mean']):>13}",
            flush=True,
        )
    rnd = descriptive["a1_event_study"]["random_matched_non_event_5d"]
    print(
        "A1 random matched non-event 5d: "
        f"mean {fmt_pct(rnd['mean'])}, 90% band [{fmt_pct(rnd['ci90_low'])}, {fmt_pct(rnd['ci90_high'])}], "
        f"draws {rnd['draws']}",
        flush=True,
    )

    print("\nA2 dose response: 5d forward returns", flush=True)
    print("bucket       n   mean    median  ci90_low  ci90_high", flush=True)
    for row in descriptive["a2_dose_response"]["buckets"]:
        print(
            f"{row['bucket']:<10} {int(row['n']):3d} "
            f"{fmt_pct(row['mean_5d']):>7} {fmt_pct(row['median_5d']):>7} "
            f"{fmt_pct(row['ci90_low']):>9} {fmt_pct(row['ci90_high']):>10}",
            flush=True,
        )

    print("\nA3 hold-day alpha: canonical V27 entry, fixed 14d path", flush=True)
    print("day  n_trades  mean    median  ci90_low  ci90_high", flush=True)
    for row in descriptive["a3_hold_day_alpha"]["profile"]:
        print(
            f"{int(row['day']):3d} {int(row['n_trades']):8d} "
            f"{fmt_pct(row['mean']):>7} {fmt_pct(row['median']):>7} "
            f"{fmt_pct(row['ci90_low']):>9} {fmt_pct(row['ci90_high']):>10}",
            flush=True,
        )

    print("\nA4 response lag: corr(dlog_mcap_t, btc_ret_t+k)", flush=True)
    print("k   full_corr  full_n   pct90_corr  pct90_n", flush=True)
    full = descriptive["a4_response_lag"]["full"]
    rest = descriptive["a4_response_lag"]["slope_pct_ge_0.90"]
    for frow, rrow in zip(full, rest):
        print(
            f"{int(frow['k']):2d} {fmt_float(frow['corr']):>10} {int(frow['n']):7d} "
            f"{fmt_float(rrow['corr']):>11} {int(rrow['n']):7d}",
            flush=True,
        )
    print("\nPart A checks", flush=True)
    for key, value in descriptive["checks"].items():
        if key == "family_direction_checks":
            continue
        print(f"{key}: {value}", flush=True)
    print(f"family_direction_checks: {descriptive['checks']['family_direction_checks']}", flush=True)


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:+.2f}%"


def fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.4f}"


def mech_family_configs(family: str) -> list[dict]:
    out = []
    exits = ["price_mom"] if family == "M5_PRICEEXIT" else F3_EXITS
    for entry in CORE_ENTRIES:
        for pct in PCT_FILTERS:
            for exit_name in exits:
                out.append({
                    "family": family,
                    "config_id": (
                        f"{DISPLAY_FAMILY[family]}_{entry}_pct{osearch.pct_label(pct)}_"
                        f"{exit_name}_confirm"
                    ),
                    "entry_kind": entry,
                    "pct_filter": pct,
                    "exit": exit_name,
                    "confirm": True,
                })
    for threshold in PCT_EVENT_THRESHOLDS:
        for exit_name in exits:
            out.append({
                "family": family,
                "config_id": f"{DISPLAY_FAMILY[family]}_pct_event_{threshold:.2f}_{exit_name}_confirm",
                "entry_kind": "pct_event",
                "entry_threshold": threshold,
                "pct_filter": None,
                "exit": exit_name,
                "confirm": True,
            })
    return out


def apply_mech_signals(cache: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = cache.copy()
    family = params["family"]
    entry = params["entry_kind"]
    if entry == "pct_event":
        raw = df["slope_pct"].values.astype(float) >= float(params["entry_threshold"])
    else:
        raw = df[f"sig_{entry}"].values.astype(bool)

    signal = raw.copy()
    if family == "M3_PERSIST":
        prev = pd.Series(raw).shift(1, fill_value=False).values.astype(bool)
        signal = signal & prev

    if params.get("confirm", True):
        if family == "M4_CATCHUP":
            signal &= df["btc_down_or_flat"].values.astype(bool)
        else:
            signal &= df["btc_up"].values.astype(bool)

    pct = params.get("pct_filter")
    if pct is not None:
        signal &= df["slope_pct"].values.astype(float) >= float(pct)

    df["signal"] = signal.astype(int)
    df["signal_short"] = 0
    return df


def mech_engine_config(_params: dict) -> StrategyConfig:
    return StrategyConfig(
        mode=LONG_ONLY,
        leverage=1.0,
        execution_delay_frac=0.0,
        fee_bps_per_side=5.0,
        long_funding_bps_per_day=3.0,
        short_funding_bps_per_day=0.0,
    )


def price_mom_exit_idx(df: pd.DataFrame, entry_idx: int, _side: int) -> tuple[int, str]:
    entry_date = pd.Timestamp(df.iloc[entry_idx]["date"])
    for j in range(entry_idx + 1, len(df)):
        days_held = int((pd.Timestamp(df.iloc[j]["date"]) - entry_date).days)
        ret3 = df.iloc[j].get("btc_ret3", np.nan)
        ret3 = float(ret3) if not pd.isna(ret3) else np.nan
        if days_held >= 3 and np.isfinite(ret3) and ret3 < 0:
            return j, "price_mom"
        if days_held >= 14:
            return j, "max_hold"
    return len(df) - 1, "open"


def mech_exit_idx(df: pd.DataFrame, entry_idx: int, side: int, params: dict, config: StrategyConfig) -> tuple[int, str]:
    if params.get("exit") == "price_mom":
        return price_mom_exit_idx(df, entry_idx, side)
    return ORIG_OSEARCH_EXIT_IDX(df, entry_idx, side, params, config)


def collect_mech_trades(signals: pd.DataFrame, params: dict, config: StrategyConfig,
                        eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> tuple[list[dict], pd.DataFrame]:
    old_exit = osearch.exit_idx
    try:
        osearch.exit_idx = mech_exit_idx
        return osearch.collect_custom_trades(signals, params, config, eval_start, eval_end)
    finally:
        osearch.exit_idx = old_exit


def mech_backtest_params(cache: pd.DataFrame, params: dict, eval_start: pd.Timestamp,
                         eval_end: pd.Timestamp) -> dict:
    config = mech_engine_config(params)
    signals = apply_mech_signals(cache, params)
    trades, merged = collect_mech_trades(signals, params, config, eval_start, eval_end)
    equity = osearch.daily_equity_curve(merged, trades, config)
    metrics = {**osearch.equity_metrics(equity, config.initial_capital), **osearch.trade_metrics(trades)}
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


def run_mech_fold(family: str, configs: list[dict], cache: pd.DataFrame, fold: osearch.Fold) -> dict:
    best_params = None
    best_train = None
    best_score = None
    for params in configs:
        result = mech_backtest_params(cache, params, fold.train_start, fold.train_end)
        score = osearch.selection_score(result["metrics"])
        if best_score is None or score > best_score:
            best_score = score
            best_params = params
            best_train = result["metrics"]
    assert best_params is not None
    assert best_train is not None

    oos = mech_backtest_params(cache, best_params, fold.test_start, fold.test_end)
    return {
        "kind": "family_fold",
        "family": family,
        "family_label": DISPLAY_FAMILY[family],
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
        "oos_daily_returns": osearch.daily_return_records(oos["equity"]),
    }


def aggregate_mech_family(family: str, fold_records: list[dict]) -> dict:
    agg = osearch.aggregate_family(family, fold_records)
    agg["family_label"] = DISPLAY_FAMILY[family]
    agg["fold_ids"] = [int(r["fold"]) for r in sorted(fold_records, key=lambda r: int(r["fold"]))]
    return agg


def aggregate_matches_folds(aggregate_record: dict, folds: list[osearch.Fold]) -> bool:
    requested = [int(f.fold) for f in folds]
    if "fold_ids" in aggregate_record:
        return [int(x) for x in aggregate_record["fold_ids"]] == requested
    recorded = aggregate_record.get("folds", [])
    return [int(r["fold"]) for r in recorded] == requested


def run_mech_family_with_ledger(family: str, cache: pd.DataFrame, folds: list[osearch.Fold],
                                completed_folds: dict[tuple[str, int], dict],
                                completed_aggs: dict[str, dict], progress: osearch.Progress,
                                logger: osearch.Logger) -> tuple[list[dict], dict]:
    configs = mech_family_configs(family)
    fold_records = []
    for fold in folds:
        key = (family, fold.fold)
        if key in completed_folds:
            rec = completed_folds[key]
            fold_records.append(rec)
            logger.log(
                f"skip {SCOPE} {DISPLAY_FAMILY[family]} fold {fold.fold}: "
                f"selected {rec['selected_config']['config_id']} ({progress.tick()})"
            )
            continue
        rec = run_mech_fold(family, configs, cache, fold)
        rec = {**rec, "scope": SCOPE}
        _append_ledger(rec)
        fold_records.append(rec)
        m = rec["oos_metrics"]
        logger.log(
            f"done {SCOPE} {DISPLAY_FAMILY[family]} fold {fold.fold}: "
            f"selected {rec['selected_config']['config_id']} "
            f"train_ret {rec['train_metrics']['total_return']:+.4f} "
            f"oos_ret {m['total_return']:+.4f} oos_trades {m['n_trades']} ({progress.tick()})"
        )

    if family in completed_aggs and aggregate_matches_folds(completed_aggs[family], folds):
        agg = completed_aggs[family]
        logger.log(
            f"skip {SCOPE} {DISPLAY_FAMILY[family]} aggregate: "
            f"ret {agg['aggregate']['total_return']:+.4f} ({progress.tick()})"
        )
        return fold_records, agg

    agg = aggregate_mech_family(family, fold_records)
    agg = {**agg, "scope": SCOPE}
    _append_ledger(agg)
    logger.log(
        f"done {SCOPE} {DISPLAY_FAMILY[family]} aggregate: "
        f"ret {agg['aggregate']['total_return']:+.4f} "
        f"sharpe {agg['aggregate']['daily_sharpe']:+.3f} "
        f"trades {agg['aggregate']['n_trades']} ({progress.tick()})"
    )
    return fold_records, agg


def load_baseline_f3() -> dict:
    path = PROJECT / "experiments" / "open-search" / "results" / "leaderboard.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["families"]["F3"]


def baseline_subset_total_return(baseline_folds: list[dict], fold_ids: set[int]) -> float:
    total = 1.0
    for rec in sorted(baseline_folds, key=lambda r: int(r["fold"])):
        if int(rec["fold"]) in fold_ids:
            total *= 1.0 + float(rec["oos_metrics"]["total_return"])
    return total - 1.0


def family_part_a_direction(family: str, checks: dict) -> bool:
    label = DISPLAY_FAMILY[family]
    return bool(checks.get("family_direction_checks", {}).get(label, False))


def family_verdict(family: str, aggregate: dict, baseline_f3: dict, folds_beating: int,
                   part_a_ok: bool, full_fold_count: bool) -> str:
    delta_ret = float(aggregate["total_return"]) - float(baseline_f3["aggregate"]["total_return"])
    delta_sharpe = float(aggregate["daily_sharpe"]) - float(baseline_f3["aggregate"]["daily_sharpe"])
    margin = delta_ret > 0.15 or delta_sharpe > 0.20
    wins_ok = full_fold_count and folds_beating >= 5
    if margin and wins_ok and part_a_ok:
        return "ACCEPT"
    if part_a_ok and not (margin and wins_ok):
        return "NOTED"
    return "REJECT"


def write_variants(actual_end: pd.Timestamp, folds: list[osearch.Fold], families: list[str],
                   family_records: dict[str, list[dict]], family_aggs: dict[str, dict],
                   descriptive: dict, smoke: bool) -> dict:
    baseline_f3 = load_baseline_f3()
    baseline_by_fold = {int(r["fold"]): r for r in baseline_f3["folds"]}
    checks = descriptive["checks"]
    result_families = {}
    leaderboard = []
    fold_ids = {int(f.fold) for f in folds}
    full_fold_count = len(folds) == len(baseline_f3["folds"])

    for family in families:
        label = DISPLAY_FAMILY[family]
        agg = family_aggs[family]["aggregate"]
        recs = sorted(family_records[family], key=lambda r: int(r["fold"]))
        folds_out = []
        wins = 0
        for rec in recs:
            fold = int(rec["fold"])
            base = baseline_by_fold[fold]
            variant_ret = float(rec["oos_metrics"]["total_return"])
            baseline_ret = float(base["oos_metrics"]["total_return"])
            beat = variant_ret > baseline_ret
            wins += int(beat)
            folds_out.append({
                "fold": fold,
                "selected_config": rec["selected_config"],
                "train_metrics": rec["train_metrics"],
                "oos_metrics": rec["oos_metrics"],
                "baseline_oos_total_return": baseline_ret,
                "beats_baseline": beat,
            })

        baseline_ret_for_delta = (
            float(baseline_f3["aggregate"]["total_return"])
            if full_fold_count
            else baseline_subset_total_return(baseline_f3["folds"], fold_ids)
        )
        delta_ret = float(agg["total_return"]) - baseline_ret_for_delta
        delta_sharpe = float(agg["daily_sharpe"]) - float(baseline_f3["aggregate"]["daily_sharpe"])
        part_a_ok = family_part_a_direction(family, checks)
        verdict = family_verdict(family, agg, baseline_f3, wins, part_a_ok, full_fold_count)
        payload = {
            "aggregate": agg,
            "folds": folds_out,
            "selected_configs": [
                {
                    "fold": row["fold"],
                    "config_id": row["selected_config"]["config_id"],
                    "selected_config": row["selected_config"],
                }
                for row in folds_out
            ],
            "folds_beating_baseline": wins,
            "folds_compared": len(recs),
            "delta_total_return_vs_baseline": round(delta_ret, 6),
            "delta_sharpe_vs_baseline": round(delta_sharpe, 6),
            "part_a_direction_ok": part_a_ok,
            "verdict": verdict,
        }
        result_families[label] = payload
        leaderboard.append({
            "family": label,
            **agg,
            "folds_beating_baseline": wins,
            "delta_total_return_vs_baseline": round(delta_ret, 6),
            "delta_sharpe_vs_baseline": round(delta_sharpe, 6),
            "verdict": verdict,
        })

    leaderboard = sorted(leaderboard, key=lambda r: float(r["total_return"]), reverse=True)
    payload = {
        "run": {
            "scope": SCOPE,
            "smoke": smoke,
            "actual_end": _date_str(actual_end),
            "fold_count": len(folds),
            "families": [DISPLAY_FAMILY[f] for f in families],
            "baseline_source": "experiments/open-search/results/leaderboard.json",
            "baseline_family": "F3",
            "baseline_fold_count": len(baseline_f3["folds"]),
            "partial_run_note": None if full_fold_count else "Total-return delta uses matching baseline folds; Sharpe delta uses full F3 aggregate.",
        },
        "baseline_f3": {
            "aggregate": baseline_f3["aggregate"],
            "folds": baseline_f3["folds"],
        },
        "part_a_checks": checks,
        "leaderboard": leaderboard,
        "families": result_families,
    }
    _atomic_write_json(VARIANTS_PATH, payload)
    return payload


def print_variants_summary(payload: dict) -> None:
    print("\nMechanism variants", flush=True)
    print("family        total_return  sharpe  trades  folds_win  d_ret    d_sharpe  verdict", flush=True)
    for row in payload["leaderboard"]:
        print(
            f"{row['family']:<13} {float(row['total_return']) * 100:+11.2f}% "
            f"{float(row['daily_sharpe']):+6.2f} {int(row['n_trades']):7d} "
            f"{int(row['folds_beating_baseline']):9d} "
            f"{float(row['delta_total_return_vs_baseline']) * 100:+7.2f}% "
            f"{float(row['delta_sharpe_vs_baseline']):+9.2f} {row['verdict']}",
            flush=True,
        )


def cache_for_family(family: str, baseline_cache: pd.DataFrame, relative_cache: pd.DataFrame) -> pd.DataFrame:
    if family == "M2_NORM":
        return relative_cache
    return baseline_cache


def run(args: argparse.Namespace) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    part = args.part.lower()
    smoke = bool(args.smoke)
    families = parse_families(args.families, smoke=smoke)
    logger = osearch.Logger(LOG_PATH)
    try:
        logger.log(f"startup part={part} smoke={smoke} families={','.join(DISPLAY_FAMILY[f] for f in families)} ledger={LEDGER_PATH}")
        data = load_full_data()
        btc, usdt, actual_end = osearch.common_data(data)
        folds = osearch.make_folds(actual_end, smoke=smoke)
        baseline_cache = add_btc_momentum_cols(osearch.build_signal_cache(btc, usdt))
        logger.log(
            f"data warmup={_date_str(osearch.WARMUP_START)} actual_end={_date_str(actual_end)} "
            f"rows={len(baseline_cache)} folds={len(folds)}"
        )

        output: dict[str, Any] = {}
        descriptive = None
        if part in ("a", "all"):
            descriptive = compute_part_a(baseline_cache, actual_end, smoke=smoke, logger=logger)
            print_part_a_tables(descriptive)
            output["descriptive"] = descriptive
        elif DESCRIPTIVE_PATH.exists():
            descriptive = json.loads(DESCRIPTIVE_PATH.read_text(encoding="utf-8"))
            logger.log(f"loaded Part A checks from {DESCRIPTIVE_PATH}")

        if part in ("b", "all"):
            if descriptive is None:
                logger.log("Part B verdicts require Part A checks; computing Part A first")
                descriptive = compute_part_a(baseline_cache, actual_end, smoke=smoke, logger=logger)
            relative_cache = build_relative_signal_cache(btc, usdt) if "M2_NORM" in families else baseline_cache
            completed_folds, completed_aggs, _completed_desc = load_mech_ledger()
            total_units = len(families) * (len(folds) + 1)
            progress = osearch.Progress(total_units)
            family_records: dict[str, list[dict]] = {}
            family_aggs: dict[str, dict] = {}
            for family in families:
                cache = cache_for_family(family, baseline_cache, relative_cache)
                records, agg = run_mech_family_with_ledger(
                    family, cache, folds, completed_folds, completed_aggs, progress, logger
                )
                family_records[family] = records
                family_aggs[family] = agg
            variants = write_variants(actual_end, folds, families, family_records, family_aggs, descriptive, smoke=smoke)
            logger.log(f"Part B wrote {VARIANTS_PATH}")
            print_variants_summary(variants)
            output["variants"] = variants

        logger.log("final summary complete")
        return output
    finally:
        logger.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["A", "B", "all"], default="all", help="which part to run (default: all)")
    parser.add_argument("--families", help="comma-separated subset: M2,M3,M4,M5")
    parser.add_argument("--smoke", action="store_true", help="Part A reduced resampling; Part B M5 folds 1..2 only")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
