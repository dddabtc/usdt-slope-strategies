"""CoinGecko data access: fetch recent data, merge into *_full.csv history.

Same data layer as dddabtc/usdt-slope-research so results are comparable.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
RATE_LIMIT_DELAY = 12  # CoinGecko free tier is strict

COINS = {
    "usdt": {"id": "tether",   "file": "usdt_market_cap.csv", "field": "market_caps", "col": "market_cap"},
    "usdc": {"id": "usd-coin", "file": "usdc_market_cap.csv", "field": "market_caps", "col": "market_cap"},
    "dai":  {"id": "dai",      "file": "dai_market_cap.csv",  "field": "market_caps", "col": "market_cap"},
    "btc":  {"id": "bitcoin",  "file": "btc_price.csv",       "field": "prices",      "col": "price"},
    "eth":  {"id": "ethereum", "file": "eth_price.csv",       "field": "prices",      "col": "price"},
}

FULL_FILES = {
    "usdt": ("usdt_market_cap_full.csv", "market_cap"),
    "usdc": ("usdc_market_cap_full.csv", "market_cap"),
    "dai":  ("dai_market_cap_full.csv",  "market_cap"),
    "btc":  ("btc_price_full.csv",       "price"),
    "eth":  ("eth_price_full.csv",       "price"),
}


def _fetch_market_chart(coin_id: str, days: str = "365") -> dict:
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    for attempt in range(5):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = (attempt + 1) * 15
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def _to_dataframe(rows: list, col_name: str) -> pd.DataFrame:
    """One row per date, valued at the day's FIRST point (the 00:00 UTC
    snapshot).  CoinGecko appends an intraday "now" point for the current
    day; taking the first point per date keeps today's settled snapshot and
    drops the drifting now-point, so today's bar is usable immediately."""
    df = pd.DataFrame(rows, columns=["timestamp_ms", col_name])
    df = df.sort_values("timestamp_ms")
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.normalize()
    df = df.groupby("date")[col_name].first().reset_index()
    return df.sort_values("date").reset_index(drop=True)


def fetch_coin_data(coin_key: str, force_refresh: bool = False) -> pd.DataFrame:
    info = COINS[coin_key]
    cache_path = DATA_DIR / info["file"]
    if not force_refresh and cache_path.exists():
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime < 12 * 3600:
            df = pd.read_csv(cache_path, parse_dates=["date"])
            print(f"  [cache] {coin_key}: {len(df)} rows ({df['date'].min().date()} → {df['date'].max().date()})")
            return df
    print(f"  [fetch] {coin_key} from CoinGecko...")
    raw = _fetch_market_chart(info["id"])
    df = _to_dataframe(raw[info["field"]], info["col"])
    df.to_csv(cache_path, index=False)
    print(f"  [saved] {coin_key}: {len(df)} rows ({df['date'].min().date()} → {df['date'].max().date()})")
    return df


def refresh_full_data(heal_days: int = 7) -> int:
    """Fetch the last 365d for every series and merge into *_full.csv.

    Rows are 00:00 UTC snapshots (see _to_dataframe), so today's bar is
    settled and merged the moment it exists.  The most recent ``heal_days``
    rows are overwritten with fresh values so a bar recorded minutes after
    midnight self-corrects on the next run.
    """
    total_new = 0
    for i, key in enumerate(COINS):
        short_df = fetch_coin_data(key, force_refresh=True)
        fname, col = FULL_FILES[key]
        full_path = DATA_DIR / fname
        if full_path.exists():
            full_df = pd.read_csv(full_path, parse_dates=["date"]).sort_values("date")
        else:
            full_df = pd.DataFrame(columns=["date", col])
        known = set(pd.to_datetime(full_df["date"]).dt.normalize())
        heal_cut = short_df["date"].max() - pd.Timedelta(days=heal_days)
        fresh = short_df[(~short_df["date"].isin(known)) | (short_df["date"] >= heal_cut)]
        n_new = int((~fresh["date"].isin(known)).sum())
        merged = (
            pd.concat([full_df, fresh], ignore_index=True)
            .sort_values("date")
            .drop_duplicates(subset="date", keep="last")
            .reset_index(drop=True)
        )
        merged.to_csv(full_path, index=False)
        total_new += n_new
        print(f"  {key}: +{n_new} rows → up to {merged['date'].max().date()}")
        if i < len(COINS) - 1:
            time.sleep(RATE_LIMIT_DELAY)
    return total_new


def load_full_data() -> dict:
    """Load *_full.csv files, aligned to the BTC∩USDT common date range."""
    data = {}
    for key, (fname, _col) in FULL_FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  ⚠️ No data file for {key}")
            continue
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        data[key] = df

    if "btc" in data and "usdt" in data:
        start = max(data["btc"]["date"].min(), data["usdt"]["date"].min())
        end = min(data["btc"]["date"].max(), data["usdt"]["date"].max())
        for key in data:
            data[key] = data[key][(data[key]["date"] >= start) & (data[key]["date"] <= end)].reset_index(drop=True)
        print(f"  Common range: {start.date()} → {end.date()}")
    return data
