"""Long-only and long/short USDT-liquidity strategy engine.

The long side is the canonical V27 strategy from dddabtc/usdt-slope-research,
re-implemented bit-for-bit (verified: identical trade list and equity with
costs disabled):

  long entry  = OR(bollinger_strict_confirmation, regime_strict_confirmation)
                AND slope 1y-percentile >= pct_threshold
  long exit   = slope-peak decay (min hold 3d, max hold 14d, 20% drop)

The short side mirrors every rule with no new fitted parameters:

  short entry = OR(bollinger lower-band break, contraction regime)
                AND BTC 3d return < 0
                AND slope 1y-percentile <= 1 - pct_threshold
                AND 60d USDT slope < 0 (bear regime only)
  short exit  = slope-trough recovery (same hold/drop constants)

The 60d regime gate on shorts mirrors the V23 production rule from the
original research ("60d USDT slope > 0, bull market only" for longs); it is
predeclared there, not fitted here.  Without it, mirrored shorts fire during
bull-market USDT dips and lose to BTC's upward drift.

On top of the original engine this adds: execution costs (fees + funding on
leveraged notional), close-based isolated-margin liquidation, and a daily
mark-to-market equity curve for both strategies.

Execution timing.  Every daily data point is a 00:00 UTC *snapshot*, not an
end-of-day aggregate: the bar labeled day T is fully known seconds after
00:00 UTC on day T (USDT issuance is on-chain observable in real time), and
BTC trades 24/7.  Execution is therefore modeled as a fill at
``execution_delay_frac`` of the way to the next snapshot:
  0.0  -> immediate execution at the signal snapshot (default headline)
  0.25 -> ~6h delay (a lazy 6-hourly cron)
  1.0  -> wait a full day (the original repo's "+1 bar" conservative bound)
Setting ``execution_delay_frac=None`` falls back to the integer
``execution_lag_bars`` semantics for exact comparability with the original
repo's two frozen artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Benchmark window (identical to usdt-slope-research V27)
# ---------------------------------------------------------------------------

RESEARCH_START = pd.Timestamp("2020-09-27")
TRAIN_END = pd.Timestamp("2021-09-27")       # signal warm-up ends, eval starts
BENCHMARK_TEST_END = pd.Timestamp("2026-03-19")  # frozen headline window end

LONG_ONLY = "long_only"
LONG_SHORT = "long_short"


@dataclass(frozen=True)
class StrategyConfig:
    mode: str = LONG_ONLY                 # long_only | long_short
    leverage: float = 1.0
    # --- frozen V27 signal parameters (predeclared, not re-fitted) ---
    slope_window: int = 3
    comparison_period: int = 30
    bb_n_std: float = 2.0
    btc_confirm_window: int = 3
    pct_lookback: int = 365
    pct_threshold: float = 0.93
    hold_days: int = 3
    max_hold_days: int = 14
    exit_drop_pct: float = 0.20
    # shorts only in a contracting-liquidity regime (predeclared V23 mirror)
    short_regime_gate: bool = True
    # --- execution / cost model ---
    # fill at this fraction of the gap to the next snapshot (0 = immediate);
    # None falls back to integer execution_lag_bars below
    execution_delay_frac: float | None = 0.0
    execution_lag_bars: int = 1
    fee_bps_per_side: float = 5.0         # taker fee + slippage, per side, on notional
    long_funding_bps_per_day: float = 3.0  # perp funding paid by longs (~0.01%/8h)
    short_funding_bps_per_day: float = 0.0  # shorts usually receive funding; modeled as 0
    maintenance_margin: float = 0.005     # exchange tier-1 maintenance, fraction of notional
    initial_capital: float = 10_000.0
    extra: dict = field(default_factory=dict)


DEFAULT_CONFIG = StrategyConfig()


def _ts(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def slice_data_window(data: dict, start=None, end=None) -> dict:
    start = _ts(start) if start is not None else None
    end = _ts(end) if end is not None else None
    out = {}
    for key, df in data.items():
        if not isinstance(df, pd.DataFrame) or "date" not in df.columns:
            out[key] = df
            continue
        x = df.copy()
        x["date"] = pd.to_datetime(x["date"]).dt.normalize()
        if start is not None:
            x = x[x["date"] >= start]
        if end is not None:
            x = x[x["date"] <= end]
        out[key] = x.sort_values("date").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling OLS slope (identical to the original repo)."""
    n = len(values)
    out = np.full(n, np.nan)
    if window > n or window < 2:
        return out
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    x_centered = x - x_mean
    shape = (n - window + 1, window)
    strides = (values.strides[0], values.strides[0])
    windows = np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides)
    y_means = windows.mean(axis=1)
    slopes = (windows - y_means[:, np.newaxis]) @ x_centered / ss_xx
    out[window - 1:] = slopes
    return out


def generate_signals(data: dict, config: StrategyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Daily signal table with long and mirrored short entry signals.

    All columns at row i use information through day i only; execution lag is
    applied later, in trade collection.
    """
    mc = data["usdt"].copy().sort_values("date").reset_index(drop=True)
    mc["date"] = pd.to_datetime(mc["date"]).dt.normalize()
    btc = data["btc"].copy().sort_values("date").reset_index(drop=True)
    btc["date"] = pd.to_datetime(btc["date"]).dt.normalize()

    values = mc["market_cap"].values.astype(float)
    slopes = _rolling_slope(values, config.slope_window)
    regime_slopes = _rolling_slope(values, 60)

    s = pd.Series(slopes)
    mid = s.rolling(config.comparison_period).mean().values
    std = s.rolling(config.comparison_period).std().values
    upper = mid + config.bb_n_std * std
    lower = mid - config.bb_n_std * std
    valid_bb = ~np.isnan(slopes) & ~np.isnan(std) & (std > 0)
    bb_long = valid_bb & (slopes > upper)
    bb_short = valid_bb & (slopes < lower)

    sos = np.diff(slopes, prepend=np.nan)
    sos_smooth = pd.Series(sos).rolling(config.slope_window).mean().values
    avg_slope = pd.Series(slopes).rolling(config.comparison_period).mean().values
    valid_rg = ~np.isnan(slopes) & ~np.isnan(sos_smooth) & ~np.isnan(avg_slope)
    accel = valid_rg & (slopes > 0) & (sos_smooth > 0) & (avg_slope > 0)
    ratio_l = np.where(accel & (avg_slope > 0), slopes / avg_slope, 0)
    regime_long = accel & (ratio_l > 1.0)
    decel = valid_rg & (slopes < 0) & (sos_smooth < 0) & (avg_slope < 0)
    ratio_s = np.where(decel & (avg_slope < 0), slopes / avg_slope, 0)
    regime_short = decel & (ratio_s > 1.0)

    lookback = config.pct_lookback
    pct_vals = np.full(len(slopes), np.nan)
    for i in range(lookback, len(slopes)):
        if np.isnan(slopes[i]):
            continue
        window = slopes[max(0, i - lookback): i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 30:
            continue
        pct_vals[i] = sp_stats.percentileofscore(valid, slopes[i]) / 100.0

    sdf = pd.DataFrame({
        "date": mc["date"],
        "slope": slopes,
        "slope_pct": pct_vals,
        "regime_slope": regime_slopes,
        "usdt_mcap": values,
        "sig_bb": bb_long.astype(int),
        "sig_regime": regime_long.astype(int),
        "sig_bb_short": bb_short.astype(int),
        "sig_regime_short": regime_short.astype(int),
    })

    confirm = btc[["date"]].copy()
    pct = btc["price"].pct_change(config.btc_confirm_window)
    confirm["btc_up"] = (pct > 0).astype(int)
    confirm["btc_down"] = (pct < 0).astype(int)
    merged = sdf.merge(confirm, on="date", how="inner")

    pre_long = merged[["sig_bb", "sig_regime"]].max(axis=1).astype(int)
    merged["signal"] = (
        (pre_long == 1) & (merged["btc_up"] == 1)
        & (merged["slope_pct"] >= config.pct_threshold)
    ).astype(int)

    pre_short = merged[["sig_bb_short", "sig_regime_short"]].max(axis=1).astype(int)
    short_ok = (
        (pre_short == 1) & (merged["btc_down"] == 1)
        & (merged["slope_pct"] <= 1.0 - config.pct_threshold)
    )
    if config.short_regime_gate:
        short_ok &= merged["regime_slope"] < 0
    merged["signal_short"] = short_ok.astype(int)

    merged["regime"] = (merged["regime_slope"] > 0).astype(int)
    return merged.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Exit rules (long: slope-peak decay; short: slope-trough recovery)
# ---------------------------------------------------------------------------

def _exit_decision(df: pd.DataFrame, entry_exec_idx: int, side: int, config: StrategyConfig):
    """Return (exit-signal index, reason).  Mirrors V27 conflict no-op logic:
    if an exit condition fires on a bar that also carries a same-side entry
    signal, the bar is conflicted and the position is held."""
    entry_date = df.iloc[entry_exec_idx]["date"]
    peak_slope = -np.inf   # long: highest slope seen
    trough_slope = np.inf  # short: lowest slope seen
    sig_col = "signal" if side > 0 else "signal_short"
    for j in range(entry_exec_idx + 1, len(df)):
        days_held = (df.iloc[j]["date"] - entry_date).days
        if days_held > config.max_hold_days:
            return j, "max_hold"
        if days_held < config.hold_days:
            continue
        s = df.iloc[j].get("slope", np.nan)
        if pd.isna(s):
            continue
        entry_conflict = int(df.iloc[j].get(sig_col, 0)) == 1
        if side > 0:
            peak_slope = max(peak_slope, float(s))
            if s < 0:
                if entry_conflict:
                    continue
                return j, "slope_negative"
            if peak_slope > 0 and s < peak_slope * (1 - config.exit_drop_pct):
                if entry_conflict:
                    continue
                return j, "slope_peak_drop"
        else:
            trough_slope = min(trough_slope, float(s))
            if s > 0:
                if entry_conflict:
                    continue
                return j, "slope_positive"
            if trough_slope < 0 and s > trough_slope * (1 - config.exit_drop_pct):
                if entry_conflict:
                    continue
                return j, "slope_trough_recovery"
    return len(df) - 1, "open"


# ---------------------------------------------------------------------------
# Trade economics: leverage, costs, liquidation
# ---------------------------------------------------------------------------

def _funding_bps(side: int, config: StrategyConfig) -> float:
    return config.long_funding_bps_per_day if side > 0 else config.short_funding_bps_per_day


def _liquidation_idx(df: pd.DataFrame, buy_idx: int, sell_idx: int, side: int,
                     config: StrategyConfig, wick_stress: float = 0.0,
                     entry_price: float | None = None):
    """First bar in (buy_idx, sell_idx] where close-based equity hits the
    maintenance margin.  Returns None if the position survives."""
    L = config.leverage
    if L <= 1.0:
        return None
    p0 = float(df.iloc[buy_idx]["price"]) if entry_price is None else float(entry_price)
    for j in range(buy_idx + 1, sell_idx + 1):
        p = float(df.iloc[j]["price"])
        if side > 0:
            p_eff = p * (1.0 - wick_stress)
            equity_ratio = 1.0 + L * (p_eff / p0 - 1.0)
        else:
            p_eff = p * (1.0 + wick_stress)
            equity_ratio = 1.0 - L * (p_eff / p0 - 1.0)
        if equity_ratio <= L * config.maintenance_margin:
            return j
    return None


def _net_trade_return(raw_ret: float, hold_days: int, side: int, config: StrategyConfig) -> float:
    L = config.leverage
    fees = 2.0 * (config.fee_bps_per_side / 10_000.0) * L
    funding = (_funding_bps(side, config) / 10_000.0) * L * max(hold_days, 0)
    return max(side * raw_ret * L - fees - funding, -0.99)


def collect_trades(signals_df: pd.DataFrame, btc_df: pd.DataFrame, config: StrategyConfig,
                   eval_start: pd.Timestamp, eval_end: pd.Timestamp,
                   right_censor_open: bool = False) -> tuple[list[dict], pd.DataFrame]:
    """Sequential trade collection over [eval_start, eval_end].

    long_only: only long entries.  long_short: when flat, a long signal takes
    priority over a short signal on the same bar; otherwise first signal wins.
    """
    sig = signals_df.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    btc = btc_df.copy()
    btc["date"] = pd.to_datetime(btc["date"]).dt.normalize()
    sig = sig[(sig["date"] >= eval_start) & (sig["date"] <= eval_end)].copy()
    btc = btc[(btc["date"] >= eval_start) & (btc["date"] <= eval_end)].copy()
    df = sig.merge(btc[["date", "price"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    if df.empty:
        return [], df

    allow_short = config.mode == LONG_SHORT
    alpha = config.execution_delay_frac
    use_frac = alpha is not None
    lag = 0 if use_frac else int(config.execution_lag_bars)

    def fill_price(idx: int) -> float:
        """Fill at alpha of the way from snapshot idx to the next one."""
        p = float(df.iloc[idx]["price"])
        if use_frac and alpha > 0 and idx + 1 < len(df):
            p_next = float(df.iloc[idx + 1]["price"])
            p = p + float(alpha) * (p_next - p)
        return p

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

        entry_signal_idx = i
        buy_idx = entry_signal_idx + lag
        if buy_idx >= len(df):
            break
        buy = df.iloc[buy_idx]
        exit_signal_idx, exit_reason = _exit_decision(df, buy_idx, side, config)
        sell_idx = min(exit_signal_idx + lag, len(df) - 1)

        buy_price = fill_price(buy_idx)
        liq_idx = _liquidation_idx(df, buy_idx, sell_idx, side, config, entry_price=buy_price)
        liquidated = liq_idx is not None
        if liquidated:
            sell_idx = liq_idx
            exit_reason = "liquidated"

        is_open = right_censor_open and exit_reason == "open"
        sell = df.iloc[sell_idx]
        # liquidation closes at the breached snapshot itself, not an alpha-fill
        sell_price = float(sell["price"]) if liquidated else fill_price(sell_idx)
        raw_ret = (sell_price - buy_price) / buy_price
        hold_days = int((sell["date"] - buy["date"]).days)
        if liquidated:
            net = -0.99
        else:
            net = _net_trade_return(raw_ret, hold_days, side, config)

        trade = {
            "side": "LONG" if side > 0 else "SHORT",
            "entry_signal_date": str(df.iloc[entry_signal_idx]["date"].date()),
            "buy_date": str(buy["date"].date()),
            "exit_signal_date": str(df.iloc[exit_signal_idx]["date"].date()) if not is_open else "",
            "sell_date": str(sell["date"].date()) if not is_open else "",
            "buy_price": round(buy_price, 2),
            "sell_price": round(sell_price, 2) if not is_open else None,
            "raw_return_pct": round(raw_ret * 100, 2) if not is_open else None,
            "return_pct": round(net * 100, 2) if not is_open else None,
            "hold_days": hold_days,
            "exit_reason": exit_reason,
            "status": "OPEN" if is_open else "CLOSED",
            "execution_delay_frac": alpha if use_frac else None,
            "execution_lag_bars": None if use_frac else lag,
        }
        if not is_open:
            capital *= (1.0 + net)
            trade["capital_after"] = round(capital, 2)
            trades.append(trade)
            i = max(sell_idx + 1, buy_idx + 1)
        else:
            trade["capital_after"] = None
            trades.append(trade)
            i = len(df)
    return trades, df


# ---------------------------------------------------------------------------
# Daily mark-to-market equity and metrics
# ---------------------------------------------------------------------------

def daily_equity_curve(merged_df: pd.DataFrame, trades: list[dict], config: StrategyConfig) -> pd.DataFrame:
    if merged_df.empty:
        return pd.DataFrame(columns=["date", "equity", "position", "daily_return", "drawdown"])
    df = merged_df[["date", "price"]].copy().sort_values("date").reset_index(drop=True)
    closed = [t for t in trades if t.get("status") != "OPEN"]
    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    by_buy = {pd.Timestamp(t["buy_date"]): t for t in closed + open_trades}
    fee = config.fee_bps_per_side / 10_000.0 * config.leverage

    cash = float(config.initial_capital)
    active = None
    rows = []
    for _, r in df.iterrows():
        date = pd.Timestamp(r["date"])
        price = float(r["price"])
        if active is None and date in by_buy:
            t = by_buy[date]
            active = {
                "trade": t,
                "side": 1 if t["side"] == "LONG" else -1,
                "buy_price": float(t.get("buy_price") or price),  # actual fill
                "entry_capital": cash * (1.0 - fee),  # entry fee paid up front
                "buy_date": date,
            }
        if active is not None:
            side = active["side"]
            raw = (price - active["buy_price"]) / active["buy_price"]
            days = (date - active["buy_date"]).days
            funding = (_funding_bps(side, config) / 10_000.0) * config.leverage * days
            equity = active["entry_capital"] * max(0.01, 1.0 + side * raw * config.leverage - funding)
            position = side
        else:
            equity = cash
            position = 0
        rows.append((date, equity, position))
        if active is not None:
            t = active["trade"]
            if t.get("sell_date") and date == pd.Timestamp(t["sell_date"]):
                # settle with the trade's own net return so curve and trade
                # capital agree exactly (incl. liquidation floor)
                cash = cash * (1.0 + t["return_pct"] / 100.0) if t["return_pct"] is not None else equity
                rows[-1] = (date, cash, 0)
                active = None

    out = pd.DataFrame(rows, columns=["date", "equity", "position"])
    out["daily_return"] = out["equity"].pct_change().fillna(0.0)
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["peak_equity"] - 1.0
    return out


def equity_metrics(equity: pd.DataFrame, initial_capital: float) -> dict:
    if equity.empty:
        return {"total_return": 0.0, "daily_sharpe": 0.0, "max_drawdown": 0.0,
                "final_capital": round(initial_capital, 2), "exposure": 0.0, "ann_return": 0.0}
    rets = equity["daily_return"].values.astype(float)
    std = np.std(rets, ddof=1) if len(rets) > 1 else 0.0
    sharpe = 0.0 if std == 0 else float(np.mean(rets) / std * np.sqrt(365))
    final = float(equity["equity"].iloc[-1])
    years = max(len(equity) / 365.0, 1e-9)
    total = final / initial_capital
    return {
        "total_return": round(total - 1.0, 4),
        "ann_return": round(total ** (1 / years) - 1.0, 4),
        "daily_sharpe": round(sharpe, 4),
        "max_drawdown": round(float(equity["drawdown"].min()), 4),
        "final_capital": round(final, 2),
        "exposure": round(float((equity["position"] != 0).mean()), 4),
    }


def trade_metrics(trades: list[dict]) -> dict:
    closed = [t for t in trades if t.get("status") != "OPEN" and t.get("return_pct") is not None]
    if not closed:
        return {"n_trades": 0, "n_long": 0, "n_short": 0, "win_rate": 0.0,
                "avg_return_per_trade": 0.0, "avg_hold_days": 0.0, "n_liquidated": 0}
    tdf = pd.DataFrame(closed)
    rets = tdf["return_pct"].astype(float) / 100.0
    return {
        "n_trades": int(len(tdf)),
        "n_long": int((tdf["side"] == "LONG").sum()),
        "n_short": int((tdf["side"] == "SHORT").sum()),
        "win_rate": round(float((rets > 0).mean()), 4),
        "avg_return_per_trade": round(float(rets.mean()), 4),
        "avg_hold_days": round(float(tdf["hold_days"].mean()), 1),
        "n_liquidated": int((tdf["exit_reason"] == "liquidated").sum()),
    }


def backtest(data: dict, config: StrategyConfig = DEFAULT_CONFIG,
             eval_start: pd.Timestamp = TRAIN_END,
             eval_end: pd.Timestamp = BENCHMARK_TEST_END,
             right_censor_open: bool = False) -> dict:
    """Full backtest over [eval_start, eval_end] with signals built from
    research_start so warm-up windows match the original repo exactly."""
    bdata = slice_data_window(data, RESEARCH_START, eval_end)
    signals = generate_signals(bdata, config)
    trades, merged = collect_trades(signals, bdata["btc"], config, _ts(eval_start), _ts(eval_end),
                                    right_censor_open=right_censor_open)
    equity = daily_equity_curve(merged, trades, config)
    bh = float(merged["price"].iloc[-1] / merged["price"].iloc[0] - 1.0) if not merged.empty else 0.0
    metrics = {**equity_metrics(equity, config.initial_capital), **trade_metrics(trades)}
    metrics.update({
        "mode": config.mode,
        "leverage": config.leverage,
        "fee_bps_per_side": config.fee_bps_per_side,
        "long_funding_bps_per_day": config.long_funding_bps_per_day,
        "short_funding_bps_per_day": config.short_funding_bps_per_day,
        "execution_delay_frac": config.execution_delay_frac,
        "execution_lag_bars": None if config.execution_delay_frac is not None else config.execution_lag_bars,
        "eval_start": str(_ts(eval_start).date()),
        "eval_end": str(_ts(eval_end).date()),
        "btc_buy_hold_return": round(bh, 4),
    })
    return {"signals": signals, "trades": trades, "equity": equity, "metrics": metrics, "merged": merged}


def post_benchmark_trades(data: dict, config: StrategyConfig,
                          benchmark_end: pd.Timestamp = BENCHMARK_TEST_END) -> list[dict]:
    """Trades after the frozen benchmark window (overlay only, OPEN-censored)."""
    latest = min(pd.to_datetime(data["btc"]["date"]).max(), pd.to_datetime(data["usdt"]["date"]).max())
    start = _ts(benchmark_end) + pd.Timedelta(days=1)
    if latest <= start:
        return []
    bdata = slice_data_window(data, RESEARCH_START, latest)
    signals = generate_signals(bdata, config)
    trades, _ = collect_trades(signals, bdata["btc"], config, start, _ts(latest), right_censor_open=True)
    for t in trades:
        t["phase"] = "Post-benchmark"
    return trades
