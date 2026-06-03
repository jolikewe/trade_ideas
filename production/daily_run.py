#!/usr/bin/env python3
"""Daily brief generator. Run daily; auto-runs at Claude session start."""
import argparse
import csv
import json
import math
from datetime import date, datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

LOG_PATH = Path("data/production/brief_log.csv")
LOG_COLS = [
    "date", "signal_date", "days_stale",
    "vix", "spy", "spy_ma200", "momentum_z",
    "vix_ok", "spy_ok", "momentum_ok", "regime_open", "regime_consec_days",
    "n_stocks", "n_buy_zone", "top_tickers",
    "ridge_train_ic",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force regenerate brief")
    parser.add_argument("--confirm-trade", action="store_true", help="Mark rebalance done")
    args = parser.parse_args()

    brief_path = Path("data/production/daily_brief.md")
    state_path = Path("data/production/rebalance_state.json")
    today = date.today().isoformat()

    if args.confirm_trade:
        state = {}
        if state_path.exists():
            state = json.loads(state_path.read_text())
        state["last_rebalance_date"] = today
        state_path.write_text(json.dumps(state, indent=2))
        print(f"Rebalance confirmed for {today}")
        return

    if not args.refresh and brief_path.exists():
        content = brief_path.read_text()
        if f"date: {today}" in content:
            print(content)
            return

    brief = _generate_brief(today, state_path)
    brief_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.write_text(brief)
    print(brief)


def _generate_brief(today: str, state_path: Path) -> str:
    import pandas as pd
    from trading_system.data.loaders import DataLoader
    from trading_system.universe.historical import PointInTimeUniverse
    from trading_system.features.mean_reversion import MeanReversionFeatures
    from trading_system.models.ridge import RidgeModel
    from trading_system.regime.detector import RegimeDetector
    from trading_system.portfolio.optimizer import FactorNeutralOptimizer
    from trading_system.utils.config import load_config
    from portfolio_state import PortfolioState

    cfg = load_config("config/backtest.yaml")["walk_forward"]

    # ── Models ────────────────────────────────────────────────────────────────
    model_dir = Path("data/models/mean_reversion")
    model_name = "model_mr_zscore_12feat"
    ridge_path = model_dir / f"{model_name}_ridge" / "production" / "model.pkl"

    if not ridge_path.exists():
        return _stub_brief(today, state_path,
                           "Production model not found. Run: python -m trading_system.cli production-train")

    ridge = RidgeModel.load(str(ridge_path))
    ridge_meta     = json.loads((ridge_path.parent / "metadata.json").read_text())
    ridge_train_ic = ridge_meta["train_ic"]
    model_train_end = ridge_meta.get("train_end", "")

    # ── Prices (incremental download built-in) ────────────────────────────────
    pit    = PointInTimeUniverse.load_or_build()
    loader = DataLoader()
    print("Updating prices...", end=" ", flush=True)
    prices = loader.yf.load_universe(pit.all_tickers, cfg["data_start_date"], today,
                                     show_progress=False)
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"done ({prices['ticker'].nunique()} tickers through {prices['date'].max().date()})")
    if prices.empty:
        return _stub_brief(today, state_path, "No price data in cache.")

    latest_date = prices["date"].max()
    days_stale  = (pd.Timestamp(today) - latest_date).days

    # ── Features & signals ────────────────────────────────────────────────────
    feats         = MeanReversionFeatures().compute_all(prices)
    # Restrict scoring to current S&P 500 members — historical tickers are for training only
    active_tickers = set(pit.get_universe(latest_date))
    latest_feats  = feats[(feats["date"] == latest_date) & feats["ticker"].isin(active_tickers)].copy()
    n_stocks      = len(latest_feats)

    X = latest_feats.set_index("ticker")[ridge.feature_names].fillna(0)
    scores   = pd.Series(ridge.predict(X), index=X.index)
    ensemble = scores.rank(pct=True)
    n_buy_zone   = int((ensemble >= 0.75).sum())

    # ── Regime ────────────────────────────────────────────────────────────────
    regime_start = str((latest_date - pd.DateOffset(days=450)).date())
    vix_series, spy_series = _fetch_regime_data(regime_start, today)

    regime_known = (vix_series is not None and spy_series is not None
                    and len(vix_series) > 10)
    if regime_known:
        vix_series = vix_series[vix_series.index <= latest_date]
        spy_series = spy_series[spy_series.index <= latest_date]
        detector   = RegimeDetector()
        regime_df  = detector.detect_composite_regime(vix_series, spy_series, prices=prices)
        latest_rd  = regime_df.index[regime_df.index.notna()].max()
        regime_row = regime_df.loc[latest_rd]
        tradeable  = bool(regime_row["tradeable"])
        vix_val    = float(vix_series.iloc[-1])
        spy_val    = float(spy_series.iloc[-1])
        spy_ma200  = float(spy_series.rolling(200).mean().iloc[-1])
        mom_z      = float(regime_row.get("momentum_z", float("nan")))
        vix_ok     = bool(regime_row["vix_ok"])
        spy_ok     = bool(regime_row["spy_ok"])
        mom_ok     = bool(regime_row["momentum_ok"])
    else:
        tradeable = False
        vix_val = spy_val = spy_ma200 = mom_z = float("nan")
        vix_ok = spy_ok = mom_ok = False

    regime_consec = _regime_consecutive_days(today, tradeable)

    # ── Portfolio optimisation ─────────────────────────────────────────────────
    weights    = pd.Series(dtype=float)
    opt_status = ""
    if tradeable:
        result     = FactorNeutralOptimizer().optimize(ensemble)
        weights    = result["weights"]
        opt_status = result["status"]

    # ── Positions P&L ─────────────────────────────────────────────────────────
    prod_cfg      = load_config("config/production.yaml")
    account_size  = prod_cfg["account"]["initial_capital"]
    ps            = PortfolioState()
    positions     = ps.load_positions()
    latest_prices = prices[prices["date"] == latest_date].set_index("ticker")["close"]
    pnl_section   = ""
    total_unrealised = 0.0
    current_weights  = pd.Series(dtype=float)

    if not positions.empty:
        mkt_vals = {}
        rows = []
        for _, r in positions.iterrows():
            tkr  = r["ticker"]
            ep   = r.get("entry_price", float("nan"))
            shrs = r.get("shares", 0)
            cp   = float(latest_prices.get(tkr, float("nan")))
            if not math.isnan(cp) and not math.isnan(ep):
                upnl = (cp - ep) * shrs
                pct  = (cp / ep - 1) * 100
                total_unrealised += upnl
                mkt_vals[tkr] = cp * shrs
                rows.append(f"| {tkr:<8} | {int(shrs):>6} | {ep:>8.2f} | {cp:>8.2f} | {upnl:>+9.2f} | {pct:>+6.1f}% |")
            else:
                mkt_vals[tkr] = 0.0
                rows.append(f"| {tkr:<8} | {int(shrs):>6} | {ep:>8.2f} | {'n/a':>8} | {'n/a':>9} | {'n/a':>7} |")
        total_mkt = sum(mkt_vals.values()) or account_size
        current_weights = pd.Series({t: v / total_mkt for t, v in mkt_vals.items()})
        pos_md = "\n".join(rows)
        pnl_section = f"""## Current Positions  (unrealised P&L: {total_unrealised:+.2f})
| Ticker   | Shares | Entry    | Now      | P&L ($)   | P&L (%) |
|----------|--------|----------|----------|-----------|---------|
{pos_md}
"""

    # ── Portfolio state ────────────────────────────────────────────────────────
    portfolio_state = json.loads(state_path.read_text()) if state_path.exists() else {}
    last_rebal      = portfolio_state.get("last_rebalance_date") or "never"
    is_rebal_day    = ps.is_rebalance_day()

    # ── Trade list ────────────────────────────────────────────────────────────
    trade_list = _compute_trade_list(
        positions, current_weights, weights, latest_prices, account_size,
        tradeable, is_rebal_day,
    )

    # ── Append to log ──────────────────────────────────────────────────────────
    top5 = "|".join(ensemble.nlargest(5).index.tolist())
    _append_log({
        "date": today, "signal_date": str(latest_date.date()),
        "days_stale": days_stale,
        "vix": round(vix_val, 2), "spy": round(spy_val, 2),
        "spy_ma200": round(spy_ma200, 2), "momentum_z": round(mom_z, 4),
        "vix_ok": int(vix_ok), "spy_ok": int(spy_ok),
        "momentum_ok": int(mom_ok), "regime_open": int(tradeable),
        "regime_consec_days": regime_consec,
        "n_stocks": n_stocks, "n_buy_zone": n_buy_zone,
        "top_tickers": top5,
        "ridge_train_ic": round(ridge_train_ic, 4),
    })

    # ── Format ─────────────────────────────────────────────────────────────────
    icon = lambda ok: "✅" if ok else "❌"

    stale_warn = ""
    if days_stale > 1:
        stale_warn += f"\n> ⚠️ **Data is {days_stale} days stale** (latest: {latest_date.date()}).\n"
    if model_train_end:
        model_age = (pd.Timestamp(today) - pd.Timestamp(model_train_end)).days
        if model_age > 35:
            stale_warn += f"\n> ⚠️ **Models are {model_age} days old** (trained through {model_train_end}). Run: `python production/retrain.py`\n"

    if regime_known:
        regime_lines = "\n".join([
            f"| VIX          | {vix_val:.1f}       | < 25      | {icon(vix_ok)} |",
            f"| SPY vs MA200 | {spy_val:.2f} / {spy_ma200:.2f} | SPY > MA  | {icon(spy_ok)} |",
            f"| Momentum Z   | {mom_z:.2f}      | < {detector.mom_z_threshold}     | {icon(mom_ok)} |",
        ])
        consec_label = f"{'open' if tradeable else 'closed'} {regime_consec} day{'s' if regime_consec != 1 else ''}"
        regime_status = f"{'OPEN 🟢' if tradeable else 'CLOSED 🔴'}  _(regime {consec_label})_"
    else:
        regime_lines = "| _(network unavailable — fetch VIX/SPY manually)_ | | | |"
        regime_status = "UNKNOWN ⚠️"

    if tradeable:
        action = ("TRADE — regime open, rebalance today" if is_rebal_day
                  else f"HOLD — next rebalance in ~{_days_until_rebal(portfolio_state)} business days")
        weights_md = "\n".join(
            f"| {t:<8} | {w:.1%} |"
            for t, w in weights.sort_values(ascending=False).items()
        )
        portfolio_section = f"""## Target Portfolio
| Ticker   | Weight |
|----------|--------|
{weights_md}

_Optimizer: {opt_status}_"""
    else:
        action = "FLAT — regime gate closed, no new trades"
        portfolio_section = "## Target Portfolio\n_Regime gate closed — no positions._"

    trade_section = trade_list.strip()

    top10    = ensemble.nlargest(10)
    top10_md = "\n".join(f"| {t:<8} | {s:.3f} |" for t, s in top10.items())
    signals_section = f"""## Top 10 Signals
| Ticker   | Score  |
|----------|--------|
{top10_md}

_Signal stats: {n_stocks} stocks scored · {n_buy_zone} in buy zone (top quartile) · ridge train IC: {ridge_train_ic:.4f}_"""

    return f"""---
date: {today}
generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
signal_date: {latest_date.date()}
---

# Daily Brief — {today}
{stale_warn}
## Regime Gate
| Gate         | Value       | Threshold | Status |
|--------------|-------------|-----------|--------|
{regime_lines}

**Overall: {regime_status}**

## Action
**{action}**

Last rebalance: {last_rebal}

{portfolio_section}

{trade_section}

{signals_section}

{pnl_section}---
_Models: {model_name} (production) · Data through {latest_date.date()}_
"""


def _compute_trade_list(
    positions,
    current_weights,
    target_weights,
    latest_prices,
    account_size: float,
    tradeable: bool,
    is_rebal_day: bool,
) -> str:
    import pandas as pd

    trade_path = Path("data/production/trade_list.md")
    held   = set(current_weights.index) if not current_weights.empty else set()
    target = set(target_weights.index)  if not target_weights.empty  else set()
    rows   = []

    if not tradeable and held:
        for tkr in sorted(held):
            pos  = positions[positions["ticker"] == tkr]
            shrs = int(pos["shares"].iloc[0]) if not pos.empty else "?"
            rows.append(f"| {tkr:<8} | SELL     | {shrs!s:>6} | regime closed |")
        summary = "SELL ALL — regime gate closed, exit all positions"

    elif tradeable and is_rebal_day:
        for tkr in sorted(held | target):
            cur_w = float(current_weights.get(tkr, 0.0))
            tgt_w = float(target_weights.get(tkr, 0.0))
            price = float(latest_prices.get(tkr, float("nan")))
            diff  = tgt_w - cur_w
            tgt_shrs = (int(tgt_w * account_size / price)
                        if tgt_w > 0 and not math.isnan(price) else "?")
            if tgt_w == 0:
                pos  = positions[positions["ticker"] == tkr]
                shrs = int(pos["shares"].iloc[0]) if not pos.empty else "?"
                rows.append(f"| {tkr:<8} | SELL     | {shrs!s:>6} | dropped from target |")
            elif cur_w == 0:
                rows.append(f"| {tkr:<8} | BUY      | {tgt_shrs!s:>6} | new → {tgt_w:.1%} |")
            elif abs(diff) >= 0.05:
                verb = "add" if diff > 0 else "trim"
                rows.append(f"| {tkr:<8} | REBALANCE| {tgt_shrs!s:>6} | {verb} {abs(diff):.1%} |")
            else:
                rows.append(f"| {tkr:<8} | HOLD     | {'—':>6} | diff {diff:+.1%} |")
        summary = "REBALANCE — execute trades below"

    else:
        summary = "HOLD — no trades today"
        for tkr in sorted(held):
            rows.append(f"| {tkr:<8} | HOLD     | {'—':>6} | |")

    header = "| Ticker   | Action   | Shares | Notes               |"
    sep    = "|----------|----------|--------|---------------------|"
    body   = "\n".join(rows) if rows else "| _(no open positions)_ | | | |"

    md = f"""## Trades
_{summary}_

{header}
{sep}
{body}

_Edit `data/production/trade_list.md` to override before executing in IB TWS._
_After executing: `python production/daily_run.py --confirm-trade`_"""

    trade_path.parent.mkdir(parents=True, exist_ok=True)
    trade_path.write_text(md.strip() + "\n")
    return md


def _fetch_regime_data(start: str, end: str):
    """Returns (vix_series, spy_series) or (None, None) on network failure."""
    import pandas as pd
    import yfinance as yf

    def _flatten(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        elif not df.empty:
            df.columns = [c.lower() for c in df.columns]
        return df

    try:
        spy_raw = _flatten(yf.download("SPY", start=start, end=end,
                                        auto_adjust=True, progress=False))
        if spy_raw.empty or "close" not in spy_raw:
            return None, None

        vix_raw = _flatten(yf.download("^VIX", start=start, end=end,
                                        auto_adjust=True, progress=False))
        if vix_raw.empty or "close" not in vix_raw:
            # yfinance ^VIX sometimes fails for longer ranges — retry with 1yr
            fallback_start = str((pd.Timestamp(end) - pd.DateOffset(days=365)).date())
            vix_raw = _flatten(yf.download("^VIX", start=fallback_start, end=end,
                                            auto_adjust=True, progress=False))
        if vix_raw.empty or "close" not in vix_raw:
            return None, None

        vix = vix_raw["close"].rename(None)
        spy = spy_raw["close"].rename(None)
        vix.index = pd.to_datetime(vix.index)
        spy.index = pd.to_datetime(spy.index)
        return vix, spy
    except Exception:
        return None, None


def _regime_consecutive_days(today: str, regime_open: bool) -> int:
    """Count consecutive days in the log where regime_open matches today's value."""
    if not LOG_PATH.exists():
        return 1
    with open(LOG_PATH) as f:
        rows = list(csv.DictReader(f))
    count = 1
    for row in reversed(rows):
        if row.get("date") == today:
            continue
        if int(row.get("regime_open", -1)) == int(regime_open):
            count += 1
        else:
            break
    return count


def _append_log(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    # Overwrite today's row if it already exists
    existing = []
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            existing = list(csv.DictReader(f))
        existing = [r for r in existing if r.get("date") != row["date"]]
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow({k: row.get(k, "") for k in LOG_COLS})


def _stub_brief(today: str, state_path: Path, reason: str) -> str:
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
    return f"""---
date: {today}
generated: {datetime.now().isoformat()}
---

# Daily Brief — {today}

## Status
{reason}

Last rebalance: {state.get("last_rebalance_date") or "never"}
"""


def _days_until_rebal(state: dict) -> int:
    import pandas as pd
    last = state.get("last_rebalance_date")
    freq = state.get("frequency_days", 5)
    if last is None:
        return 0
    elapsed = len(pd.bdate_range(pd.Timestamp(last), date.today())) - 1
    return max(0, freq - elapsed)


if __name__ == "__main__":
    main()
