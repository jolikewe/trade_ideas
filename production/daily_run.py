#!/usr/bin/env python3
"""Daily brief generator. Auto-runs at Claude session start via .claude/session-start.sh."""
import argparse
import json
from datetime import date, datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
    import yfinance as yf
    from trading_system.data.loaders import DataLoader
    from trading_system.universe.historical import PointInTimeUniverse
    from trading_system.features.mean_reversion import MeanReversionFeatures, FEATURE_COLS
    from trading_system.models.ridge import RidgeModel
    from trading_system.models.lightgbm_model import LightGBMModel
    from trading_system.backtest.cross_sectional.signal_generator import ICWeightedSignalGenerator
    from trading_system.regime.detector import RegimeDetector
    from trading_system.portfolio.optimizer import FactorNeutralOptimizer
    from trading_system.utils.config import load_config

    cfg = load_config("config/backtest.yaml")["walk_forward"]

    # Load production models
    model_dir = Path("data/models/mean_reversion")
    model_name = "model_mr_zscore_12feat"
    ridge_path = model_dir / f"{model_name}_ridge" / "production" / "model.pkl"
    lgb_path = model_dir / f"{model_name}_lightgbm" / "production" / "model.json"

    if not ridge_path.exists() or not lgb_path.with_suffix(".lgb").exists():
        return _stub_brief(today, state_path, "Production models not found. Run: python -m trading_system.cli production-train")

    ridge = RidgeModel.load(str(ridge_path))
    lgb_model = LightGBMModel.load(str(lgb_path))

    # Load prices from cache — use today so we pick up any freshly downloaded data
    pit = PointInTimeUniverse.load_or_build()
    loader = DataLoader()
    prices = loader.load_prices(pit.all_tickers, cfg["data_start_date"], today)
    if prices.empty:
        return _stub_brief(today, state_path, "No price data in cache.")

    # Compute features
    feats = MeanReversionFeatures().compute_all(prices)
    latest_date = feats["date"].max()
    latest_feats = feats[feats["date"] == latest_date].copy()

    # Generate signals
    sig_gen = ICWeightedSignalGenerator()
    ridge_meta_path = ridge_path.parent / "metadata.json"
    lgb_meta_path = lgb_path.parent / "metadata.json"
    ridge_val_ic = json.loads(ridge_meta_path.read_text()).get("val_ic") or 0.0
    lgb_val_ic = json.loads(lgb_meta_path.read_text()).get("val_ic") or 0.0

    X_r = latest_feats.set_index("ticker")[ridge.feature_names].fillna(0)
    X_l = latest_feats.set_index("ticker")[lgb_model.feature_names].fillna(0)
    ridge_scores = pd.Series(ridge.predict(X_r), index=X_r.index)
    lgb_scores = pd.Series(lgb_model.predict(X_l), index=X_l.index)

    # For production models there's no val_ic — use equal weighting
    if ridge_val_ic == 0.0 and lgb_val_ic == 0.0:
        ensemble = (ridge_scores.rank(pct=True) + lgb_scores.rank(pct=True)) / 2
    else:
        val_ics = {"ridge": ridge_val_ic, "lightgbm": lgb_val_ic}
        ensemble = sig_gen.generate({"ridge": ridge_scores, "lightgbm": lgb_scores}, val_ics)

    ensemble = ensemble.rank(pct=True)

    # Fetch VIX and SPY for regime gate — need 200+ trading days before latest_date
    regime_start = str((latest_date - pd.DateOffset(days=450)).date())
    vix_series, spy_series = _fetch_regime_data(regime_start, today)

    # Regime detection — cap VIX/SPY to latest price date so momentum z aligns
    regime_known = vix_series is not None and spy_series is not None and len(vix_series) > 10
    if regime_known:
        vix_series = vix_series[vix_series.index <= latest_date]
        spy_series = spy_series[spy_series.index <= latest_date]
        detector = RegimeDetector()
        regime = detector.detect_composite_regime(vix_series, spy_series, prices=prices)
        latest_regime_date = regime.index[regime.index.notna()].max()
        regime_row = regime.loc[latest_regime_date]
        tradeable = bool(regime_row["tradeable"])
        vix_val = float(vix_series.iloc[-1])
        spy_val = float(spy_series.iloc[-1])
        spy_ma200 = float(spy_series.rolling(200).mean().iloc[-1])
        mom_z = float(regime_row.get("momentum_z", float("nan")))
    else:
        tradeable = False
        vix_val = spy_val = spy_ma200 = mom_z = float("nan")

    # Portfolio weights (only if regime open)
    weights = pd.Series(dtype=float)
    opt_status = ""
    if tradeable:
        optimizer = FactorNeutralOptimizer(target_n=6)
        result = optimizer.optimize(ensemble)
        weights = result["weights"]
        opt_status = result["status"]

    # Portfolio state
    portfolio_state = {}
    if state_path.exists():
        portfolio_state = json.loads(state_path.read_text())
    last_rebal = portfolio_state.get("last_rebalance_date") or "never"
    sys.path.insert(0, str(Path(__file__).parent))
    from portfolio_state import PortfolioState
    ps = PortfolioState()
    is_rebal_day = ps.is_rebalance_day()
    positions = ps.load_positions()

    # Format brief
    gate_icon = lambda ok: "✅" if ok else "❌"
    if regime_known:
        regime_lines = [
            f"| VIX            | {vix_val:.1f}   | <25      | {gate_icon(regime_row['vix_ok'])} |",
            f"| SPY vs MA200   | {spy_val:.2f} / {spy_ma200:.2f} | SPY>MA200 | {gate_icon(regime_row['spy_ok'])} |",
            f"| Momentum Z     | {mom_z:.2f}   | <0.5     | {gate_icon(regime_row['momentum_ok'])} |",
        ]
        regime_status = "OPEN 🟢" if tradeable else "CLOSED 🔴"
    else:
        regime_lines = ["| _(network unavailable — fetch VIX/SPY manually)_ |  |  |  |"]
        regime_status = "UNKNOWN ⚠️"

    if tradeable:
        action = "TRADE — regime open, rebalance today" if is_rebal_day else f"HOLD — next rebalance due in ~{_days_until_rebal(portfolio_state)} business days"
        weights_md = "\n".join(
            f"| {t:<8} | {w:.1%} |" for t, w in weights.sort_values(ascending=False).items()
        )
        portfolio_section = f"""## Target Portfolio
| Ticker   | Weight |
|----------|--------|
{weights_md}

Optimizer status: {opt_status}"""
    else:
        action = "FLAT — regime gate closed, no new trades"
        portfolio_section = "## Target Portfolio\n_Regime gate closed — no positions._"

    top10 = ensemble.nlargest(10)
    top10_md = "\n".join(f"| {t:<8} | {s:.3f} |" for t, s in top10.items())

    signals_section = f"""## Top 10 Signals (as of {latest_date.date()})
| Ticker   | Score  |
|----------|--------|
{top10_md}"""

    positions_section = ""
    if not positions.empty:
        pos_md = "\n".join(
            f"| {r['ticker']:<8} | {r['shares']:>6} | {r.get('entry_price', 'n/a')} |"
            for _, r in positions.iterrows()
        )
        positions_section = f"""## Current Positions
| Ticker   | Shares | Entry |
|----------|--------|-------|
{pos_md}
"""

    brief = f"""---
date: {today}
generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
---

# Daily Brief — {today}

## Regime Gate
| Gate           | Value  | Threshold | Status |
|----------------|--------|-----------|--------|
{chr(10).join(regime_lines)}

**Overall: {regime_status}**

## Action
**{action}**

Last rebalance: {last_rebal}

{portfolio_section}

{signals_section}

{positions_section}---
_Models: {model_name} (production). Signal date: {latest_date.date()}._
"""
    return brief


def _fetch_regime_data(start: str, end: str):
    """Returns (vix_series, spy_series) or (None, None) on network failure."""
    import pandas as pd
    import yfinance as yf

    def _flatten(df):
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df

    try:
        vix_raw = _flatten(yf.download("^VIX", start=start, end=end,
                                        auto_adjust=True, progress=False))
        spy_raw = _flatten(yf.download("SPY", start=start, end=end,
                                        auto_adjust=True, progress=False))
        if vix_raw.empty or spy_raw.empty or "close" not in vix_raw or "close" not in spy_raw:
            return None, None
        vix = vix_raw["close"].rename(None)
        spy = spy_raw["close"].rename(None)
        vix.index = pd.to_datetime(vix.index)
        spy.index = pd.to_datetime(spy.index)
        return vix, spy
    except Exception:
        return None, None


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

Last rebalance: {state.get("last_rebalance_date", "never")}
"""


def _days_until_rebal(state: dict) -> int:
    import pandas as pd
    from datetime import date
    last = state.get("last_rebalance_date")
    freq = state.get("frequency_days", 5)
    if last is None:
        return 0
    elapsed = len(pd.bdate_range(pd.Timestamp(last), date.today())) - 1
    return max(0, freq - elapsed)


if __name__ == "__main__":
    main()
