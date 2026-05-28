import argparse

def main():
    parser = argparse.ArgumentParser(prog="trading-system",
                                      description="Trading System v2 CLI")
    subparsers = parser.add_subparsers(dest="command")

    dl = subparsers.add_parser("download", help="Download market data")
    dl.add_argument("--start", default="2010-01-01")
    dl.add_argument("--end", default="2025-12-31")
    dl.add_argument("--tickers", default="sp500", choices=["sp500", "sp1500"])
    dl.add_argument("--force", action="store_true")

    tr = subparsers.add_parser("train", help="Train models")
    tr.add_argument("--models", default="ridge", choices=["ridge"])
    tr.add_argument("--feature-sets", default="all")
    tr.add_argument("--windows", default="all")
    tr.add_argument("--verbose", "-v", action="store_true")
    tr.add_argument("--parallel", "-p", action="store_true")
    tr.add_argument("--n-jobs", type=int, default=4)
    tr.add_argument("--no-cache", action="store_true")
    tr.add_argument("--synthetic", action="store_true")

    bt = subparsers.add_parser("backtest", help="Run backtest")
    bt.add_argument("--window", type=int, required=True)
    bt.add_argument("--verbose", "-v", action="store_true")
    bt.add_argument("--no-regime", action="store_true")

    pt = subparsers.add_parser("production-train", help="Train production model on full history")
    pt.add_argument("--verbose", "-v", action="store_true")

    subparsers.add_parser("spy-timing", help="SPY timing backtest (v1 compat)")

    args = parser.parse_args()

    if args.command == "download":
        _cmd_download(args)
    elif args.command == "train":
        _cmd_train(args)
    elif args.command == "backtest":
        _cmd_backtest(args)
    elif args.command == "production-train":
        _cmd_production_train(args)
    elif args.command == "spy-timing":
        _cmd_spy_timing(args)
    else:
        parser.print_help()

def _cmd_download(args):
    from .universe.historical import PointInTimeUniverse
    from .data.yfinance_loader import YFinanceLoader
    print(f"Downloading {args.tickers} data {args.start} → {args.end}")
    pit = PointInTimeUniverse.load_or_build()
    tickers = pit.all_tickers
    loader = YFinanceLoader()
    df = loader.load_universe(tickers, args.start, args.end, force_refresh=args.force)
    print(f"Downloaded {len(df)} rows for {df['ticker'].nunique()} tickers")

def _cmd_train(args):
    from .data.synthetic import SyntheticDataGenerator
    from .features.mean_reversion import MeanReversionFeatures
    from .labels.builder import LabelBuilder
    from .models.trainer import ModelTrainer
    from .utils.config import load_config

    cfg = load_config("config/backtest.yaml")["walk_forward"]
    start = cfg["data_start_date"]
    end = cfg["data_end_date"]

    if args.synthetic:
        print("Using synthetic data...")
        gen = SyntheticDataGenerator(seed=42)
        prices = gen.generate_prices(n_stocks=50, start_date=start, end_date=end)
    else:
        from .data.loaders import DataLoader
        from .universe.historical import PointInTimeUniverse
        pit = PointInTimeUniverse.load_or_build()
        loader = DataLoader()
        prices = loader.load_prices(pit.all_tickers, start, end,
                                    force_refresh=args.no_cache)

    print("Computing features...")
    feats = MeanReversionFeatures().compute_all(prices)
    labels = LabelBuilder().build_labels(prices)

    # PIT filter: only train on rows where ticker was in the S&P 500
    import pandas as pd
    pit_df = pit.df[["ticker", "added_date", "removed_date"]].copy()
    feats["date"] = pd.to_datetime(feats["date"])
    feats = feats.merge(pit_df, on="ticker", how="left")
    feats = feats[
        (feats["date"] >= feats["added_date"]) &
        (feats["removed_date"].isna() | (feats["date"] < feats["removed_date"]))
    ].drop(columns=["added_date", "removed_date"])
    if args.verbose:
        print(f"PIT filter: {feats['ticker'].nunique()} tickers, {len(feats):,} rows")

    trainer = ModelTrainer()
    windows = _parse_windows(args.windows, cfg)

    for w_id, (ts, te, vs, ve) in enumerate(windows, 1):
        if args.verbose:
            print(f"Window {w_id}: train {ts}→{te}, val {vs}→{ve}")
        results = trainer.train_window(feats, labels, ts, te, vs, ve, w_id)
        if args.verbose:
            for m, r in results.items():
                print(f"  {m}: train_ic={r['train_ic']:.4f}, val_ic={r['val_ic']:.4f}")

def _cmd_backtest(args):
    import json
    from datetime import datetime
    from pathlib import Path

    import pandas as pd
    import yfinance as yf

    from .backtest.cross_sectional.engine import BacktestEngine
    from .data.loaders import DataLoader
    from .features.mean_reversion import MeanReversionFeatures
    from .models.ridge import RidgeModel
    from .universe.historical import PointInTimeUniverse
    from .utils.config import load_config

    cfg = load_config("config/backtest.yaml")["walk_forward"]
    windows = _parse_windows("all", cfg)

    w_idx = args.window - 1
    if w_idx < 0 or w_idx >= len(windows):
        print(f"Error: window must be 1-{len(windows)}")
        return

    ts, te, vs, ve = windows[w_idx]
    test_start = ve
    test_end = str((pd.Timestamp(ve) + pd.DateOffset(years=cfg["test_years"])).date())
    print(f"Window {args.window}: train {ts}→{te}, val {vs}→{ve}, test {test_start}→{test_end}")
    print(f"Regime gate: {'on' if not args.no_regime else 'off'}")

    model_dir = Path("data/models/mean_reversion")
    model_name = "model_mr_zscore_12feat"
    ridge_path = model_dir / f"{model_name}_ridge" / f"window_{args.window}" / "model.pkl"

    if not ridge_path.exists():
        print(f"Error: Ridge model for window {args.window} not found. Run 'train' first.")
        return

    ridge = RidgeModel.load(str(ridge_path))
    print(f"Model: ridge val_ic={ridge.val_ic:.4f}")

    pit = PointInTimeUniverse.load_or_build()
    loader = DataLoader()
    print("Loading prices...")
    prices = loader.load_prices(pit.all_tickers, cfg["data_start_date"], cfg["data_end_date"])

    print("Computing features...")
    feats = MeanReversionFeatures().compute_all(prices)
    test_feats = feats[(feats["date"] >= test_start) & (feats["date"] < test_end)].copy()

    print("Generating signals...")
    signals: dict = {}
    for date, group in test_feats.groupby("date"):
        active = set(pit.get_universe(pd.Timestamp(date)))
        group = group[group["ticker"].isin(active)]
        if len(group) < 10:
            continue
        X = group.set_index("ticker")[ridge.feature_names].fillna(0)
        signals[date] = pd.Series(ridge.predict(X), index=X.index)

    signals_df = pd.DataFrame(signals).T
    signals_df.index = pd.DatetimeIndex(signals_df.index)

    vix_series = None
    if not args.no_regime:
        print("Fetching VIX...")
        vix_raw = yf.download("^VIX", start=cfg["data_start_date"], end=test_end,
                              auto_adjust=True, progress=False)
        if not vix_raw.empty:
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = [c[0].lower() for c in vix_raw.columns]
            else:
                vix_raw.columns = [c.lower() for c in vix_raw.columns]
            vix_series = vix_raw["close"].rename(None)
            vix_series.index = pd.to_datetime(vix_series.index)

    pre_start = pd.bdate_range(end=test_start, periods=401)[0]
    extended_prices = prices[
        (prices["date"] >= pre_start) & (prices["date"] < pd.Timestamp(test_end))
    ].copy()
    n_days = extended_prices[extended_prices["date"] >= pd.Timestamp(test_start)]["date"].nunique()
    print(f"Running backtest on {n_days} trading days (+ {(pd.Timestamp(test_start) - pre_start).days}d warmup)...")
    engine = BacktestEngine()
    result = engine.run(extended_prices, signals_df, test_start, test_end,
                        use_regime=not args.no_regime, vix_series=vix_series)

    out_dir = Path("data/results/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"window_{args.window}_{ts_str}.json"
    payload = {
        "window": args.window, "train_start": ts, "train_end": te,
        "val_start": vs, "val_end": ve, "test_start": test_start, "test_end": test_end,
        "use_regime": not args.no_regime, "val_ic": ridge.val_ic,
        "sharpe": result["sharpe"], "max_drawdown": result["max_drawdown"],
        "dsr": result["dsr"], "returns": result["returns"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nResults — window {args.window}:")
    print(f"  Sharpe:       {result['sharpe']:.4f}")
    print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"  DSR:          {result['dsr']:.4f}")
    print(f"  Saved:        {out_path}")

def _cmd_production_train(args):
    from .data.loaders import DataLoader
    from .universe.historical import PointInTimeUniverse
    from .features.mean_reversion import MeanReversionFeatures
    from .labels.builder import LabelBuilder
    from .models.trainer import ModelTrainer
    from .utils.config import load_config

    cfg = load_config("config/backtest.yaml")["walk_forward"]
    train_start = cfg["data_start_date"]
    train_end = cfg["data_end_date"]

    import pandas as pd

    pit = PointInTimeUniverse.load_or_build()
    loader = DataLoader()
    print("Loading prices...")
    prices = loader.load_prices(pit.all_tickers, train_start, train_end)

    pit_df = pit.df[["ticker", "added_date", "removed_date"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.merge(pit_df, on="ticker", how="left")
    prices = prices[
        (prices["date"] >= prices["added_date"]) &
        (prices["removed_date"].isna() | (prices["date"] < prices["removed_date"]))
    ].drop(columns=["added_date", "removed_date"])

    print("Computing features...")
    feats = MeanReversionFeatures().compute_all(prices)
    labels = LabelBuilder().build_labels(prices)

    print(f"Training production models on {train_start} → {train_end}...")
    trainer = ModelTrainer()
    results = trainer.train_production(feats, labels, train_start, train_end)

    print("Production training complete:")
    for model, metrics in results.items():
        print(f"  {model}: train_ic={metrics['train_ic']:.4f}")
    print("Saved to data/models/mean_reversion/*/production/")

def _cmd_spy_timing(args):
    from .backtest.directional.spy_timing import SPYTimingBacktest
    from .data.synthetic import SyntheticDataGenerator
    gen = SyntheticDataGenerator(seed=0)
    spy = gen.generate_spy()
    result = SPYTimingBacktest().run(spy)
    print(f"SPY Timing — Sharpe: {result['sharpe']:.2f}, MaxDD: {result['max_drawdown']:.2%}")

def _parse_windows(windows_arg: str, cfg: dict) -> list[tuple]:
    import pandas as pd
    start = pd.Timestamp(cfg["data_start_date"])
    train_y = cfg["train_years"]
    val_y = cfg["val_years"]
    test_y = cfg["test_years"]
    result = []
    t = start
    while True:
        te = t + pd.DateOffset(years=train_y)
        vs = te
        ve = vs + pd.DateOffset(years=val_y)
        if ve > pd.Timestamp(cfg["data_end_date"]):
            break
        result.append((str(t.date()), str(te.date()), str(vs.date()), str(ve.date())))
        t += pd.DateOffset(years=1)
        if len(result) >= 11:
            break
    return result

if __name__ == "__main__":
    main()
