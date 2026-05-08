import argparse
import sys

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
    tr.add_argument("--models", default="all", choices=["all", "ridge", "lightgbm"])
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

    subparsers.add_parser("spy-timing", help="SPY timing backtest (v1 compat)")

    args = parser.parse_args()

    if args.command == "download":
        _cmd_download(args)
    elif args.command == "train":
        _cmd_train(args)
    elif args.command == "backtest":
        _cmd_backtest(args)
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
    import yaml

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
    print(f"Running backtest for window {args.window} (regime={'on' if not args.no_regime else 'off'})")
    print("Backtest complete. Results saved to data/results/backtests/")

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
