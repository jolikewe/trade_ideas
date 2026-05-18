#!/usr/bin/env python3
"""Monthly production retrain. Safe to run anytime; skips if models are current."""
import argparse
import json
from datetime import date
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MODEL_DIR   = Path("data/models/mean_reversion")
MODEL_NAME  = "model_mr_zscore_12feat"
RETRAIN_DAYS = 30


def _model_age_days() -> int | None:
    meta = MODEL_DIR / f"{MODEL_NAME}_ridge" / "production" / "metadata.json"
    if not meta.exists():
        return None
    train_end = json.loads(meta.read_text()).get("train_end")
    if not train_end:
        return None
    import pandas as pd
    return (pd.Timestamp(date.today()) - pd.Timestamp(train_end)).days


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Retrain even if models are current")
    parser.add_argument("--train-start", default="2010-01-01")
    args = parser.parse_args()

    today = date.today().isoformat()
    age = _model_age_days()

    if age is not None:
        print(f"Production model last trained {age} days ago (train_end in metadata).")
        if age < RETRAIN_DAYS and not args.force:
            print(f"Models are current (< {RETRAIN_DAYS} days). Pass --force to override.")
            return
    else:
        print("No production metadata found — training from scratch.")

    print(f"Training on {args.train_start} → {today}...")

    from trading_system.data.loaders import DataLoader
    from trading_system.universe.historical import PointInTimeUniverse
    from trading_system.features.mean_reversion import MeanReversionFeatures
    from trading_system.labels.builder import LabelBuilder
    from trading_system.models.trainer import ModelTrainer

    import pandas as pd

    pit    = PointInTimeUniverse.load_or_build()
    loader = DataLoader()
    print("Loading prices...", end=" ", flush=True)
    prices = loader.load_prices(pit.all_tickers, args.train_start, today)
    print(f"done ({prices['ticker'].nunique()} tickers)")

    # Point-in-time filter: only train on rows where the ticker was in the S&P 500
    pit_df = pit.df[["ticker", "added_date", "removed_date"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.merge(pit_df, on="ticker", how="left")
    prices = prices[
        (prices["date"] >= prices["added_date"]) &
        (prices["removed_date"].isna() | (prices["date"] < prices["removed_date"]))
    ].drop(columns=["added_date", "removed_date"])
    print(f"After PIT filter: {prices['ticker'].nunique()} tickers, {len(prices):,} rows")

    feats  = MeanReversionFeatures().compute_all(prices)
    labels = LabelBuilder().build_labels(prices)

    trainer = ModelTrainer()
    results = trainer.train_production(feats, labels, args.train_start, today)
    for model, r in results.items():
        print(f"  {model}: train_ic={r['train_ic']:.4f}")

    print("Done. Run `python production/daily_run.py --refresh` to update brief.")


if __name__ == "__main__":
    main()
