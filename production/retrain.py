#!/usr/bin/env python3
"""Monthly production retrain. Trains on all available data, saves as window_12."""
import argparse
from datetime import date
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser(description="Monthly production retrain")
    parser.add_argument("--dry-run", action="store_true", help="Preview splits without training")
    parser.add_argument("--train-start", default="2010-01-01", help="Training start date")
    parser.add_argument("--val-months", type=int, default=12, help="Validation window in months")
    args = parser.parse_args()

    today = date.today().isoformat()
    from dateutil.relativedelta import relativedelta
    import pandas as pd
    val_start = (pd.Timestamp(today) - pd.DateOffset(months=args.val_months)).date().isoformat()

    print(f"Production Retrain")
    print(f"  Train: {args.train_start} → {val_start}")
    print(f"  Val:   {val_start} → {today}")

    if args.dry_run:
        print("Dry run — no training performed.")
        return

    from trading_system.data.loaders import DataLoader
    from trading_system.universe.historical import PointInTimeUniverse
    from trading_system.features.mean_reversion import MeanReversionFeatures
    from trading_system.labels.builder import LabelBuilder
    from trading_system.models.trainer import ModelTrainer

    print("Loading universe...")
    pit = PointInTimeUniverse.load_or_build()
    loader = DataLoader()
    prices = loader.load_prices(pit.all_tickers, args.train_start, today)

    print("Computing features and labels...")
    feats = MeanReversionFeatures().compute_all(prices)
    labels = LabelBuilder().build_labels(prices)

    print("Training production model (window 12)...")
    trainer = ModelTrainer()
    results = trainer.train_window(feats, labels, args.train_start, val_start,
                                    val_start, today, window_id=12)

    for model, r in results.items():
        print(f"  {model}: train_ic={r['train_ic']:.4f}, val_ic={r['val_ic']:.4f}")

    print("Retrain complete. Run `python production/daily_run.py --refresh` to update brief.")

if __name__ == "__main__":
    main()
