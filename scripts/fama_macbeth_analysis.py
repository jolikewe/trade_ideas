#!/usr/bin/env python3
"""Fama-MacBeth regression for feature validation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    import pandas as pd
    import numpy as np
    from trading_system.data.synthetic import SyntheticDataGenerator
    from trading_system.features.mean_reversion import MeanReversionFeatures, FEATURE_COLS
    from trading_system.labels.builder import LabelBuilder

    print("Running Fama-MacBeth analysis on synthetic data...")
    gen = SyntheticDataGenerator(seed=42)
    prices = gen.generate_prices(n_stocks=50, start_date="2018-01-01", end_date="2022-12-31")
    feats = MeanReversionFeatures().compute_all(prices)
    labels = LabelBuilder().build_labels(prices)
    data = feats.merge(labels, on=["ticker", "date"], how="inner").dropna()

    t_stats = {}
    for col in FEATURE_COLS:
        if col not in data.columns:
            continue
        cross_betas = data.groupby("date").apply(
            lambda g: g[[col, "label"]].corr().iloc[0, 1]
        )
        t_stat = cross_betas.mean() / (cross_betas.std() / len(cross_betas) ** 0.5)
        t_stats[col] = t_stat

    print("\nFama-MacBeth t-statistics:")
    for feat, t in sorted(t_stats.items(), key=lambda x: abs(x[1]), reverse=True):
        sig = "***" if abs(t) > 3 else "**" if abs(t) > 2 else "*" if abs(t) > 1.96 else ""
        print(f"  {feat:<25} t={t:+.2f} {sig}")

if __name__ == "__main__":
    main()
