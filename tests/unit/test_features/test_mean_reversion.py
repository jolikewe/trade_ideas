import pytest
import pandas as pd
import numpy as np
from trading_system.data.synthetic import SyntheticDataGenerator
from trading_system.features.mean_reversion import MeanReversionFeatures, FEATURE_COLS

@pytest.fixture
def prices():
    gen = SyntheticDataGenerator(seed=42)
    return gen.generate_prices(n_stocks=10, start_date="2020-01-01", end_date="2022-12-31")

def test_feature_count(prices):
    feats = MeanReversionFeatures().compute_all(prices)
    computed = [c for c in FEATURE_COLS if c in feats.columns]
    assert len(computed) == 12

def test_rsi_bounds(prices):
    feats = MeanReversionFeatures().compute_all(prices)
    rsi = feats["rsi_14d"].dropna()
    assert rsi.between(0, 100).all()

def test_zscores_finite(prices):
    feats = MeanReversionFeatures().compute_all(prices)
    for col in ["zscore_5d", "zscore_20d"]:
        vals = feats[col].dropna()
        assert np.isfinite(vals).all()

def test_no_ticker_date_na(prices):
    feats = MeanReversionFeatures().compute_all(prices)
    assert feats["ticker"].notna().all()
    assert feats["date"].notna().all()
