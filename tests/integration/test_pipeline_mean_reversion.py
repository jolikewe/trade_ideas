import pytest
import pandas as pd
from trading_system.data.synthetic import SyntheticDataGenerator
from trading_system.features.mean_reversion import MeanReversionFeatures
from trading_system.labels.builder import LabelBuilder
from trading_system.models.ridge import RidgeModel
from trading_system.features.mean_reversion import FEATURE_COLS

@pytest.fixture
def synthetic_prices():
    gen = SyntheticDataGenerator(seed=99)
    return gen.generate_prices(n_stocks=20, start_date="2018-01-01", end_date="2022-12-31")

def test_full_pipeline(synthetic_prices):
    prices = synthetic_prices
    feats = MeanReversionFeatures().compute_all(prices)
    labels = LabelBuilder(horizon_days=5).build_labels(prices)

    data = feats.merge(labels, on=["ticker", "date"], how="inner").dropna()
    assert len(data) > 100

    split = data["date"].quantile(0.8)
    train = data[data["date"] <= split]
    val = data[data["date"] > split]

    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    model = RidgeModel()
    model.fit(train[feature_cols], train["label"],
              val[feature_cols], val["label"])
    assert model.val_ic > -1.0
