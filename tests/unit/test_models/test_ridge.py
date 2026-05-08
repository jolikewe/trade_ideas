import pytest
import pandas as pd
import numpy as np
from trading_system.models.ridge import RidgeModel

@pytest.fixture
def train_val_data():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(X.iloc[:, 0] * 0.1 + rng.standard_normal(n) * 0.5)
    return X[:400], y[:400], X[400:], y[400:]

def test_ridge_trains(train_val_data):
    X_tr, y_tr, X_val, y_val = train_val_data
    model = RidgeModel()
    model.fit(X_tr, y_tr, X_val, y_val)
    assert model.model is not None

def test_ridge_ic_in_range(train_val_data):
    X_tr, y_tr, X_val, y_val = train_val_data
    model = RidgeModel()
    model.fit(X_tr, y_tr, X_val, y_val)
    assert -1.0 <= model.val_ic <= 1.0

def test_ridge_predict_shape(train_val_data):
    X_tr, y_tr, X_val, y_val = train_val_data
    model = RidgeModel()
    model.fit(X_tr, y_tr, X_val, y_val)
    preds = model.predict(X_val)
    assert len(preds) == len(X_val)
