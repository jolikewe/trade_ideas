import pytest
import numpy as np
import pandas as pd
from trading_system.portfolio.optimizer import FactorNeutralOptimizer

@pytest.fixture
def signals():
    rng = np.random.default_rng(42)
    return pd.Series(rng.uniform(0, 1, 20), index=[f"T{i:02d}" for i in range(20)])

def test_weights_sum_to_one(signals):
    opt = FactorNeutralOptimizer(target_n=6)
    result = opt.optimize(signals)
    assert abs(result["weights"].sum() - 1.0) < 1e-4

def test_max_weight(signals):
    opt = FactorNeutralOptimizer(target_n=6, max_weight=0.25)
    result = opt.optimize(signals)
    assert (result["weights"] <= 0.26).all()

def test_n_positions(signals):
    opt = FactorNeutralOptimizer(target_n=6)
    result = opt.optimize(signals)
    assert 1 <= len(result["weights"]) <= 20

def test_speed(signals):
    import time
    opt = FactorNeutralOptimizer(target_n=6)
    t0 = time.time()
    opt.optimize(signals)
    assert time.time() - t0 < 5.0
