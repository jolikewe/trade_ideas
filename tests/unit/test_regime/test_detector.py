import pytest
import pandas as pd
from trading_system.data.synthetic import SyntheticDataGenerator
from trading_system.regime.detector import RegimeDetector

@pytest.fixture
def spy_vix():
    gen0 = SyntheticDataGenerator(seed=0)
    gen1 = SyntheticDataGenerator(seed=1)
    spy = gen0.generate_spy("2015-01-01", "2021-12-31")
    vix = gen1.generate_vix("2015-01-01", "2021-12-31")
    return spy, vix

def test_regime_columns(spy_vix, tmp_path):
    spy, vix = spy_vix
    detector = RegimeDetector("config/regime/detector.yaml")
    regime = detector.detect_composite_regime(vix, spy)
    assert {"vix_ok", "spy_ok", "tradeable"}.issubset(regime.columns)

def test_tradeable_pct_range(spy_vix):
    spy, vix = spy_vix
    detector = RegimeDetector("config/regime/detector.yaml")
    tradeable = detector.get_tradeable_dates(vix, spy)
    pct = tradeable.mean()
    assert 0.2 <= pct <= 0.9

def test_vix_gate(spy_vix):
    spy, vix = spy_vix
    detector = RegimeDetector("config/regime/detector.yaml")
    regime = detector.detect_composite_regime(vix, spy)
    high_vix = vix[vix > 25].index
    assert (regime.loc[regime.index.isin(high_vix), "vix_ok"] == 0).all()
