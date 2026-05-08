import pandas as pd
from .mean_reversion import MeanReversionFeatures

class FeatureBuilder:
    def __init__(self, feature_set: str = "model_mr_zscore_12feat"):
        self.feature_set = feature_set
        self.mr = MeanReversionFeatures()

    def build(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self.mr.compute_all(prices)
