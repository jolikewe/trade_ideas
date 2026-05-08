import pandas as pd
import numpy as np
from .base import BaseFeature

class VolumeVolatility(BaseFeature):
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        df = prices[["ticker", "date", "close", "volume"]].copy()
        df["vol_20d"] = df.groupby("ticker")["close"].transform(
            lambda s: np.log(s / s.shift(1)).rolling(20).std() * np.sqrt(252)
        )
        df["rel_volume"] = df.groupby("ticker")["volume"].transform(
            lambda s: s / s.rolling(20).mean()
        )
        return df[["ticker", "date", "vol_20d", "rel_volume"]]
