import pandas as pd
import numpy as np
from .base import BaseFeature

class CrossSectionalReversal(BaseFeature):
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        df = prices[["ticker", "date", "close"]].copy()
        df["ret_1d"] = df.groupby("ticker")["close"].pct_change().shift(1)
        df["cs_reversal_1d"] = df.groupby("date")["ret_1d"].transform(
            lambda s: -(s - s.mean()) / (s.std() + 1e-8)
        )
        return df[["ticker", "date", "cs_reversal_1d"]]
