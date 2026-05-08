import pandas as pd
from .base import BaseFeature

class MarketContext(BaseFeature):
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        if "vix_close" not in prices.columns:
            return prices[["ticker", "date"]].copy()
        df = prices[["ticker", "date", "vix_close"]].copy()
        df["vix_level"] = df["vix_close"]
        return df[["ticker", "date", "vix_level"]]
