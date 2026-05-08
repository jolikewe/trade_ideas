import pandas as pd

class LabelBuilder:
    def __init__(self, horizon_days: int = 5):
        self.horizon = horizon_days

    def build_labels(self, prices: pd.DataFrame) -> pd.DataFrame:
        df = prices[["ticker", "date", "close"]].copy()
        df = df.sort_values(["ticker", "date"])
        df["label"] = df.groupby("ticker")["close"].transform(
            lambda s: s.shift(-self.horizon) / s - 1
        )
        df = df.dropna(subset=["label"])
        return df[["ticker", "date", "label"]]
