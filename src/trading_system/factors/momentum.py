import pandas as pd

def compute_momentum_12m1m(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices[["ticker", "date", "close"]].copy()
    df["mom_12m1m"] = df.groupby("ticker")["close"].transform(
        lambda s: s.shift(21) / s.shift(252) - 1
    )
    return df[["ticker", "date", "mom_12m1m"]]
