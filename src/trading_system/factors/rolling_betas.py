import numpy as np
import pandas as pd
from .ken_french import KenFrenchLoader, FACTORS

class RollingBetaComputer:
    def __init__(self, window: int = 252):
        self.window = window
        self.kf = KenFrenchLoader()

    def compute(self, prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        ff = self.kf.load(start, end)
        results = []
        for ticker, grp in prices.groupby("ticker"):
            grp = grp.set_index("date").sort_index()
            ret = grp["close"].pct_change()
            merged = ret.to_frame("ret").join(ff, how="left").dropna()
            betas = self._rolling_ols(merged, self.window)
            betas["ticker"] = ticker
            results.append(betas.reset_index())
        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def _rolling_ols(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        y = df["ret"].values
        X = df[FACTORS].values
        n = len(y)
        cols = {f: np.full(n, np.nan) for f in FACTORS}
        for i in range(window, n):
            yi = y[i - window:i]
            Xi = np.column_stack([np.ones(window), X[i - window:i]])
            try:
                b = np.linalg.lstsq(Xi, yi, rcond=None)[0]
                for j, f in enumerate(FACTORS):
                    cols[f][i] = b[j + 1]
            except np.linalg.LinAlgError:
                pass
        result = pd.DataFrame(cols, index=df.index)
        return result
