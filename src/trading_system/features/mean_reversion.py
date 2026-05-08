import pandas as pd
import numpy as np
from .base import FeatureSet
from .transforms import cross_section_winsorize, fill_cross_section_median

FEATURE_COLS = [
    "zscore_5d", "zscore_10d", "zscore_20d", "zscore_60d",
    "ret_zscore_5d", "ret_zscore_10d", "ret_zscore_20d",
    "bb_distance_20d", "rsi_14d", "dev_from_52w_mean",
    "distance_52w_high", "realized_vol_20d",
]

class MeanReversionFeatures(FeatureSet):
    def __init__(self, winsorize_pct: float = 0.01):
        self.winsorize_pct = winsorize_pct

    def compute_all(self, prices: pd.DataFrame) -> pd.DataFrame:
        out = []
        for ticker, grp in prices.groupby("ticker"):
            grp = grp.sort_values("date").copy()
            close = grp["close"].shift(1)  # lag 1 to avoid lookahead

            rows = grp[["ticker", "date"]].copy()

            for lb in [5, 10, 20, 60]:
                roll = close.rolling(lb)
                mu = roll.mean()
                sigma = roll.std().replace(0, np.nan)
                rows[f"zscore_{lb}d"] = (close - mu) / sigma

            for lb in [5, 10, 20]:
                ret = close.pct_change(lb)
                hist_std = ret.rolling(252).std().replace(0, np.nan)
                rows[f"ret_zscore_{lb}d"] = ret / hist_std

            roll20 = close.rolling(20)
            mu20 = roll20.mean()
            sigma20 = roll20.std()
            rows["bb_distance_20d"] = (close - mu20) / (2 * sigma20 + 1e-8)

            rows["rsi_14d"] = self._rsi(close, 14)

            mu252 = close.rolling(252).mean()
            rows["dev_from_52w_mean"] = (close - mu252) / (mu252 + 1e-8)

            high252 = close.rolling(252).max()
            rows["distance_52w_high"] = (close - high252) / (high252 + 1e-8)

            log_ret = np.log(close / close.shift(1))
            rows["realized_vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)

            out.append(rows)

        df = pd.concat(out, ignore_index=True)
        df = fill_cross_section_median(df, FEATURE_COLS)
        df = cross_section_winsorize(df, FEATURE_COLS, self.winsorize_pct, 1 - self.winsorize_pct)
        return df

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
