import numpy as np
import pandas as pd

class SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_prices(self, n_stocks: int = 20, start_date: str = "2018-01-01",
                        end_date: str = "2022-12-31") -> pd.DataFrame:
        dates = pd.bdate_range(start_date, end_date)
        records = []
        for i in range(n_stocks):
            ticker = f"SYN{i:02d}"
            log_returns = self.rng.normal(0.0002, 0.015, len(dates))
            prices = 50.0 * np.exp(np.cumsum(log_returns))
            vix = np.abs(self.rng.normal(18, 5, len(dates))).clip(10, 80)
            for j, d in enumerate(dates):
                hi = prices[j] * (1 + abs(self.rng.normal(0, 0.005)))
                lo = prices[j] * (1 - abs(self.rng.normal(0, 0.005)))
                records.append({
                    "ticker": ticker, "date": d,
                    "open": prices[j], "high": hi, "low": lo,
                    "close": prices[j], "volume": int(self.rng.integers(500_000, 5_000_000)),
                    "vix_close": vix[j],
                })
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    def generate_spy(self, start_date: str = "2015-01-01", end_date: str = "2021-12-31") -> pd.Series:
        dates = pd.bdate_range(start_date, end_date)
        returns = self.rng.normal(0.0003, 0.01, len(dates))
        prices = 300.0 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates, name="SPY")

    def generate_vix(self, start_date: str = "2015-01-01", end_date: str = "2021-12-31") -> pd.Series:
        dates = pd.bdate_range(start_date, end_date)
        vix = np.abs(self.rng.normal(18, 5, len(dates))).clip(10, 80)
        return pd.Series(vix, index=dates, name="VIX")
