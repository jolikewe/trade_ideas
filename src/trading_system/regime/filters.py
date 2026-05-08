import pandas as pd

class RegimeGate:
    def __init__(self, regime_df: pd.DataFrame):
        self.regime = regime_df

    def is_tradeable(self, date: pd.Timestamp) -> bool:
        if date not in self.regime.index:
            return False
        return bool(self.regime.loc[date, "tradeable"])

    def filter_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        tradeable = self.regime["tradeable"].reindex(dates).fillna(0)
        return dates[tradeable.values == 1]
