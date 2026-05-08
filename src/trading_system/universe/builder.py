from .historical import PointInTimeUniverse
from .filters import PriceFilter, ADRFilter
import pandas as pd

class UniverseBuilder:
    def __init__(self, min_price: float = 5.0, min_avg_volume: int = 100_000,
                 exclude_adrs: bool = True):
        self.price_filter = PriceFilter(min_price, min_avg_volume)
        self.adr_filter = ADRFilter() if exclude_adrs else None

    def build(self, prices: pd.DataFrame, date: pd.Timestamp,
              pit: PointInTimeUniverse | None = None) -> list[str]:
        tickers = self.price_filter.apply(prices)
        if self.adr_filter:
            tickers = self.adr_filter.apply(tickers)
        if pit is not None:
            pit_set = set(pit.get_universe(date))
            tickers = [t for t in tickers if t in pit_set]
        return tickers
