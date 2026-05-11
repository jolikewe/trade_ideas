import pandas as pd
from .yfinance_loader import YFinanceLoader

class DataLoader:
    def __init__(self, raw_dir: str = "data/raw/yfinance"):
        self.yf = YFinanceLoader(cache_dir=raw_dir)

    def load_prices(self, tickers: list[str], start: str, end: str,
                    force_refresh: bool = False) -> pd.DataFrame:
        return self.yf.load_universe(tickers, start, end, force_refresh)
