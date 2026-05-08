import pandas as pd
import pandas_datareader.data as web
from pathlib import Path
from datetime import datetime, timedelta

FACTORS = ["MktRF", "SMB", "HML", "RMW", "CMA", "MOM"]

class KenFrenchLoader:
    def __init__(self, cache_dir: str = "data/raw/ken_french", cache_ttl_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = cache_ttl_days

    def load(self, start: str = "2010-01-01", end: str = "2026-01-01") -> pd.DataFrame:
        path = self.cache_dir / "ff5mom_daily.parquet"
        if path.exists() and not self._is_stale(path):
            return pd.read_parquet(path)
        return self._download(start, end, path)

    def _is_stale(self, path: Path) -> bool:
        df = pd.read_parquet(path)
        latest = pd.to_datetime(df.index).max()
        return (datetime.now() - latest.to_pydatetime()).days > self.ttl

    def _download(self, start: str, end: str, path: Path) -> pd.DataFrame:
        try:
            ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench",
                                  start=start, end=end)[0] / 100
            mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench",
                                  start=start, end=end)[0] / 100
            mom.columns = ["MOM"]
            df = ff5.join(mom, how="inner")
            df.index = pd.to_datetime(df.index)
            df.to_parquet(path)
            return df
        except Exception as e:
            if path.exists():
                return pd.read_parquet(path)
            raise RuntimeError(f"Cannot load Ken French data: {e}")
