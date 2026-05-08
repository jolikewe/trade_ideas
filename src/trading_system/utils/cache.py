import pandas as pd
from pathlib import Path

class DataCache:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, name: str, **kwargs) -> Path:
        parts = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        fname = f"{name}_{parts}.parquet" if parts else f"{name}.parquet"
        return self.cache_dir / fname

    def exists(self, name: str, **kwargs) -> bool:
        return self._key(name, **kwargs).exists()

    def load(self, name: str, **kwargs) -> pd.DataFrame:
        return pd.read_parquet(self._key(name, **kwargs))

    def save(self, df: pd.DataFrame, name: str, **kwargs) -> None:
        df.to_parquet(self._key(name, **kwargs), index=True)
