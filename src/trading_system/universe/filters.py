import re
import pandas as pd

_ADR_PATTERNS = re.compile(
    r"(\b(ADR|ADS|SPONSORED|SPONS)\b|\.A$|N\.V\.|S\.A\.|PLC$)", re.IGNORECASE
)

class PriceFilter:
    def __init__(self, min_price: float = 5.0, min_avg_volume: int = 100_000):
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume

    def apply(self, df: pd.DataFrame) -> list[str]:
        stats = df.groupby("ticker").agg(
            avg_close=("close", "mean"),
            avg_vol=("volume", "mean"),
        )
        mask = (stats["avg_close"] >= self.min_price) & (stats["avg_vol"] >= self.min_avg_volume)
        return stats[mask].index.tolist()

class ADRFilter:
    def is_adr(self, ticker: str, name: str = "") -> bool:
        return bool(_ADR_PATTERNS.search(name)) or ticker.endswith("Y")

    def apply(self, tickers: list[str], names: dict[str, str] | None = None) -> list[str]:
        names = names or {}
        return [t for t in tickers if not self.is_adr(t, names.get(t, ""))]
