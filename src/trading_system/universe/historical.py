import pandas as pd
from pathlib import Path

_DEFAULT_PATH = "data/raw/sp500_pit_constituents.csv"

class PointInTimeUniverse:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df["added_date"] = pd.to_datetime(self.df["added_date"])
        self.df["removed_date"] = pd.to_datetime(self.df.get("removed_date", pd.NaT))

    @classmethod
    def load_or_build(cls, path: str = _DEFAULT_PATH) -> "PointInTimeUniverse":
        p = Path(path)
        if p.exists():
            return cls(pd.read_csv(p))
        return cls._build_from_wikipedia(path)

    @classmethod
    def _build_from_wikipedia(cls, save_path: str) -> "PointInTimeUniverse":
        import io
        import requests
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (compatible; trading-system-bot/1.0)"},
            timeout=30,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), match="Symbol", flavor="lxml")
        current = tables[0][["Symbol", "Date added"]].rename(
            columns={"Symbol": "ticker", "Date added": "added_date"}
        )
        current["removed_date"] = pd.NaT
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        current.to_csv(save_path, index=False)
        return cls(current)

    def get_universe(self, date: pd.Timestamp) -> list[str]:
        mask = (self.df["added_date"] <= date) & (
            self.df["removed_date"].isna() | (self.df["removed_date"] > date)
        )
        return self.df[mask]["ticker"].tolist()

    def get_training_universe(self, train_start: pd.Timestamp,
                               train_end: pd.Timestamp) -> list[str]:
        mask = (self.df["added_date"] <= train_end) & (
            self.df["removed_date"].isna() | (self.df["removed_date"] > train_start)
        )
        return self.df[mask]["ticker"].tolist()

    def summary(self) -> str:
        total = len(self.df)
        active = self.df["removed_date"].isna().sum()
        return f"PIT Universe: {total} total tickers, {active} active, {total-active} historical"

    @property
    def all_tickers(self) -> list[str]:
        return self.df["ticker"].tolist()
