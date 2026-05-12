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

        # Table 0: current constituents
        current = tables[0][["Symbol", "Date added"]].rename(
            columns={"Symbol": "ticker", "Date added": "added_date"}
        )
        current["removed_date"] = pd.NaT

        # Table 1: historical changes — pull removed tickers to reduce survivorship bias
        historical_rows = []
        if len(tables) > 1:
            try:
                changes = tables[1].copy()
                # Flatten MultiIndex columns (Wikipedia uses grouped headers)
                if isinstance(changes.columns, pd.MultiIndex):
                    changes.columns = [" ".join(c).strip() for c in changes.columns]
                cols_lower = {c: c.lower().replace(" ", "_") for c in changes.columns}
                changes = changes.rename(columns=cols_lower)

                date_col     = next((c for c in changes.columns if c.startswith("date")), None)
                rem_tkr_col  = next((c for c in changes.columns if "removed" in c and "ticker" in c), None)
                add_tkr_col  = next((c for c in changes.columns if "added" in c and "ticker" in c), None)

                if date_col and rem_tkr_col:
                    changes[date_col] = pd.to_datetime(changes[date_col], errors="coerce")
                    removals = (changes[[date_col, rem_tkr_col]]
                                .dropna(subset=[rem_tkr_col])
                                .rename(columns={date_col: "removed_date", rem_tkr_col: "ticker"}))
                    removals = removals[removals["ticker"].str.strip() != ""]

                    # Best-effort add dates from the additions column
                    add_dates = {}
                    if add_tkr_col:
                        additions = changes[[date_col, add_tkr_col]].dropna(subset=[add_tkr_col])
                        additions = additions[additions[add_tkr_col].str.strip() != ""]
                        add_dates = (additions.groupby(add_tkr_col)[date_col].min()
                                     .to_dict())

                    current_tickers = set(current["ticker"])
                    for _, row in removals.iterrows():
                        tkr = row["ticker"].strip()
                        if tkr in current_tickers:
                            continue  # still active; already in current table
                        historical_rows.append({
                            "ticker": tkr,
                            "added_date": add_dates.get(tkr, pd.Timestamp("2000-01-01")),
                            "removed_date": row["removed_date"],
                        })
            except Exception:
                pass  # fall back to current-only if parsing fails

        all_df = pd.concat(
            [current, pd.DataFrame(historical_rows)], ignore_index=True
        ).drop_duplicates("ticker", keep="first")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(save_path, index=False)
        return cls(all_df)

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
