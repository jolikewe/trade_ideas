import json
from pathlib import Path

import pandas as pd

from ..features.mean_reversion import FEATURE_COLS
from .ridge import RidgeModel

class ModelTrainer:
    def __init__(self, model_dir: str = "data/models/mean_reversion",
                 purge_days: int = 5, embargo_pct: float = 0.10):
        self.model_dir = Path(model_dir)
        self.purge_days = purge_days
        self.embargo_pct = embargo_pct

    def train_window(self, features: pd.DataFrame, labels: pd.DataFrame,
                     train_start: str, train_end: str, val_start: str, val_end: str,
                     window_id: int, model_name: str = "model_mr_zscore_12feat") -> dict:
        features = features.sort_values(["date", "ticker"])
        labels = labels.sort_values(["date", "ticker"])
        data = features.merge(labels, on=["ticker", "date"], how="inner")

        train_end_dt = pd.Timestamp(train_end)
        val_start_dt = pd.Timestamp(val_start)
        purge_end = train_end_dt + pd.Timedelta(days=self.purge_days)

        train = data[(data["date"] >= train_start) & (data["date"] <= train_end_dt)]
        val_all = data[(data["date"] >= val_start_dt) & (data["date"] <= val_end)]
        embargo_end = val_start_dt + (val_all["date"].max() - val_start_dt) * self.embargo_pct
        val = val_all[val_all["date"] > embargo_end]

        feature_cols = [c for c in FEATURE_COLS if c in data.columns]
        X_train = train[feature_cols].fillna(0)
        y_train = train["label"]
        X_val = val[feature_cols].fillna(0)
        y_val = val["label"]

        results = {}

        ridge = RidgeModel()
        ridge.fit(X_train, y_train, X_val, y_val)
        out_dir = self.model_dir / f"{model_name}_ridge" / f"window_{window_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ridge.save(str(out_dir / "model.pkl"))
        self._save_metadata(out_dir, ridge.train_ic, ridge.val_ic, feature_cols,
                            train_start, train_end, val_start, val_end)
        results["ridge"] = {"train_ic": ridge.train_ic, "val_ic": ridge.val_ic}

        return results

    def train_production(self, features: pd.DataFrame, labels: pd.DataFrame,
                         train_start: str, train_end: str,
                         model_name: str = "model_mr_zscore_12feat") -> dict:
        features = features.sort_values(["date", "ticker"])
        labels = labels.sort_values(["date", "ticker"])
        data = features.merge(labels, on=["ticker", "date"], how="inner")
        train = data[(data["date"] >= train_start) & (data["date"] <= train_end)]

        feature_cols = [c for c in FEATURE_COLS if c in data.columns]
        X_train = train[feature_cols].fillna(0)
        y_train = train["label"]

        results = {}

        ridge = RidgeModel()
        ridge.fit(X_train, y_train)
        out_dir = self.model_dir / f"{model_name}_ridge" / "production"
        out_dir.mkdir(parents=True, exist_ok=True)
        ridge.save(str(out_dir / "model.pkl"))
        self._save_metadata(out_dir, ridge.train_ic, None, feature_cols,
                            train_start, train_end, None, None)
        results["ridge"] = {"train_ic": ridge.train_ic}

        return results

    def _save_metadata(self, out_dir: Path, train_ic: float, val_ic: float | None,
                       features: list[str], train_start: str | None, train_end: str | None,
                       val_start: str | None, val_end: str | None) -> None:
        meta = {"train_ic": train_ic, "val_ic": val_ic, "feature_list": features,
                "train_start": train_start, "train_end": train_end,
                "val_start": val_start, "val_end": val_end}
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
