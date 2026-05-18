import numpy as np
import pandas as pd
import lightgbm as lgb
import json
from pathlib import Path

FIXED_PARAMS = {
    "num_iterations": 500,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "objective": "regression",
}

class LightGBMModel:
    def __init__(self, max_depth: int = 4, num_leaves: int = 20,
                 learning_rate: float = 0.01, min_child_samples: int = 300):
        self.params = {**FIXED_PARAMS, "max_depth": max_depth,
                       "num_leaves": num_leaves, "learning_rate": learning_rate,
                       "min_child_samples": min_child_samples}
        self.booster: lgb.Booster | None = None
        self.train_ic: float = 0.0
        self.val_ic: float = 0.0
        self.feature_names: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> None:
        self.feature_names = list(X_train.columns)
        dtrain = lgb.Dataset(X_train.values, label=y_train.values,
                             feature_name=self.feature_names)
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val.values, label=y_val.values, reference=dtrain)
            self.booster = lgb.train(self.params, dtrain,
                                     valid_sets=[dval], valid_names=["val"])
            self.val_ic = self._ic(self.booster.predict(X_val.values), y_val.values)
        else:
            self.booster = lgb.train(self.params, dtrain)
        self.train_ic = self._ic(self.booster.predict(X_train.values), y_train.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X.values)

    def rank_scores(self, X: pd.DataFrame) -> pd.Series:
        scores = self.predict(X)
        return pd.Series(scores, index=X.index).rank(pct=True)

    @staticmethod
    def _ic(pred: np.ndarray, actual: np.ndarray) -> float:
        if len(pred) < 2:
            return 0.0
        return float(pd.Series(pred).corr(pd.Series(actual), method="spearman"))

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(p.with_suffix(".lgb")))
        meta = {"train_ic": self.train_ic, "val_ic": self.val_ic,
                "feature_names": self.feature_names, "params": self.params}
        with open(p.with_name("model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        p = Path(path)
        obj = cls()
        obj.booster = lgb.Booster(model_file=str(p.with_suffix(".lgb")))
        with open(p.with_name("model_meta.json")) as f:
            meta = json.load(f)
        obj.train_ic = meta["train_ic"]
        obj.val_ic = meta["val_ic"]
        obj.feature_names = meta["feature_names"]
        return obj
