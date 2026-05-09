import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import pickle

class RidgeModel:
    def __init__(self, alpha_grid: list[float] | None = None):
        self.alpha_grid = alpha_grid or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self.model: RidgeCV | None = None
        self.scaler = StandardScaler()
        self.train_ic: float = 0.0
        self.val_ic: float = 0.0
        self.feature_names: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> None:
        self.feature_names = list(X_train.columns)
        X_tr = self.scaler.fit_transform(X_train.values)
        self.model = RidgeCV(alphas=self.alpha_grid, cv=5)
        self.model.fit(X_tr, y_train.values)
        self.train_ic = self._ic(self.model.predict(X_tr), y_train.values)
        if X_val is not None and y_val is not None:
            self.val_ic = self._ic(self.model.predict(self.scaler.transform(X_val.values)), y_val.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X.values))

    def rank_scores(self, X: pd.DataFrame) -> pd.Series:
        scores = self.predict(X)
        return pd.Series(scores, index=X.index).rank(pct=True)

    @staticmethod
    def _ic(pred: np.ndarray, actual: np.ndarray) -> float:
        if len(pred) < 2:
            return 0.0
        return float(pd.Series(pred).corr(pd.Series(actual), method="spearman"))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "feature_names": self.feature_names,
                         "train_ic": self.train_ic, "val_ic": self.val_ic}, f)

    @classmethod
    def load(cls, path: str) -> "RidgeModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_names = data["feature_names"]
        obj.train_ic = data["train_ic"]
        obj.val_ic = data["val_ic"]
        return obj
