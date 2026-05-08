import pandas as pd
from .optimizer import FactorNeutralOptimizer

class PortfolioConstructor:
    def __init__(self, target_n: int = 6, max_weight: float = 0.25):
        self.target_n = target_n
        self.max_weight = max_weight

    def construct(self, signals: pd.Series) -> pd.Series:
        top = signals.nlargest(self.target_n)
        weights = top / top.sum()
        return weights

class FactorNeutralConstructor(PortfolioConstructor):
    def __init__(self, target_n: int = 6, max_weight: float = 0.25):
        super().__init__(target_n, max_weight)
        self.optimizer = FactorNeutralOptimizer(target_n, max_weight)

    def construct(self, signals: pd.Series, betas: pd.DataFrame | None = None) -> pd.Series:
        result = self.optimizer.optimize(signals, betas)
        return result["weights"]
