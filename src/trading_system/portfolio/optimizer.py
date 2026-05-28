import numpy as np
import pandas as pd
import cvxpy as cp

class FactorNeutralOptimizer:
    def __init__(self, target_n: int = 20, max_weight: float = 0.08,
                 mktrf_threshold: float = 0.15, factor_threshold: float = 0.10):
        self.target_n = target_n
        self.max_weight = max_weight
        self.mktrf_threshold = mktrf_threshold
        self.factor_threshold = factor_threshold

    def optimize(self, signals: pd.Series,
                 betas: pd.DataFrame | None = None) -> dict:
        top = signals.nlargest(max(self.target_n * 2, 20)).index
        sig = signals.loc[top].values
        n = len(sig)

        w = cp.Variable(n, nonneg=True)
        objective = cp.Maximize(sig @ w)
        constraints = [cp.sum(w) == 1, w <= self.max_weight]

        if betas is not None:
            b = betas.loc[betas.index.isin(top)].reindex(top).fillna(0)
            for factor in b.columns:
                beta_vec = b[factor].values
                thresh = self.mktrf_threshold if factor == "MktRF" else self.factor_threshold
                constraints.append(cp.abs(beta_vec @ w) <= thresh)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            prob.solve(verbose=False)

        if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            equal = np.ones(self.target_n) / self.target_n
            chosen = signals.nlargest(self.target_n).index
            return {"weights": pd.Series(equal, index=chosen), "status": "fallback"}

        weights = pd.Series(w.value, index=top)
        weights = weights[weights > 1e-4]
        weights = weights / weights.sum()
        return {"weights": weights, "status": prob.status}
