import numpy as np
import pandas as pd
from scipy.stats import norm

def sharpe_ratio(returns: pd.Series, annualize: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annualize))

def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())

def deflated_sharpe_ratio(returns: pd.Series, n_trials: int = 50) -> float:
    T = len(returns)
    if T < 10:
        return 0.0
    sr_hat = sharpe_ratio(returns)
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())
    euler_gamma = 0.5772156649
    sr_star = np.sqrt(2 * np.log(n_trials)) * (
        (1 - euler_gamma) / np.sqrt(np.log(np.log(n_trials)))
    ) if n_trials > 1 else 0.0
    denom = 1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2
    if denom <= 0:
        return 0.0
    z = (sr_hat - sr_star) * np.sqrt(T - 1) / np.sqrt(denom)
    return float(norm.cdf(z))
