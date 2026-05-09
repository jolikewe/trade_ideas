import pandas as pd
import numpy as np
from ..analytics import sharpe_ratio, max_drawdown, deflated_sharpe_ratio
from ...regime.detector import RegimeDetector
from ...features.mean_reversion import MeanReversionFeatures, FEATURE_COLS
from ...labels.builder import LabelBuilder
from ...portfolio.optimizer import FactorNeutralOptimizer
from .execution import IBCommissionSimulator

class BacktestEngine:
    def __init__(self, config_path: str = "config/regime/detector.yaml",
                 initial_capital: float = 100_000, rebalance_freq: int = 5,
                 max_positions: int = 6):
        self.regime_detector = RegimeDetector(config_path)
        self.capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.max_positions = max_positions
        self.optimizer = FactorNeutralOptimizer(target_n=max_positions)
        self.execution = IBCommissionSimulator()

    def run(self, prices: pd.DataFrame, signals: pd.DataFrame,
            start: str, end: str, use_regime: bool = True,
            vix_series: pd.Series | None = None) -> dict:
        dates = pd.bdate_range(start, end)
        prices_pivot = prices.pivot(index="date", columns="ticker", values="close")
        returns_pivot = prices_pivot.pct_change()

        spy_series = prices_pivot.mean(axis=1)

        if use_regime:
            if vix_series is None and "vix_close" in prices.columns:
                vix_series = (prices[["date", "vix_close"]]
                              .drop_duplicates("date")
                              .set_index("date")["vix_close"])
            if vix_series is not None:
                vix_series = vix_series.reindex(spy_series.index).ffill()
                regime = self.regime_detector.detect_composite_regime(vix_series, spy_series)
            else:
                regime = pd.DataFrame({"tradeable": 1}, index=spy_series.index)
        else:
            regime = pd.DataFrame({"tradeable": 1}, index=spy_series.index)

        portfolio_returns = []
        current_weights = pd.Series(dtype=float)
        last_rebalance = None

        for i, date in enumerate(dates):
            if date not in prices_pivot.index:
                continue
            day_ret = returns_pivot.loc[date]

            if len(current_weights) > 0:
                port_ret = (current_weights * day_ret.reindex(current_weights.index)).sum()
                portfolio_returns.append({"date": date, "return": port_ret})

            should_rebalance = (last_rebalance is None or
                                (date - last_rebalance).days >= self.rebalance_freq * 1.4)
            tradeable = regime["tradeable"].get(date, 0) == 1

            if should_rebalance and tradeable and date in signals.index:
                day_signals = signals.loc[date]
                if len(day_signals) >= self.max_positions:
                    result = self.optimizer.optimize(day_signals)
                    current_weights = result["weights"]
                    last_rebalance = date

        if not portfolio_returns:
            return {"sharpe": 0.0, "max_drawdown": 0.0, "dsr": 0.0, "returns": []}

        ret_series = pd.DataFrame(portfolio_returns).set_index("date")["return"]
        return {
            "sharpe": sharpe_ratio(ret_series),
            "max_drawdown": max_drawdown(ret_series),
            "dsr": deflated_sharpe_ratio(ret_series),
            "returns": ret_series.tolist(),
        }
