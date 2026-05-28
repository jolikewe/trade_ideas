import numpy as np
import pandas as pd

from ...portfolio.optimizer import FactorNeutralOptimizer
from ...regime.detector import RegimeDetector
from ..analytics import deflated_sharpe_ratio, max_drawdown, sharpe_ratio
from .execution import IBCommissionSimulator

class BacktestEngine:
    def __init__(self, config_path: str = "config/regime/detector.yaml",
                 initial_capital: float = 100_000, rebalance_freq: int = 5,
                 max_positions: int = 20, stop_loss_pct: float = 0.10,
                 tight_stop_pct: float = 0.05):
        self.regime_detector = RegimeDetector(config_path)
        self.capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.tight_stop_pct = tight_stop_pct
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
                regime = self.regime_detector.detect_composite_regime(vix_series, spy_series, prices=prices)
            else:
                regime = pd.DataFrame({"tradeable": 1}, index=spy_series.index)
        else:
            regime = pd.DataFrame({"tradeable": 1}, index=spy_series.index)

        portfolio_returns = []
        current_weights = pd.Series(dtype=float)
        last_rebalance = None
        position_highs: dict[str, float] = {}
        entry_prices: dict[str, float] = {}
        regime_open = True

        for i, date in enumerate(dates):
            if date not in prices_pivot.index:
                continue
            day_ret = returns_pivot.loc[date]
            day_prices = prices_pivot.loc[date]

            if len(current_weights) > 0:
                port_ret = (current_weights * day_ret.reindex(current_weights.index)).sum()
                portfolio_returns.append({"date": date, "return": port_ret})

            tradeable = regime["tradeable"].get(date, 0) == 1

            # Update HWMs for held positions
            for ticker in list(position_highs.keys()):
                price = day_prices.get(ticker)
                if price is not None and not np.isnan(price):
                    position_highs[ticker] = max(position_highs[ticker], price)

            # Per-position trailing stop-loss
            if len(current_weights) > 0:
                stop_pct = self.tight_stop_pct if not regime_open else self.stop_loss_pct
                stopped_out = []
                for ticker in current_weights.index:
                    hwm = position_highs.get(ticker)
                    price = day_prices.get(ticker)
                    if hwm is None or price is None or np.isnan(price):
                        continue
                    if price <= hwm * (1 - stop_pct):
                        stopped_out.append(ticker)
                if stopped_out:
                    current_weights = current_weights.drop(stopped_out)
                    for t in stopped_out:
                        position_highs.pop(t, None)
                        entry_prices.pop(t, None)
                    if len(current_weights) > 0:
                        current_weights = current_weights / current_weights.sum()

            # Regime gate: Option A — liquidate losers, tighten stops on winners
            if not tradeable and regime_open and len(current_weights) > 0:
                losers = []
                for ticker in current_weights.index:
                    entry = entry_prices.get(ticker)
                    price = day_prices.get(ticker)
                    if entry is None or price is None or np.isnan(price):
                        losers.append(ticker)
                        continue
                    if price <= entry:
                        losers.append(ticker)
                if losers:
                    current_weights = current_weights.drop(losers)
                    for t in losers:
                        position_highs.pop(t, None)
                        entry_prices.pop(t, None)
                    if len(current_weights) > 0:
                        current_weights = current_weights / current_weights.sum()

            regime_open = tradeable

            should_rebalance = (last_rebalance is None or
                                (date - last_rebalance).days >= self.rebalance_freq * 1.4)

            if should_rebalance and tradeable and date in signals.index:
                day_signals = signals.loc[date]
                if len(day_signals) >= self.max_positions:
                    result = self.optimizer.optimize(day_signals)
                    new_weights = result["weights"]
                    # Update entry prices and HWMs for new/changed positions
                    for ticker in new_weights.index:
                        price = day_prices.get(ticker)
                        if price is not None and not np.isnan(price):
                            if ticker not in current_weights.index:
                                entry_prices[ticker] = price
                                position_highs[ticker] = price
                    # Remove tracking for exited positions
                    for ticker in set(current_weights.index) - set(new_weights.index):
                        position_highs.pop(ticker, None)
                        entry_prices.pop(ticker, None)
                    current_weights = new_weights
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
