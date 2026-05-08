import pandas as pd
import numpy as np

class BacktestPortfolio:
    def __init__(self, initial_capital: float = 100_000):
        self.capital = initial_capital
        self.positions: dict[str, float] = {}
        self.history: list[dict] = []

    def rebalance(self, target_weights: pd.Series, prices: pd.Series,
                   date: pd.Timestamp) -> None:
        self.positions = {}
        for ticker, w in target_weights.items():
            if ticker in prices.index:
                self.positions[ticker] = w

        self.history.append({
            "date": date,
            "n_positions": len(self.positions),
            "capital": self.capital,
        })

    def mark_to_market(self, prices: pd.Series) -> float:
        return sum(w * prices.get(t, 1.0) for t, w in self.positions.items())
