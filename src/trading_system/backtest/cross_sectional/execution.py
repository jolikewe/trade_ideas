import pandas as pd

class IBCommissionSimulator:
    def __init__(self, per_share: float = 0.005, min_per_order: float = 1.0,
                 max_pct: float = 0.01, slippage_bps: float = 5):
        self.per_share = per_share
        self.min_per_order = min_per_order
        self.max_pct = max_pct
        self.slippage = slippage_bps / 10_000

    def cost(self, shares: float, price: float) -> float:
        trade_val = abs(shares * price)
        commission = max(abs(shares) * self.per_share, self.min_per_order)
        commission = min(commission, trade_val * self.max_pct)
        slippage_cost = trade_val * self.slippage
        return commission + slippage_cost
