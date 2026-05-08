#!/usr/bin/env python3
"""Position tracking, P&L, and rebalance state."""
import json
from datetime import date
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class PortfolioState:
    def __init__(self, positions_path: str = "data/production/positions.csv",
                 state_path: str = "data/production/rebalance_state.json",
                 equity_log: str = "data/production/equity_log.csv"):
        self.positions_path = Path(positions_path)
        self.state_path = Path(state_path)
        self.equity_log = Path(equity_log)

    def load_positions(self) -> pd.DataFrame:
        if not self.positions_path.exists():
            return pd.DataFrame(columns=["ticker", "shares", "entry_price", "entry_date"])
        return pd.read_csv(self.positions_path, parse_dates=["entry_date"])

    def load_state(self) -> dict:
        if not self.state_path.exists():
            return {"last_rebalance_date": None, "frequency_days": 5}
        return json.loads(self.state_path.read_text())

    def is_rebalance_day(self) -> bool:
        state = self.load_state()
        last = state.get("last_rebalance_date")
        freq = state.get("frequency_days", 5)
        if last is None:
            return True
        last_dt = pd.Timestamp(last)
        business_days = len(pd.bdate_range(last_dt, date.today())) - 1
        return business_days >= freq

    def mark_to_market(self, current_prices: dict[str, float]) -> pd.DataFrame:
        pos = self.load_positions()
        if pos.empty:
            return pos
        pos["current_price"] = pos["ticker"].map(current_prices)
        pos["unrealised_pnl"] = (pos["current_price"] - pos["entry_price"]) * pos["shares"]
        return pos

if __name__ == "__main__":
    state = PortfolioState()
    positions = state.load_positions()
    print(f"Positions: {len(positions)}")
    print(f"Is rebalance day: {state.is_rebalance_day()}")
