import pandas as pd
from ..analytics import sharpe_ratio, max_drawdown

class SPYTimingBacktest:
    """v1 legacy: simple SPY 200-day MA timing strategy."""

    def run(self, spy: pd.Series, ma_window: int = 200) -> dict:
        ma = spy.rolling(ma_window).mean()
        signal = (spy > ma).shift(1)
        spy_ret = spy.pct_change()
        strategy_ret = spy_ret * signal
        return {
            "sharpe": sharpe_ratio(strategy_ret.dropna()),
            "max_drawdown": max_drawdown(strategy_ret.dropna()),
        }
