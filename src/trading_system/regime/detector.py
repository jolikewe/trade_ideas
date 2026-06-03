import pandas as pd
import numpy as np
from pathlib import Path
from ..utils.config import load_config

class RegimeDetector:
    def __init__(self, config_path: str = "config/regime/detector.yaml"):
        cfg = load_config(config_path)
        self.vix_threshold = cfg["vix_gate"]["threshold"]
        self.spy_ma_window = cfg["spy_gate"]["ma_window"]
        mom_cfg = cfg["momentum_gate"]
        self.mom_enabled = mom_cfg.get("enabled", True)
        self.mom_z_threshold = mom_cfg["z_threshold"]
        self.mom_lookback_long = mom_cfg["lookback_long"]
        self.mom_lookback_skip = mom_cfg["lookback_skip"]
        self.mom_smooth_window = mom_cfg["smooth_window"]
        self.mom_top_pct = mom_cfg["top_pct"]

    def detect_composite_regime(self, vix: pd.Series, spy: pd.Series,
                                  prices: pd.DataFrame | None = None) -> pd.DataFrame:
        idx = vix.index.intersection(spy.index)
        vix = vix.loc[idx]
        spy = spy.loc[idx]

        vix_ok = (vix < self.vix_threshold).astype(int)
        spy_ma = spy.rolling(self.spy_ma_window).mean()
        spy_ok = (spy > spy_ma).astype(int)

        if self.mom_enabled and prices is not None:
            mom_z = self._compute_momentum_z(prices)
            mom_z = mom_z.reindex(idx)
            mom_ok = (mom_z < self.mom_z_threshold).fillna(0).astype(int)
        else:
            mom_ok = pd.Series(1, index=idx)
            mom_z = pd.Series(np.nan, index=idx)

        tradeable = (vix_ok & spy_ok & mom_ok)

        return pd.DataFrame({
            "vix_ok": vix_ok, "spy_ok": spy_ok,
            "momentum_ok": mom_ok, "tradeable": tradeable,
            "momentum_z": mom_z,
        }, index=idx)

    def get_tradeable_dates(self, vix: pd.Series, spy: pd.Series,
                             prices: pd.DataFrame | None = None,
                             min_score: int = 1) -> pd.Series:
        regime = self.detect_composite_regime(vix, spy, prices)
        return (regime["tradeable"] >= min_score)

    def _compute_momentum_z(self, prices: pd.DataFrame) -> pd.Series:
        close = prices.pivot(index="date", columns="ticker", values="close")
        # 12-1 momentum: return from (lookback_long ago) to (lookback_skip ago)
        lagged_far = close.shift(self.mom_lookback_long)
        lagged_near = close.shift(self.mom_lookback_skip)
        mom_ret = lagged_near / lagged_far - 1

        spread_series = {}
        for d in mom_ret.index:
            row = mom_ret.loc[d].dropna()
            if len(row) < 10:
                continue
            top_thresh = row.quantile(1 - self.mom_top_pct)
            bot_thresh = row.quantile(self.mom_top_pct)
            spread_series[d] = row[row >= top_thresh].mean() - row[row <= bot_thresh].mean()

        if not spread_series:
            return pd.Series(dtype=float)
        spread = pd.Series(spread_series).sort_index()
        smoothed = spread.rolling(self.mom_smooth_window).mean()
        history_std = smoothed.rolling(self.mom_lookback_long).std()
        history_mean = smoothed.rolling(self.mom_lookback_long).mean()
        z = (smoothed - history_mean) / (history_std + 1e-8)
        return z
