# Setup & Reference — Trading System v2

## Installation

**Requirements:** Python 3.12, git. macOS or Linux.

```bash
# Clone and set up venv
git clone https://github.com/jolikewe/mean_reversion.git trade
cd trade
python3 -m venv .env
source .env/bin/activate
pip install -e .

# Verify
pytest tests/          # 15 tests, uses synthetic data, no internet needed
trading-system --help
```

## First-Time Data Setup

```bash
# Download S&P 500 price history (501 tickers, ~95MB, ~3 min)
python -m trading_system.cli download --start 2010-01-01 --end <today>

# Train walk-forward models (11 windows)
python -m trading_system.cli train --verbose

# Train production model (full history)
python -m trading_system.cli production-train

# Set up production state files
mkdir -p data/production
echo 'ticker,shares,entry_price,entry_date' > data/production/positions.csv
echo '{"last_rebalance_date": null, "frequency_days": 5}' > data/production/rebalance_state.json

# First daily brief
python production/daily_run.py --refresh
```

## Daily Use

```bash
source .env/bin/activate
python production/daily_run.py --refresh     # incremental price update + brief
python production/daily_run.py --confirm-trade  # after executing trades
```

Run at ~9am GMT+8 (after NYSE close at 4am GMT+8). Act before NYSE open at 9:30pm GMT+8.

## Monthly Retrain

```bash
python production/retrain.py        # skips if models trained < 30 days ago
python production/retrain.py --force
```

The brief warns when models are > 35 days old. Retrain refreshes the production model on all history through today.

## Configuration

| File | Purpose |
|------|---------|
| `config/backtest.yaml` | Walk-forward windows, train/val/test years |
| `config/regime/detector.yaml` | VIX threshold, SPY MA window, momentum z threshold |
| `config/portfolio.yaml` | Optimizer settings (n positions, max weight, factor thresholds) |
| `config/execution.yaml` | Commission model |
| `config/production.yaml` | Account size, rebalance frequency |
| `config/universe.yaml` | Universe filters |
| `config/features/mean_reversion/model_mr_zscore_12feat.yaml` | Active feature set |

## Data Layout

```
data/
  raw/yfinance/         Per-ticker parquet files (incremental, ~190KB each)
  models/mean_reversion/
    model_mr_zscore_12feat_ridge/
      window_1/ … window_11/   model.pkl, metadata.json
      production/              model.pkl, metadata.json
    model_mr_zscore_12feat_lightgbm/
      window_1/ … window_11/   model.lgb, model_meta.json, metadata.json
      production/              model.lgb, model_meta.json, metadata.json
  results/backtests/    window_N_<timestamp>.json
  production/
    daily_brief.md      Latest brief (overwritten daily)
    brief_log.csv       Historical log (appended daily)
    trade_list.md       Today's trades (editable before executing in IB TWS)
    positions.csv       Current holdings — edit after executing trades
    rebalance_state.json
```

## Key Algorithms

**Features (12):** `zscore_5/10/20/60d`, `ret_zscore_5/10/20d`, `bb_distance_20d`, `rsi_14d`, `dev_from_52w_mean`, `distance_52w_high`, `realized_vol_20d`. All use `shift(1)` — no lookahead.

**Labels:** 5-day forward return.

**Walk-forward:** 4yr train / 1yr val / 1yr test, 5-day purge gap, 10% embargo. 11 windows covering 2010–2025.

**Ensemble:** Equal-weight rank average of Ridge and LightGBM percentile scores. (Production models have no val set so IC-weighted ensemble is not applicable.)

**Portfolio:** cvxpy/ECOS factor-neutral optimizer. Long-only, max 25% per position, 6 positions target.

**Regime gate (AND logic — all must pass):**
- VIX < 25
- SPY > 200-day moving average
- Cross-sectional momentum z-score < 0.5

## Known Issues

- `BRK.B` and `BF.B` always fail to download from yfinance — skipped automatically, no output shown.
- Universe includes historical S&P 500 removals from Wikipedia's changes table (~2000–present). Pre-2000 delistings are not captured — residual survivorship bias exists for that period.
