# Trade Ideas: ML Mean Reversion on U.S. Equities

Low-frequency, regime-gated mean-reversion strategy on S&P 500 stocks, with walk-forward validation and a daily production workflow.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue) ![scikit-learn](https://img.shields.io/badge/model-ridge%20regression-orange) ![cvxpy](https://img.shields.io/badge/optimizer-cvxpy-green)

## Overview

The system scores every S&P 500 stock each day with a Ridge regression on 12 mean-reversion features (z-scores, Bollinger distance, RSI, 52-week deviation, volatility), trained to predict 5-day forward returns. Scores are ranked, and a long-only, factor-neutral cvxpy optimizer picks the portfolio. A composite regime gate (VIX, SPY trend, cross-sectional momentum) switches trading off in unfavourable markets. The same code path drives the walk-forward backtest and the daily production brief, which is a markdown trade list to execute manually in Interactive Brokers.

## Strategy

Values below come from the code defaults and `config/`.

| | |
|---|---|
| Universe | S&P 500 (~501 tickers via yfinance) |
| Signal | Mean reversion: z-score oversold → buy |
| Model | Ridge regression (12 features) |
| Positions | 20 stocks, 8% max weight (cvxpy) |
| Holding period | 5 days |
| Stop-loss | 10% trailing per position (tightened to 5% when regime closes) |
| Rebalance | Every 5 business days, regime-gated |
| Regime gate | VIX < 25 AND SPY > 200-day MA AND cross-sectional momentum z < 1.5 (`config/regime/detector.yaml`) |
| Regime close | Losers liquidated immediately, winners held with tightened stop |
| Validation | 11 walk-forward windows, 4yr train / 1yr val / 1yr test (2010–2025) |
| Account | $2,000 (Interactive Brokers) |

## Getting Started

**Requirements:** Python 3.12+, git. macOS or Linux.

```bash
python3 -m venv .env
source .env/bin/activate
pip install -e .            # or: conda env create -f environment.yaml

pytest tests/               # 15 tests on synthetic data, no network needed
trading-system --help
```

### First-time data setup

```bash
# Download S&P 500 price history into a per-ticker parquet cache
python -m trading_system.cli download --start 2010-01-01 --end <today>

# Train the 11 walk-forward models
python -m trading_system.cli train --verbose

# Train the production model (full history, no validation set)
python -m trading_system.cli production-train

# Create production state files
mkdir -p data/production
echo 'ticker,shares,entry_price,entry_date' > data/production/positions.csv
echo '{"last_rebalance_date": null, "frequency_days": 5}' > data/production/rebalance_state.json

# First daily brief
python production/daily_run.py --refresh
```

### Daily workflow

Run after the NYSE close (about 9am GMT+8) and act before the next open.

```bash
python production/daily_run.py --refresh         # incremental price update + brief
python production/daily_run.py --confirm-trade   # after executing trades in IB TWS
```

The brief is written to `data/production/daily_brief.md` and appended to `data/production/brief_log.csv`.

### Monthly retrain

```bash
python production/retrain.py            # skips if models are < 30 days old
python production/retrain.py --force    # retrain unconditionally
```

The brief warns when models are more than 35 days old.

### CLI reference

```bash
python -m trading_system.cli download --start 2010-01-01 --end <date> [--force]
python -m trading_system.cli train [--verbose]
python -m trading_system.cli backtest --window <1-11> [--no-regime]
python -m trading_system.cli production-train
```

## Repository Structure

```
config/
  backtest.yaml                 # Walk-forward windows (2010-2025)
  regime/detector.yaml          # VIX / SPY / momentum gate thresholds
  production.yaml               # Account size, rebalance frequency, paths
  portfolio.yaml, execution.yaml, universe.yaml
  features/mean_reversion/model_mr_zscore_12feat.yaml
  models/                       # ridge / lightgbm / ensemble settings
src/trading_system/
  cli.py                        # Entry point (download, train, backtest, production-train)
  data/                         # yfinance loader (incremental cache), DataLoader, synthetic data
  features/                     # 12 mean-reversion features, transforms
  labels/builder.py             # 5-day forward return labels
  models/                       # Ridge (active), LightGBM, trainer
  portfolio/optimizer.py        # Factor-neutral optimizer (cvxpy/ECOS)
  regime/detector.py            # Composite regime gate
  backtest/
    cross_sectional/            # Backtest engine, signal generator, IB commission model
    directional/spy_timing.py   # SPY timing backtest
    analytics.py
  universe/historical.py        # Point-in-time S&P 500 universe
  utils/                        # Cache and config helpers
production/
  daily_run.py                  # Daily brief generator
  portfolio_state.py            # Position tracking and P&L
  retrain.py                    # Monthly production retrain
scripts/
  fama_macbeth_analysis.py
  show_results_table.py
tests/
  unit/                         # Features, models, portfolio, regime
  integration/                  # Full pipeline test
data/                           # gitignored, created at runtime
  raw/yfinance/                 # Per-ticker parquet cache
  models/mean_reversion/        # Model artefacts (window_1 ... window_11, production)
  results/backtests/            # Backtest result JSON
  production/                   # Brief, brief log, trade list, positions, rebalance state
```

Only `backtest.yaml`, `regime/detector.yaml` and the account size in `production.yaml` are read at runtime. Position count (20), max weight (8%) and stop-loss levels are defaults in `portfolio/optimizer.py` and `backtest/cross_sectional/engine.py`; `portfolio.yaml` lists different values (6 positions, 25%) that the code does not currently use.

## How It Works

- **Features (12):** `zscore_5/10/20/60d`, `ret_zscore_5/10/20d`, `bb_distance_20d`, `rsi_14d`, `dev_from_52w_mean`, `distance_52w_high`, `realized_vol_20d`. All use `shift(1)` to avoid lookahead.
- **Labels:** 5-day forward return.
- **Walk-forward:** 4yr train / 1yr val / 1yr test, 5-day purge gap, 10% embargo; 11 windows covering 2010-2025. Universe membership is point-in-time.
- **Model:** Ridge regression; scores are ranked to percentiles for signal generation.
- **Portfolio:** long-only cvxpy/ECOS optimizer, 20 target positions, 8% max weight, 10% trailing stop (5% once the regime closes).
- **Regime gate (all must pass):** VIX < 25, SPY above its 200-day MA, momentum z < 1.5.

## Notes and Limitations

- `BRK.B` and `BF.B` always fail to download from yfinance and are skipped.
- The universe includes S&P 500 removals from Wikipedia's changes table (about 2000 to present). Pre-2000 delistings are not captured, so some survivorship bias remains for that period.
- Data and trained models live under `data/` and are not committed.

## Development

```bash
pytest tests/ -v
ruff check src/ tests/
```
