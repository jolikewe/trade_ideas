# Trading System v2

Low-frequency ML-driven mean reversion on U.S. equities.

## Strategy

| | |
|---|---|
| Universe | S&P 500 (~501 tickers via yfinance) |
| Signal | Mean reversion: z-score oversold → buy |
| Model | Ridge regression (12 features) |
| Positions | 20 stocks, 8% max weight (cvxpy) |
| Holding period | 5 days |
| Stop-loss | 10% trailing per position (tightened to 5% when regime closes) |
| Rebalance | Every 5 business days, regime-gated |
| Regime gate | VIX < 25 AND SPY > 200-day MA AND momentum z < 1.5 |
| Regime close | Losers liquidated immediately, winners held with tightened stop |
| Validation | 11 walk-forward windows, 4yr train / 1yr val / 1yr test (2010–2025) |
| Account | $2,000 (Interactive Brokers) |

## Quick Start

```bash
# 1. Create venv and install
python3 -m venv .env
source .env/bin/activate
pip install -e .

# 2. Run tests (no data needed — uses synthetic data)
pytest tests/

# 3. Download price data
python -m trading_system.cli download --start 2010-01-01 --end <today>

# 4. Train models (11 walk-forward windows)
python -m trading_system.cli train --verbose

# 5. Train production model (full history, no val set)
python -m trading_system.cli production-train

# 6. Daily brief (downloads latest prices + generates signals)
python production/daily_run.py --refresh
```

## Daily Workflow (9am GMT+8)

```bash
source .env/bin/activate
python production/daily_run.py --refresh   # incremental download + brief

# After executing trades in IB TWS:
python production/daily_run.py --confirm-trade
```

## Monthly Retrain

```bash
python production/retrain.py              # skips if models are < 30 days old
python production/retrain.py --force      # retrain unconditionally
```

The brief will warn when models are > 35 days old.

Brief is saved to `data/production/daily_brief.md`. Historical log at `data/production/brief_log.csv`.

## CLI Reference

```bash
python -m trading_system.cli download --start 2010-01-01 --end <date> [--force]
python -m trading_system.cli train [--verbose]
python -m trading_system.cli backtest --window <1-11> [--no-regime]
python -m trading_system.cli production-train
```

## Repository Structure

```
config/
  backtest.yaml                 # Walk-forward windows (11 windows, 2010–2025)
  regime/detector.yaml          # VIX/SPY/momentum thresholds
  features/mean_reversion/
    model_mr_zscore_12feat.yaml # Active feature config
  portfolio.yaml, execution.yaml, production.yaml, universe.yaml
src/trading_system/
  cli.py                        # Entry point
  data/
    yfinance_loader.py          # Incremental per-ticker parquet cache
    loaders.py                  # DataLoader facade
    synthetic.py                # Synthetic data for tests
  features/mean_reversion.py    # 12 MR features
  labels/builder.py             # 5-day forward return labels
  models/
    ridge.py, trainer.py
  portfolio/optimizer.py        # Factor-neutral optimizer (cvxpy/ECOS)
  regime/detector.py            # Composite regime gate
  backtest/
    cross_sectional/engine.py   # Backtest engine
    cross_sectional/signal_generator.py
    analytics.py
  universe/historical.py        # Point-in-time S&P 500 universe
  utils/cache.py, config.py
production/
  daily_run.py                  # Daily brief generator
  portfolio_state.py            # Position tracking and P&L
  retrain.py                    # Monthly production retrain
scripts/
  fama_macbeth_analysis.py
  show_results_table.py
data/                           # gitignored
  raw/yfinance/                 # Per-ticker parquet cache
  models/mean_reversion/        # Trained model artefacts
  results/backtests/            # Backtest results JSON
  production/                   # Brief, positions, log
tests/
  unit/                         # Features, models, portfolio, regime
  integration/                  # Full pipeline test
```

## Development

```bash
pytest tests/ -v
ruff check src/ tests/
```

See **CLAUDE.md** for project rules, CLI reference, and model artefact paths.
See **setup.md** for full installation and configuration details.
