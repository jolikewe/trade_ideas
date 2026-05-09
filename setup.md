# Trading System v2 — Complete Setup & Reference Guide

**Purpose:** Master setup document for deploying this trading system on a new machine or handing off to a new Claude instance. Also serves as compact codebase reference covering directory layout, module interfaces, key algorithms, and infrastructure.

**Last updated:** 2026-04-23

---

## Table of Contents

1. [Required Files for Setup/Export](#1-required-files-for-setupexport)
2. [Quick Start (TL;DR)](#2-quick-start-tldr)
3. [Prerequisites & Installation](#3-prerequisites--installation)
4. [Key Development Practices](#4-key-development-practices)
5. [Project Overview](#5-project-overview)
6. [Trading System Design Principles](#6-trading-system-design-principles)
7. [Verification & Testing](#7-verification--testing)
8. [How to Export This System](#8-how-to-export-this-system)
9. [Production Trading Workflow](#9-production-trading-workflow)
10. [Directory Structure](#10-directory-structure)
11. [Source Code Module Map](#11-source-code-module-map)
12. [Key Algorithms](#12-key-algorithms)
13. [Active Feature Set](#13-active-feature-set)
14. [Key Config Values](#14-key-config-values)
15. [Data Schemas](#15-data-schemas)
16. [Infrastructure Config](#16-infrastructure-config)

---

## 1. Required Files for Setup/Export

### 1.1 Essential Documentation (Read in This Order)

1. **SETUP.md** (this file) — start here, complete machine setup + codebase reference
2. **README.md** — project overview and quick reference
3. **docs/design/trading_plan.md** — full trading system specification

### 1.2 Required Repository Files

Configuration files (version controlled):

```
environment.yaml              # Conda environment definition
pyproject.toml                # Package metadata and dependencies
.gitignore
config/
  universe.yaml               # Stock universe filters (+ ADR heuristic)
  portfolio.yaml              # Portfolio construction (+ factor_neutral section)
  execution.yaml              # Execution and cost settings
  backtest.yaml               # Backtest validation windows (11 windows, 2010-2025)
  production.yaml             # Live config (account size, model window, rebal frequency)
  features/mean_reversion/
    model_mr_zscore_12feat.yaml   # ACTIVE: 12-feature production model
    model_mr_zscore.yaml          # Baseline: 15 MR features
    model_mr_crosssec.yaml        # Deferred
    model_mr_pairs.yaml           # Deferred
  models/
    ridge.yaml                # Ridge alpha grid
    lightgbm.yaml             # LGB hyperparams + regularization
    ensemble.yaml             # IC weighting config
  factors/
    ken_french.yaml           # FF5+MOM factor constraints
    in_house.yaml             # Momentum + quality factors
  regime/
    detector.yaml             # VIX thresholds + SPY MA config
```

Source code (version controlled):

```
src/trading_system/           # All Python source code
tests/                        # All test files
scripts/                      # Download and analysis scripts
production/
  daily_run.py                # Daily brief generator — auto-runs at Claude session start
  portfolio_state.py          # Position tracking, P&L, rebalance state
  retrain.py                  # Monthly retrain — auto-detects latest data, full pipeline
```

Documentation (version controlled):

```
SETUP.md                      # This file
README.md                     # Project overview
docs/design/trading_plan.md   # Full specification
```

### 1.3 Machine-Specific Files (Create Per Machine, NOT in Git)

These files are gitignored and must be created on each machine:

```
config/data_sources.yaml      # Data source config (see Section 3.7)
.env/                         # Conda environment (created during setup)
data/                         # Data files (too large for git)
logs/                         # Log files
```

---

## 2. Quick Start (TL;DR)

```bash
# 1. Navigate to repo
cd "/path/to/trade v2"

# 2. Create conda environment (locally in repo)
conda env create -f environment.yaml -p ./.env

# 3. Activate environment
conda activate ./.env

# 4. Install package
pip install -e .

# 5. Create data directories
mkdir -p data/raw/{yfinance,ken_french,sec_edgar}
mkdir -p data/cache data/models/mean_reversion data/results/{training,backtests} logs

# 6. Create config/data_sources.yaml (adapt to your setup — see Section 3.7)

# 7. Download data (internet required)
python scripts/download_yfinance_data.py --start 2010-01-01 --end 2025-12-31

# 8. Run tests (uses synthetic data, no internet required)
pytest tests/

# 9. Verify
trading-system --help
ruff check

# 10. Production setup (after training + backtest are complete)
mkdir -p data/production
echo 'ticker,shares,entry_price,entry_date' > data/production/positions.csv
echo '{"last_rebalance_date": null, "frequency_days": 5}' > data/production/rebalance_state.json
python production/retrain.py      # train production model on all available data
python production/daily_run.py    # generate first daily brief
```

---

## 3. Prerequisites & Installation

### 3.1 System Requirements

**Operating System:**
- Linux (CentOS 7+, Ubuntu 20.04+, RHEL 8+) — PREFERRED
- macOS (10.15+) — SUPPORTED
- Windows with WSL2 — NOT TESTED

**Hardware:**
- CPU: 4+ cores recommended
- RAM: 16 GB minimum, 32 GB recommended for backtesting
- Disk: 100 GB free space minimum (raw price data is ~20–50 GB)

**Required Software (install BEFORE proceeding):**
- Git 2.0+ — verify: `git --version`
- Conda or Miniconda — download: https://docs.conda.io/en/latest/miniconda.html — verify: `conda --version`

**Optional but Recommended:**
- screen/tmux for long-running download processes
- SSH keys for Git

### 3.2 Data Requirements

v2 uses free public data sources — no proprietary data required.

**Automatic downloads (via scripts):**
- OHLCV price data: yfinance (free, ~20–50 GB for full S&P 500 history)
- SPY prices: yfinance (SPY)
- VIX: yfinance (^VIX)
- Fama-French factors: pandas-datareader (Ken French data library)
- Fundamentals: SEC EDGAR REST API (free, rate-limited)

**Universe list:** `data/raw/universe_tickers.csv` — created automatically by `scripts/download_yfinance_data.py` (fetches S&P 500 tickers from Wikipedia). Optional: replace with a custom list (CSV with `ticker` column).

**Storage planning:**

| Path | Size |
|------|------|
| `data/raw/yfinance/` | ~20–50 GB (full S&P 500, 15 years) |
| `data/raw/ken_french/` | ~5 MB |
| `data/raw/sec_edgar/` | ~10–30 GB (company facts JSON by CIK) |
| `data/cache/` | ~10–20 GB (processed Parquet files) |
| `data/models/` | ~1–5 GB (trained model artifacts) |
| `data/results/` | ~100 MB (training/backtest JSON) |

### 3.3 Step 1: Navigate to Repository

```bash
cd "/path/to/trade v2"
ls -la
# Must see: README.md, environment.yaml, pyproject.toml, src/, config/, tests/

# Verification
test -f environment.yaml && echo "OK environment.yaml" || echo "MISSING environment.yaml"
test -f pyproject.toml   && echo "OK pyproject.toml"   || echo "MISSING pyproject.toml"
test -d src              && echo "OK src/"              || echo "MISSING src/"
test -d config           && echo "OK config/"           || echo "MISSING config/"
```

### 3.4 Step 2: Create Conda Environment

**IMPORTANT:** The environment is created INSIDE the repo at `./.env`, not in the global conda directory.

```bash
pwd
# Should show: /path/to/trade v2

conda env create -f environment.yaml -p ./.env
# Takes 5–15 minutes

# Verification
test -d ./.env && echo "OK Conda environment created" || echo "FAIL Environment creation failed"
```

**What gets installed (from environment.yaml):**
- Python 3.12
- Core data science: pandas, numpy, scipy
- Machine learning: scikit-learn, lightgbm
- Portfolio optimization: cvxpy (with ECOS solver)
- Data sources: yfinance≥0.2.0, pandas-datareader≥0.10.0, requests≥2.28.0
- Data storage: pyarrow, h5py
- Statistical analysis: statsmodels
- Visualization: matplotlib, seaborn
- Testing: pytest, pytest-cov
- Code quality: ruff
- Utilities: pyyaml, tqdm, python-dateutil, pytz

### 3.5 Step 3: Activate Environment & Install Package

```bash
# Activate (do this every session)
conda activate ./.env
# Prompt should change to show (.env)

python --version     # Should be Python 3.12.x
which pip            # Should be /path/to/trade v2/.env/bin/pip

pip install -e .

# Verification
pip list | grep trading-system
# Should show: trading-system 0.2.0 /path/to/trade v2/src

which trading-system
# Should show: /path/to/trade v2/.env/bin/trading-system

trading-system --help
# Should show help text with: download, train, backtest, spy-timing, monitor, report

# Test imports
python -c "from trading_system.data.yfinance_loader import YFinanceLoader; print('OK yfinance_loader')"
python -c "from trading_system.regime.detector import RegimeDetector; print('OK RegimeDetector')"
python -c "from trading_system.portfolio.optimizer import FactorNeutralOptimizer; print('OK optimizer')"
python -c "import cvxpy; print('OK cvxpy', cvxpy.__version__)"
python -c "import yfinance; print('OK yfinance', yfinance.__version__)"
```

### 3.6 Step 4: Create Required Directories

```bash
mkdir -p data/raw/yfinance
mkdir -p data/raw/ken_french
mkdir -p data/raw/sec_edgar
mkdir -p data/cache
mkdir -p data/models/mean_reversion
mkdir -p data/results/training
mkdir -p data/results/backtests
mkdir -p data/production
mkdir -p logs

# Initialise production state files
echo 'ticker,shares,entry_price,entry_date' > data/production/positions.csv
echo '{"last_rebalance_date": null, "frequency_days": 5}' > data/production/rebalance_state.json

ls -la data/
```

### 3.7 Step 5: Configure Data Sources

Create `config/data_sources.yaml` (this file is gitignored — machine-specific):

```yaml
primary_source: yfinance

yfinance:
  cache_dir: data/raw/yfinance
  threads: 10
  bulk_download: true
  universe_source: csv           # data/raw/universe_tickers.csv (auto-created)
  auto_adjust: true              # handles splits + dividends

stooq:
  use_for_fallback: true
  tickers: [SPY, "^VIX"]        # fallback for SPY/VIX if yfinance fails

ken_french:
  datasets:
    - F-F_Research_Data_5_Factors_2x3_daily
    - F-F_Momentum_Factor_daily
  cache_dir: data/raw/ken_french
  cache_ttl_days: 7              # refresh weekly

sec_edgar:
  user_agent: "YourName youremail@example.com"  # REQUIRED: SEC mandates this header
  cache_dir: data/raw/sec_edgar
  rate_limit: 10                 # max requests/second
  cache_ttl_days: 90             # refresh quarterly
  use_filing_date: true          # point-in-time correctness
```

**Note:** Replace `user_agent` with your actual name/email — the SEC requires this header.

### 3.8 Step 6: Download Data

```bash
# Quick test: S&P 500 tickers, recent 5 years
python scripts/download_yfinance_data.py \
  --start 2019-01-01 --end 2025-12-31 \
  --tickers sp500

# Full download: S&P 500, 2010-2025
python scripts/download_yfinance_data.py \
  --start 2010-01-01 --end 2025-12-31 \
  --tickers sp500

# S&P 1500 (larger universe, more data)
python scripts/download_yfinance_data.py \
  --start 2010-01-01 --end 2025-12-31 \
  --tickers sp1500

# Download SEC fundamentals (optional; large dataset; long-running — hours)
python scripts/download_sec_fundamentals.py
```

Ken French factors are downloaded automatically on first use (small, fast). Cache is auto-refreshed if >7 days stale.

Point-in-time S&P 500 universe is built automatically during the download step from Wikipedia's S&P 500 changes history and saved to `data/raw/sp500_pit_constituents.csv`.

### 3.9 Step 7: Run Tests

All tests use synthetic data — no internet or downloaded data required:

```bash
pytest tests/ -v                                                    # all tests
pytest tests/unit/ -v                                               # unit tests only
pytest tests/integration/ -v                                        # integration tests (slower)
pytest --cov=src/trading_system tests/ -v                          # with coverage
pytest tests/unit/test_features/test_mean_reversion.py -v
pytest tests/unit/test_regime/test_detector.py -v
pytest tests/unit/test_portfolio/test_optimizer.py -v
```

**Expected passing tests:**
- `test_features/test_mean_reversion.py`: z-scores bounded, RSI [0,100], 15 features computed
- `test_regime/test_detector.py`: COVID 2020 = crisis, gate blocks ~20–60% of dates
- `test_portfolio/test_optimizer.py`: weights sum to 1, factor betas |β| < 0.20, speed < 5 s
- `test_pipeline_mean_reversion.py`: full pipeline end-to-end with synthetic data

### 3.10 Step 8: First Training Run

```bash
# Test mode (1 window, fast validation)
# Temporarily edit config/backtest.yaml:
#   data_start_date: "2019-01-01"
#   data_end_date:   "2024-12-31"
python -m trading_system.cli train --verbose

# Full mode (11 windows, 2010–2025)
python -m trading_system.cli train --verbose --parallel
```

---

## 4. Key Development Practices

### 4.1 Always Activate Environment
```bash
conda activate ./.env
```

### 4.2 Run Tests Before and After Changes
```bash
pytest tests/ -v
```

### 4.3 Lint and Format
```bash
ruff check src/ tests/
ruff format src/ tests/
```

### 4.4 Data Integrity
- **Never** use `period_end_date` for fundamentals — always use `filing_date` for point-in-time correctness
- **Never** use future data in features — all rolling calculations use `shift(1)` (lag 1) to avoid lookahead
- **Always** use `auto_adjust=True` with yfinance to handle splits and dividends

### 4.5 SEC EDGAR Rate Limiting
The SEC EDGAR API enforces 10 requests/second. `SECEdgarLoader` handles this automatically with exponential backoff. For bulk downloads, use `scripts/download_sec_fundamentals.py` which batches requests and caches locally.

### 4.6 yfinance Caching
yfinance downloads are cached as Parquet files in `data/raw/yfinance/`. To force a re-download, pass `force_refresh=True` to `YFinanceLoader` methods or use `--no-cache` flag in CLI.

---

## 5. Project Overview

### 5.1 Strategy

Mean reversion: when a stock's price drops significantly below its rolling mean (negative z-score), it tends to revert upward over a 5-day horizon. The system:

1. Computes 12 mean-reversion features (z-scores, RSI, Bollinger bands, 52w-high distance, realized vol)
2. Trains Ridge + LightGBM models to predict 5-day forward returns
3. Applies regime gate (VIX + SPY 200-day MA + momentum z-score) to avoid trading in crises
4. Constructs factor-neutral portfolio using cvxpy (neutralizes FF5+MOM exposure)

### 5.2 Walk-Forward Validation

11 windows, each with:
- Train: 4 years
- Validation: 1 year (with 10% embargo)
- Test: 1 year
- Purge: 5 days (= label horizon, prevents train/val leakage)
- Roll: 1 year forward per window

Windows cover 2015–2025 test years.

### 5.3 Regime Gate

Three independent AND-gates — ALL must pass to allow trading:

| Gate | Condition | Config key |
|------|-----------|------------|
| VIX | VIX < 25 | `vix_gate.threshold` |
| SPY trend | SPY close > 200-day MA | `spy_gate.ma_window` |
| Momentum | momentum_z < 0.5 | `momentum_gate.z_threshold` |

`momentum_z` = z-score of cross-sectional momentum spread (top/bottom 20% stocks, 12m-1m returns), 60-day smoothed, vs trailing 252-day history.

On gated dates, hold current positions without rebalancing. Config: `config/regime/detector.yaml`.

### 5.4 Factor Neutralization

The portfolio optimizer (cvxpy) maximizes signal-weighted returns subject to:
- `sum(w) = 1` (fully invested)
- `0 ≤ w_i ≤ max_weight` (no shorting, position cap)
- `|β_k · w| ≤ threshold_k` for each factor k

Factors: MktRF (≤0.15), SMB, HML, RMW, CMA, MOM (all ≤0.10). Solver: ECOS.

---

## 6. Trading System Design Principles

These principles are inherited from v1 and remain unchanged in v2:

1. **No lookahead bias:** All features computed with lag-1 data; labels use t+horizon close prices
2. **Purged CV:** Training data is purged of label overlap period
3. **Embargo:** Validation data removes first 10% to prevent bleed from training noise
4. **Point-in-time fundamentals:** SEC data uses `filing_date` not `period_end_date`
5. **Regime as gate, not feature:** VIX/SPY data decides WHETHER to trade, not used as a model input feature (avoids overfitting to regime-specific patterns)
6. **Survivorship bias mitigation:** `PointInTimeUniverse` (868 tickers, Wikipedia S&P 500 changes history) used at train and backtest time to filter to tickers actually in the index on each date. Residual bias acknowledged (pre-2010 removals may be incomplete)
7. **Cost-aware:** All signals evaluated after realistic transaction costs (IB ~0.5–0.9% round trip)

---

## 7. Verification & Testing

### 7.1 All Imports Work

```python
python -c "
from trading_system.data.loaders import DataLoader
from trading_system.data.yfinance_loader import YFinanceLoader
from trading_system.features.mean_reversion import MeanReversionFeatures
from trading_system.regime.detector import RegimeDetector
from trading_system.portfolio.optimizer import FactorNeutralOptimizer
from trading_system.data.synthetic import SyntheticDataGenerator
from trading_system.universe.historical import PointInTimeUniverse
print('All imports OK')
"
```

```python
# PIT universe (requires data download first)
python -c "
from trading_system.universe.historical import PointInTimeUniverse
pit = PointInTimeUniverse.load_or_build()
print(pit.summary())
"
# Expected: ~868 total tickers, ~503 active today, ~365 historical
```

### 7.2 Synthetic Pipeline Works

```python
python -c "
from trading_system.data.synthetic import SyntheticDataGenerator
from trading_system.features.mean_reversion import MeanReversionFeatures
from trading_system.labels.builder import LabelBuilder
gen = SyntheticDataGenerator(seed=42)
prices = gen.generate_prices(n_stocks=10, start_date='2020-01-01', end_date='2022-12-31')
feats = MeanReversionFeatures().compute_all(prices)
labels = LabelBuilder(horizon_days=5).build_labels(prices)
print(f'Prices: {len(prices)} rows')
print(f'Features: {len(feats.columns) - 2} feature columns')
print(f'Labels: {len(labels)} rows')
"
```

### 7.3 Regime Detector Works

```python
python -c "
from trading_system.regime.detector import RegimeDetector
from trading_system.data.synthetic import SyntheticDataGenerator
gen = SyntheticDataGenerator(seed=0)
spy = gen.generate_spy(start_date='2015-01-01', end_date='2021-12-31')
vix = SyntheticDataGenerator(seed=1).generate_vix(start_date='2015-01-01', end_date='2021-12-31')
detector = RegimeDetector(config_path='config/regime/detector.yaml')
regime = detector.detect_composite_regime(vix, spy)
tradeable_pct = detector.get_tradeable_dates(vix, spy, min_score=1).mean()
print(f'Regime rows: {len(regime)}')
print(f'Tradeable: {tradeable_pct:.1%}')
"
```

### 7.4 Optimizer Works

```python
python -c "
import numpy as np
import pandas as pd
from trading_system.portfolio.optimizer import FactorNeutralOptimizer
signals = pd.Series(np.random.uniform(0, 1, 20), index=[f'TICK{i:02d}' for i in range(20)])
opt = FactorNeutralOptimizer(target_n=6)
result = opt.optimize(signals)
w = result['weights']
print(f'Positions: {len(w)}, sum={w.sum():.4f}')
"
```

---

## 8. How to Export This System

### 8.1 Git-Tracked Files (Version Controlled)

All source code, configs, tests, and docs are version controlled. Clone the repo.

### 8.2 Data Files (NOT in Git — Must Re-Download or Transfer)

```bash
# Option A: Re-download from scratch (internet required)
python scripts/download_yfinance_data.py --start 2010-01-01 --end 2025-12-31
python scripts/download_sec_fundamentals.py   # optional, takes hours

# Option B: Transfer existing cache via rsync/scp
rsync -av data/raw/   new_machine:/path/to/trade\ v2/data/raw/
rsync -av data/cache/ new_machine:/path/to/trade\ v2/data/cache/
rsync -av data/models/ new_machine:/path/to/trade\ v2/data/models/
```

### 8.3 Machine-Specific Files (Create on Each Machine)

```bash
# Create config/data_sources.yaml (see Section 3.7 template)
conda env create -f environment.yaml -p ./.env
conda activate ./.env
pip install -e .
```

### 8.4 For Claude Instance Handoff

Read these files in order:
1. `SETUP.md` (this file)
2. `README.md`
3. `docs/design/trading_plan.md`
4. Check `.claude/` directory for any hooks or best-practices files

---

## 9. Production Trading Workflow

### 9.1 Overview

The system generates a daily brief at every Claude session start and supports live trading through Interactive Brokers (IB TWS). The brief is cached per day — the first session regenerates it, subsequent sessions read from cache.

### 9.2 Daily Routine

```bash
# Brief is auto-generated when Claude session starts.
# To regenerate manually (e.g. late in the day after market close):
python production/daily_run.py --refresh
```

**Brief contains:**
- Regime status (VIX, SPY trend, momentum z-score) — all 3 gates must pass to trade
- Portfolio positions marked-to-market
- Rebalance status and next rebalance date
- Top signals + BUY/SELL/HOLD list (on rebalance days when regime is open)

### 9.3 Rebalance Day (Regime Open)

```bash
# 1. Read brief — it shows exact BUY/SELL/HOLD list
python production/daily_run.py

# 2. Execute trades manually in IB TWS

# 3. Update positions file with executed trades
nano data/production/positions.csv    # add/remove rows

# 4. Mark rebalance done
python production/daily_run.py --confirm-trade
```

**`data/production/positions.csv` format:**
```
ticker,shares,entry_price,entry_date
AAPL,10,175.50,2026-04-22
```

### 9.4 Monthly Retrain

Run on the 1st of each month to incorporate the latest market data:

```bash
# Retrain production model (window 12) on all data up to today.
# - Auto-detects latest available data in cache
# - Downloads incremental data from yfinance if needed (~20-30 min for monthly increment)
# - Rebuilds feature/label caches if range not covered
# - Trains Ridge + LGB in parallel, saves to window_12/
# - Auto-updates config/production.yaml to point at new model
python production/retrain.py

# After retrain, regenerate brief with new model
python production/daily_run.py --refresh
```

**First-time retrain (or after a long gap):** incremental download checks all 868 tickers for new data — expect ~1–2 hours the first time, ~20–30 min for monthly increments thereafter.

**Retrain options:**
```bash
python production/retrain.py --dry-run                      # preview splits without training
python production/retrain.py --train-start 2015-01-01       # skip early QE regime data
python production/retrain.py --val-months 6                 # shorter validation window
```

### 9.5 Production Files

| File | Purpose |
|------|---------|
| `production/daily_run.py` | Daily brief generator — auto-runs at Claude session start |
| `production/portfolio_state.py` | Position tracking, P&L, rebalance state |
| `production/retrain.py` | Monthly retrain — auto-detects latest data, full pipeline |
| `config/production.yaml` | Live config (account size, model window, rebal frequency) |
| `data/production/positions.csv` | Open positions — edit after each trade |
| `data/production/rebalance_state.json` | Last rebalance date — updated by `--confirm-trade` |
| `data/production/daily_brief.md` | Cached daily brief — auto-generated |

---

## 10. Directory Structure

```
trade v2/
├── .claude/
│   ├── settings.json           # SessionStart hook (runs session-start.sh)
│   ├── settings.local.json     # Per-machine bash permissions (not committed)
│   └── session-start.sh        # Generates daily brief + loads README.md on session start
├── config/
│   ├── backtest.yaml           # Walk-forward window definitions (11 windows, 2010-2025)
│   ├── data_sources.yaml       # Machine-specific data source paths (gitignored)
│   ├── execution.yaml          # IB commission model
│   ├── portfolio.yaml          # cvxpy optimizer constraints
│   ├── production.yaml         # Live trading config (account, model, rebal frequency)
│   ├── universe.yaml           # Universe filters (price, ADR, rebal freq)
│   ├── features/mean_reversion/
│   │   ├── model_mr_zscore_12feat.yaml   # ACTIVE: 12-feature production model
│   │   ├── model_mr_zscore.yaml          # Baseline 15-feature config
│   │   ├── model_mr_zscore_10feat.yaml   # Pruned 10-feature (retired)
│   │   ├── model_mr_crosssec.yaml        # Cross-sectional (deferred)
│   │   └── model_mr_pairs.yaml           # Pairs trading (deferred)
│   ├── models/
│   │   ├── ridge.yaml                    # Ridge alpha grid
│   │   ├── lightgbm.yaml                 # LGB hyperparams + regularization
│   │   └── ensemble.yaml                 # IC weighting config
│   ├── factors/
│   │   ├── ken_french.yaml               # FF5+MOM factor constraints
│   │   └── in_house.yaml                 # Momentum + quality factors
│   └── regime/
│       └── detector.yaml                 # Gate thresholds (VIX 25, SPY 200d MA, momentum_z 0.5)
├── data/                       # gitignored
│   ├── cache/                  # Parquet caches (prices, features, labels, betas, regime)
│   ├── models/mean_reversion/
│   │   └── model_mr_zscore_12feat_{ridge,lightgbm}/window_N/
│   ├── production/
│   │   ├── positions.csv
│   │   ├── daily_brief.md
│   │   └── rebalance_state.json
│   └── raw/
│       ├── yfinance/           # Per-ticker Parquet (868 tickers)
│       ├── ken_french/         # FF5+MOM Parquet cache
│       └── sec_edgar/          # Company facts JSON by CIK
├── production/
│   ├── daily_run.py            # Daily brief generator
│   ├── portfolio_state.py      # Position tracking, mark-to-market, rebalance state
│   └── retrain.py              # Monthly retrain
├── scripts/
│   ├── OHLCV/
│   │   └── download_yfinance_data.py     # Download 868-ticker universe
│   ├── download_sec_fundamentals.py      # Batch SEC EDGAR company facts
│   ├── fama_macbeth_analysis.py          # FM regression for feature validation
│   └── show_results_table.py            # Cross-window results table + DSR
├── src/trading_system/         # Main package (see Section 11)
├── tests/
│   ├── integration/
│   │   └── test_pipeline_mean_reversion.py
│   └── unit/
│       ├── test_features/test_mean_reversion.py
│       ├── test_models/test_ridge.py
│       ├── test_portfolio/test_optimizer.py
│       └── test_regime/test_detector.py
├── docs/design/trading_plan.md
├── environment.yaml
├── pyproject.toml
├── README.md
└── SETUP.md
```

---

## 11. Source Code Module Map

`src/trading_system/`

| Module | Description |
|--------|-------------|
| `cli.py` | Main CLI: train, backtest, download, spy-timing |
| **backtest/** | |
| `backtest/analytics.py` | Sharpe, drawdown, `deflated_sharpe_ratio()` |
| `backtest/cross_sectional/engine.py` | `BacktestEngine`: regime-gated, PIT-filtered rebalance loop |
| `backtest/cross_sectional/signal_generator.py` | IC-weighted ensemble scoring |
| `backtest/cross_sectional/execution.py` | IB commission simulator |
| `backtest/cross_sectional/portfolio.py` | In-backtest portfolio tracking |
| `backtest/directional/spy_timing.py` | SPY timing backtest (v1 legacy) |
| **data/** | |
| `data/loaders.py` | `DataLoader` facade |
| `data/yfinance_loader.py` | Bulk OHLCV download + per-ticker facts |
| `data/point_in_time.py` | Point-in-time data access |
| `data/synthetic.py` | `SyntheticDataGenerator` for testing (no internet needed) |
| `data/validators.py` | Data quality checks |
| `data/corporate_actions.py` | Stub (yfinance `auto_adjust=True` handles splits) |
| **factors/** | |
| `factors/ken_french.py` | Ken French FF5+MOM; auto-refresh |
| `factors/rolling_betas.py` | `RollingBetaComputer`: rolling OLS betas |
| `factors/momentum.py` | 12m-1m cross-sectional momentum |
| `factors/sec_fundamentals.py` | B/P, E/P, ROE, gross profitability (point-in-time) |
| **features/** | |
| `features/base.py` | `BaseFeature`, `FeatureSet` ABCs |
| `features/builder.py` | `FeatureBuilder`: loads config, orchestrates compute |
| `features/transforms.py` | Winsorize, rank-normalize, cross-sectional z-score |
| `features/mean_reversion.py` | 12 MR features (see Section 13) |
| `features/cross_sectional_reversal.py` | Loser-effect CS z-scores |
| `features/price_momentum.py` | Momentum control features |
| `features/volume_volatility.py` | Volume/vol features |
| `features/market_context.py` | Market context features |
| **labels/** | |
| `labels/builder.py` | `LabelBuilder`: 5-day forward return labels (shift=-5, no lookahead) |
| **models/** | |
| `models/trainer.py` | `ModelTrainer`: walk-forward train/val split + PIT filter |
| `models/ridge.py` | `RidgeModel`: sklearn Ridge + IC scoring |
| `models/lightgbm_model.py` | `LightGBMModel`: lgb.train + IC scoring |
| **portfolio/** | |
| `portfolio/optimizer.py` | `FactorNeutralOptimizer`: cvxpy, ECOS solver |
| `portfolio/constructor.py` | `PortfolioConstructor` + `FactorNeutralConstructor` |
| **regime/** | |
| `regime/detector.py` | `RegimeDetector`: 3-gate AND (VIX, SPY MA, momentum_z) |
| `regime/filters.py` | `RegimeGate`: tradeable date filter |
| **universe/** | |
| `universe/builder.py` | `UniverseBuilder` |
| `universe/filters.py` | `PriceFilter`, `ADRFilter` (heuristic regex) |
| `universe/historical.py` | `PointInTimeUniverse` (868 tickers, Wikipedia S&P 500 changes) |
| **utils/** | |
| `utils/cache.py` | `DataCache`: Parquet read/write with key-based naming |
| `utils/config.py` | YAML config loading utilities |

---

## 12. Key Algorithms

### IC-Weighted Ensemble
(`backtest/cross_sectional/signal_generator.py`)

```python
for each model m with val_ic[m]:
    if val_ic[m] < 0:
        exclude  # negative IC = worse than random
    weight[m] = val_ic[m] / sum(val_ic for included models)

ensemble_score = sum(weight[m] * rank_score[m] for m in included)
```

Models below `min_ic_threshold=0.02` are warned but not excluded (still IC-weighted).

### Deflated Sharpe Ratio
(`backtest/analytics.py` → `deflated_sharpe_ratio()`)

Bailey & Lopez de Prado (2014). Adjusts for multiple testing and non-normality:

```
SR_hat = mean(r) / std(r) * sqrt(252)
SR*    = sqrt(2 * log(N)) * (1 - euler_gamma) / sqrt(log(log(N)))
Z      = (SR_hat - SR*) * sqrt(1 - sqrt(T-1) / skew*SR_hat + (kurt-1)/4 * SR_hat^2)
DSR    = Phi(Z)
```

Default N=50 trials. DSR > 0.95 = significant at 5% after multiple-testing correction.

### Point-in-Time Universe
(`universe/historical.py` → `PointInTimeUniverse`)

```python
# Ticker in S&P 500 on date if:
added_date <= date AND (removed_date > date OR removed_date is NaT)

# Training filter: any presence in [train_start, train_end]:
added_date <= train_end AND (removed_date > train_start OR NaT)
```

Built from Wikipedia S&P 500 changes table (walk backwards from current members). Saved to `data/raw/sp500_pit_constituents.csv` (868 tickers, 394 change events). Load via `PointInTimeUniverse.load_or_build()`.

### Regime Gate
(`regime/detector.py`)

All three must pass (AND — any veto blocks trading):

1. VIX < 25
2. SPY close > 200-day MA
3. momentum_z < 0.5 (z-score of top20%−bottom20% 12m-1m spread, 60d smoothed, vs 252d history)

### Factor Neutralization
(`portfolio/optimizer.py`)

```
maximize  signal · w
subject to:
  sum(w) = 1,  0 ≤ w_i ≤ max_weight
  |β_MktRF · w| ≤ 0.15
  |β_k · w|     ≤ 0.10   for k in {SMB, HML, RMW, CMA, MOM}
```

Solver: ECOS. Betas from 252-day rolling OLS vs Ken French FF5+MOM factors.

---

## 13. Active Feature Set

**Model:** `model_mr_zscore_12feat`

| # | Feature | Type | Key param |
|---|---------|------|-----------|
| 1 | `zscore_5d` | Price z-score | lookback=5 |
| 2 | `zscore_10d` | Price z-score | lookback=10 |
| 3 | `zscore_20d` | Price z-score | lookback=20 |
| 4 | `zscore_60d` | Price z-score | lookback=60 |
| 5 | `ret_zscore_5d` | Return z-score | lookback=5, history=252 |
| 6 | `ret_zscore_10d` | Return z-score | lookback=10, history=252 |
| 7 | `ret_zscore_20d` | Return z-score | lookback=20, history=252 |
| 8 | `bb_distance_20d` | Bollinger | lookback=20, n_std=2.0 |
| 9 | `rsi_14d` | RSI | period=14 |
| 10 | `dev_from_52w_mean` | Long-term deviation | lookback=252 |
| 11 | `distance_52w_high` | 52w high distance | lookback=252, FM t=13.09 |
| 12 | `realized_vol_20d` | Realized vol | window=20, FM t=−6.66 |

All features: winsorized 1%/99% cross-sectionally, missing → cross-sectional median.

---

## 14. Key Config Values

### `config/models/lightgbm.yaml`

```yaml
hyperparameters:
  max_depth: [4, 5]
  num_leaves: [15, 31, 63]
  learning_rate: [0.01]
  min_child_samples: [50]          # raised from 20 to reduce overfitting

fixed_params:
  num_iterations: 300
  reg_lambda: 1.0                  # L2 regularization
  subsample: 0.8                   # row subsampling per tree
  subsample_freq: 1
  colsample_bytree: 0.8            # feature subsampling per tree
  random_state: 42

early_stopping:
  enabled: false                   # disabled — RMSE unsuitable for low-IC signals
```

### `config/regime/detector.yaml`

```yaml
vix_gate:
  threshold: 25
spy_gate:
  ma_window: 200
momentum_gate:
  enabled: true
  z_threshold: 0.5
  lookback_long: 252
  lookback_skip: 21
  smooth_window: 60
  top_pct: 0.2
```

### `config/production.yaml`

```yaml
account:
  initial_capital: 2000.0
  max_positions: 6
model:
  window: 12
  feature_set: model_mr_zscore_12feat
rebalance:
  frequency_days: 5
data:
  lookback_days: 400
regime:
  vix_threshold: 25.0
  momentum_z_threshold: 0.5
paths:
  positions_file: data/production/positions.csv
  equity_log: data/production/equity_log.csv
  daily_brief: data/production/daily_brief.md
  rebalance_state: data/production/rebalance_state.json
```

### `config/backtest.yaml` (walk-forward windows)

```yaml
walk_forward:
  data_start_date: "2010-01-01"
  data_end_date: "2025-12-31"
  train_years: 4
  val_years: 1
  test_years: 1
  purge_days: 5                    # = label horizon, prevents train/val leakage
  embargo_pct: 0.10
```

---

## 15. Data Schemas

### Parquet Caches (`data/cache/`)

Named by params: `{type}_end=YYYY-MM-DD_start=YYYY-MM-DD.parquet`

| Cache | Key columns |
|-------|-------------|
| `prices_*` | ticker, date, open, high, low, close, volume, vix_close |
| `model_mr_zscore_*` | ticker, date, + 15 base feature cols |
| `labels_*` | ticker, date, label (5d fwd return) |
| `rolling_betas_*` | ticker, date, MktRF, SMB, HML, RMW, CMA, MOM |
| `regime_scores_*` | date, vix_ok, spy_ok, momentum_ok, tradeable, momentum_z |

### Production Files (`data/production/`)

| File | Schema / Description |
|------|---------------------|
| `positions.csv` | ticker, shares, entry_price, entry_date |
| `rebalance_state.json` | `{"last_rebalance_date": "YYYY-MM-DD", "frequency_days": 5}` |
| `equity_log.csv` | date, realised_pnl |
| `daily_brief.md` | Cached markdown brief, regenerated once per calendar day |

### Model Artifacts (`data/models/mean_reversion/{model_name}/window_N/`)

| File | Description |
|------|-------------|
| `model.pkl` | Ridge trained model (pickle) |
| `model.lgb` | LightGBM trained model (native LGB format) |
| `metadata.json` | Ridge: train_ic, val_ic, feature_list, train/val dates, hyperparams |
| `model_meta.json` | LightGBM: train_ic, val_ic, feature_list, hyperparams |

### PIT Constituent DB (`data/raw/sp500_pit_constituents.csv`)

Columns: ticker, added_date, removed_date (NaT = still active)  
868 rows: 503 current + 365 historical/delisted

---

## 16. Infrastructure Config

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "trading_system"
version = "0.2.0"
requires-python = ">=3.12"

dependencies = [
    "pandas>=2.0.0", "numpy>=1.24.0", "scipy>=1.10.0",
    "scikit-learn>=1.3.0", "lightgbm>=4.0.0",
    "pyarrow>=12.0.0", "statsmodels>=0.14.0",
    "matplotlib>=3.7.0", "seaborn>=0.12.0",
    "pyyaml>=6.0", "tqdm>=4.65.0",
    "python-dateutil>=2.8.2", "pytz>=2023.3",
    "yfinance>=0.2.0", "pandas-datareader>=0.10.0",
    "cvxpy>=1.3.0", "requests>=2.28.0",
]

[project.scripts]
trading-system = "trading_system.cli:main"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP"]
ignore = ["E501", "N803", "N806"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

### `.claude/settings.json`

```json
{
  "hooks": [
    {
      "type": "command",
      "command": "bash .claude/session-start.sh"
    }
  ],
  "SessionStart": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "bash .claude/session-start.sh"
        }
      ]
    }
  ]
}
```

### `.claude/session-start.sh`

Runs on every Claude session start:

1. If `data/production/daily_brief.md` already has today's date tag → print cached brief
2. Otherwise: run `python production/daily_run.py` to generate fresh brief (~30 s)
3. Load and print `README.md`
4. Print instruction to confirm files have been read
