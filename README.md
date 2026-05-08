# Trading System v2

A low-frequency, ML-driven mean reversion trading system for retail-tradeable U.S. stocks.
Evolved from v1 (momentum) — same walk-forward infrastructure, new strategy.

New machine or new Claude instance? Read **SETUP.md** first.

## Quick Start

```bash
conda env create -f environment.yaml -p ./.env
conda activate ./.env
pip install -e .
python -m trading_system.cli --help
```

## System Overview

| Aspect | Detail |
|--------|--------|
| Account size | $2,000 |
| Universe | U.S. common stocks (NYSE/NASDAQ), 800–1,500 stocks |
| Positions | 6 stocks, factor-neutral weighted |
| Holding period | 5–30 days |
| Rebalance frequency | Every 5–7 days (regime-gated) |
| Signal | Mean reversion: z-score oversold → buy |
| Models | Ridge + LightGBM ensemble (12 MR features, model_mr_zscore_12feat) |
| Validation | 4-year rolling walk-forward (11 windows, 2010–2025) |
| Data sources | yfinance (OHLCV), Ken French (factors), SEC EDGAR (fundamentals) |
| Transaction costs | Interactive Brokers (~0.5–0.9% per round trip) |
| Risk limits | $150 stop/position, $500 max drawdown |
| Regime gate | 3-gate AND: VIX < 25, SPY > 200MA, cross-sectional momentum z < 0.5 |
| Factor neutrality | Ken French FF5+MOM; portfolio \|β_k\| < 0.10 |

Full specification: `docs/design/trading_plan.md`

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Infrastructure & Data | ✅ Complete |
| 2 | Universe & Labels | ✅ Complete |
| 3 | Mean Reversion Features (12) | ✅ Complete |
| 4 | Training & Baseline Backtest | ✅ Complete |
| 5 | Factor Neutralization — Layer 1 (Ken French) | ✅ Complete |
| 6 | Layer 2 + Regime Detection | ✅ Complete |
| 7 | Bias Fixes, Statistical Rigor & Production | ✅ Complete |

## Training Results

12-feat model, PIT universe, post-regularization:

| Model | Avg Val IC | Windows Saved | Notes |
|-------|-----------|---------------|-------|
| model_mr_zscore_12feat_ridge | ~0.036 | 9/11 | W2, W9, W11 below save threshold |
| model_mr_zscore_12feat_lightgbm | ~0.018 | 11/11 | W6, W10, W11 below 0.02 quality gate (saved but warned) |

LGB regularized: `subsample=0.8, colsample_bytree=0.8, min_child_samples=50, reg_lambda=1.0`

## Backtest Results

12-feat model, PIT universe, avg Sharpe across 11 windows (excl. W6 100%-gated):

| Scenario | Avg Sharpe |
|----------|-----------|
| Ridge | +0.66 |
| LGB (regularized) | +0.45 |
| Ensemble (IC-weighted) | +0.63 |

DSR = 0.000 (SR\* = 0.84 for N=50 trials). Not statistically significant after multiple-testing correction.

## Repository Structure

```
trade v2/
├── config/
│   ├── features/mean_reversion/
│   │   ├── model_mr_zscore_12feat.yaml   # ACTIVE: 12 MR features
│   │   ├── model_mr_zscore.yaml          # Baseline: 15 MR features
│   │   ├── model_mr_crosssec.yaml        # Cross-sectional reversal (deferred)
│   │   └── model_mr_pairs.yaml           # Pairs trading (deferred)
│   ├── models/
│   │   ├── ridge.yaml, lightgbm.yaml, ensemble.yaml
│   ├── factors/
│   │   ├── ken_french.yaml               # FF5+MOM constraints
│   │   └── in_house.yaml                 # Momentum + quality factors
│   ├── regime/
│   │   └── detector.yaml                 # VIX thresholds + SPY MA config
│   ├── backtest.yaml                     # Walk-forward windows (11 windows, 2010-2025)
│   ├── universe.yaml                     # Stock universe filters + ADR heuristics
│   ├── portfolio.yaml                    # Factor-neutral optimizer config
│   ├── execution.yaml                    # Execution and cost settings
│   ├── data_sources.yaml                 # Machine-specific (gitignored)
│   └── production.yaml                   # Live config (account, model, rebal freq)
├── src/trading_system/
│   ├── cli.py
│   ├── data/
│   │   ├── loaders.py                    # DataLoader facade
│   │   ├── yfinance_loader.py            # Bulk OHLCV + per-ticker facts
│   │   ├── point_in_time.py
│   │   ├── synthetic.py                  # SyntheticDataGenerator (no internet needed)
│   │   └── validators.py, corporate_actions.py
│   ├── features/
│   │   ├── base.py, builder.py, transforms.py
│   │   ├── mean_reversion.py             # 12 MR features (primary)
│   │   ├── cross_sectional_reversal.py   # Loser-effect CS z-scores
│   │   ├── price_momentum.py             # Momentum controls
│   │   ├── volume_volatility.py
│   │   └── market_context.py
│   ├── factors/
│   │   ├── ken_french.py                 # FF5+MOM loader + portfolio betas
│   │   ├── rolling_betas.py              # Per-stock rolling OLS betas
│   │   ├── sec_fundamentals.py           # B/P, E/P, ROE, gross profitability
│   │   └── momentum.py                   # 12m-1m momentum factor
│   ├── regime/
│   │   ├── detector.py                   # VIX regimes + SPY 200-day MA composite gate
│   │   └── filters.py                    # RegimeGate (tradeable date filter)
│   ├── labels/
│   │   └── builder.py                    # 5-day forward return labels
│   ├── models/
│   │   ├── trainer.py                    # Walk-forward ModelTrainer
│   │   ├── ridge.py                      # Ridge regression model
│   │   └── lightgbm_model.py             # LightGBM model
│   ├── portfolio/
│   │   ├── optimizer.py                  # FactorNeutralOptimizer (cvxpy/ECOS)
│   │   └── constructor.py                # PortfolioConstructor + FactorNeutralConstructor
│   ├── backtest/
│   │   ├── analytics.py                  # Shared performance metrics + deflated_sharpe_ratio()
│   │   ├── cross_sectional/
│   │   │   ├── engine.py                 # BacktestEngine with regime gate
│   │   │   ├── execution.py              # IB commission simulator
│   │   │   ├── portfolio.py              # Portfolio management
│   │   │   └── signal_generator.py       # IC-weighted ensemble signal generation
│   │   └── directional/
│   │       └── spy_timing.py             # SPY timing backtest (v1 compat)
│   ├── universe/
│   │   ├── builder.py, filters.py
│   │   └── historical.py                 # PointInTimeUniverse (868-ticker PIT constituent DB)
│   └── utils/
│       ├── cache.py                      # Parquet-based DataCache
│       └── config.py                     # Config loading utilities
├── scripts/
│   ├── download_yfinance_data.py         # Download universe OHLCV + SPY/VIX
│   ├── download_sec_fundamentals.py      # Batch SEC EDGAR company facts
│   ├── fama_macbeth_analysis.py          # Factor validation (Fama-MacBeth)
│   └── show_results_table.py             # Cross-window results table + DSR
├── production/
│   ├── daily_run.py                      # Daily brief generator — auto-runs at Claude session start
│   ├── portfolio_state.py                # Position tracking, P&L, rebalance state
│   └── retrain.py                        # Monthly production retrain
├── data/                                 # gitignored
│   ├── cache/                            # Parquet cache (prices, features, labels, betas, regime)
│   ├── raw/
│   │   ├── yfinance/                     # Cached yfinance bulk downloads
│   │   ├── ken_french/                   # Cached Fama-French factor data
│   │   └── sec_edgar/                    # Cached SEC EDGAR company facts JSON
│   ├── models/mean_reversion/            # Trained model artifacts per window
│   ├── results/                          # Training/backtest results JSON
│   └── production/                       # positions.csv, daily_brief.md, rebalance_state.json
├── tests/
│   ├── unit/
│   │   ├── test_features/                # MR feature tests (z-score, RSI, BB)
│   │   ├── test_factors/                 # Ken French, rolling betas, SEC
│   │   ├── test_regime/                  # Regime detector + gate tests
│   │   ├── test_portfolio/               # Factor-neutral optimizer tests
│   │   └── test_models/
│   └── integration/
│       └── test_pipeline_mean_reversion.py   # Full pipeline end-to-end
├── docs/design/trading_plan.md           # Trading plan specification
├── environment.yaml
├── pyproject.toml
├── README.md
└── SETUP.md
```

## CLI Reference

### Download Data

```bash
# Download S&P 500 universe (yfinance)
python -m trading_system.cli download --start 2010-01-01 --end 2025-12-31 --tickers sp500

# Download S&P 1500 (larger universe)
python -m trading_system.cli download --tickers sp1500

# Force re-download (ignore cache)
python -m trading_system.cli download --force
```

### Train

**Recommended — parallel across all windows (one process per window):**

```bash
for w in 1 2 3 4 5 6 7 8 9 10 11; do
    python -m trading_system.cli train --models all --windows $w --parallel --n-jobs 4 \
    > /tmp/train_w${w}.log 2>&1 &
done
# Monitor: tail -f /tmp/train_w1.log
```

**Single-process alternatives:**

```bash
# All windows sequentially (models parallelized within each window)
python -m trading_system.cli train --models all --parallel --verbose

# Specific windows only
python -m trading_system.cli train --windows 1,2,3 --parallel --verbose

# Use synthetic data (no yfinance required)
python -m trading_system.cli train --synthetic --verbose
```

**Flags:**

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--models` | all, ridge, lightgbm | all | Models to train |
| `--feature-sets` | model_mr_zscore | all | Feature set to use |
| `--windows` | 1-11 (comma-separated) | all | Walk-forward windows |
| `--verbose / -v` | — | off | Live progress output |
| `--parallel / -p` | — | off | Parallel model training |
| `--no-cache` | — | off | Force reload from yfinance |
| `--synthetic` | — | off | Use synthetic data |

**Config** (`config/backtest.yaml`):

```yaml
# TEST MODE — 1 window, fast iteration
# data_start_date: "2019-01-01"
# data_end_date:   "2024-12-31"

# FULL MODE — 11 windows, 2010-2025
data_start_date: "2010-01-01"
data_end_date:   "2025-12-31"
```

**Outputs:**
- Models: `data/models/mean_reversion/{model_name}/window_{id}/`
- Results: `data/results/training/training_results_{timestamp}.json`
- Cache: `data/cache/` (prices, features, labels as Parquet)

### Backtest

```bash
# Recommended — all windows in parallel
for w in 1 2 3 4 5 6 7 8 9 10 11; do
    python -m trading_system.cli backtest --window $w > /tmp/bt_w${w}.log 2>&1 &
done
# Monitor: tail -f /tmp/bt_w1.log
# Results: python scripts/show_results_table.py --full

# Single window
python -m trading_system.cli backtest --window 1 --verbose
python -m trading_system.cli backtest --window 1 --no-regime --verbose   # disable regime gate
```

Runs 6 scenarios per window: ridge, lgb, ensemble, ic_ensemble, persistence, tiered_vix. Results saved to `data/results/backtests/`.

### Production

```bash
# Daily brief (regime + portfolio + signals) — auto-runs on Claude session start
python production/daily_run.py              # generate or use cached brief
python production/daily_run.py --refresh    # force regenerate
python production/daily_run.py --confirm-trade  # mark rebalance done after executing trades

# Monthly retrain — run on the 1st of each month
# Downloads incremental price data, rebuilds caches if needed, trains Ridge + LGB
python production/retrain.py               # full retrain to latest available data
python production/retrain.py --dry-run     # preview date splits without training
python production/daily_run.py --refresh   # regenerate brief after retrain
```

**Rebalance day workflow:**

1. Read brief → execute BUY/SELL/HOLD list in IB TWS
2. Edit `data/production/positions.csv` with executed trades
3. Run `python production/daily_run.py --confirm-trade`

Monthly retrain auto-detects the latest available data in cache (downloading incrementally if needed), trains on all history with the last 12 months as validation, saves to `window_12`. First run after a long gap takes ~1–2 hours (868 tickers × incremental download); monthly increments take ~20–30 min.

### Other Commands

```bash
python -m trading_system.cli spy-timing    # SPY timing backtest (v1 compat)
```

### Development

```bash
# Tests
pytest
pytest --cov=src/trading_system tests/     # with coverage
pytest tests/unit/test_features/ -v        # specific module
pytest tests/integration/ -v               # integration only

# Lint and format
ruff check src/ tests/
ruff format src/ tests/
```

## v1 → v2 Changes

| Component | v1 | v2 |
|-----------|----|----|
| Data source | Elastic (SMILE/EODPrime) | yfinance + pandas-datareader + SEC EDGAR |
| Strategy | Price momentum | Mean reversion (z-score oversold) |
| Features | 25 momentum features (model_a/b/c/d/e) | 12 MR features (model_mr_zscore_12feat) |
| Portfolio | Equal weight, top-6 by score | Factor-neutral (cvxpy, FF5+MOM constraints) |
| Trading gate | Always-on | Regime gate: VIX + SPY 200-day MA + momentum z |
| Factor control | None | Ken French FF5+MOM \|β_k\| < 0.10 |
| ADR filter | Metadata API (is_adr column) | Heuristic regex patterns |
| Model save path | data/models/cross_sectional/ | data/models/mean_reversion/ |

Walk-forward infrastructure, CLI interface, label builder, analytics, and config formats are unchanged from v1.

## Changelog

### 2026-04-22 — Production System & Monthly Retrain (Session 5)

- **Production daily brief** (`production/daily_run.py` NEW): Auto-generated at each Claude session start via `.claude/session-start.sh`. Shows regime status (VIX/SPY/momentum gates), portfolio P&L, rebalance status, top signals, and BUY/SELL/HOLD list. Cached per day — first session generates fresh (~30 s), subsequent sessions use cache (~5 s).
- **Portfolio state tracking** (`production/portfolio_state.py` NEW): Reads/writes `data/production/positions.csv`. Marks positions to market, computes unrealised P&L, determines rebalance day based on business-day count since last rebalance.
- **Monthly production retrain** (`production/retrain.py` NEW): Self-contained retrain script. Trains Ridge + LGB on all available historical data (default 2010-01-01 → latest available), validated on the most recent 12 months. Auto-detects latest data from cache; downloads incremental yfinance data and rebuilds feature/label caches if the range isn't covered. Saves as `window_12`, auto-updates `config/production.yaml`. No walk-forward windows — one final model trained on everything.
- **Production model (window 12):** Train 2010-01-01 → 2025-04-16, val 2025-04-17 → 2026-04-17. Ridge val_ic=0.055, LGB val_ic=0.017. Lower ICs reflect Apr 2025–Apr 2026 was a high-momentum period where mean reversion struggled — consistent with current gated regime. Ridge dominant; ensemble weights Ridge ~3× LGB.
- **Production config** (`config/production.yaml` NEW): Account size $2,000, 6 positions, 5-day rebalance frequency, model window, regime gate thresholds.

### 2026-04-22 — Bias Fixes & Statistical Rigor (Session 4)

- **Survivorship bias fix — PointInTimeUniverse** (`src/trading_system/universe/historical.py` NEW): Wikipedia S&P 500 changes history reconstructed into a point-in-time constituent DB (868 tickers: 503 current + 365 historical/delisted, 394 change events). Applied at both backtest time (per-date filtering) and training time (window-range filtering). Residual bias acknowledged: pre-2010 removals may be incomplete, delisted stocks not fetchable from yfinance silently skipped.
- **Ken French data freshness** (`src/trading_system/data/datareader_loader.py`): Auto-refresh added — if cached factor data is >7 days old, re-download. KF publishes monthly with ~4–6 week lag; `_is_cache_stale()` checks latest date in cached Parquet vs today.
- **Deflated Sharpe Ratio** (`src/trading_system/backtest/analytics.py`): `deflated_sharpe_ratio(returns, n_trials=50)` added. Bailey & Lopez de Prado (2014) formula accounting for multiple testing (N trials) and return non-normality (skew, excess kurtosis). DSR = 0.000 for current strategy (SR\* = 0.84 for N=50 trials over 11 years vs observed avg Sharpe +0.52). Result does not meet 5% significance threshold after multiple-testing correction. Daily returns now saved in backtest JSON; DSR shown in all `show_results_table.py` views.
- **show_results_table.py** new flags: `--compare-each` (3 per-experiment tables). DSR row shown automatically in all table views.
- **Factor neutralization:** Always enabled (cvxpy, FF5+MOM). No-factor-neutral scenario removed — factor neutral is the only portfolio construction mode.
- **Full pipeline re-run completed (2026-04-22):** All 11 windows retrained + rebacktested with PIT universe. Bias-corrected results: Ridge +0.66, LGB +0.45, Ensemble +0.63.

### 2026-04-21 — Model & Regime Improvements (Session 3)

- **Consolidated to 12-feat model only** — retired 15-feat and 10-feat configs from training and backtest scenarios
- **LGB regularization** — added `subsample=0.8, colsample_bytree=0.8, min_child_samples=50, reg_lambda=1.0`; LGB avg Sharpe improved from +0.49 → +0.58
- 3 experimental regime scenarios — `ic_ensemble` (strict IC gate), `persistence` (N=5 consecutive open days), `tiered_vix` (VIX 20–25 → 3 positions)
- `show_results_table.py --compare` — new flag to compare baseline vs 3 experiments
- Cleaned up stale models — deleted 10-feat and 15-feat model directories

### 2026-04-16 — Walk-Forward Backtest & Feature Engineering (Session 2)

- Added Ridge to backtest scenarios; created `scripts/show_results_table.py`
- Evaluated 3 candidate features; pruned 5 non-significant features
- Added `distance_52w_high` (FM t=13.09) and `realized_vol_20d` (FM t=−6.66) → `model_mr_zscore_12feat`
- Ran ablation: 12-feat mean val IC = 0.0362 vs 15-feat baseline 0.0246 (+47%)
- Backtested all 11 windows for 3 feature configs

### 2026-04-13 — v2 Inception (Session 1)

- Initialized Trade v2 from v1 architecture; pivoted strategy from momentum to mean reversion
- Pivoted data source from Elastic → yfinance + pandas-datareader + SEC EDGAR
- New modules: `regime/`, `factors/`, `data/yfinance_loader.py`, `data/datareader_loader.py`, `data/sec_loader.py`
- New: `features/mean_reversion.py` — 15 features: PriceZScore, ReturnZScore, BollingerBand, RSI, CrossSectionalZScore, AutocorrReturn, DeviationFrom52wMean, SimpleReturn
- New: `portfolio/optimizer.py` — FactorNeutralOptimizer with cvxpy, ECOS solver, |β_k| < 0.10 constraints
- New: `regime/detector.py` — VIX regimes (low <15, normal <25, high <35, crisis ≥35); SPY 200-day MA trend; composite score gate
- New: `universe/filters.py` ADRFilter using heuristic patterns (no metadata API required)
- Adapted: `data/loaders.py` interface preserved; internals replaced with yfinance backend
- Adapted: `backtest/cross_sectional/engine.py` with regime gate (hold-on-gate, skip rebalance)
- Adapted: `models/trainer.py` feature set loop: model_mr_zscore replaces model_a/b/c/d/e
- Configs added: `config/data_sources.yaml`, `config/factors/`, `config/regime/`, `config/features/mean_reversion/`
- Scripts added: `scripts/download_yfinance_data.py`, `scripts/download_sec_fundamentals.py`
- Tests: unit tests for MR features, regime detector, factor-neutral optimizer; integration test for full pipeline
