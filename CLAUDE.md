# CLAUDE.md — Trading System v2

## Behavior Rules

### Always Plan Before Implementing
Before writing any code or making any changes:
1. State what you understand the task to be
2. List what you need to clarify — ask all questions upfront, not one at a time
3. Present a concrete plan: which files change, what each change does, why
4. Wait for explicit approval before proceeding

No exceptions — even for small changes. A one-line fix still gets a one-sentence plan and a confirmation.

### No Guessing or Fabricating
- If you don't know something, say so directly.
- Never invent config values, file paths, function signatures, or results you haven't verified by reading the actual file.
- If a file doesn't exist yet, say it doesn't exist.
- If asked about backtest results, training metrics, or any numbers — only report what is in actual files on disk.
- If you're unsure whether a module, flag, or behavior exists, grep for it or read the file before answering.

### Clarify Ambiguity Before Acting
If the request is unclear, incomplete, or could be interpreted multiple ways:
- Ask for clarification before doing anything
- List the specific ambiguities — don't ask open-ended questions
- Propose your default interpretation and confirm it before using it

### Git & Commits
- Never `git add`, `git commit`, or `git push` without an explicit request.
- Never include `Co-Authored-By` lines in commit messages.
- Always show the proposed commit message and wait for approval before committing.
- Always ask for explicit approval before pushing to remote.

### Formatting Conventions
- Write a full calendar year as e.g. `2015`, not `2015-2016`. A test window labelled "2015" means 2015-01-01 → 2016-01-01.

### Memory System
- Do not save project-specific information to the Claude memory system. Put it here in CLAUDE.md instead so it travels with the repo and is always loaded.

---

## Project Context

**Strategy:** Mean reversion on U.S. equities. Ridge regression predicts 5-day forward returns. 20-position long-only portfolio via cvxpy (8% max weight per position). Regime gate (VIX + SPY 200-day MA + momentum z-score) blocks new trades — on regime close, losing positions are liquidated and winning positions get a tightened 5% trailing stop. Per-position 10% trailing stop-loss active at all times. All three regime gates use AND logic — all must pass.

**Universe:** S&P 500 current constituents (~501 tickers) via `PointInTimeUniverse`. BRK.B and BF.B always fail to download — expected, ignore those warnings.

**Active model:** `model_mr_zscore_12feat` — 12 features defined in `src/trading_system/features/mean_reversion.py::FEATURE_COLS`.

**Walk-forward:** 11 windows, 4yr train / 1yr val / 1yr test, 5-day purge, 10% embargo. Window dates driven by `config/backtest.yaml`.

**Key constraint:** Point-in-time correctness everywhere. No lookahead bias. All features use `shift(1)`.

**Environment:** Python 3.12 venv at `.env/`. Activate with `source .env/bin/activate`. Run tests with `.env/bin/pytest tests/ -v`.

**Data:** Not in git. Price cache is in `data/raw/yfinance/` (per-ticker parquet) and `data/cache/` (merged parquet keyed by date range). Run the download command below to populate or refresh.

---

## CLI Reference

All commands run from repo root with the venv active:

```bash
# Download / refresh price data
python -m trading_system.cli download --start 2010-01-01 --end <YYYY-MM-DD> --force

# Walk-forward training (all 11 windows)
python -m trading_system.cli train --verbose

# Backtest a single window (1–11)
python -m trading_system.cli backtest --window 3

# Train production model on full history (Ridge only, no val set)
python -m trading_system.cli production-train

# Daily brief (generates + saves to data/production/daily_brief.md)
python production/daily_run.py --refresh

# Confirm a rebalance was executed
python production/daily_run.py --confirm-trade
```

**Daily workflow:**
1. `python production/daily_run.py --refresh` — downloads any missing price data incrementally, then generates brief
2. Review brief, place trades in IB TWS if regime open and rebalance day
3. `python production/daily_run.py --confirm-trade` — after executing trades

---

## Model Artefacts

| Path | Contents |
|------|----------|
| `data/models/mean_reversion/model_mr_zscore_12feat_ridge/window_N/model.pkl` | Ridge model (pickle) |
| `data/models/mean_reversion/model_mr_zscore_12feat_ridge/window_N/metadata.json` | Trainer metadata (train/val dates, val_ic) |
| `data/models/mean_reversion/model_mr_zscore_12feat_ridge/production/` | Production model, no val_ic (trained on all data) |
| `data/results/backtests/window_N_<timestamp>.json` | Backtest results per window |

---

## Production Pipeline

| File | Purpose |
|------|---------|
| `production/daily_run.py` | Daily brief generator — signals, regime, portfolio |
| `production/portfolio_state.py` | Position tracking and P&L |
| `data/production/daily_brief.md` | Latest brief (overwritten daily) |
| `data/production/brief_log.csv` | Historical log — one row per day |
| `data/production/positions.csv` | Current holdings (edit after executing trades) |
| `data/production/rebalance_state.json` | Last rebalance date and frequency |

---

## Development Workflow

### Before any code change
1. Read the relevant file(s) first
2. Check if tests exist for the area you're touching
3. Present the plan and wait for approval

### After any code change
- Run `pytest tests/ -v` and report the actual output
- Run `ruff check src/ tests/` and fix any errors before marking done

### Keep Documentation in Sync
After any change that affects behaviour, interfaces, config, or file formats:
- Update **README.md** if the change affects CLI usage, results, or repository structure
- Update **SETUP.md** if the change affects installation, config values, or data schemas
- Update **CLAUDE.md** if the change affects project context, workflow, or artefact paths
- Update **`.session_context.md`** (when asked) with what was changed and what's next

Minor internal bug fixes (no interface change) do not require doc updates. When in doubt, flag it.

### Never do without asking first
- Delete files or directories
- Modify `config/production.yaml` (live trading config)
- Push to remote
- Change `data/production/positions.csv`
- Alter walk-forward window definitions in `config/backtest.yaml`

---

## Session Context (`.session_context.md`)

`.session_context.md` is a persistent, gitignored file loaded at every session start. It tracks work state across sessions.

**When asked to update it**, rewrite only the relevant section(s) — do not wipe the whole file:

- **Current Task / WIP** — what is actively being worked on right now
- **To-Do** — prioritized list of next steps (use `- [ ]` checkboxes)
- **Completed** — append dated bullet(s) for anything finished this session (format: `- YYYY-MM-DD: what was done`)
- **Notes** — any important context, decisions, or constraints that don't fit elsewhere

**Rules:**
- Only update when explicitly asked
- Never fabricate entries — only log things that actually happened
- Do not reset or clear old Completed entries — accumulate them

---

## What Good Looks Like

**Good:**
> "I need to clarify two things before proceeding: (1) should the new feature use the same winsorization as the existing ones, and (2) which config file controls the feature list? Once confirmed, my plan is: add the feature to `src/trading_system/features/mean_reversion.py`, add it to `FEATURE_COLS`, and add a test in `tests/unit/test_features/test_mean_reversion.py`."

**Bad:**
> "I'll add the feature now." *(no plan, no questions)*
> "The val_ic for window 7 is around 0.03." *(fabricated — not read from a file)*
> "The function probably takes a `lookback` parameter." *(guessing without reading)*
