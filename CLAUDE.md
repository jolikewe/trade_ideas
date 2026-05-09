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
- If you don't know something, say so directly. Do not fill gaps with plausible-sounding details.
- Never invent config values, file paths, function signatures, or results you haven't verified by reading the actual file.
- If a file doesn't exist yet, say it doesn't exist. Don't describe it as if it does.
- If asked about backtest results, training metrics, or any numbers — only report what is in actual files on disk. Do not estimate or extrapolate.
- If you're unsure whether a module, flag, or behavior exists, grep for it or read the file before answering.

### Clarify Ambiguity Before Acting
If the request is unclear, incomplete, or could be interpreted multiple ways:
- Ask for clarification before doing anything
- List the specific ambiguities — don't ask open-ended questions
- Propose your default interpretation and confirm it before using it

---

## Project Context

**Strategy:** Mean reversion on U.S. equities. ML ensemble (Ridge + LightGBM) predicts 5-day forward returns. Factor-neutral portfolio via cvxpy. Regime gate (VIX + SPY 200-day MA + momentum z-score) blocks trading in bad markets.

**Active model:** `model_mr_zscore_12feat` — 12 features, see `config/features/mean_reversion/model_mr_zscore_12feat.yaml`

**Key constraint:** Point-in-time correctness everywhere. No lookahead bias. Use `filing_date` not `period_end_date` for fundamentals. All features use `shift(1)`.

**Environment:** Python 3.12 venv at `.env/`. Activate with `source .env/bin/activate`. Run tests with `.env/bin/pytest tests/ -v`.

**Data:** Not in git. Must be downloaded via `scripts/download_yfinance_data.py`. Do not assume data files exist unless you verify with `ls` or `find`.

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
- Update **README.md** if the change affects CLI usage, results, or the repository structure section
- Update **SETUP.md** if the change affects installation, config values, data schemas, or infrastructure
- Update **`.session_context.md`** (when asked) with what was changed and what's next

Minor internal bug fixes (no interface change) do not require doc updates. When in doubt, flag it: "Does this need a doc update?"

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
- Only update it when explicitly asked ("update session context", "log that", "mark that done", etc.)
- Never fabricate entries — only log things that actually happened in the session
- Keep Completed entries short and factual
- Do not reset or clear old Completed entries — accumulate them

---

## What Good Looks Like

**Good:**
> "I need to clarify two things before proceeding: (1) should the new feature use the same winsorization as the existing ones, and (2) which config file controls the feature list? Once confirmed, my plan is: add the feature to `src/trading_system/features/mean_reversion.py`, add it to `FEATURE_COLS`, and add a test in `tests/unit/test_features/test_mean_reversion.py`."

**Bad:**
> "I'll add the feature now." *(no plan, no questions)*
> "The val_ic for window 7 is around 0.03." *(fabricated — not read from a file)*
> "The function probably takes a `lookback` parameter." *(guessing without reading)*
