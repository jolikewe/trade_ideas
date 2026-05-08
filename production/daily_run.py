#!/usr/bin/env python3
"""Daily brief generator. Auto-runs at Claude session start via .claude/session-start.sh."""
import argparse
import json
from datetime import date, datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force regenerate brief")
    parser.add_argument("--confirm-trade", action="store_true", help="Mark rebalance done")
    args = parser.parse_args()

    brief_path = Path("data/production/daily_brief.md")
    state_path = Path("data/production/rebalance_state.json")
    today = date.today().isoformat()

    if args.confirm_trade:
        state = {}
        if state_path.exists():
            state = json.loads(state_path.read_text())
        state["last_rebalance_date"] = today
        state_path.write_text(json.dumps(state, indent=2))
        print(f"Rebalance confirmed for {today}")
        return

    if not args.refresh and brief_path.exists():
        content = brief_path.read_text()
        if f"date: {today}" in content:
            print(content)
            return

    brief = _generate_brief(today, state_path)
    brief_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.write_text(brief)
    print(brief)

def _generate_brief(today: str, state_path: Path) -> str:
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())

    last_rebal = state.get("last_rebalance_date", "never")
    freq = state.get("frequency_days", 5)

    brief = f"""---
date: {today}
generated: {datetime.now().isoformat()}
---

# Daily Brief — {today}

## Regime Status
_Run `python production/daily_run.py` with live data to get regime status._

## Portfolio Status
- Last rebalance: {last_rebal}
- Rebalance frequency: every {freq} business days

## Positions
_Edit data/production/positions.csv after executing trades._

## Action
- Review positions.csv
- Check regime gates (VIX, SPY 200-day MA, momentum z)
- On rebalance day: execute trades in IB TWS, then run --confirm-trade
"""
    return brief

if __name__ == "__main__":
    main()
