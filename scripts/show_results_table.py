#!/usr/bin/env python3
"""Display cross-window training and backtest results."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare-each", action="store_true")
    args = parser.parse_args()

    results_dir = Path("data/results")
    training = sorted((results_dir / "training").glob("*.json")) if (results_dir / "training").exists() else []
    backtests = sorted((results_dir / "backtests").glob("*.json")) if (results_dir / "backtests").exists() else []

    if not training and not backtests:
        print("No results found. Run training and backtest first.")
        return

    for path in training[-5:]:
        data = json.loads(path.read_text())
        print(f"\n{path.name}")
        for k, v in data.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
