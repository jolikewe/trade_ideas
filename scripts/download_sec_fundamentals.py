#!/usr/bin/env python3
"""Batch download SEC EDGAR company facts for all universe tickers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    from trading_system.universe.historical import PointInTimeUniverse
    from trading_system.factors.sec_fundamentals import SECEdgarLoader
    import time

    pit = PointInTimeUniverse.load_or_build()
    loader = SECEdgarLoader()
    tickers = pit.all_tickers
    print(f"Downloading SEC facts for {len(tickers)} tickers (this takes hours)...")
    for i, ticker in enumerate(tickers):
        try:
            loader.load_company_facts(ticker)
            if i % 50 == 0:
                print(f"  {i}/{len(tickers)} done")
        except Exception as e:
            print(f"  Skip {ticker}: {e}")

if __name__ == "__main__":
    main()
