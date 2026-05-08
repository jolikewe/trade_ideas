#!/usr/bin/env python3
"""Download OHLCV data for universe tickers via yfinance."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--tickers", default="sp500", choices=["sp500", "sp1500"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from trading_system.universe.historical import PointInTimeUniverse
    from trading_system.data.yfinance_loader import YFinanceLoader

    print(f"Building PIT universe...")
    pit = PointInTimeUniverse.load_or_build()
    tickers = pit.all_tickers
    print(f"Universe: {len(tickers)} tickers")

    loader = YFinanceLoader(threads=10)
    print(f"Downloading {args.start} → {args.end}...")
    df = loader.load_universe(tickers, args.start, args.end, force_refresh=args.force)
    print(f"Done: {len(df)} rows, {df['ticker'].nunique()} tickers")

if __name__ == "__main__":
    main()
