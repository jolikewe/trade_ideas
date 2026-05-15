import contextlib
import io
import pandas as pd
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

_SKIP_TICKERS = {"BRK.B", "BF.B"}  # always fail on yfinance free tier

class YFinanceLoader:
    def __init__(self, cache_dir: str = "data/raw/yfinance", threads: int = 10,
                 auto_adjust: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.threads = threads
        self.auto_adjust = auto_adjust

    def _ticker_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.parquet"

    def load_ticker(self, ticker: str, start: str, end: str,
                    force_refresh: bool = False) -> pd.DataFrame | None:
        path = self._ticker_path(ticker)
        existing = None
        if path.exists() and not force_refresh:
            existing = pd.read_parquet(path)
            if existing.empty:
                return None
            last_date = pd.to_datetime(existing["date"]).max()
            last_bday = pd.Timestamp.now().normalize() - pd.offsets.BDay(1)
            if last_date >= last_bday:
                return existing  # already current
            # incremental: only fetch missing range
            start = str((last_date + pd.Timedelta(days=1)).date())

        try:
            _sink = io.StringIO()
            with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
                raw = yf.download(ticker, start=start, end=end,
                                  auto_adjust=self.auto_adjust, progress=False)
            if raw.empty:
                return existing  # no new data, return what we have
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [price.lower() for price, _ in raw.columns]
            else:
                raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
            raw = raw.reset_index()
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
            raw["ticker"] = ticker
            if existing is not None:
                raw = pd.concat([existing, raw], ignore_index=True).drop_duplicates("date")
            raw.to_parquet(path, index=False)
            return raw
        except Exception:
            return existing

    def load_universe(self, tickers: list[str], start: str, end: str,
                      force_refresh: bool = False,
                      show_progress: bool = True) -> pd.DataFrame:
        frames = []
        tickers = [t for t in tickers if t not in _SKIP_TICKERS]
        for ticker in tqdm(tickers, desc="Loading prices", disable=not show_progress):
            df = self.load_ticker(ticker, start, end, force_refresh)
            if df is not None:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        return result.sort_values(["ticker", "date"]).reset_index(drop=True)
