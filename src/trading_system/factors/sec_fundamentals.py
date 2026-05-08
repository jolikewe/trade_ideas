import json
import pandas as pd
import requests
from pathlib import Path

class SECEdgarLoader:
    BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"
    HEADERS = {"User-Agent": "trading-system user@example.com"}

    def __init__(self, cache_dir: str = "data/raw/sec_edgar",
                 rate_limit: int = 10, cache_ttl_days: int = 90):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._delay = 1.0 / rate_limit

    def load_company_facts(self, cik: str) -> dict:
        path = self.cache_dir / f"CIK{cik.zfill(10)}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        import time
        time.sleep(self._delay)
        resp = requests.get(f"{self.BASE_URL}/CIK{cik.zfill(10)}.json",
                            headers=self.HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        with open(path, "w") as f:
            json.dump(data, f)
        return data
