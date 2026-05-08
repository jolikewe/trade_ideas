import pandas as pd
import numpy as np

def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

def cross_section_winsorize(df: pd.DataFrame, cols: list[str],
                             lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = df.groupby("date")[col].transform(lambda s: winsorize(s, lower, upper))
    return df

def cross_section_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = df.groupby("date")[col].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-8)
        )
    return df

def rank_normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = df.groupby("date")[col].transform(
            lambda s: s.rank(pct=True) * 2 - 1
        )
    return df

def fill_cross_section_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = df.groupby("date")[col].transform(lambda s: s.fillna(s.median()))
    return df
