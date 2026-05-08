from abc import ABC, abstractmethod
import pandas as pd

class BaseFeature(ABC):
    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        ...

class FeatureSet(ABC):
    @abstractmethod
    def compute_all(self, prices: pd.DataFrame) -> pd.DataFrame:
        ...
