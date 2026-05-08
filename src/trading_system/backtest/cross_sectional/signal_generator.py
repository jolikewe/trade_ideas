import numpy as np
import pandas as pd

class ICWeightedSignalGenerator:
    def __init__(self, min_ic_threshold: float = 0.02):
        self.min_ic_threshold = min_ic_threshold

    def generate(self, model_scores: dict[str, pd.Series],
                 val_ics: dict[str, float]) -> pd.Series:
        included = {m: ic for m, ic in val_ics.items() if ic > 0}
        if not included:
            included = val_ics
        total_ic = sum(included.values())
        if total_ic == 0:
            return list(model_scores.values())[0]
        ensemble = sum(
            model_scores[m] * (ic / total_ic)
            for m, ic in included.items()
            if m in model_scores
        )
        return ensemble
