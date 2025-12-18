from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass
class RFParams:
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 20
    max_features: str = "sqrt"
    random_state: int = 42
    n_jobs: int = -1


def fit_random_forest(
    X: np.ndarray, y: np.ndarray, params: RFParams
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        min_samples_split=params.min_samples_split,
        max_features=params.max_features,
        random_state=params.random_state,
        n_jobs=params.n_jobs,
        oob_score=True,
    )
    model.fit(X, y)
    return model


def predict(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
