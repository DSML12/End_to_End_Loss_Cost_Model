"""drift.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

_EPS = 1e-6  


def bin_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, qs))
    edges[0], edges[-1] = -np.inf, np.inf  # catch out-of-range values
    return edges


def numeric_distribution(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(x, bins=edges)[0].astype(float)
    return (counts + _EPS) / (counts.sum() + _EPS * len(counts))


def categorical_distribution(x: pd.Series, categories: list) -> np.ndarray:
    counts = x.value_counts()
    raw = np.array([counts.get(c, 0.0) for c in categories], dtype=float)
    return (raw + _EPS) / (raw.sum() + _EPS * len(categories))


def psi_from_distributions(r: np.ndarray, a: np.ndarray) -> float:
    return float(np.sum((a - r) * np.log(a / r)))


def psi(reference: np.ndarray, actual: np.ndarray, edges: np.ndarray) -> float:
    r = numeric_distribution(reference, edges)
    a = numeric_distribution(actual, edges)
    return psi_from_distributions(r, a)


def psi_categorical(reference: pd.Series, actual: pd.Series) -> float:
    categories = reference.astype("category").cat.categories.tolist()
    r = categorical_distribution(reference, categories)
    a = categorical_distribution(actual, categories)
    return psi_from_distributions(r, a)


def band(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "material"


def csi_by_feature(
    ref: pd.DataFrame, act: pd.DataFrame, features: list[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in features:
        if is_numeric_dtype(ref[f]):
            edges = bin_edges(ref[f].to_numpy())
            out[f] = psi(ref[f].to_numpy(), act[f].to_numpy(), edges)
        else:
            out[f] = psi_categorical(ref[f], act[f])
    return out
