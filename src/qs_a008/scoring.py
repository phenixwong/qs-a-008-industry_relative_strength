from __future__ import annotations

from statistics import NormalDist

import pandas as pd

from .rs import calculate_weighted_score


def cross_sectional_percentile(values: pd.Series) -> pd.Series:
    """Convert a cross-section to the normal CDF of population z-scores."""
    values = values.astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(50.0, index=values.index, dtype=float).where(values.notna())
    z = (values - mean) / std
    return z.map(lambda value: NormalDist().cdf(value) * 100.0 if pd.notna(value) else float("nan"))


def score_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add raw weighted score and 0-100 cross-sectional RS score."""
    result = frame.copy()
    result["raw_score"] = calculate_weighted_score(result)
    result["rs_score"] = cross_sectional_percentile(result["raw_score"])
    return result
