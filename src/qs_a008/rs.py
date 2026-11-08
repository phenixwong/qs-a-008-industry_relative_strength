from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_relative_strength(target_close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    """Return target close relative to the benchmark, scaled by 100."""
    return target_close.astype(float).div(benchmark_close.astype(float)).mul(100.0)


def calculate_excess_return(rs_line: pd.Series, window: int) -> pd.Series:
    """Return the N-trading-day percentage change of an RS line."""
    return rs_line.astype(float).div(rs_line.shift(window)).sub(1.0)


def calculate_state(rs_line: pd.Series, ma_line: pd.Series) -> pd.Series:
    """Classify an RS line as strong when it is above its moving average."""
    return pd.Series(np.where(rs_line > ma_line, "强", "弱"), index=rs_line.index)


def calculate_state_duration(state: pd.Series) -> pd.Series:
    """Count consecutive days in the current state."""
    groups = state.ne(state.shift()).cumsum()
    return state.groupby(groups, sort=False).cumcount().add(1)


def detect_transition_signal(state: pd.Series) -> pd.Series:
    """Mark the first day of a strong/weak regime transition."""
    previous = state.shift()
    return pd.Series(
        np.select(
            [(previous == "弱") & (state == "强"), (previous == "强") & (state == "弱")],
            ["新转强", "新转弱"],
            default="",
        ),
        index=state.index,
    )


def calculate_weighted_score(frame: pd.DataFrame) -> pd.Series:
    """Calculate the raw weighted excess-momentum score."""
    return (
        frame["excess20d"].astype(float) * 0.5
        + frame["excess60d"].astype(float) * 0.3
        + frame["excess120d"].astype(float) * 0.2
    )

