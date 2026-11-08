import numpy as np
import pandas as pd
import pytest

from qs_a008.rs import (
    calculate_excess_return,
    calculate_relative_strength,
    calculate_state,
    calculate_state_duration,
    calculate_weighted_score,
    detect_transition_signal,
)


def test_relative_strength_uses_benchmark_ratio():
    result = calculate_relative_strength(
        pd.Series([110.0, 120.0]), pd.Series([100.0, 100.0])
    )
    np.testing.assert_allclose(result.to_numpy(), [110.0, 120.0])


def test_excess_return_is_change_in_rs_line():
    result = calculate_excess_return(pd.Series([100.0, 110.0, 121.0]), 2)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(0.21)


def test_state_and_duration_track_current_regime():
    rs = pd.Series([99.0, 101.0, 102.0, 98.0])
    ma = pd.Series([100.0, 100.0, 100.0, 100.0])
    state = calculate_state(rs, ma)
    assert state.tolist() == ["弱", "强", "强", "弱"]
    assert calculate_state_duration(state).tolist() == [1, 1, 2, 1]


def test_transition_signal_marks_only_the_switch_day():
    state = pd.Series(["弱", "弱", "强", "强", "弱"])
    assert detect_transition_signal(state).tolist() == ["", "", "新转强", "", "新转弱"]


def test_weighted_score_uses_report_weights():
    frame = pd.DataFrame(
        {"excess20d": [0.10, 0.00], "excess60d": [0.20, 0.00], "excess120d": [0.30, 0.00]}
    )
    result = calculate_weighted_score(frame)
    assert result.iloc[0] > result.iloc[1]
