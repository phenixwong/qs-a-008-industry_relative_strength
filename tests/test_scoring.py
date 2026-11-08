import pandas as pd
import pytest

from qs_a008.scoring import cross_sectional_percentile, score_indicators


def test_cross_sectional_percentile_centers_group_average_at_50():
    result = cross_sectional_percentile(pd.Series([0.0, 1.0, 2.0]))
    assert result.iloc[1] == pytest.approx(50.0)
    assert result.iloc[0] < 50 < result.iloc[2]


def test_score_indicators_returns_report_columns():
    data = pd.DataFrame(
        {
            "name": ["银行", "通信"],
            "excess20d": [0.10, -0.10],
            "excess60d": [0.20, -0.20],
            "excess120d": [0.30, -0.30],
        }
    )
    result = score_indicators(data)
    assert {"raw_score", "rs_score"}.issubset(result.columns)
    assert result.loc[result["name"] == "银行", "rs_score"].iloc[0] > 50
    assert result.loc[result["name"] == "通信", "rs_score"].iloc[0] < 50

