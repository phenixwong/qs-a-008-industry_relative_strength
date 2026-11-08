import pandas as pd
import pytest

from qs_a008.charts import (
    normalize_rs_line,
    render_industry_rs_panel,
    render_market_rs_grid,
    render_momentum_quadrant,
    render_rotation_heatmap,
)


def test_normalize_rs_line_starts_at_100():
    result = normalize_rs_line(pd.Series([20.0, 22.0, 18.0]))
    assert result.iloc[0] == 100.0
    assert result.iloc[1] == pytest.approx(110.0)
    assert result.iloc[2] == pytest.approx(90.0)


def test_rotation_heatmap_contains_rows_and_dates():
    matrix = pd.DataFrame(
        [[10.0, 50.0], [90.0, 70.0]],
        index=["银行", "通信"],
        columns=["2026-10-29", "2026-10-30"],
    )
    svg = render_rotation_heatmap(matrix)
    assert "银行" in svg and "通信" in svg
    assert "2026-10-30" in svg
    assert "rect" in svg


def test_momentum_quadrant_contains_labels_and_zero_axes():
    frame = pd.DataFrame(
        {"name": ["银行", "上证指数"], "excess20d": [0.1, -0.1], "excess60d": [0.2, -0.2], "group": ["industry", "market"], "state": ["强", "弱"]}
    )
    svg = render_momentum_quadrant(frame)
    assert "银行" in svg and "上证指数" in svg
    assert "stroke-dasharray" in svg
    assert "fill='#d23f2e'" in svg
    assert "fill='#2e8b57'" in svg


def test_momentum_quadrant_colors_by_excess_sign_not_state():
    frame = pd.DataFrame(
        {
            "name": ["双正", "双负", "分化", "沪深300"],
            "excess20d": [0.1, -0.1, 0.1, 0.0],
            "excess60d": [0.2, -0.2, -0.2, 0.0],
            "group": ["industry", "industry", "industry", "market"],
            "state": ["弱", "强", "强", "基准"],
        }
    )
    svg = render_momentum_quadrant(frame)
    assert "双正" in svg and "fill='#d23f2e'" in svg
    assert "双负" in svg and "fill='#2e8b57'" in svg
    assert "分化" in svg and "fill='#9aa5b1'" in svg


def test_heatmap_color_scale_runs_from_green_weak_to_red_strong():
    matrix = pd.DataFrame([[0.0], [100.0]], index=["弱", "强"], columns=["2026-10-30"])
    svg = render_rotation_heatmap(matrix)
    assert "fill='#006837'" in svg
    assert "fill='#a50026'" in svg


def test_industry_rs_lines_share_one_plot_and_include_benchmark_panel():
    dates = pd.date_range("2026-01-01", periods=3)
    lines = pd.DataFrame({"银行": [100, 101, 102], "通信": [100, 99, 98]}, index=dates)
    benchmark = pd.Series([100, 101, 100], index=dates)
    svg = render_industry_rs_panel(lines, benchmark)
    assert svg.count("<polyline") >= 3
    assert "RS线（起点=100" in svg
    assert "沪深300走势" in svg


def test_market_rs_grid_renders_index_and_ma_traces():
    dates = pd.date_range("2026-01-01", periods=25)
    sample = {"name": "上证50", "state": "强", "rs": pd.Series(range(100, 125), index=dates)}
    svg = render_market_rs_grid([sample], "1/2")
    assert "上证50" in svg
    assert svg.count("<polyline") >= 2
    assert "100" in svg
