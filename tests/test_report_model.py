import pandas as pd
import pytest

from qs_a008.analytics import build_daily_metrics, build_heatmap_matrix
from qs_a008.universe import CORE_MARKET_INDICES, MARKET_INDICES


def test_market_mapping_uses_the_index_code_named_shenzhen_1000():
    item = next(item for item in MARKET_INDICES if item["name"] == "深证1000")
    assert item["ts_code"] == "399011.SZ"


def test_daily_metrics_include_delta_signal_and_absolute_40d_momentum():
    dates = pd.bdate_range("2026-01-01", periods=160)
    benchmark = pd.Series(range(1000, 1160), index=dates, dtype=float)
    industry = benchmark * pd.Series([1 + i * 0.0005 for i in range(160)], index=dates)
    prices = pd.DataFrame(
        [(d, "沪深300", "benchmark", benchmark.loc[d]) for d in dates]
        + [(d, "银行", "industry", industry.loc[d]) for d in dates],
        columns=["trade_date", "name", "group", "close"],
    )
    result = build_daily_metrics(prices, "沪深300")
    bank = result[result.name == "银行"].iloc[-1]
    assert bank.momentum40 == pytest.approx(industry.iloc[-1] / industry.iloc[-41] - 1)
    assert "score_delta" in result.columns
    assert bank.state in {"强", "弱"}


def test_daily_metrics_include_single_day_relative_return_and_raw_score():
    dates = pd.bdate_range("2026-01-01", periods=125)
    benchmark = pd.Series([1000.0] * 124 + [1020.0], index=dates)
    industry = pd.Series([100.0] * 124 + [103.0], index=dates)
    prices = pd.DataFrame(
        [(d, "沪深300", "benchmark", benchmark.loc[d]) for d in dates]
        + [(d, "银行", "industry", industry.loc[d]) for d in dates],
        columns=["trade_date", "name", "group", "close"],
    )
    result = build_daily_metrics(prices, "沪深300")
    bank = result[(result.name == "银行")].iloc[-1]
    base = result[(result.name == "沪深300")].iloc[-1]
    assert bank.excess1d == pytest.approx(1.03 / 1.02 - 1)
    assert bank.return1d == pytest.approx(0.03)
    assert bank.raw_score == pytest.approx(
        0.5 * bank.excess20d + 0.3 * bank.excess60d + 0.2 * bank.excess120d
    )
    assert base.excess1d == 0
    assert base.return1d == pytest.approx(0.02)


def test_report_tables_show_raw_score_before_rs_score_and_daily_hint():
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_first_report.py"
    spec = importlib.util.spec_from_file_location("build_first_report", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    frame = pd.DataFrame(
        [{"name": "银行", "group": "industry", "rs_score": 70.0, "raw_score": .1234,
          "score_delta": 1.0, "state": "弱", "duration": 3, "signal": "",
          "excess20d": .02, "excess60d": .01, "excess120d": .03, "momentum40": .04,
          "excess1d": .025, "return1d": .04, "prev_state": "弱"}]
    )
    table = module.render_table(frame)
    assert "<td>+12.34%</td><td>70</td>" in table
    daily = module.render_daily_relative_table(frame)
    assert "+2.50%" in daily
    assert "弱势反弹" in daily
    benchmark = frame.iloc[0].copy()
    benchmark.update({"name": "沪深300", "group": "benchmark", "state": "弱", "prev_state": "基准", "excess1d": 0.0})
    benchmark_html = module.render_daily_relative_table(pd.DataFrame([benchmark]))
    assert "<td>基准</td><td>基准</td><td>基准</td>" in benchmark_html


def test_heatmap_is_combined_cross_section_and_orders_latest_strongest_first():
    dates = pd.bdate_range("2026-01-01", periods=240)
    rows = []
    for name, growth, group in [("银行", 0.002, "industry"), ("通信", -0.002, "industry"), ("沪深300", 0.0, "benchmark")]:
        rows.extend((d, name, group, 100 * (1 + growth) ** i) for i, d in enumerate(dates))
    prices = pd.DataFrame(rows, columns=["trade_date", "name", "group", "close"])
    metrics = build_daily_metrics(prices, "沪深300")
    heat = build_heatmap_matrix(metrics, ["银行", "通信", "沪深300"], periods=120)
    assert list(heat.index) == ["银行", "沪深300", "通信"]
    assert heat.shape == (3, 120)
    assert heat.loc["银行"].iloc[-1] > heat.loc["通信"].iloc[-1]


def test_original_core_indices_are_the_six_named_benchmarks():
    assert [item["name"] for item in CORE_MARKET_INDICES] == [
        "上证指数", "中证2000", "中证1000", "中证500", "科创50", "创业板指"
    ]
