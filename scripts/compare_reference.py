from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from qs_a008.analytics import build_daily_metrics, build_heatmap_matrix
from qs_a008.universe import BENCHMARK, INDUSTRIES, MARKET_INDICES


DEFAULT_REFERENCE = Path("/Users/huangzhicheng/个人系统/00_收件箱/rs_report_20261030_from_cai.html")
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "generated"
RAW = ROOT / "data" / "raw"


def _plot_payload(source: str, plot_id: str) -> tuple[list, dict]:
    match = re.search(r"Plotly\.newPlot\(\s*\"" + re.escape(plot_id) + r"\",", source)
    if match is None:
        raise ValueError(f"原报告缺少图表: {plot_id}")
    decoder = json.JSONDecoder()
    data_start = source.index("[", match.end())
    traces, used = decoder.raw_decode(source[data_start:])
    layout_start = data_start + used
    while source[layout_start].isspace() or source[layout_start] == ",":
        layout_start += 1
    layout, _ = decoder.raw_decode(source[layout_start:])
    return traces, layout


def _array(value) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value)
    if not isinstance(value, dict) or "bdata" not in value:
        return np.asarray(value)
    types = {"f8": "<f8", "f4": "<f4", "i4": "<i4", "i2": "<i2", "i1": "i1", "u1": "u1"}
    dtype = np.dtype(types[value["dtype"]])
    array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=dtype)
    shape = value.get("shape")
    if shape:
        shape = tuple(map(int, shape.split(","))) if isinstance(shape, str) else tuple(shape)
        array = array.reshape(shape)
    return array


def _assert_series(name: str, actual: np.ndarray, expected: np.ndarray, checks: list[str]) -> None:
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-10, equal_nan=True):
        difference = np.nanmax(np.abs(actual - expected)) if actual.shape == expected.shape else "shape differs"
        raise AssertionError(f"{name}: 原图序列不一致，最大偏差={difference}")
    checks.append(name)


def compare(reference: Path) -> list[str]:
    source = reference.read_text(encoding="utf-8")
    tables = pd.read_html(str(reference))
    current = pd.read_csv(REPORTS / "latest_metrics.csv")
    daily = pd.read_csv(REPORTS / "daily_metrics.csv", parse_dates=["trade_date"])
    universe = [BENCHMARK, *INDUSTRIES, *MARKET_INDICES]
    raw_frames = [
        pd.read_csv(RAW / f"{item['ts_code'].replace('.', '_')}.csv", parse_dates=["trade_date"])
        for item in universe
    ]
    raw = pd.concat(raw_frames, ignore_index=True)
    checks: list[str] = []

    for table_idx, group, labels in [(0, "industry", [item["name"] for item in INDUSTRIES]), (1, "market", [item["name"] for item in MARKET_INDICES])]:
        reference_table = tables[table_idx]
        result = current[current.group == group].set_index("name")
        ordered = result.sort_values("rs_score", ascending=False, kind="stable").index.tolist()
        if ordered != reference_table["名称"].tolist():
            raise AssertionError(f"{group}: 排名顺序与原报告不一致")
        checks.append(f"{group} rank/order ({len(labels)})")
        fields = {
            "RS分": ("rs_score", 0), "分Δ": ("score_delta", 1),
            "超额20d": ("excess20d", 1), "超额60d": ("excess60d", 1),
            "超额120d": ("excess120d", 1),
        }
        if group == "industry":
            fields["动量40d(≥3%)"] = ("momentum40", 1)
        for _, ref_row in reference_table.iterrows():
            row = result.loc[ref_row["名称"]]
            for field, (column, digits) in fields.items():
                value = str(ref_row[field]).replace("✓", "").replace("%", "").replace("+", "").strip()
                expected = float(value)
                actual = float(row[column])
                if field.startswith("超额") or field == "动量40d(≥3%)":
                    actual *= 100.0
                if round(expected, digits) != round(actual, digits):
                    raise AssertionError(f"{group} {ref_row['名称']} {field}: 原版={expected}, 复算={actual}")
            for field in ("状态", "持续", "今日信号"):
                expected = str(ref_row[field]).strip()
                actual = str(row[{"状态": "state", "持续": "duration", "今日信号": "signal"}[field]])
                if field == "今日信号":
                    expected = expected.removeprefix("↓").removeprefix("↑")
                if expected not in {"nan", ""} and expected != actual:
                    raise AssertionError(f"{group} {ref_row['名称']} {field}: 原版={expected}, 复算={actual}")
        checks.append(f"{group} displayed indicators")

    # Heatmap: all 38 rows and all 120 dates, including one joint industry/core-index ranking.
    heat_traces, _ = _plot_payload(source, "heat")
    heat_trace = heat_traces[0]
    names = [str(name).split(" ", 1)[1] for name in heat_trace["y"]]
    data_names = [BENCHMARK["name"] if name == "沪深300·基准" else name for name in names]
    dates = pd.to_datetime(heat_trace["x"])
    heat = build_heatmap_matrix(daily, data_names, periods=120).reindex(index=data_names)
    heat_values = heat.reindex(columns=dates).to_numpy(dtype=float)
    _assert_series("heatmap 38x120", _array(heat_trace["z"]), heat_values, checks)

    # Quadrant data: exact 20/60-day excess coordinates for 31 industries + six core indices + benchmark.
    quad_traces, _ = _plot_payload(source, "quad")
    current_date = daily.trade_date.max()
    today = daily[daily.trade_date == current_date].set_index("name")
    for trace in quad_traces:
        for idx, label in enumerate(trace["text"]):
            name = str(label).removeprefix("◆")
            if name == "沪深300·基准":
                name = BENCHMARK["name"]
            row = today.loc[name]
            _assert_series(f"quadrant {name}", np.asarray([_array(trace["x"])[idx], _array(trace["y"])[idx]]), np.asarray([row.excess60d * 100, row.excess20d * 100]), checks)

    prices = raw.pivot(index="trade_date", columns="name", values="close").sort_index()
    benchmark = prices[BENCHMARK["name"]]

    # Industry panel: each selected series is normalized on the shared 250-session window.
    rs_traces, _ = _plot_payload(source, "rs_tb")
    industry_names = [trace["name"] for trace in rs_traces if trace.get("name") and not trace["name"].endswith("·MA") and trace["name"] != "沪深300(基准)"]
    industry_dates = pd.to_datetime(next(trace["x"] for trace in rs_traces if trace.get("name") in industry_names))
    for trace in rs_traces:
        name = trace.get("name", "")
        if not name or name == "沪深300(基准)":
            continue
        is_ma = name.endswith("·MA")
        base_name = name.removesuffix("·MA")
        if base_name not in industry_names:
            continue
        trace_dates = pd.to_datetime(trace["x"])
        if is_ma:
            full_relative = prices[base_name].div(benchmark)
            scale = full_relative.reindex(industry_dates).iloc[0]
            expected = full_relative.div(scale).mul(100).rolling(20, min_periods=20).mean().reindex(trace_dates)
            _assert_series(f"industry MA20 {base_name}", _array(trace["y"]), expected.to_numpy(), checks)
            continue
        relative = prices[name].div(benchmark)
        scale = relative.reindex(industry_dates).iloc[0]
        normalized = relative.div(scale).mul(100).reindex(trace_dates)
        _assert_series(f"industry RS {name}", _array(trace["y"]), normalized.to_numpy(), checks)
    base_trace = next(trace for trace in rs_traces if trace.get("name") == "沪深300(基准)")
    base = benchmark.reindex(pd.to_datetime(base_trace["x"]))
    _assert_series("industry-panel benchmark", _array(base_trace["y"]), base.div(base.iloc[0]).mul(100).to_numpy(), checks)

    # Index RS grids: exact 250-session ratio, MA20, and 100 baseline against the reference traces.
    expected_market_order = tables[1]["名称"].tolist()
    for plot_id, members, has_benchmark in [("mkt_grid", expected_market_order[:8], True), ("mkt_grid2", expected_market_order[8:], False)]:
        traces, layout = _plot_payload(source, plot_id)
        if has_benchmark:
            dates = pd.to_datetime(traces[0]["x"])
            base = benchmark.reindex(dates)
            _assert_series("market-grid benchmark", _array(traces[0]["y"]), base.div(base.iloc[0]).mul(100).to_numpy(), checks)
            first_pair = 1
        else:
            first_pair = 0
        for offset, name in enumerate(members):
            line, ma = traces[first_pair + offset * 2:first_pair + offset * 2 + 2]
            dates = pd.to_datetime(line["x"])
            full_relative = prices[name].div(benchmark)
            base = full_relative.reindex(dates).iloc[0]
            normalized_full = full_relative.div(base).mul(100)
            normalized = normalized_full.reindex(dates)
            _assert_series(f"market RS {name}", _array(line["y"]), normalized.to_numpy(), checks)
            expected_ma = normalized_full.rolling(20, min_periods=20).mean().reindex(dates)
            _assert_series(f"market MA20 {name}", _array(ma["y"]), expected_ma.to_numpy(), checks)

    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", nargs="?", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    checks = compare(args.reference)
    print(f"PASS: {len(checks)} reference checks")
    for item in checks:
        print(f"  {item}")


if __name__ == "__main__":
    main()
