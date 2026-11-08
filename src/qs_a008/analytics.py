from __future__ import annotations


import pandas as pd

from .rs import (
    calculate_excess_return,
    calculate_relative_strength,
    calculate_state,
    calculate_state_duration,
    detect_transition_signal,
)
from .scoring import calculate_weighted_score, cross_sectional_percentile


def build_daily_metrics(prices: pd.DataFrame, benchmark_name: str) -> pd.DataFrame:
    """Calculate daily relative-strength indicators for each asset and peer group."""
    required = {"trade_date", "name", "group", "close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"价格数据缺少字段: {sorted(missing)}")
    data = prices.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["trade_date", "name", "close"])
    if data.duplicated(["trade_date", "name"]).any():
        raise ValueError("价格数据存在重复的交易日和指数名称")
    wide = data.pivot(index="trade_date", columns="name", values="close").sort_index()
    if benchmark_name not in wide:
        raise ValueError(f"缺少基准指数价格: {benchmark_name}")
    benchmark = wide[benchmark_name]
    groups = data.drop_duplicates("name").set_index("name")["group"].to_dict()
    frames = []
    for name in wide.columns:
        if name == benchmark_name:
            continue
        close = wide[name]
        rs = calculate_relative_strength(close, benchmark)
        features = pd.DataFrame(index=wide.index)
        features["return1d"] = close.pct_change()
        features["excess1d"] = rs.pct_change()
        features["excess20d"] = calculate_excess_return(rs, 20)
        features["excess60d"] = calculate_excess_return(rs, 60)
        features["excess120d"] = calculate_excess_return(rs, 120)
        features["raw_score"] = calculate_weighted_score(features)
        features["rs_line"] = rs
        features["ma20"] = rs.rolling(20, min_periods=20).mean()
        features["state"] = calculate_state(rs, features["ma20"])
        features["prev_state"] = features["state"].shift(1)
        features["duration"] = calculate_state_duration(features["state"])
        features["signal"] = detect_transition_signal(features["state"])
        features["momentum40"] = close.div(close.shift(40)).sub(1.0)
        features["name"] = name
        features["group"] = groups[name]
        features.index.name = "trade_date"
        frames.append(features.reset_index())

    # Include the benchmark in the combined heatmap universe and momentum view.
    benchmark_frame = pd.DataFrame({
        "trade_date": wide.index,
        "name": benchmark_name,
        "group": "benchmark",
        "return1d": benchmark.pct_change(),
        "excess1d": 0.0,
        "excess20d": 0.0,
        "excess60d": 0.0,
        "excess120d": 0.0,
        "raw_score": 0.0,
        "rs_line": 100.0,
        "ma20": 100.0,
        "state": "弱",
        "prev_state": "基准",
        "duration": 1,
        "signal": "",
        "momentum40": 0.0,
    })
    result = pd.concat([*frames, benchmark_frame], ignore_index=True)
    scored = []
    for group, subset in result[result.group.isin(["industry", "market"])].groupby("group", sort=False):
        subset = subset.sort_values(["name", "trade_date"]).copy()
        subset["rs_score"] = subset.groupby("trade_date", sort=False)["raw_score"].transform(
            lambda values: cross_sectional_percentile(values)
        )
        subset["score_delta"] = subset.groupby("name", sort=False)["rs_score"].diff()
        scored.append(subset)
    result = pd.concat([*scored, benchmark_frame.assign(rs_score=50.0, score_delta=0.0)], ignore_index=True)
    return result.sort_values(["trade_date", "group", "name"]).reset_index(drop=True)


def build_heatmap_matrix(
    metrics: pd.DataFrame, names: list[str], periods: int = 120
) -> pd.DataFrame:
    """Score the selected industry/benchmark set together for each trading day."""
    subset = metrics[metrics.name.isin(names)].copy()
    raw = subset.pivot(index="trade_date", columns="name", values="raw_score").reindex(columns=names)
    raw = raw.dropna(how="any").tail(periods)
    scored = raw.apply(cross_sectional_percentile, axis=1)
    latest = scored.iloc[-1].sort_values(ascending=False, kind="stable").index
    return scored.loc[:, latest].T
