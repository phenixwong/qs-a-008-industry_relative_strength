from __future__ import annotations

import argparse
import html
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from qs_a008.analytics import build_daily_metrics, build_heatmap_matrix
from qs_a008.charts import (
    normalize_rs_line,
    render_industry_rs_panel,
    render_market_rs_grid,
    render_momentum_quadrant,
    render_rotation_heatmap,
)
from qs_a008.config import get_tushare_token
from qs_a008.data import TushareIndexProvider, write_csv
from qs_a008.universe import BENCHMARK, CORE_MARKET_INDICES, INDUSTRIES, MARKET_INDICES


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports" / "generated"


def fetch_universe(start_date: str, end_date: str) -> pd.DataFrame:
    provider = TushareIndexProvider(get_tushare_token())
    items = [BENCHMARK, *INDUSTRIES, *MARKET_INDICES]
    frames: list[pd.DataFrame] = []
    for item in items:
        cache = RAW_DIR / f"{item['ts_code'].replace('.', '_')}.csv"
        try:
            fetcher = provider.fetch_industry_daily if item["group"] == "industry" else provider.fetch_index_daily
            frame = fetcher(item["ts_code"], item["name"], start_date, end_date)
            if frame.empty:
                print(f"WARN empty: {item['name']} {item['ts_code']}")
                continue
            frame["group"] = item["group"]
            write_csv(frame, cache)
            frames.append(frame)
            print(f"OK {item['name']}: {len(frame)} rows")
        except Exception as exc:
            print(f"WARN failed {item['name']} {item['ts_code']}: {exc}")
            if cache.exists():
                frames.append(pd.read_csv(cache, parse_dates=["trade_date"]))
    if not frames:
        raise RuntimeError("没有获取到任何指数数据")
    return pd.concat(frames, ignore_index=True)


def build_metrics(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    metrics = build_daily_metrics(raw, BENCHMARK["name"])
    latest = metrics.trade_date.max()
    current = metrics[metrics.trade_date == latest].copy()
    close_wide = raw.pivot(index="trade_date", columns="name", values="close").sort_index()
    benchmark_close = close_wide[BENCHMARK["name"]]
    series = metrics.pivot(index="trade_date", columns="name", values="rs_line").tail(250)
    return current, series, benchmark_close


def build_heatmap(metrics: pd.DataFrame) -> pd.DataFrame:
    names = [item["name"] for item in INDUSTRIES]
    names += [item["name"] for item in CORE_MARKET_INDICES]
    names += [BENCHMARK["name"]]
    return build_heatmap_matrix(metrics, names, periods=120)


def render_table(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("rs_score", ascending=False).reset_index(drop=True)
    body = []
    for i, row in ordered.iterrows():
        state_cls = "strong" if row.state == "强" else "weak"
        signal = row.signal
        if signal == "新转弱":
            signal = "↓新转弱"
        elif signal == "新转强":
            signal = "↑新转强"
        body.append(
            "<tr>"
            f"<td>{i + 1}</td><td>{html.escape(row['name'])}</td>"
            f"<td>{row.raw_score:+.2%}</td><td>{row.rs_score:.0f}</td><td>{row.score_delta:+.1f}</td>"
            f"<td><span class='pill {state_cls}'>{row.state}</span></td>"
            f"<td>{row.duration}</td><td>{html.escape(signal)}</td>"
            f"<td class='{ 'pos' if row.excess20d >= 0 else 'neg' }'>{row.excess20d:+.1%}</td>"
            f"<td class='{ 'pos' if row.excess60d >= 0 else 'neg' }'>{row.excess60d:+.1%}</td>"
            f"<td class='{ 'pos' if row.excess120d >= 0 else 'neg' }'>{row.excess120d:+.1%}</td>"
            + (f"<td class='{ 'pos' if row.momentum40 >= 0 else 'neg' }'>{row.momentum40:+.1%} {'✓' if row.momentum40 >= .03 else ''}</td>" if row.group == "industry" else "")
            + "</tr>"
        )
    return "".join(body)


def _relative_day_hint(row: pd.Series) -> str:
    if row.group == "benchmark":
        return "基准"
    if row.signal == "新转强":
        return "趋势转强"
    if row.signal == "新转弱":
        return "趋势转弱"
    if row.state == "弱" and row.excess1d > 0:
        return "弱势反弹"
    if row.state == "强" and row.excess1d < 0:
        return "强势回撤"
    if row.excess1d > 0:
        return "相对走强"
    if row.excess1d < 0:
        return "相对走弱"
    return "相对持平"


def render_daily_relative_table(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("excess1d", ascending=False, kind="stable")
    body = []
    for _, row in ordered.iterrows():
        relative_class = "pos" if row.excess1d >= 0 else "neg"
        return_class = "pos" if row.return1d >= 0 else "neg"
        prior_state = "基准" if row.group == "benchmark" else str(row.prev_state)
        current_state = "基准" if row.group == "benchmark" else str(row.state)
        body.append(
            "<tr>"
            f"<td>{html.escape(row['name'])}</td>"
            f"<td class='{return_class}'>{row.return1d:+.2%}</td>"
            f"<td class='{relative_class}'>{row.excess1d:+.2%}</td>"
            f"<td>{html.escape(prior_state)}</td>"
            f"<td>{html.escape(current_state)}</td>"
            f"<td>{html.escape(_relative_day_hint(row))}</td>"
            "</tr>"
        )
    return "".join(body)


def render_report(metrics: pd.DataFrame, series: pd.DataFrame, ma_series: pd.DataFrame, benchmark: pd.Series, heatmap: pd.DataFrame) -> Path:
    latest = benchmark.index.max().date().isoformat()
    industries = metrics[metrics.group == "industry"]
    markets = metrics[metrics.group == "market"]
    strongest = industries.nlargest(5, "rs_score")
    weakest = industries.nsmallest(5, "rs_score")
    quadrant = metrics[["name", "group", "excess20d", "excess60d"]].copy()
    selected_names = pd.concat([strongest, weakest]).name.tolist()
    rs_lines = series[selected_names].copy()
    scales = rs_lines.iloc[0]
    rs_lines = rs_lines.div(scales).mul(100)
    ma_lines = ma_series[selected_names].div(scales).mul(100)
    plot_dates = rs_lines.index
    bench_window = benchmark.reindex(plot_dates).dropna()
    benchmark_normalized = normalize_rs_line(bench_window)
    market_ranked = markets.sort_values("rs_score", ascending=False)
    market_items = []
    for _, row in market_ranked.iterrows():
        raw_rs = series[row["name"]].dropna()
        scale = raw_rs.iloc[0]
        rs = raw_rs.div(scale).mul(100)
        ma = ma_series[row["name"]].reindex(raw_rs.index).div(scale).mul(100)
        market_items.append({"name": row["name"], "state": row["state"], "rs": rs, "ma": ma})
    base_item = {"name": "沪深300·基准", "state": "基准", "rs": benchmark_normalized}
    grid1 = render_market_rs_grid([base_item, *market_items[:8]], "1/2")
    grid2 = render_market_rs_grid(market_items[8:], "2/2")
    new_weak = industries.loc[industries.signal == "新转弱", "name"].tolist()
    new_strong = industries.loc[industries.signal == "新转强", "name"].tolist()
    signal_text = "；".join(
        part for part in [
            f"今日新转强: {'、'.join(new_strong)}" if new_strong else "",
            f"今日新转弱: {'、'.join(new_weak)}" if new_weak else "",
        ] if part
    ) or "今日无状态切换"
    html_doc = f"""<!doctype html><html lang='zh'><head><meta charset='utf-8'>
<title>相对强弱仪表盘 · {latest}</title><style>
body{{margin:0;color:#1f2328;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;background:#fff}}
.wrap{{max-width:1520px;margin:auto;padding:20px}}h1{{margin:0 0 4px}}h2{{margin:24px 0 8px;font-size:19px}}.meta{{color:#57606a;font-size:13px;margin-bottom:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #eaeef2;padding:6px 8px;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#f6f8fa;font-weight:600}}
.pill{{padding:2px 9px;border-radius:11px;font-size:12px}}.strong{{background:#fdecea;color:#d23f2e}}.weak{{background:#e8f4ec;color:#2e8b57}}.pos{{color:#d23f2e}}.neg{{color:#2e8b57}}
.chart-svg{{display:block;width:100%;height:auto;margin:8px 0 18px}}.heatmap-svg{{min-height:650px}}.quadrant-svg{{min-height:600px}}.industry-rs-svg{{min-height:600px}}.market-grid-svg{{min-height:500px}}.axis-label{{font-size:11px;fill:#57606a}}.heat-label{{font-size:12px;fill:#34404a}}
.foot{{color:#8b949e;font-size:12px;line-height:1.7;margin-top:36px}}
</style></head><body><main class='wrap'><h1>相对强弱仪表盘</h1><div class='meta'>数据日期：{latest} · 基准：沪深300 · 首版工程报告</div>
<h2>① 行业相对强弱榜单</h2><div class='meta'>{html.escape(signal_text)}</div>
<table><thead><tr><th>#</th><th>名称</th><th>RS原始分</th><th>RS分</th><th>分Δ</th><th>状态</th><th>持续</th><th>今日信号</th><th>超额20d</th><th>超额60d</th><th>超额120d</th><th>动量40d(≥3%)</th></tr></thead><tbody>{render_table(industries)}</tbody></table>
<h2>② 行业 × 宽基轮动热力图</h2><div class='meta'>近120个交易日；行业与核心宽基合并横截面，红色强、绿色弱</div>{render_rotation_heatmap(heatmap)}
<h2>③ 超额动量象限图</h2>{render_momentum_quadrant(quadrant)}
<h2>④ 最强 / 最弱 5 行业 RS 线（近250交易日）</h2>{render_industry_rs_panel(rs_lines, benchmark_normalized, ma_lines)}
<h2>⑤ 大盘 / 宽基指数相对强弱（近250交易日，按RS分降序）</h2>{grid1}{grid2}
<table><thead><tr><th>#</th><th>名称</th><th>RS原始分</th><th>RS分</th><th>分Δ</th><th>状态</th><th>持续</th><th>今日信号</th><th>超额20d</th><th>超额60d</th><th>超额120d</th></tr></thead><tbody>{render_table(markets)}</tbody></table>
<h2>⑥ 单日相对涨幅 / 反转提示</h2><div class='meta'>按单日相对沪深300涨幅排序；“弱势反弹 / 强势回撤”表示单日方向与当前RS状态相反，不等同于趋势已反转。</div>
<table><thead><tr><th>行业 / 宽基</th><th>指数单日涨幅</th><th>单日相对涨幅</th><th>前日RS状态</th><th>当日RS状态</th><th>提示</th></tr></thead><tbody>{render_daily_relative_table(metrics[(metrics.group == 'industry') | metrics.name.isin([item['name'] for item in CORE_MARKET_INDICES]) | (metrics.group == 'benchmark')])}</tbody></table>
<div class='foot'>口径: RS线 = 指数收盘 / 沪深300 收盘; 状态 = RS线 vs 其 MA20; 超额N日 = RS线 N 日变化;<br>RS原始分 = 0.5·超额20d + 0.3·超额60d + 0.2·超额120d; RS分 = 原始分当日横截面 z 分数的正态分位×100 (连续 0–100, 50=组内平均);<br>单日相对涨幅 = RS线(t) / RS线(t-1) - 1; “弱势反弹 / 强势回撤”仅为单日提示，趋势状态仍由RS线与MA20判定。<br>动量40d = 行业指数绝对收益, ≥3% 与 new_high 策略行业动量门槛 (W_MOM=40, ITHR=0.03) 对齐.<br>数据: a-stock-data-updaters (stock-main-index-data / sw-index-data), 每日收盘后随数据更新重新生成.<br>强/弱配色遵循 A 股习惯 (红强绿弱). 本报告仅为研究工具, 不构成投资建议.</div></main></body></html>"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"rs_report_{latest}.html"
    path.write_text(html_doc, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-date", help="数据截止日，格式 YYYY-MM-DD；默认今天")
    args = parser.parse_args()
    end = date.fromisoformat(args.end_date) if args.end_date else date.today()
    start = end - timedelta(days=500)
    raw = fetch_universe(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    all_metrics = build_daily_metrics(raw, BENCHMARK["name"])
    latest_date = all_metrics.trade_date.max()
    metrics = all_metrics[all_metrics.trade_date == latest_date].copy()
    series = all_metrics.pivot(index="trade_date", columns="name", values="rs_line").tail(250)
    ma_series = all_metrics.pivot(index="trade_date", columns="name", values="ma20").tail(250)
    close_wide = raw.pivot(index="trade_date", columns="name", values="close").sort_index()
    benchmark = close_wide[BENCHMARK["name"]]
    heatmap = build_heatmap(all_metrics)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics[metrics.group != "benchmark"].to_csv(REPORT_DIR / "latest_metrics.csv", index=False, encoding="utf-8-sig")
    all_metrics.to_csv(REPORT_DIR / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    report = render_report(metrics, series, ma_series, benchmark, heatmap)
    print(f"REPORT={report}")


if __name__ == "__main__":
    main()
