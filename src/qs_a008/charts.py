from __future__ import annotations

import html

import numpy as np
import pandas as pd


def normalize_rs_line(values: pd.Series) -> pd.Series:
    """Normalize an RS series so its first valid observation equals 100."""
    values = values.astype(float)
    first = values.dropna().iloc[0]
    if first == 0:
        raise ValueError("RS线归一化起点不能为0")
    return values.div(first).mul(100.0)


def _heat_color(value: float) -> str:
    stops = [
        (0.0, (0, 104, 55)), (0.1, (26, 152, 80)), (0.2, (102, 189, 99)),
        (0.3, (166, 217, 106)), (0.4, (217, 239, 139)), (0.5, (255, 255, 191)),
        (0.6, (254, 224, 139)), (0.7, (253, 174, 97)), (0.8, (244, 109, 67)),
        (0.9, (215, 48, 39)), (1.0, (165, 0, 38)),
    ]
    position = max(0.0, min(100.0, float(value))) / 100.0
    for (lo, c0), (hi, c1) in zip(stops, stops[1:]):
        if position <= hi:
            ratio = (position - lo) / (hi - lo)
            rgb = tuple(round(a + (b - a) * ratio) for a, b in zip(c0, c1))
            return "#%02x%02x%02x" % rgb
    return "#a50026"


def render_rotation_heatmap(matrix: pd.DataFrame, width: int = 1480) -> str:
    """Render the 120-session, green-weak to red-strong relative-strength heatmap."""
    if matrix.empty:
        return f"<svg viewBox='0 0 {width} 120'></svg>"
    cols = matrix.columns
    left, right, top, cell_h = 120, 105, 32, 18
    cell_w = (width - left - right) / len(cols)
    height = top + len(matrix.index) * cell_h + 56
    parts = [f"<svg viewBox='0 0 {width:.0f} {height:.0f}' class='chart-svg heatmap-svg' role='img'>"]
    for row_idx, (name, row) in enumerate(matrix.iterrows()):
        y = top + row_idx * cell_h
        label = f"{row_idx + 1:02d} {name}"
        parts.append(f"<text class='heat-label' x='{left - 8:.1f}' y='{y + 13:.1f}' text-anchor='end'>{html.escape(label)}</text>")
        for col_idx, value in enumerate(row):
            x = left + col_idx * cell_w
            if pd.isna(value):
                color = "#f0f2f4"
                tooltip = f"{name} {cols[col_idx]}: 缺失"
            else:
                color = _heat_color(value)
                tooltip = f"{name} {cols[col_idx]}: RS分 {float(value):.2f}"
            parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{cell_w + .25:.1f}' height='{cell_h + .25:.1f}' fill='{color}'><title>{html.escape(tooltip)}</title></rect>")
    tick_count = min(14, len(cols))
    for idx in np.linspace(0, len(cols) - 1, tick_count, dtype=int):
        x = left + (idx + .5) * cell_w
        stamp = pd.Timestamp(cols[idx]).strftime("%m-%d")
        parts.append(f"<text class='axis-label' x='{x:.1f}' y='{top + len(matrix.index) * cell_h + 20:.1f}' text-anchor='middle'>{stamp}</text>")
    legend_y = height - 18
    parts.append(f"<defs><linearGradient id='rs-grad' x1='0' x2='1'><stop offset='0%' stop-color='#006837'/><stop offset='50%' stop-color='#ffffbf'/><stop offset='100%' stop-color='#a50026'/></linearGradient></defs><rect x='{width-320}' y='{legend_y-10}' width='230' height='10' fill='url(#rs-grad)'/><text x='{width-326}' y='{legend_y}' text-anchor='end'>弱</text><text x='{width-80}' y='{legend_y}' text-anchor='start'>强</text>")
    parts.append("</svg>")
    return "".join(parts)


def render_momentum_quadrant(frame: pd.DataFrame, width: int = 1480, height: int = 760) -> str:
    """Render an offline SVG excess-20d vs excess-60d quadrant chart."""
    if frame.empty:
        return f"<svg viewBox='0 0 {width} {height}'></svg>"
    core_names = {"沪深300", "上证指数", "中证2000", "中证1000", "中证500", "科创50", "创业板指"}
    frame = frame[(frame.group == "industry") | frame.name.isin(core_names)].copy()
    margin = {"left": 90, "right": 35, "top": 38, "bottom": 74}
    x0, x1 = float(frame.excess60d.abs().max()), float(frame.excess60d.abs().max())
    y0, y1 = float(frame.excess20d.abs().max()), float(frame.excess20d.abs().max())
    x0, x1 = -max(x0 * 1.2, .05), max(x1 * 1.2, .05)
    y0, y1 = -max(y0 * 1.2, .05), max(y1 * 1.2, .05)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    sx = lambda value: margin["left"] + (value - x0) / (x1 - x0) * plot_w
    sy = lambda value: margin["top"] + (y1 - value) / (y1 - y0) * plot_h
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart-svg quadrant-svg' role='img'>"]
    for value in np.linspace(x0, x1, 7):
        x = sx(value)
        parts.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{margin['top']}' y2='{height-margin['bottom']}' stroke='#edf0f2'/><text class='axis-label' x='{x:.1f}' y='{height-margin['bottom']+22}' text-anchor='middle'>{value*100:.0f}</text>")
    for value in np.linspace(y0, y1, 7):
        y = sy(value)
        parts.append(f"<line x1='{margin['left']}' x2='{width-margin['right']}' y1='{y:.1f}' y2='{y:.1f}' stroke='#edf0f2'/><text class='axis-label' x='{margin['left']-12}' y='{y+4:.1f}' text-anchor='end'>{value*100:.0f}</text>")
    parts.append(f"<line x1='{sx(0):.1f}' x2='{sx(0):.1f}' y1='{margin['top']}' y2='{height-margin['bottom']}' stroke='#707983' stroke-dasharray='5 5'/>")
    parts.append(f"<line x1='{margin['left']}' x2='{width-margin['right']}' y1='{sy(0):.1f}' y2='{sy(0):.1f}' stroke='#707983' stroke-dasharray='5 5'/>")
    for _, row in frame.iterrows():
        x, y = sx(float(row.excess60d)), sy(float(row.excess20d))
        is_benchmark = row["name"] == "沪深300"
        state = row.get("state", "")
        excess20, excess60 = float(row.excess20d), float(row.excess60d)
        if is_benchmark or (excess20 >= 0) != (excess60 >= 0):
            color = "#9aa5b1"
        else:
            color = "#d23f2e" if excess20 >= 0 else "#2e8b57"
        if row.get("group") == "industry":
            shape = f"<circle cx='{x:.1f}' cy='{y:.1f}' r='6' fill='{color}' stroke='white' stroke-width='1.5'/>"
        else:
            shape = f"<rect x='{x-5:.1f}' y='{y-5:.1f}' width='10' height='10' fill='{color}' transform='rotate(45 {x:.1f} {y:.1f})' stroke='white' stroke-width='1.2'/>"
        label_name = "沪深300·基准" if is_benchmark else str(row["name"])
        label = f"◆{label_name}" if row.get("group") != "industry" else label_name
        title = f"{label_name} · 状态 {state} · 20日 {excess20:+.1%} · 60日 {excess60:+.1%} · {row.get('signal', '')}"
        parts.append(shape + f"<circle cx='{x:.1f}' cy='{y:.1f}' r='11' fill='transparent'><title>{html.escape(title)}</title></circle><text x='{x+9:.1f}' y='{y-9:.1f}'>{html.escape(label)}</text>")
    parts.append(f"<text x='{width/2:.1f}' y='{height-10}' text-anchor='middle'>超额60d (%)</text><text transform='translate(14 {height/2}) rotate(-90)' text-anchor='middle'>超额20d (%)</text>")
    parts.append("</svg>")
    return "".join(parts)


def _polyline(values: pd.Series, x: float, y: float, width: float, height: float, lo: float, hi: float) -> str:
    values = values.astype(float).dropna()
    if values.empty:
        return ""
    span = hi - lo or 1.0
    n = len(values)
    points = " ".join(
        f"{x + i * width / max(n - 1, 1):.2f},{y + height - (v - lo) / span * height:.2f}"
        for i, v in enumerate(values)
        if pd.notna(v)
    )
    return f"<polyline points='{points}' fill='none'"


def render_industry_rs_panel(
    lines: pd.DataFrame,
    benchmark: pd.Series,
    ma_lines: pd.DataFrame | None = None,
    width: int = 1480,
) -> str:
    """Render ten normalized industry RS lines on shared axes plus benchmark below."""
    top, plot_h, lower_y, lower_h = 40, 390, 490, 145
    left, right = 70, 250
    plot_w = width - left - right
    all_values = pd.concat([lines.stack(), pd.Series([100.0])])
    lo, hi = float(all_values.min()), float(all_values.max())
    pad = max((hi - lo) * .08, 2.0)
    lo, hi = lo - pad, hi + pad
    parts = [f"<svg viewBox='0 0 {width} 670' class='chart-svg industry-rs-svg' role='img'>"]
    parts.append(f"<text x='{left}' y='22'>RS线（起点=100，10条曲线共用同一坐标轴）</text>")
    for v in np.linspace(lo, hi, 6):
        y = top + plot_h - (v - lo) / (hi - lo) * plot_h
        parts.append(f"<line x1='{left}' x2='{width-right}' y1='{y:.1f}' y2='{y:.1f}' stroke='#edf0f2'/><text class='axis-label' x='{left-8}' y='{y+4:.1f}' text-anchor='end'>{v:.0f}</text>")
    y100 = top + plot_h - (100 - lo) / (hi - lo) * plot_h
    parts.append(f"<line x1='{left}' x2='{width-right}' y1='{y100:.1f}' y2='{y100:.1f}' stroke='#9aa5b1' stroke-dasharray='5 5'/>")
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666", "#66c2a5", "#fc8d62"]
    for idx, name in enumerate(lines.columns):
        series = lines[name].dropna()
        color = palette[idx % len(palette)]
        poly = _polyline(series, left, top, plot_w, plot_h, lo, hi)
        if poly:
            ma_values = ma_lines[name].reindex(series.index) if ma_lines is not None else series.rolling(20, min_periods=20).mean()
            ma_points = " ".join(
                f"{left + i * plot_w / max(len(series)-1,1):.2f},{top + plot_h - (v-lo)/(hi-lo)*plot_h:.2f}"
                for i, v in enumerate(ma_values) if pd.notna(v)
            )
            parts.append(poly + f" stroke='{color}' stroke-width='2'/><polyline points='{ma_points}' fill='none' stroke='{color}' stroke-width='1.2' stroke-dasharray='5 4'/>")
            parts.append(f"<text x='{width-right+8}' y='{top + 17 + idx*20}' fill='{color}'>{html.escape(name)}</text>")
    parts.append(f"<text x='{left}' y='{lower_y-12}'>沪深300走势（同窗口归一化至100）</text>")
    b = normalize_rs_line(benchmark.dropna())
    blo, bhi = float(b.min()), float(b.max()); bpad = max((bhi-blo)*.08, 2); blo-=bpad; bhi+=bpad
    bpoly = _polyline(b, left, lower_y, plot_w, lower_h, blo, bhi)
    if bpoly:
        parts.append(bpoly + " stroke='#31577d' stroke-width='2'/>")
    parts.append("</svg>")
    return "".join(parts)


def render_market_rs_grid(items: list[dict], panel: str, width: int = 1480) -> str:
    """Render an eight-index, four-column RS grid with MA20 and the 100 baseline."""
    cols = 4
    rows = 3 if len(items) > 8 else 2
    height = 825 if rows == 3 else 590
    gap_x, gap_y, top, left = 18, 24, 48, 35
    panel_w = (width - left * 2 - gap_x * (cols - 1)) / cols
    panel_h = (height - top - 35 - gap_y * (rows - 1)) / rows
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart-svg market-grid-svg' role='img'><text x='{left}' y='24'>大盘 / 宽基相对RS线 · {html.escape(panel)}（虚线=MA20，灰线=100）</text>"]
    for idx, item in enumerate(items[:12]):
        col, row = idx % cols, idx // cols
        x0, y0 = left + col * (panel_w + gap_x), top + row * (panel_h + gap_y)
        rs = item["rs"].dropna().astype(float)
        ma = item.get("ma")
        ma = ma.reindex(rs.index).dropna() if ma is not None else rs.rolling(20, min_periods=20).mean().dropna()
        vals = pd.concat([rs, ma, pd.Series([100.0])])
        lo, hi = float(vals.min()), float(vals.max()); pad=max((hi-lo)*.12,2);lo-=pad;hi+=pad
        gx, gy, gw, gh = x0+25, y0+24, panel_w-35, panel_h-42
        color = "#31577d" if item.get("state") == "基准" else ("#d23f2e" if item.get("state") == "强" else "#2e8b57")
        parts.append(f"<text x='{x0+4:.1f}' y='{y0+14:.1f}' fill='{color}'>{html.escape(item['name'])}</text>")
        ybase = gy + gh - (100-lo)/(hi-lo)*gh
        parts.append(f"<line x1='{gx}' x2='{gx+gw}' y1='{ybase:.1f}' y2='{ybase:.1f}' stroke='#9aa5b1' stroke-dasharray='3 3'/>")
        line = _polyline(rs, gx, gy, gw, gh, lo, hi)
        if line: parts.append(line + f" stroke='{color}' stroke-width='1.7'/>")
        ma_line = _polyline(ma, gx, gy, gw, gh, lo, hi)
        if ma_line: parts.append(ma_line + f" stroke='{color}' stroke-width='1.1' stroke-dasharray='4 3'/>")
        parts.append(f"<text class='axis-label' x='{gx}' y='{gy+gh+15:.1f}'>100</text><title>{html.escape(item['name'])}: 当前RS {rs.iloc[-1]:.2f}，状态 {item.get('state','')}</title>")
    parts.append("</svg>")
    return "".join(parts)
