# QS-A-008 行业相对强弱观测

根据申万行业和宽基指数日线数据，计算相对沪深300的强弱指标并生成离线 HTML 报告。原版指标规格见 `docs/spec.md`。

## 环境

- Python 3.11+
- `TUSHARE_TOKEN` 环境变量
- 安装依赖：`python -m pip install -e '.[dev]'`

## 生成报告

```bash
PYTHONPATH=src python scripts/build_first_report.py
```

固定历史截止日（原版复核）：

```bash
PYTHONPATH=src python scripts/build_first_report.py --end-date 2026-10-30
```

报告和当前指标快照写入 `reports/generated/`；原始日线缓存在 `data/raw/`。

## 测试与原版核验

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider
PYTHONPATH=src python scripts/compare_reference.py
```

比较脚本默认读取收件箱中的 2026-10-30 原始报告，也可以把其他原版 HTML 路径作为参数传入。
