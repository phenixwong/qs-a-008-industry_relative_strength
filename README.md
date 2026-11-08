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

## 自动更新与网站发布

`.github/workflows/publish-report.yml` 在每周一至周五北京时间 22:00 运行，也可在 GitHub Actions 页面手动触发。流程先运行测试，再从 Tushare 获取数据、生成报告并校验行业与宽基数据行数完整，最后将最新报告发布到 GitHub Pages；取数或校验失败时不会覆盖线上页面。

首次启用前，在仓库 **Settings → Secrets and variables → Actions** 添加名为 `TUSHARE_TOKEN` 的 Repository secret；再到 **Settings → Pages** 将 **Build and deployment → Source** 设为 **GitHub Actions**。工作流首次成功部署后，网站地址为 `https://phenixwong.github.io/qs-a-008-industry_relative_strength/`。网站可公开访问，请勿在报告中放入敏感信息。
