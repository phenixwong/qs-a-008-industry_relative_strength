from __future__ import annotations

import os


def get_tushare_token() -> str:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "未找到 TUSHARE_TOKEN，请在当前环境变量或本项目 .env 中配置 Tushare token。"
        )
    return token

