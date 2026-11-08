from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["trade_date", "ts_code", "name", "close"]


def normalize_index_daily(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize a Tushare index_daily response into the local schema."""
    missing = {"trade_date", "ts_code", "close"}.difference(raw.columns)
    if missing:
        raise ValueError(f"指数日线缺少字段: {sorted(missing)}")
    result = raw[["trade_date", "ts_code", "close"]].copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"].astype(str), format="%Y%m%d")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result["name"] = name
    result = result.dropna(subset=["trade_date", "close"])
    return result[REQUIRED_COLUMNS].sort_values("trade_date").reset_index(drop=True)


class TushareIndexProvider:
    """Thin adapter around Tushare's index_daily endpoint."""

    def __init__(self, token: str):
        import tushare as ts

        self.pro = ts.pro_api(token)

    def fetch_index_daily(
        self, ts_code: str, name: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        raw = self.pro.index_daily(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )
        return normalize_index_daily(raw, name)

    def fetch_industry_daily(
        self, ts_code: str, name: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        raw = self.pro.sw_daily(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )
        return normalize_index_daily(raw, name)


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
