import pandas as pd

from qs_a008.data import normalize_index_daily


def test_normalize_index_daily_sorts_and_keeps_required_fields():
    raw = pd.DataFrame(
        {
            "trade_date": ["20261030", "20261029"],
            "ts_code": ["000300.SH", "000300.SH"],
            "close": [4600.0, 4550.0],
        }
    )
    result = normalize_index_daily(raw, "沪深300")
    assert result["trade_date"].tolist() == [pd.Timestamp("2026-10-29"), pd.Timestamp("2026-10-30")]
    assert result["name"].tolist() == ["沪深300", "沪深300"]
    assert result["close"].tolist() == [4550.0, 4600.0]

