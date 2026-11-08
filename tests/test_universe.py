from qs_a008.universe import BENCHMARK, INDUSTRIES, MARKET_INDICES


def test_universe_has_report_benchmark_and_expected_groups():
    assert BENCHMARK["name"] == "沪深300"
    assert BENCHMARK["ts_code"] == "000300.SH"
    assert len(INDUSTRIES) == 31
    assert len(MARKET_INDICES) == 16
    assert len({item["ts_code"] for item in INDUSTRIES}) == 31
    assert len({item["ts_code"] for item in MARKET_INDICES}) == 16

