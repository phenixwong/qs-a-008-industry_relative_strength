import pytest

from qs_a008.config import get_tushare_token


def test_get_tushare_token_reads_environment(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    assert get_tushare_token() == "test-token"


def test_get_tushare_token_fails_with_actionable_message(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="TUSHARE_TOKEN"):
        get_tushare_token()

