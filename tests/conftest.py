"""All pytest tests use disposable state and prohibit external connections."""
import os
import socket
import pytest
import requests

# Applied before importing application modules or reading .env settings.
os.environ["TRADING_MODE"] = "paper"


@pytest.fixture(autouse=True)
def offline_test_runtime(tmp_path, monkeypatch):
    from config import settings, TradingMode
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "trading_mode", TradingMode.PAPER)
    def blocked(*args, **kwargs):
        raise AssertionError("External connections are forbidden in automated tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(requests.Session, "request", blocked)
    try:
        import httpx
        monkeypatch.setattr(httpx.Client, "send", blocked)
    except ImportError:
        pass
    try:
        from curl_cffi.requests import Session
        monkeypatch.setattr(Session, "request", blocked)
    except ImportError:
        pass


def pytest_collection_modifyitems(items):
    # Legacy manual diagnostics contact a real account / local inference server.
    # They are not unit tests and must never run as part of an unattended suite.
    for item in items:
        if item.name in {
            "test_live_broker_reconciliation",
            "test_live_ai_pipeline",
            "test_multi_llm_committee_fast_mode",
            "test_stock_picker_scoring",
        }:
            item.add_marker(pytest.mark.skip(reason="Manual external-service diagnostic; excluded from offline suite"))
