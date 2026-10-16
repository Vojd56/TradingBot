from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests

from config import TransactionCostConfig, settings, TradingMode
from src.broker.t212_client import Trading212Client, Trading212APIError
from src.execution.submission_journal import SubmissionJournal
from src.execution.broker_adapter import PaperBrokerAdapter, T212BrokerAdapter
from src.quant.risk_engine import RiskGatekeeper, PortfolioState, OrderIntent, PositionSizingResult
from src.quant.execution_engine import ExecutionEngine
from src.quant.prediction_model import TripleBarriers
from src.quant.features import CausalFeatureEngine
from src.quant.backtester import IntradayBacktester, ExitReason
from src.strategy.intraday_data import validate_ohlcv_integrity, ParquetBarStore


def intent(**changes):
    args = dict(ticker="AAPL", entry_price=100, stop_loss_price=98, take_profit_price=104, desired_shares=5)
    args.update(changes)
    return OrderIntent(**args)


def account(**changes):
    args = dict(equity=10000, cash=10000)
    args.update(changes)
    return PortfolioState(**args)


def test_zero_cash_rejected():
    assert not RiskGatekeeper().evaluate_order_intent(intent(), account(cash=0)).approved


def test_loss_limit_permits_reducing_exit_and_rejects_oversell():
    state = account(daily_realized_pnl=-1000, open_positions={"AAPL": 5})
    assert RiskGatekeeper().evaluate_order_intent(intent(side="SELL"), state).approved
    assert not RiskGatekeeper().evaluate_order_intent(intent(side="SELL", desired_shares=6), state).approved
    assert not RiskGatekeeper().evaluate_order_intent(intent(), state).approved


@pytest.mark.parametrize("changes", [{"take_profit_price": 102}, {"side": "BOGUS"}, {"desired_shares": float("nan")}])
def test_invalid_intent_rejected(changes):
    assert not RiskGatekeeper().evaluate_order_intent(intent(**changes), account()).approved


def test_paper_idempotency_and_cash_limit():
    order = RiskGatekeeper().evaluate_order_intent(intent(), account())
    broker = PaperBrokerAdapter(1000)
    assert broker.place_order(order) == broker.place_order(order)
    assert broker.cash == 500
    assert broker.positions == {"AAPL": 5}
    assert not broker.cancel_order(order.client_order_id)
    with pytest.raises(ValueError):
        PaperBrokerAdapter(100).place_order(order)


def test_http_post_timeout_never_retries():
    client = Trading212Client(api_key="fake", api_secret="fake")
    client.session.request = Mock(side_effect=requests.Timeout("uncertain"))
    with pytest.raises(Trading212APIError):
        client._request("POST", "/api/v0/equity/orders/market")
    assert client.session.request.call_count == 1


def test_submission_survives_restart_and_requires_matching_terminal_order(tmp_path):
    path = tmp_path / "journal.sqlite"
    journal = SubmissionJournal(path)
    journal.reserve("AAPL_US_EQ", {"quantity": 5})
    with pytest.raises(RuntimeError):
        SubmissionJournal(path).reserve("AAPL_US_EQ", {"quantity": 5})
    journal.acknowledge("AAPL_US_EQ", 123)
    assert not journal.reconcile_terminal("AAPL_US_EQ", {"id": 999, "status": "FILLED"})
    assert not journal.reconcile_terminal("AAPL_US_EQ", {"id": 123, "status": "NEW"})
    assert journal.reconcile_terminal("AAPL_US_EQ", {"id": 123, "status": "FILLED"})


def test_timeout_blocks_new_client_submission(tmp_path):
    path = str(tmp_path / "journal.sqlite")
    client = Trading212Client(api_key="fake", api_secret="fake", journal_path=path)
    client._request = Mock(side_effect=Trading212APIError(0, "timeout"))
    with pytest.raises(Trading212APIError):
        client.place_market_order("AAPL_US_EQ", 5)
    restarted = Trading212Client(api_key="fake", api_secret="fake", journal_path=path)
    restarted._request = Mock()
    with pytest.raises(RuntimeError):
        restarted.place_market_order("AAPL_US_EQ", 5)
    restarted._request.assert_not_called()


def test_adapter_does_not_retry_extended_hours_failure():
    client = Mock()
    client.place_market_order.side_effect = Trading212APIError(0, "timeout")
    with pytest.raises(RuntimeError):
        T212BrokerAdapter(client).place_order(RiskGatekeeper().evaluate_order_intent(intent(), account()))
    assert client.place_market_order.call_count == 1


def test_demo_cannot_target_live_host():
    with pytest.raises(ValueError):
        Trading212Client(api_key="fake", base_url="https://live.trading212.com")


def test_backtest_cannot_submit(monkeypatch):
    monkeypatch.setattr(settings, "trading_mode", TradingMode.BACKTEST)
    with pytest.raises(RuntimeError):
        Trading212Client(api_key="fake").place_market_order("AAPL_US_EQ", 1)


def test_quant_ack_is_not_fill_and_no_quantity_retry():
    sizing = PositionSizingResult(ticker="AAPL", target_shares=5, target_value=500,
        portfolio_weight_pct=5, risk_per_trade_cash=10, risk_per_trade_pct=.1,
        current_shares=0, shares_diff=5, is_rebalance_required=True,
        approved_for_execution=True, rationale="fixture")
    barriers = TripleBarriers(entry_price=100, stop_loss_price=98, take_profit_price=104,
        max_holding_bars=8, take_profit_pct=4, stop_loss_pct=2, rr_ratio=2)
    client = Mock()
    client.place_order.return_value = {"id": 12, "status": "NEW"}
    engine = ExecutionEngine(client=client, is_live=True)
    order = engine.execute_sizing_order(sizing, barriers, 1000)
    assert order.status == "SUBMITTED"
    assert order.executed_price is None
    client.place_order.side_effect = Trading212APIError(400, "min-quantity-exceeded must trade at least 100")
    engine = ExecutionEngine(client=client, is_live=True)
    order = engine.execute_sizing_order(sizing, barriers, 1000)
    assert order.status == "REJECTED"
    assert client.place_order.call_count == 2


def bars():
    return pd.DataFrame(dict(Open=[100.] * 5, High=[101.] * 5, Low=[99.] * 5,
                             Close=[100.] * 5, Volume=[1000.] * 5),
                        index=pd.date_range("2025-06-02 15:35", periods=5, freq="5min", tz="America/New_York"))


@pytest.mark.parametrize("damage", ["missing", "negative", "geometry", "nan", "duplicate"])
def test_corrupt_data_rejected_without_mutation(damage):
    df = bars()
    if damage == "missing":
        df = df.drop(columns="Open")
    elif damage == "negative":
        df.iloc[0, 4] = -1
    elif damage == "geometry":
        df.iloc[0, 1] = 90
    elif damage == "nan":
        df.iloc[0, 3] = np.nan
    else:
        df = pd.concat([df, df.iloc[:1]])
    before = df.copy(deep=True)
    with pytest.raises(ValueError):
        validate_ohlcv_integrity(df)
    pd.testing.assert_frame_equal(df, before)


def test_corrupt_store_never_overwritten(tmp_path):
    store = ParquetBarStore(tmp_path)
    path = store.save_bars("AAPL", "5m", bars())
    path.write_bytes(b"broken parquet")
    with pytest.raises(ValueError):
        store.save_bars("AAPL", "5m", bars())
    assert path.read_bytes() == b"broken parquet"


def test_session_vwap_variance_matches_weighted_population():
    df = bars().iloc[:2].copy()
    df.loc[:, ["Open", "High", "Low", "Close"]] = [[100]*4, [102]*4]
    result = CausalFeatureEngine.compute_session_vwap(df)
    assert result.vwap.iloc[-1] == 101
    assert result.vwap_std.iloc[-1] == pytest.approx(1)


def test_stop_gap_on_last_bar_precedes_eod_and_costs_reconcile():
    df = bars()
    df.iloc[-1, :4] = [95, 96, 94, 95]
    signals = pd.Series([1, 0, 0, 0, 0], index=df.index)
    sl = pd.Series(98., index=df.index)
    tp = pd.Series(104., index=df.index)
    tester = IntradayBacktester(cost_config=TransactionCostConfig(spread_bps=5, slippage_bps=5, fx_conversion_bps=0))
    result = tester.run(df, signals, symbol="AAPL", sl_series=sl, tp_series=tp)
    trade = result.trades[0]
    assert trade.exit_reason == ExitReason.STOP_LOSS
    assert trade.raw_exit_price == 95
    assert trade.gross_pnl - trade.total_cost_paid == pytest.approx(trade.net_pnl, abs=.02)
