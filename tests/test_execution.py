import os
import sys
from pathlib import Path

# Zajisteni importu z korene repozitare
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TradingMode, settings
from src.broker.t212_client import Trading212Client
from src.execution import (
    OrderExecutor,
    PortfolioReconciler,
    PortfolioTracker,
    TickerMapper,
)
from src.strategy.models import SignalAction, TradingSignal


def test_ticker_mapper():
    print("\n--- [1] Test Ticker Mappera ---")
    assert TickerMapper.to_t212_ticker("NVDA") == "NVDA_US_EQ"
    assert TickerMapper.to_t212_ticker("AAPL") == "AAPL_US_EQ"
    assert TickerMapper.to_t212_ticker("BRK-B") == "BRKb_US_EQ"
    assert TickerMapper.to_standard_ticker("NVDA_US_EQ") == "NVDA"
    assert TickerMapper.to_standard_ticker("BRKb_US_EQ") == "BRK-B"
    print("  [OK] Obousměrný převod tickerů a speciálních případů funguje bezchybně.")


def test_portfolio_tracker():
    print("\n--- [2] Test Perzistentního Trackeru ---")
    test_file = "data/test_portfolio_state.json"
    if Path(test_file).exists():
        Path(test_file).unlink()

    tracker = PortfolioTracker(storage_path=test_file)

    # 1. Záznam nákupu
    rec = tracker.record_buy(
        ticker="AAPL",
        quantity=0.05,
        entry_price_usd=300.0,
        stop_loss_price=275.0,
        take_profit_price=340.0,
        allocated_eur=16.67,
    )
    assert rec.ticker == "AAPL"
    assert len(tracker.get_open_positions()) == 1
    print("  [OK] Nákup zaevidován do perzistentního stavu.")

    # 2. Záznam prodeje se ziskem
    closed = tracker.record_sell(
        ticker="AAPL",
        close_price_usd=330.0,
        close_reason="TAKE_PROFIT",
    )
    assert closed is not None
    assert closed.realized_pnl_usd == round((330.0 - 300.0) * 0.05, 2)
    assert len(tracker.get_open_positions()) == 0
    print(f"  [OK] Prodej zaevidován, realizovaný PnL: {closed.realized_pnl_usd:+.2f} USD.")

    # Úklid testovacího souboru
    if Path(test_file).exists():
        Path(test_file).unlink()


def test_safety_guards():
    print("\n--- [3] Test Bezpečnostních Pojistek ---")
    # Ověříme, že v aktuální konfiguraci běžíme v demo/paper režimu
    assert settings.trading_mode in (TradingMode.PAPER, "demo", "paper"), "Test musí běžet v demo/paper režimu!"
    executor = OrderExecutor()
    print("  [OK] OrderExecutor bezpečně zkontroloval DEMO režim.")

    # Ověření kurzu EUR/USD
    rate = executor.get_eur_usd_rate()
    assert 0.8 < rate < 1.5, f"Neočekávaný kurz EUR/USD: {rate}"
    print(f"  [OK] Tržní kurz EUR/USD: {rate:.4f}")


def test_live_broker_reconciliation():
    print("\n--- [4] Test Živé Synchronizace s Trading 212 Demo ---")
    client = Trading212Client()
    reconciler = PortfolioReconciler(client=client)

    broker_positions = reconciler.sync_broker_portfolio()
    print(f"  [OK] Z brokera načteno {len(broker_positions)} otevřených pozic.")
    for p in broker_positions:
        print(f"       - {p.get('ticker')}: {p.get('quantity')} ks @ {p.get('currentPrice')} USD")

    open_in_tracker = reconciler.tracker.get_open_positions()
    print(f"  [OK] Tracker úspěšně synchronizoval pozice: {list(open_in_tracker.keys())}")


def test_gatekeeper_daily_loss_kill_switch():
    print("\n--- [5] Test RiskGatekeeper: Daily Kill-Switch (-2 %) ---")
    from src.quant.risk_engine import OrderIntent, PortfolioState, RiskGatekeeper

    gatekeeper = RiskGatekeeper(max_daily_loss_pct=0.02)
    # Portfolio s 2.5% denní ztrátou (250 EUR z 10 000 EUR)
    portfolio = PortfolioState(
        equity=10_000.0,
        cash=5_000.0,
        daily_realized_pnl=-250.0,
        daily_unrealized_pnl=0.0,
        open_positions_count=0,
    )
    intent = OrderIntent(
        ticker="AAPL",
        side="BUY",
        entry_price=150.0,
        stop_loss_price=145.0,
    )

    result = gatekeeper.evaluate_order_intent(intent, portfolio)
    assert not result.approved, "Příkaz měl být zamítnut kvůli překročení denní ztráty!"
    assert "DAILY_KILL_SWITCH" in result.reason
    print(f"  [OK] Daily Kill-Switch správně zablokoval nákup: {result.reason}")


def test_gatekeeper_stale_data_guard():
    print("\n--- [6] Test RiskGatekeeper: Stale Data Guard (> 120 s) ---")
    from datetime import datetime, timedelta
    from src.quant.risk_engine import MarketBar, OrderIntent, PortfolioState, RiskGatekeeper

    gatekeeper = RiskGatekeeper(max_stale_seconds=120)
    portfolio = PortfolioState(equity=10_000.0, cash=5_000.0)
    intent = OrderIntent(ticker="NVDA", side="BUY", entry_price=100.0, stop_loss_price=95.0)

    # Zastaralá svíčka (150 sekund stará)
    stale_bar = MarketBar(
        ticker="NVDA",
        close=100.0,
        high=101.0,
        low=99.0,
        timestamp=datetime.now() - timedelta(seconds=150),
    )

    result = gatekeeper.evaluate_order_intent(intent, portfolio, market_bar=stale_bar)
    assert not result.approved, "Příkaz měl být zamítnut kvůli zastaralým datům!"
    assert "STALE_DATA" in result.reason
    print(f"  [OK] Stale Data Guard správně zablokoval zastaralou svíčku: {result.reason}")


def test_gatekeeper_duplicate_order_guard():
    print("\n--- [7] Test RiskGatekeeper: Duplicate client_order_id Guard ---")
    from datetime import datetime
    from src.quant.risk_engine import OrderIntent, PortfolioState, RiskGatekeeper

    gatekeeper = RiskGatekeeper()
    portfolio = PortfolioState(equity=10_000.0, cash=5_000.0)
    now_ts = datetime(2025, 1, 15, 10, 30)

    intent = OrderIntent(
        ticker="MSFT",
        side="BUY",
        entry_price=300.0,
        stop_loss_price=290.0,
        desired_shares=5.0,
        timestamp=now_ts,
    )

    # 1. pokus: musí projít
    first_res = gatekeeper.evaluate_order_intent(intent, portfolio)
    assert first_res.approved, "První pokus musí být schválen!"

    # 2. pokus s identickými parametry ve stejný čas: musí být zamítnut jako duplicita
    second_res = gatekeeper.evaluate_order_intent(intent, portfolio)
    assert not second_res.approved, "Druhý pokus musí být zamítnut jako duplicita!"
    assert "DUPLICATE_ORDER" in second_res.reason
    print(f"  [OK] Idempotence správně detekovala duplicitní příkaz: {second_res.reason}")


def test_gatekeeper_inertia_buffer():
    print("\n--- [8] Test RiskGatekeeper: 20% No-Trade Inertia Buffer ---")
    from src.quant.risk_engine import OrderIntent, PortfolioState, RiskGatekeeper

    gatekeeper = RiskGatekeeper(inertia_buffer_pct=20.0)
    # Držíme 10 ks MSFT
    portfolio = PortfolioState(
        equity=100_000.0,
        cash=50_000.0,
        open_positions={"MSFT": 10.0},
        open_positions_count=1,
    )

    # Chceme upravit na 11 ks (změna o 10 % < 20 % buffer)
    intent = OrderIntent(
        ticker="MSFT",
        side="BUY",
        entry_price=300.0,
        stop_loss_price=290.0,
        desired_shares=11.0,
    )

    res = gatekeeper.evaluate_order_intent(intent, portfolio)
    assert not res.approved
    assert "NO_TRADE_INERTIA_BUFFER" in res.reason
    print(f"  [OK] No-Trade Buffer zablokoval zbytečné přebalancování: {res.reason}")


def test_order_executor_blocks_unapproved_order():
    print("\n--- [9] Test OrderExecutor: Blokování neschváleného příkazu ---")
    from src.quant.risk_engine import PortfolioState, RiskGatekeeper

    # Gatekeeper s denní ztrátou
    gatekeeper = RiskGatekeeper(max_daily_loss_pct=0.01)
    # Simulujeme denní ztrátu
    gatekeeper.evaluate_order_intent = lambda intent, port, bar=None: __import__("src.quant.risk_engine", fromlist=["RejectedOrder"]).RejectedOrder(
        ticker=intent.ticker, side=intent.side, reason="TEST_REJECTION", rejected_at=__import__("datetime").datetime.now()
    )

    executor = OrderExecutor(gatekeeper=gatekeeper, dry_run=True)
    signal = TradingSignal(
        ticker="AAPL",
        action=SignalAction.BUY,
        total_score=80.0,
        technical_score=40.0,
        sentiment_score=40.0,
        confidence=0.85,
        current_price=150.0,
        stop_loss_price=145.0,
        suggested_allocation_eur=20.0,
    )

    result = executor.execute_buy(signal)
    assert result is None, "OrderExecutor nesmí odeslat pokyn bez ApprovedOrder!"
    print("  [OK] OrderExecutor striktně odmítl odeslání bez razítka ApprovedOrder.")



def test_paper_broker_adapter():
    print("\n--- [10] Test PaperBrokerAdapter: Virtuální exekuce ---")
    from datetime import datetime
    from src.execution.broker_adapter import PaperBrokerAdapter
    from src.quant.risk_engine import ApprovedOrder

    adapter = PaperBrokerAdapter(initial_cash=50_000.0)
    order = ApprovedOrder(
        client_order_id="TEST_PAPER_001",
        ticker="AMD",
        side="BUY",
        shares=10.0,
        entry_price=120.0,
        stop_loss=115.0,
        take_profit=130.0,
        risk_per_trade_cash=50.0,
        risk_per_trade_pct=0.5,
        allocation_cash=1200.0,
        approved_at=datetime.now(),
    )

    res = adapter.place_order(order)
    assert res["status"] == "FILLED"
    assert adapter.get_account_cash() == 50_000.0 - 1200.0
    assert adapter.get_open_positions()["AMD"] == 10.0
    print("  [OK] PaperBrokerAdapter provedl virtuální nákup a správně aktualizoval hotovost i pozice.")


if __name__ == "__main__":
    try:
        test_ticker_mapper()
        test_portfolio_tracker()
        test_safety_guards()
        test_live_broker_reconciliation()
        test_gatekeeper_daily_loss_kill_switch()
        test_gatekeeper_stale_data_guard()
        test_gatekeeper_duplicate_order_guard()
        test_gatekeeper_inertia_buffer()
        test_order_executor_blocks_unapproved_order()
        test_paper_broker_adapter()
        print("\n==========================================")
        print("  ETAPA 3 (AG-04) ÚSPĚŠNĚ DOKONČENA A OVĚŘENA")
        print("==========================================\n")
    except Exception as exc:
        print(f"\nCHYBA V TESTU FÁZE 4: {exc}")
        sys.exit(1)

