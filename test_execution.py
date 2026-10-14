import os
import sys
from pathlib import Path
from config import settings
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
    # Ověříme, že v aktuální konfiguraci běžíme v demo režimu
    assert settings.trading_mode == "demo", "Test musí běžet v demo režimu!"
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


if __name__ == "__main__":
    try:
        test_ticker_mapper()
        test_portfolio_tracker()
        test_safety_guards()
        test_live_broker_reconciliation()
        print("\n==========================================")
        print("  FÁZE 4 ÚSPĚŠNĚ DOKONČENA A OVĚŘENA")
        print("==========================================\n")
    except Exception as exc:
        print(f"\nCHYBA V TESTU FÁZE 4: {exc}")
        sys.exit(1)
