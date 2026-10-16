import sys
from pathlib import Path

# Zajisteni importu z korene repozitare
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.data.market_data import TechnicalMetrics
from src.llm.sentiment_engine import LLMStockVerdict
from src.quant.regime_detector import MarketRegime, MarketRegimeType
from src.quant.signal_engine import IntradaySignalEngine
from src.strategy import (
    ExitReason,
    PositionExitSignal,
    RiskManager,
    SignalAction,
    StockPicker,
    TradingSignal,
)
from src.strategy.opportunity_scanner import OpportunityScanner


def test_risk_manager_logic():
    print("\n--- [1] Test Risk Managementu a alokace kapitálu ---")
    rm = RiskManager(total_capital_eur=50.0, max_positions=3, atr_sl_multiplier=2.0, atr_tp_multiplier=3.5)

    # 1. Velikost pozice
    assert rm.max_position_size_eur == 16.67, f"Očekáváno 16.67 EUR, získáno {rm.max_position_size_eur}"
    print(f"  [OK] Max velikost pozice z 50 EUR: {rm.max_position_size_eur} EUR")

    # 2. Alokace při různých úrovních přesvědčení (Conviction Sizing)
    alloc_tier1 = rm.calculate_allocation(0, available_cash_eur=50.0, signal_score=85.0)
    assert alloc_tier1 == 21.67, f"Chyba při Tier 1 (85 b): {alloc_tier1}"

    alloc_tier2 = rm.calculate_allocation(0, available_cash_eur=50.0, signal_score=78.0)
    assert alloc_tier2 == 16.67, f"Chyba při Tier 2 (78 b): {alloc_tier2}"

    alloc_tier3 = rm.calculate_allocation(0, available_cash_eur=50.0, signal_score=70.0)
    assert alloc_tier3 == 10.84, f"Chyba při Tier 3 (70 b): {alloc_tier3}"

    alloc_limit = rm.calculate_allocation(3, available_cash_eur=100.0, signal_score=85.0)
    assert alloc_limit == 0.0, f"Chyba při dosažení limitu 3 pozic: {alloc_limit}"
    print("  [OK] Dynamická conviction alokace (Tier 1: 21.67 EUR, Tier 2: 16.67 EUR, Tier 3: 10.84 EUR) a limit pozic fungují správně.")

    # 3. Přepočet frakčních akcií (EUR -> USD)
    shares = rm.calculate_shares_quantity(allocated_eur=16.67, stock_price=228.0, eur_usd_rate=1.08)
    assert shares > 0, "Počet akcií musí být kladný"
    print(f"  [OK] Frakční akcie pro 16.67 EUR při ceně 228 USD: {shares} ks")

    # 4. ATR Stop Loss a Take Profit
    sl, tp = rm.calculate_stops(current_price=200.0, atr_14=5.0)
    assert sl == 190.0, f"Očekávaný SL 190.0, získáno {sl}"
    assert tp == 217.5, f"Očekávaný TP 217.5, získáno {tp}"
    print(f"  [OK] ATR Stop Loss: {sl} USD, Take Profit: {tp} USD")

    # 5. Výstupní podmínky (Stop loss, Take profit, SMA breakdown, AI avoid)
    dummy_metrics = TechnicalMetrics(
        ticker="NVDA",
        current_price=200.0,
        sma_50=190.0,
        sma_200=170.0,
        rsi_14=55.0,
        atr_14=5.0,
        trend_bullish=True,
        rsi_neutral_or_pullback=True,
    )

    # Scénář A: Zásah Stop Lossu (cena klesla na 188 USD pod SL 190 USD)
    exit_sl = rm.evaluate_position_exit(
        ticker="NVDA",
        current_price=188.0,
        entry_price=200.0,
        metrics=dummy_metrics,
        custom_stop_loss=190.0,
        custom_take_profit=217.5,
    )
    assert exit_sl.should_exit and exit_sl.reason == ExitReason.STOP_LOSS
    print("  [OK] Detekce Stop Lossu funguje správně.")

    # Scénář B: Zásah Take Profitu (cena stoupla na 220 USD nad TP 217.5 USD)
    exit_tp = rm.evaluate_position_exit(
        ticker="NVDA",
        current_price=220.0,
        entry_price=200.0,
        metrics=dummy_metrics,
        custom_stop_loss=190.0,
        custom_take_profit=217.5,
    )
    assert exit_tp.should_exit and exit_tp.reason == ExitReason.TAKE_PROFIT
    print("  [OK] Detekce Take Profitu funguje správně.")

    # Scénář C: Zlomení trendu pod SMA 50
    broken_metrics = dummy_metrics.model_copy(update={"sma_50": 205.0})
    exit_trend = rm.evaluate_position_exit(
        ticker="NVDA",
        current_price=200.0,
        entry_price=200.0,
        metrics=broken_metrics,
        custom_stop_loss=180.0,
        custom_take_profit=250.0,
    )
    assert exit_trend.should_exit and exit_trend.reason == ExitReason.TREND_BREAK
    print("  [OK] Detekce prolomení 50denního SMA funguje správně.")

    # Scénář D: Zhoršení AI verdiktu na AVOID
    avoid_verdict = LLMStockVerdict(
        ticker="NVDA",
        sentiment_score=-0.7,
        recommendation="AVOID",
        confidence=0.85,
        key_drivers=[],
        risks=["Ztráta klíčového trhu"],
        reasoning="Závažná negativa."
    )
    exit_ai = rm.evaluate_position_exit(
        ticker="NVDA",
        current_price=200.0,
        entry_price=200.0,
        metrics=dummy_metrics,
        ai_verdict=avoid_verdict,
        custom_stop_loss=180.0,
        custom_take_profit=250.0,
    )
    assert exit_ai.should_exit and exit_ai.reason == ExitReason.AI_AVOID
    print("  [OK] Detekce zhoršení sentimentu na AI AVOID funguje správně.")


def test_stock_picker_scoring():
    print("\n--- [2] Test Kvantitativního Scoringu a Výběru Akcií ---")
    picker = StockPicker()

    # Živý sken 2 vybraných akcií
    test_tickers = ["NVDA", "AAPL"]
    signals = picker.scan_watchlist(custom_tickers=test_tickers)

    assert len(signals) >= 1, f"Očekáván alespoň 1 signál z tržních dat, získáno {len(signals)}"

    for s in signals:
        print(f"\n==========================================")
        print(f"Výsledný signál: {s.ticker}")
        print(f"  Akce:              {s.action.value}")
        print(f"  Celkové skóre:     {s.total_score:.1f} / 100")
        print(f"  Technická složka:  {s.technical_score:.1f} / 50")
        print(f"  Sentiment složka:  {s.sentiment_score:.1f} / 50")
        print(f"  Doporučená alokace: {s.suggested_allocation_eur:.2f} EUR")
        if s.stop_loss_price and s.take_profit_price:
            print(f"  Úrovně: Stop Loss = {s.stop_loss_price} USD | Take Profit = {s.take_profit_price} USD")
        print(f"  Důvody:            {'; '.join(s.reasons)}")


def test_opportunity_scanner_and_intraday_signals():
    print("\n--- [3] Test OpportunityScanner a IntradaySignalEngine ---")
    scanner = OpportunityScanner()
    engine = IntradaySignalEngine()
    regime = MarketRegime(
        regime=MarketRegimeType.TRENDING_BULL,
        allow_trading=True,
        risk_multiplier=1.0,
    )
    df_bull = pd.DataFrame(
        {
            "Open": [100.0] * 10,
            "High": [101.0, 101.5, 101.8, 101.2, 103.5, 104.0, 104.2, 104.5, 104.8, 105.0],
            "Low": [99.5, 99.8, 99.7, 100.0, 101.0, 102.0, 102.5, 103.0, 103.5, 104.0],
            "Close": [100.5, 101.2, 101.5, 101.0, 103.0, 103.8, 104.0, 104.2, 104.5, 104.8],
            "Volume": [10000] * 10,
            "session": ["REGULAR"] * 10,
            "minutes_from_open": [i * 5 for i in range(10)],
            "vwap": [100.8] * 10,
            "orb_high": [np.nan, np.nan, np.nan, 101.8, 101.8, 101.8, 101.8, 101.8, 101.8, 101.8],
            "orb_low": [np.nan, np.nan, np.nan, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
            "orb_width": [np.nan, np.nan, np.nan, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3],
            "atr_14": [4.0] * 10,
            "rvol_at_time": [1.0, 1.0, 1.0, 1.2, 1.9, 2.0, 1.8, 1.5, 1.4, 1.3],
            "intraday_rs_zscore": [0.5] * 10,
        },
        index=pd.date_range("2026-03-10 09:30", periods=10, freq="5min", tz="America/New_York"),
    )
    sig = engine.evaluate(df_bull, regime, current_idx=4, ticker="TSLA")
    assert sig is not None and sig.strategy_name == "ORB_15M_MOMENTUM"
    print(f"  [OK] IntradaySignalEngine vygeneroval signál: {sig.strategy_name} pro {sig.ticker}")

    bias = scanner.compute_directional_bias(df_bull)
    assert bias > 0.4
    print(f"  [OK] OpportunityScanner Directional Bias: {bias:+.2f}")


if __name__ == "__main__":
    try:
        test_risk_manager_logic()
        test_opportunity_scanner_and_intraday_signals()
        test_stock_picker_scoring()
        print("\n==========================================")
        print("  FÁZE 3 ÚSPĚŠNĚ DOKONČENA A OVĚŘENA")
        print("==========================================\n")
    except Exception as exc:
        print(f"\nCHYBA V TESTU FÁZE 3: {exc}")
        sys.exit(1)
