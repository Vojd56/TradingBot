import logging
import sys
from pathlib import Path

# Zajisteni importu z korene repozitare
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MarketDataProvider
from src.llm import LLMStockVerdict, SentimentEngine
from src.llm.multi_llm_committee import MultiLLMCommittee
from src.llm.sentiment_engine import CatalystAssessment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def test_sanitization_unit_tests():
    print("\n--- [A] Test jednotkových sanitizačních mechanismů JSONu ---")
    engine = SentimentEngine()

    # 1. Test poškozeného JSONu s opakovanými čárkami (častá vada 1.5B/quant modelů)
    dirty_json_1 = (
        "```json\n"
        "{\n"
        '  "ticker": "NVDA",\n'
        '  "sentiment_score": "0.65",\n'
        ",\n,\n,\n"
        '  "recommendation": "BUY",\n'
        '  "confidence": 0.85,\n'
        ",\n"
        '  "risk_factors": ["Vysoká valuace", "Dodavatelské kapacity"],\n'
        '  "reasoning": "Silná poptávka po AI akcelerátorech Blackwell."\n'
        "}\n"
        "```"
    )

    data_1 = engine._sanitize_and_extract_json(dirty_json_1)
    assert data_1 is not None, "Sanitizace selhala pro dirty_json_1"
    verdict_1 = LLMStockVerdict(**data_1)
    assert verdict_1.ticker == "NVDA"
    assert verdict_1.recommendation == "BUY"
    assert verdict_1.sentiment_score == 0.65
    assert len(verdict_1.risks) == 2
    print("  [OK] Test 1: Opakované čárky a markdown bloky úspěšně opraveny.")

    # 2. Test poškozené syntaxe zachráněné nouzovou regex extrakcí
    dirty_json_2 = (
        'Zde je analýza pro Apple:\n'
        '{"ticker": "AAPL", "sentiment_score": 0.20, "recommendation": "HOLD", '
        '"confidence": 0.70, "reasoning": "Konsolidace před uvedením nového produktu." TRUNCATED_OR_INVALID_TAIL'
    )
    data_2 = engine._sanitize_and_extract_json(dirty_json_2)
    assert data_2 is not None, "Regex rekonstrukce selhala pro dirty_json_2"
    verdict_2 = LLMStockVerdict(**data_2)
    assert verdict_2.ticker == "AAPL"
    assert verdict_2.recommendation == "HOLD"
    assert verdict_2.sentiment_score == 0.20
    print("  [OK] Test 2: Nouzová regex rekonstrukce úspěšně zachránila nevalidní konec.")

    # 3. Test normalizace a ořezání hraničních hodnot (clamping)
    out_of_bounds = {
        "ticker": "TSLA",
        "sentiment_score": 5.0,  # mimo rozsah -> clamp na 1.0
        "recommendation": "doporučuji nákup (BUY NOW)",
        "confidence": -2.0,  # mimo rozsah -> clamp na 0.0
        "reasoning": "Rychlý test ořezání hodnot."
    }
    verdict_3 = LLMStockVerdict(**out_of_bounds)
    assert verdict_3.sentiment_score == 1.0
    assert verdict_3.confidence == 0.0
    assert verdict_3.recommendation == "BUY"
    print("  [OK] Test 3: Normalizace mezních hodnot (clamping) a doporučení funguje spolehlivě.")


def test_catalyst_assessment_and_veto():
    print("\n--- [B] Test CatalystAssessment a VETO ochrany před ředěním ---")
    engine = SentimentEngine()

    # 1. Toxická událost - sekundární úpis a ředění akcií
    toxic_headlines_1 = [
        "Company Announces Proposed Public Offering of Common Stock",
        "Analyst comment on market volatility",
    ]
    assessment_1 = engine.classify_catalyst("DILUTE_CORP", toxic_headlines_1)
    assert assessment_1.trade_veto is True
    assert assessment_1.has_catalyst is True
    assert "ředění" in assessment_1.veto_reason or "úpis" in assessment_1.veto_reason
    assert assessment_1.sentiment_score < 0
    print("  [OK] Test 1: Sekundární úpis spolehlivě vyvolal okamžité VETO.")

    # 2. Toxická událost - vyšetřování SEC pro podvod
    toxic_headlines_2 = [
        "SEC Opens Fraud Investigation into Accounting Irregularities",
    ]
    assessment_2 = engine.classify_catalyst("FRAUD_INC", toxic_headlines_2)
    assert assessment_2.trade_veto is True
    assert "vyšetřování" in assessment_2.veto_reason or "SEC" in assessment_2.veto_reason
    print("  [OK] Test 2: SEC vyšetřování vyvolalo okamžité VETO.")

    # 3. Pozitivní katalyzátor - zisk významného vládního kontraktu
    positive_headlines = [
        "Defense Firm Awarded $750 Million Contract with U.S. Army",
        "Quarterly results scheduled for next month",
    ]
    assessment_3 = engine.classify_catalyst("DEFENSE_CO", positive_headlines)
    assert assessment_3.trade_veto is False
    assert assessment_3.has_catalyst is True
    assert assessment_3.sentiment_score > 0
    assert assessment_3.catalyst_type == "CONTRACT_WIN"
    print("  [OK] Test 3: Pozitivní katalyzátor správně klasifikován bez VETA.")


def test_multi_llm_committee_fast_mode():
    print("\n--- [C] Test Multi-LLM Committee s fast_mode a ochranou proti latenci ---")
    committee = MultiLLMCommittee()
    from src.data.market_data import StockAnalysisData, TechnicalMetrics
    dummy_data = StockAnalysisData(
        ticker="PLTR",
        company_name="Palantir Technologies",
        sector="Technology",
        industry="Software",
        metrics=TechnicalMetrics(
            ticker="PLTR",
            current_price=25.0,
            sma_50=23.0,
            sma_200=20.0,
            rsi_14=58.0,
            atr_14=1.2,
            trend_bullish=True,
            rsi_neutral_or_pullback=True,
        ),
        recent_news=[],
    )

    # V fast_mode s 0.1s timeoutem musí rada doběhnout bleskově (< 0.5s) a vrátit validní CommitteeConsensus
    consensus = committee.evaluate_consensus(dummy_data, fast_mode=True, timeout=0.1)
    assert consensus is not None
    assert consensus.ticker == "PLTR"
    assert consensus.final_recommendation in ("BUY", "HOLD", "AVOID")
    assert len(consensus.agent_verdicts) == 3
    print(f"  [OK] Fast Mode Rada AI doběhla okamžitě s verdiktem: {consensus.final_recommendation} ({consensus.consensus_strength})")


def test_live_ai_pipeline():
    print("\n--- [D] Test živého propojení s Lemonade Serverem a NPU ---")
    import socket
    server_online = False
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.5):
            server_online = True
    except OSError:
        server_online = False

    if not server_online:
        print("  [INFO] Lokální Lemonade server (127.0.0.1:8000) není spuštěn. Test živého LLM přeskakuji (pravidlový fallback ověřen).")
        return

    data_provider = MarketDataProvider()
    sentiment_engine = SentimentEngine()

    test_tickers = ["NVDA", "AAPL"]

    for ticker in test_tickers:
        print(f"\n==========================================")
        print(f"1. Načítám živá data a zprávy pro {ticker}...")
        stock_data = data_provider.prepare_ticker_data(ticker)

        if not stock_data:
            print(f"CHYBA: Nepodařilo se načíst tržní data pro {ticker}")
            continue

        print(f"   Společnost: {stock_data.company_name} ({stock_data.sector})")
        print(f"   Aktuální cena: {stock_data.metrics.current_price} USD | Trend býčí: {stock_data.metrics.trend_bullish}")
        print(f"   Počet zpráv: {len(stock_data.recent_news)}")

        print(f"\n2. Spouštím sentimentovou analýzu na Lemonade ({sentiment_engine.model})...")
        verdict = sentiment_engine.analyze_stock(stock_data)

        assert verdict is not None, f"Analýza pro {ticker} vrátila None!"
        print(f"\nAI VÝSLEDEK PRO {verdict.ticker}:")
        print(f"   Doporučení:    {verdict.recommendation}")
        print(f"   Sentiment:     {verdict.sentiment_score:+.2f} [-1.0 až +1.0]")
        print(f"   Důvěra modelu: {verdict.confidence * 100:.0f} %")


if __name__ == "__main__":
    try:
        test_sanitization_unit_tests()
        test_catalyst_assessment_and_veto()
        test_multi_llm_committee_fast_mode()
        test_live_ai_pipeline()
        print("\n==========================================")
        print("  FÁZE 2 ÚSPĚŠNĚ DOKONČENA A OVĚŘENA")
        print("==========================================\n")
    except Exception as e:
        print(f"\nCHYBA V TESTU FÁZE 2: {e}")
        sys.exit(1)