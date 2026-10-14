import logging
from typing import List, Optional
from src.data.market_data import MarketDataProvider, StockAnalysisData
from src.llm.sentiment_engine import LLMStockVerdict, SentimentEngine
from src.strategy.intelligence_cache import IntelligenceCache
from src.strategy.models import SignalAction, TradingSignal
from src.strategy.opportunity_scanner import OpportunityScanner
from src.strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# Výběr vysoce růstových a volatilních titulů pro aktivní stock picking s vyšším potenciálem
DEFAULT_WATCHLIST: List[str] = [
    "NVDA",   # AI polovodiče & datacentra (Blackwell lídr)
    "TSLA",   # Autonomní technologie, robotika & vysoká volatilita
    "PLTR",   # Enterprise AI (AIP platforma, silné momentum)
    "MSTR",   # Bitcoin treasury proxy s vysokou betou a procentními zisky
    "COIN",   # Kryptoměnová burzovní & custody infrastruktura
    "AMD",    # AI akcelerátory MI325X/MI350 & serverové CPU
    "ARM",    # AI architektonické licence v edge i cloudu
    "AVGO",   # Custom AI ASICs pro hyperscalery & vysokorychlostní sítě
    "META",   # Monetizace AI doporučování & Llama open-source
    "AMZN",   # AWS cloud re-akcelerace & e-commerce
    "AAPL",   # Globální ekosystém & spotřebitelská AI
]


class StockPicker:
    """
    Kvantitativní hodnotící engine kombinující technickou analýzu a AI sentiment.
    Provádí hodnocení jednotlivých akcií a řazení kandidátů pro nákup.
    """

    def __init__(
        self,
        risk_manager: Optional[RiskManager] = None,
        sentiment_engine: Optional[SentimentEngine] = None,
        data_provider: Optional[MarketDataProvider] = None,
        opportunity_scanner: Optional[OpportunityScanner] = None,
        intelligence_cache: Optional[IntelligenceCache] = None,
        watchlist: Optional[List[str]] = None,
    ):
        self.risk_manager = risk_manager or RiskManager()
        self.sentiment_engine = sentiment_engine or SentimentEngine()
        self.data_provider = data_provider or MarketDataProvider()
        self.opportunity_scanner = opportunity_scanner or OpportunityScanner()
        self.intelligence_cache = intelligence_cache or IntelligenceCache()
        self.watchlist = watchlist or DEFAULT_WATCHLIST

    def calculate_technical_score(self, stock_data: StockAnalysisData) -> float:
        """
        Vypočte technické skóre v rozmezí 0.0 až 50.0 bodů:
        - Shoda klouzavých průměrů (Cena > SMA 50 > SMA 200): až 25 bodů
        - RSI momentum a absence přehřátí (RSI 40-68): až 15 bodů
        - Přiměřená ATR volatilita vůči ceně: až 10 bodů
        """
        m = stock_data.metrics
        score = 0.0

        # 1. Trendový filtr (SMA)
        if m.current_price > m.sma_50 and m.sma_50 > m.sma_200:
            score += 25.0
        elif m.current_price > m.sma_50:
            score += 15.0
        elif m.current_price > m.sma_200:
            score += 8.0

        # 2. RSI filtr
        if 42.0 <= m.rsi_14 <= 58.0:
            # Ideální konsolidace před dalším impulsem
            score += 15.0
        elif 58.0 < m.rsi_14 <= 68.0:
            # Silné zdravé momentum
            score += 12.0
        elif 35.0 <= m.rsi_14 < 42.0:
            # Pullback v rámci trendu
            score += 8.0
        elif m.rsi_14 > 72.0:
            # Extrémně překoupeno -> penalizace
            score += 0.0
        elif m.rsi_14 < 35.0:
            # Propad momenta
            score = max(0.0, score - 8.0)

        # 3. ATR volatilita (poměr ATR k ceně akcie)
        if m.current_price > 0:
            atr_pct = (m.atr_14 / m.current_price) * 100
            if 1.2 <= atr_pct <= 4.0:
                score += 10.0
            elif 4.0 < atr_pct <= 6.0:
                score += 6.0
            else:
                score += 3.0

        return min(50.0, max(0.0, round(score, 2)))

    def calculate_sentiment_score(self, verdict: LLMStockVerdict) -> float:
        """
        Vypočte sentimentové skóre v rozmezí 0.0 až 50.0 bodů:
        - Doporučení (BUY: 25 b, HOLD: 10 b, AVOID: 0 b)
        - Sentiment skóre (-1.0 až +1.0 převedeno lineárně): až 25 b
        - Váženo důvěrou modelu (confidence)
        """
        base_recommendation = 0.0
        if verdict.recommendation == "BUY":
            base_recommendation = 25.0
        elif verdict.recommendation == "HOLD":
            base_recommendation = 10.0
        else:
            base_recommendation = 0.0

        # Škálování sentiment_score (kladná hodnota přináší až 25 bodů)
        if verdict.sentiment_score > 0:
            sentiment_component = verdict.sentiment_score * 25.0
        else:
            sentiment_component = 0.0

        raw_score = base_recommendation + sentiment_component
        # Vážení mírou jistoty modelu
        weighted_score = raw_score * max(0.4, verdict.confidence)
        return min(50.0, max(0.0, round(weighted_score, 2)))

    def evaluate_stock(
        self,
        stock_data: StockAnalysisData,
        ai_verdict: LLMStockVerdict
    ) -> TradingSignal:
        """
        Propojí technická data a AI verdikt do celkového TradingSignal.
        """
        tech_score = self.calculate_technical_score(stock_data)
        sent_score = self.calculate_sentiment_score(ai_verdict)
        total_score = round(tech_score + sent_score, 2)

        reasons = []
        m = stock_data.metrics
        f = stock_data.fundamentals

        # 1. Vyhodnocení fundamentálních valuačních rizik a předností
        is_valuation_bubble = False
        is_revenue_declining = False
        valuation_reasons = []

        if f:
            # Ochrana před nákupem extrémně předražených bublin (Forward P/E > 85 bez hyper-růstu)
            if f.forward_pe and f.forward_pe > 85.0 and (f.revenue_growth_yoy is None or f.revenue_growth_yoy < 45.0):
                is_valuation_bubble = True
                valuation_reasons.append(f"Napnutá valuace (Forward P/E {f.forward_pe:.1f} bez hyper-růstu)")
            elif f.trailing_pe and f.trailing_pe > 150.0 and (f.revenue_growth_yoy is None or f.revenue_growth_yoy < 45.0):
                is_valuation_bubble = True
                valuation_reasons.append(f"Vysoké valuační riziko (Trailing P/E {f.trailing_pe:.1f})")

            # Klesající obrat
            if f.revenue_growth_yoy is not None and f.revenue_growth_yoy < -3.0:
                is_revenue_declining = True
                valuation_reasons.append(f"Meziroční pokles tržeb ({f.revenue_growth_yoy:+.1f} %)")

            # Atraktivní GARP (Growth at a Reasonable Price)
            if f.forward_pe and f.forward_pe < 25.0 and f.revenue_growth_yoy and f.revenue_growth_yoy > 20.0:
                valuation_reasons.append(f"Atraktivní GARP profil (Forward P/E {f.forward_pe:.1f} při růstu tržeb {f.revenue_growth_yoy:+.1f} %)")
            elif f.peg_ratio and 0.5 <= f.peg_ratio <= 1.4:
                valuation_reasons.append(f"Výborný poměr valuace k růstu zisků (PEG: {f.peg_ratio:.2f})")

        # 2. Přísné podmínky pro nový BUY (žádné nákupy na překoupených vrcholech RSI > 68)
        is_tech_buyable = (
            (m.trend_bullish and 38.0 <= m.rsi_14 <= 68.0) or
            (m.current_price > m.sma_50 and 38.0 <= m.rsi_14 <= 65.0)
        )
        is_ai_bullish = ai_verdict.recommendation == "BUY" and ai_verdict.sentiment_score >= 0.25

        if is_tech_buyable and is_ai_bullish and not is_valuation_bubble and not is_revenue_declining and total_score >= 70.0:
            action = SignalAction.BUY
            if m.trend_bullish:
                reasons.append("Potvrzen silný technický růstový trend (Cena > SMA 50 > SMA 200)")
            else:
                reasons.append("Cena nad klíčovou podporou SMA 50 ve zdravém růstovém pullbacku")
            reasons.append(f"Pozitivní AI fundamentální analýza ({ai_verdict.sentiment_score:+.2f})")
            reasons.append(f"Zdravé momentum bez překoupení (RSI: {m.rsi_14:.1f})")
            reasons.extend(valuation_reasons)
        elif is_valuation_bubble or is_revenue_declining or m.rsi_14 > 70.0:
            # HOLD: Dobrá firma, ale předražená nebo krátkodobě překoupená
            action = SignalAction.HOLD
            if m.rsi_14 > 70.0:
                reasons.append(f"Krátkodobě překoupený trh (RSI: {m.rsi_14:.1f}) – vyčkání na zdravý pullback")
            reasons.extend(valuation_reasons)
        elif ai_verdict.recommendation == "AVOID" or (not m.trend_bullish and m.current_price < m.sma_50):
            action = SignalAction.SELL
            reasons.append("Technický trend nebo fundamentální sentiment vykazují oslabení")
            reasons.extend(valuation_reasons)
        else:
            action = SignalAction.HOLD
            reasons.append("Podmínky pro nový nákup nejsou plně splněny (vyčkání na optimální vstup)")
            reasons.extend(valuation_reasons)

        # Výpočet dynamických Stop Loss a Take Profit úrovní
        sl_price, tp_price = self.risk_manager.calculate_stops(m.current_price, m.atr_14)

        alloc_eur = 0.0
        if action == SignalAction.BUY:
            alloc_eur = self.risk_manager.calculate_allocation(
                current_open_positions_count=0,
                available_cash_eur=self.risk_manager.total_capital_eur,
                signal_score=total_score,
                atr_14=m.atr_14,
                current_price=m.current_price,
            )

        return TradingSignal(
            ticker=stock_data.ticker,
            action=action,
            total_score=total_score,
            technical_score=tech_score,
            sentiment_score=sent_score,
            confidence=round((ai_verdict.confidence + (tech_score / 50.0)) / 2.0, 2),
            current_price=m.current_price,
            currency=getattr(m, "currency", "USD"),
            suggested_allocation_eur=alloc_eur,
            stop_loss_price=sl_price if action == SignalAction.BUY else None,
            take_profit_price=tp_price if action == SignalAction.BUY else None,
            atr_14=m.atr_14,
            reasons=reasons,
        )

    def scan_opportunities(
        self,
        current_holdings: Optional[List[str]] = None,
        top_candidates_count: int = 8
    ) -> List[TradingSignal]:
        """
        Autonomní globální skenování trhu:
        1. Rychlý kvantitativní screening univerza (30+ akcií) vybere TOP kandidáty s nejvyšším technickým momentem.
        2. K těmto kandidátům připojí držené pozice (pro audit jejich pokračujícího potenciálu).
        3. Pro tyto klíčové tituly spustí hloubkovou lokální AI analýzu na AMD NPU.
        4. Vrátí seřazené signály.
        """
        logger.info("=== ZAHÁJENÍ AUTONOMNÍHO SKENOVÁNÍ TRHU ===")
        # 1. Rychlý kvantitativní screening
        top_candidates = self.opportunity_scanner.fast_quant_screen(top_n=top_candidates_count)
        candidate_tickers = [c.ticker for c in top_candidates]

        # 2. Zařazení držených pozic do auditu, pokud již nejsou mezi kandidáty
        tickers_to_evaluate = list(candidate_tickers)
        if current_holdings:
            for h in current_holdings:
                clean_h = h.strip().upper()
                if clean_h not in tickers_to_evaluate:
                    tickers_to_evaluate.append(clean_h)

        logger.info(
            f"Vybráno {len(tickers_to_evaluate)} klíčových titulů pro hloubkovou AI analýzu na NPU: "
            f"{', '.join(tickers_to_evaluate)}"
        )

        # 3. Hloubková analýza a lokální AI evaluace (s prioritním využitím výzkumné cache)
        signals: List[TradingSignal] = []
        for ticker in tickers_to_evaluate:
            stock_data = self.data_provider.prepare_ticker_data(ticker)
            if not stock_data:
                logger.warning(f"Přeskakuji {ticker}: data nedostupná.")
                continue

            # Nejprve ověříme čerstvý výzkum v perzistentní cache (zpracovaný 24/7 researcherem)
            verdict = self.intelligence_cache.get_verdict(ticker, max_age_hours=36.0)
            if verdict:
                logger.info(
                    f"[Cache Hit] Používám hloubkový výzkum z cache pro {ticker} "
                    f"({verdict.recommendation}, sentiment: {verdict.sentiment_score:+.2f})."
                )
            else:
                logger.info(f"Analyzuji tržní data a spouštím AI predikci na NPU pro {ticker}...")
                verdict = self.sentiment_engine.analyze_stock(stock_data)
                if verdict:
                    # Uložíme do cache pro budoucí okamžité použití
                    self.intelligence_cache.save_verdict(
                        ticker=ticker,
                        verdict=verdict,
                        company_name=stock_data.company_name,
                        sector=stock_data.sector,
                        article_count=len(stock_data.recent_news),
                        latest_headline=stock_data.recent_news[0].title if stock_data.recent_news else ""
                    )

            if not verdict:
                logger.warning(f"Přeskakuji {ticker}: lokální LLM nevrátil validní verdikt.")
                continue

            signal = self.evaluate_stock(stock_data, verdict)
            signals.append(signal)

        # 4. Seřazení podle skóre sestupně
        signals.sort(key=lambda s: s.total_score, reverse=True)
        return signals

    def scan_watchlist(
        self,
        custom_tickers: Optional[List[str]] = None,
        current_holdings: Optional[List[str]] = None
    ) -> List[TradingSignal]:
        """
        Provede skenování. Pokud nejsou zadány custom_tickers,
        spustí plně autonomní vyhledávání příležitostí přes OpportunityScanner.
        """
        if custom_tickers:
            signals: List[TradingSignal] = []
            logger.info(f"Skenuji specifikované tickery ({len(custom_tickers)} titulů)...")
            for ticker in custom_tickers:
                stock_data = self.data_provider.prepare_ticker_data(ticker)
                if not stock_data:
                    continue
                verdict = self.sentiment_engine.analyze_stock(stock_data)
                if not verdict:
                    continue
                signal = self.evaluate_stock(stock_data, verdict)
                signals.append(signal)
            signals.sort(key=lambda s: s.total_score, reverse=True)
            return signals

        # Výchozí autonomní režim
        return self.scan_opportunities(current_holdings=current_holdings)
