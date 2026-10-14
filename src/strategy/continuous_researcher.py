import logging
import threading
import time
from typing import Dict, List, Optional
from src.data.market_data import MarketDataProvider
from src.llm.sentiment_engine import SentimentEngine
from src.strategy.autonomous_discovery import AutonomousDiscoveryEngine
from src.strategy.intelligence_cache import IntelligenceCache
from src.strategy.opportunity_scanner import OpportunityScanner

logger = logging.getLogger(__name__)


class ContinuousResearcher:
    """
    Nepřetržitý výzkumný modul (Continuous Research Engine).
    Běží 24/7 na pozadí a bez časového tlaku prochází web,
    stahuje nejnovější články, zprávy a fundamentální data pro akcie
    a nechává je analyzovat lokálním modelem na AMD Ryzen AI NPU.
    
    Výsledky ukládá do IntelligenceCache, odkud si je exekuční modul
    okamžitě vytáhne při otevření trhu bez jakéhokoliv zpoždění.
    """

    def __init__(
        self,
        cache: Optional[IntelligenceCache] = None,
        data_provider: Optional[MarketDataProvider] = None,
        sentiment_engine: Optional[SentimentEngine] = None,
        opportunity_scanner: Optional[OpportunityScanner] = None,
        discovery_engine: Optional[AutonomousDiscoveryEngine] = None,
    ):
        self.cache = cache or IntelligenceCache()
        self.data_provider = data_provider or MarketDataProvider()
        self.sentiment_engine = sentiment_engine or SentimentEngine()
        self.opportunity_scanner = opportunity_scanner or OpportunityScanner()
        self.discovery_engine = discovery_engine or AutonomousDiscoveryEngine(scanner=self.opportunity_scanner)

        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._cooldown_tickers: Dict[str, float] = {}
        self._last_discovery_ts: float = 0.0

    def research_stock(self, ticker: str) -> bool:
        """
        Provede hloubkový průzkum jedné akcie:
        - Stáhne zprávy a technická data
        - Spustí lokální AI analýzu na NPU
        - Uloží výsledek do cache
        """
        clean = ticker.strip().upper()
        logger.info(f"🔎 [Continuous Research] Zahajuji hloubkový průzkum webu pro {clean}...")

        stock_data = self.data_provider.prepare_ticker_data(clean)
        if not stock_data:
            logger.warning(f"[Continuous Research] Tržní data pro {clean} nejsou dostupná.")
            return False

        # Získání verdiktu z lokálního NPU
        verdict = self.sentiment_engine.analyze_stock(stock_data)
        if not verdict:
            logger.warning(f"[Continuous Research] Lokální LLM nevrátil validní odpověď pro {clean}.")
            return False

        # Uložení do perzistentní cache
        latest_headline = stock_data.recent_news[0].title if stock_data.recent_news else ""
        self.cache.save_verdict(
            ticker=clean,
            verdict=verdict,
            company_name=stock_data.company_name,
            sector=stock_data.sector,
            article_count=len(stock_data.recent_news),
            latest_headline=latest_headline,
        )
        return True

    def run_one_cycle(self, universe: Optional[List[str]] = None) -> Optional[str]:
        """
        Vybere jeden titul z univerza, který nebyl nejdéle zkoumán,
        a provede jeho hloubkovou analýzu na NPU.
        Pokud některý titul selže (např. dočasně chybí data), zařadí ho do cooldownu
        na 45 minut, aby neblokoval výzkum ostatních titulů.
        """
        all_tickers = universe or self.opportunity_scanner.get_all_universe_tickers()
        now_ts = time.time()

        # Vyfiltrujeme tickery, které nedávno selhaly (cooldown 45 minut)
        active_candidates = [
            t for t in all_tickers
            if (now_ts - self._cooldown_tickers.get(t.strip().upper(), 0.0)) > 2700.0
        ]

        # Pokud by byly v cooldownu všechny, použijeme kompletní univerzum
        candidates_to_use = active_candidates if active_candidates else all_tickers
        next_tickers = self.cache.get_next_tickers_to_research(candidates_to_use, limit=1)

        if not next_tickers:
            return None

        target_ticker = next_tickers[0]
        success = self.research_stock(target_ticker)

        if not success:
            clean = target_ticker.strip().upper()
            self._cooldown_tickers[clean] = now_ts
            logger.info(
                f"⏳ [Continuous Research] Ticker {clean} dočasně zařazen do cooldownu "
                f"(pokračuji ve výzkumu dalších {len(all_tickers) - 1} titulů v pořadí)."
            )

        return target_ticker if success else None

    def _worker_loop(self, pause_between_research_seconds: float = 25.0) -> None:
        """Pracovní smyčka běžící ve vlastním vlákně."""
        logger.info(
            f"🚀 [Continuous Research Worker] Odstartován (pauza mezi akcemi: {pause_between_research_seconds}s). "
            "LLM nepřetržitě analyzuje web a zprávy na pozadí bez časového tlaku."
        )

        while not self._stop_event.is_set():
            try:
                now_ts = time.time()
                # Každých 30 minut (1800 s) prohledáme média na nově zmíněné AI/tech tituly a IPO
                if (now_ts - self._last_discovery_ts) >= 1800.0:
                    self._last_discovery_ts = now_ts
                    logger.info("📡 [Continuous Research] Zahajuji periodické autonomní skenování zpráv na nové firmy a IPO...")
                    try:
                        new_opps = self.discovery_engine.discover_and_register_opportunities(max_new_per_run=3)
                        for opp in new_opps:
                            if self._stop_event.is_set():
                                break
                            logger.info(
                                f"⚡ [Autonomous Discovery] Přednostně zařazuji čerstvý objev {opp.ticker} "
                                f"({opp.company_name}) do okamžité AI analýzy na NPU..."
                            )
                            self.research_stock(opp.ticker)
                    except Exception as disc_exc:
                        logger.warning(f"[Continuous Research] Chyba při autonomním skenování zpráv: {disc_exc}")

                researched = self.run_one_cycle()
                if researched:
                    logger.debug(f"[Continuous Research Worker] Dokončen výzkum pro: {researched}")
            except Exception as exc:
                logger.error(f"[Continuous Research Worker] Chyba v cyklu výzkumu: {exc}")

            # Čekání s možností okamžitého probuzení při stop_event
            self._stop_event.wait(pause_between_research_seconds)

        logger.info("🛑 [Continuous Research Worker] Vlákno výzkumu bylo bezpečně zastaveno.")

    def start_background(self, pause_seconds: float = 25.0) -> None:
        """Nastartuje výzkumníka v pozadí jako daemon thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("[Continuous Research Worker] Vlákno již běží.")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(pause_seconds,),
            name="ContinuousResearcherThread",
            daemon=True
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Zastaví výzkumné vlákno."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
