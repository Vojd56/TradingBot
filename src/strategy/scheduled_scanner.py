from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import schedule

from src.data.market_data import MarketDataProvider, StockAnalysisData
from src.llm.multi_llm_committee import CommitteeConsensus, MultiLLMCommittee
from src.strategy.autonomous_discovery import AutonomousDiscoveryEngine
from src.strategy.intelligence_cache import IntelligenceCache
from src.strategy.market_hours import MarketHours
from src.strategy.opportunity_scanner import OpportunityScanner

logger = logging.getLogger(__name__)

QUALIFIED_OPPORTUNITIES_FILE = Path("data") / "qualified_opportunities.json"


class ScheduledMarketScanner:
    """
    Dvoufázový autonomní skener globálního trhu (Scheduled Market & Multi-LLM Scanner).
    
    Odděluje strategické hledání příležitostí od samotného tradingu:
    - 1. Ranní evropský sken (09:30 CET): Projde evropské tituly a globální compoundery.
    - 2. Odpolední americký sken (16:00 CET): Projde americké tech giganty, páková ETF a AI novinky.
    
    Každý kandidát je podroben křížovému zkoumání v Radě AI (Multi-LLM Committee):
    - Fundamentální analytik (hodnota a kvalita)
    - Skeptický auditor (kontrola rizik a klamavého růstu - právo VETA)
    - Katalytický specialista (zprávy a momentum)
    
    Pouze tituly s potvrzeným konsenzem jsou uloženy do souboru 'data/qualified_opportunities.json'
    pro okamžitou spotřebu exekučním traderem.
    """

    def __init__(
        self,
        opportunity_scanner: Optional[OpportunityScanner] = None,
        data_provider: Optional[MarketDataProvider] = None,
        committee: Optional[MultiLLMCommittee] = None,
        intelligence_cache: Optional[IntelligenceCache] = None,
        discovery_engine: Optional[AutonomousDiscoveryEngine] = None,
        output_file: Optional[Path] = None,
    ):
        self.opportunity_scanner = opportunity_scanner or OpportunityScanner()
        self.data_provider = data_provider or MarketDataProvider()
        self.committee = committee or MultiLLMCommittee()
        self.intelligence_cache = intelligence_cache or IntelligenceCache()
        self.discovery_engine = discovery_engine or AutonomousDiscoveryEngine(scanner=self.opportunity_scanner)
        self.output_file = output_file or QUALIFIED_OPPORTUNITIES_FILE

    def run_deep_scan(
        self,
        session_label: str = "MANUAL",
        top_quant_candidates: int = 12,
        include_discovery: bool = True,
    ) -> List[CommitteeConsensus]:
        """
        Provede kompletní hloubkový sken trhu s Multi-LLM konsenzem:
        1. Autonomní discovery nových titulů ze světových médií.
        2. Kvantitativní screening celého univerza.
        3. Rada AI modelů prověří top kandidáty a otestuje pravdivost hypotézy.
        4. Uloží schválené příležitosti do perzistentní databáze pro trading engine.
        """
        now = datetime.now()
        print("\n" + "=" * 80)
        print(f"  🔍 HLOUBKOVÝ SKEN TRHU & RADA AI ANALYTIKŮ [{session_label}] ({now.strftime('%d.%m.%Y %H:%M:%S')})")
        print("=" * 80)

        # 1. Autonomní discovery ze zpráv
        if include_discovery:
            print("1. Prohledávám globální zprávy na nové AI firmy a kontrakty...")
            try:
                new_disc = self.discovery_engine.discover_and_register_opportunities(max_new_per_run=4)
                if new_disc:
                    disc_tickers = ", ".join([d.ticker for d in new_disc])
                    print(f"   ✨ Nově objeveno v médiích: {disc_tickers}")
                else:
                    print("   Žádná nová IPO/kontrakty nezaznamenány, univerzum je aktuální.")
            except Exception as e:
                logger.debug(f"Discovery chyba: {e}")

        # 2. Rychlý kvantitativní screening trhu
        print(f"\n2. Kvantitativní screening 65+ globálních titulů (výběr TOP {top_quant_candidates} momentum kandidátů)...")
        candidates = self.opportunity_scanner.fast_quant_screen(top_n=top_quant_candidates)
        print(f"   Vybráno {len(candidates)} nejsilnějších kandidátů pro zasedání Rady AI.")

        # 3. Zasedání Rady AI a ověření pravdivosti (Multi-LLM Consensus)
        print("\n3. Spouštím zasedání Rady AI (Fundamentální analytik vs. Skeptický auditor vs. Katalyzátory)...")
        approved_consensuses: List[CommitteeConsensus] = []
        all_consensuses: List[CommitteeConsensus] = []

        for idx, cand in enumerate(candidates, 1):
            ticker = cand.ticker
            print(f"   [{idx}/{len(candidates)}] Prověřuji {ticker:<7} (Technické skóre: {cand.score:.1f}/50)...", end="", flush=True)

            stock_data = self.data_provider.prepare_ticker_data(ticker)
            if not stock_data:
                print(" ❌ Data nedostupná")
                continue

            consensus = self.committee.evaluate_consensus(stock_data)
            all_consensuses.append(consensus)

            # Uložení do IntelligenceCache pro zpětnou vazbu celého systému
            v_fund = consensus.agent_verdicts.get("FUNDAMENTAL_ANALYST")
            if v_fund:
                self.intelligence_cache.save_verdict(
                    ticker=ticker,
                    verdict=v_fund.model_dump(mode="python"),  # type: ignore
                    company_name=stock_data.company_name,
                    sector=stock_data.sector,
                    article_count=len(stock_data.recent_news),
                    latest_headline=stock_data.recent_news[0].title if stock_data.recent_news else "",
                )

            if consensus.approved_for_trading:
                approved_consensuses.append(consensus)
                print(f" 🟢 SCHVÁLENO ({consensus.consensus_strength} - Shoda: {consensus.agreement_percentage:.0f} %)")
            else:
                veto_str = " (🛑 VETO)" if consensus.consensus_strength == "VETO_TRIGGERED" else ""
                print(f" ⚪ ZAMÍTNUTO / VYČKÁVÁNÍ{veto_str} ({consensus.final_recommendation})")

        # 4. Uložení schválených kandidátů do JSONu pro Trading Engine
        self._save_qualified_opportunities(approved_consensuses, session_label)

        # 5. Zobrazení souhrnné tabulky pro uživatele
        self._print_scan_summary_table(all_consensuses, approved_consensuses)

        return approved_consensuses

    def _save_qualified_opportunities(
        self,
        consensuses: List[CommitteeConsensus],
        session_label: str
    ) -> None:
        """Uloží schválené příležitosti s konsenzem do JSON souboru pro okamžité použití tradingem."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_scan_timestamp": time.time(),
            "last_scan_iso": datetime.now().isoformat(),
            "session_label": session_label,
            "qualified_count": len(consensuses),
            "opportunities": [c.model_dump(mode="json") for c in consensuses],
        }
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Schválené příležitosti ({len(consensuses)}) uloženy do {self.output_file}")

    @classmethod
    def load_qualified_tickers(cls, max_age_hours: float = 24.0) -> List[str]:
        """Načte seznam schválených tickerů pro trading engine."""
        if not QUALIFIED_OPPORTUNITIES_FILE.exists():
            return []
        try:
            with open(QUALIFIED_OPPORTUNITIES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            last_ts = data.get("last_scan_timestamp", 0.0)
            if (time.time() - last_ts) > (max_age_hours * 3600):
                logger.warning("Seznam schválených příležitostí je starší než 24 hodin.")
                return []
            opps = data.get("opportunities", [])
            return [str(o.get("ticker")) for o in opps if o.get("approved_for_trading")]
        except Exception as exc:
            logger.debug(f"Chyba při načítání schválených příležitostí: {exc}")
            return []

    def _print_scan_summary_table(
        self,
        all_consensuses: List[CommitteeConsensus],
        approved: List[CommitteeConsensus]
    ) -> None:
        """Vytiskne přehlednou tabulku výsledků zasedání Rady AI."""
        print("\n" + "=" * 90)
        print("          🏛️ VÝSLEDKY ZASEDÁNÍ RADY AI ANALYTIKŮ & KONSENZUS           ")
        print("=" * 90)
        print(f"{'Ticker':<7} | {'Verdikt':<10} | {'Shoda Rady':<12} | {'Sentiment':<10} | {'Důvěra':<8} | {'Syntéza a klíčový verdikt'}")
        print("-" * 90)

        for c in all_consensuses:
            status_badge = "🟢 BUY" if c.approved_for_trading else ("🛑 VETO" if c.consensus_strength == "VETO_TRIGGERED" else "⚪ HOLD")
            shoda_str = f"{c.agreement_percentage:.0f} % ({c.consensus_strength[:8]})"
            driver = c.consolidated_catalysts[0] if c.consolidated_catalysts else c.consensus_synthesis[:40]
            print(
                f"{c.ticker:<7} | {status_badge:<10} | {shoda_str:<12} | "
                f"{c.average_sentiment:+6.2f}     | {c.average_confidence*100:4.0f} %  | {driver[:35]}"
            )

        print("-" * 90)
        print(f"Schváleno pro trading: {len(approved)} z {len(all_consensuses)} analyzovaných společností.")
        if approved:
            tickers_str = ", ".join([f"{a.ticker} ({a.agreement_percentage:.0f} % shoda)" for a in approved])
            print(f"✨ Schválené tituly připravené pro trading: {tickers_str}")
            print(f"📁 Uloženo do: {self.output_file} (připraveno pro bleskovou exekuci)")
        else:
            print("ℹ️ Žádný titul nezískal potřebnou většinu a shodu bez výhrad (kapitál zůstává 100% ochráněn).")
        print("=" * 90 + "\n")

    def run_morning_europe_scan(self) -> List[CommitteeConsensus]:
        """Ranní evropský sken (09:30 CET): XETRA, LSE, Euronext, globální obrana a compoundery."""
        return self.run_deep_scan(session_label="RANNÍ EVROPSKÝ SKEN (09:30 CET)", top_quant_candidates=10)

    def run_afternoon_us_scan(self) -> List[CommitteeConsensus]:
        """Odpolední americký sken (16:00 CET): Wall Street (NYSE/NASDAQ), AI lídři, páková ETF a zprávy."""
        return self.run_deep_scan(session_label="ODPOLEDNÍ AMERICKÝ SKEN (16:00 CET)", top_quant_candidates=12)

    def start_scheduler_loop(self) -> None:
        """
        Nastartuje nepřetržitý plánovač pro běh 2x denně v optimální tržní časy:
        - 09:30 CET: Ranní evropská seance
        - 16:00 CET: Odpolední americká seance Wall Street
        """
        print("\n" + "=" * 70)
        print("          ⏰ PLÁNOVAČ AUTOMATICKÉHO SKENOVÁNÍ TRHU (2x DENNĚ)          ")
        print("=" * 70)
        print("  1. Ranní evropský sken:    Každý všední den v 09:30 CET")
        print("  2. Odpolední americký sken: Každý všední den v 16:00 CET")
        print("  Výstupní soubor:           data/qualified_opportunities.json")
        print("  Multi-LLM Rada:            AKTIVNÍ (Fundament + Skeptik + Katalyzátory)")
        print("=" * 70)
        print("Stiskněte Ctrl+C pro ukončení plánovače.\n")

        # Nastavení plánovače
        schedule.every().monday.at("09:30").do(self.run_morning_europe_scan)
        schedule.every().tuesday.at("09:30").do(self.run_morning_europe_scan)
        schedule.every().wednesday.at("09:30").do(self.run_morning_europe_scan)
        schedule.every().thursday.at("09:30").do(self.run_morning_europe_scan)
        schedule.every().friday.at("09:30").do(self.run_morning_europe_scan)

        schedule.every().monday.at("16:00").do(self.run_afternoon_us_scan)
        schedule.every().tuesday.at("16:00").do(self.run_afternoon_us_scan)
        schedule.every().wednesday.at("16:00").do(self.run_afternoon_us_scan)
        schedule.every().thursday.at("16:00").do(self.run_afternoon_us_scan)
        schedule.every().friday.at("16:00").do(self.run_afternoon_us_scan)

        # Provedeme úvodní sken ihned při startu, pokud soubor ještě neexistuje
        if not self.output_file.exists():
            print("🚀 Provádím úvodní inicializační sken trhu...")
            self.run_deep_scan(session_label="INICIALIZAČNÍ SKEN", top_quant_candidates=10)

        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n🛑 Plánovač skenování byl bezpečně ukončen.")
