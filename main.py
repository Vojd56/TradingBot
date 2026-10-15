import os
from pathlib import Path
import subprocess
import sys

# Automatické přepnutí na virtuální prostředí venv, pokud byl skript spuštěn globálním pythonem
if not (hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)):
    venv_py = Path(__file__).resolve().parent / "venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        try:
            res = subprocess.run([str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])
            sys.exit(res.returncode)
        except KeyboardInterrupt:
            sys.exit(0)

import argparse
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import time
from typing import Any, Dict, Optional
import schedule

from config import settings
from src.broker.t212_client import Trading212Client
from src.data.market_data import MarketDataProvider
from src.execution import (
    OrderExecutor,
    PortfolioReconciler,
    PortfolioTracker,
    TickerMapper,
)
from src.llm.multi_llm_committee import MultiLLMCommittee
from src.llm.sentiment_engine import SentimentEngine
from src.strategy import RiskManager, StockPicker
from src.strategy.autonomous_discovery import AutonomousDiscoveryEngine
from src.strategy.continuous_researcher import ContinuousResearcher
from src.strategy.day_trader import DayTraderEngine
from src.strategy.intelligence_cache import IntelligenceCache
from src.strategy.market_hours import MarketHours
from src.strategy.scheduled_scanner import ScheduledMarketScanner
from src.strategy.trade_reflection import TradeReflectionEngine

# Vytvoření adresáře pro logy a reporty
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = LOGS_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Konfigurace rotujícího logování
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

file_handler = RotatingFileHandler(
    LOGS_DIR / "trading_bot.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger("TradingBot")


class TradingBotApp:
    """
    Hlavní řídicí aplikace AI Trading Bota.
    Sdružuje datovou vrstvu, AI sentiment engine, kvantitativní strategii a exekuční modul.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

        logger.info("Inicializuji komponenty Trading Bota...")
        self.client = Trading212Client()
        self.data_provider = MarketDataProvider()
        self.sentiment_engine = SentimentEngine()
        self.risk_manager = RiskManager()
        self.tracker = PortfolioTracker()
        self.intelligence_cache = IntelligenceCache()
        self.picker = StockPicker(
            risk_manager=self.risk_manager,
            sentiment_engine=self.sentiment_engine,
            data_provider=self.data_provider,
            intelligence_cache=self.intelligence_cache,
        )
        self.reconciler = PortfolioReconciler(
            client=self.client,
            tracker=self.tracker,
            risk_manager=self.risk_manager,
            data_provider=self.data_provider,
            sentiment_engine=self.sentiment_engine,
            dry_run=self.dry_run,
        )
        self.discovery_engine = AutonomousDiscoveryEngine(scanner=self.picker.opportunity_scanner)
        self.committee = MultiLLMCommittee(sentiment_engine=self.sentiment_engine)
        self.market_scanner = ScheduledMarketScanner(
            opportunity_scanner=self.picker.opportunity_scanner,
            data_provider=self.data_provider,
            committee=self.committee,
            intelligence_cache=self.intelligence_cache,
            discovery_engine=self.discovery_engine,
        )
        self.researcher = ContinuousResearcher(
            cache=self.intelligence_cache,
            data_provider=self.data_provider,
            sentiment_engine=self.sentiment_engine,
            opportunity_scanner=self.picker.opportunity_scanner,
            discovery_engine=self.discovery_engine,
        )

    def print_banner(self) -> None:
        """Vytiskne úvodní banner aplikace s konfigurací."""
        mode_str = f"[{settings.trading_mode.upper()} REŽIM]"
        if settings.trading_mode == "live":
            mode_str += " ⚠️ REÁLNÉ PENÍZE ⚠️"
        else:
            mode_str += " (Cvičný demo účet)"

        print("\n" + "=" * 70)
        print("          AI QUANTITATIVE TRADING BOT (Trading 212 & AMD NPU)      ")
        print("=" * 70)
        print(f"  Prostředí:         {mode_str}")
        print(f"  Broker API:        {settings.t212_base_url}")
        print(f"  LLM Server:        {settings.lemonade_base_url}")
        print(f"  LLM Model:         {settings.lemonade_model}")
        print(f"  Testovací kapitál: {settings.initial_capital_eur} EUR")
        print(f"  Max pozic:         {settings.max_positions}")
        print(f"  Dry-run režim:     {'AKTIVNÍ (žádné reálné příkazy)' if self.dry_run else 'NE (reálná exekuce na demo)'}")
        print("=" * 70 + "\n")

    def show_status(self) -> None:
        """Zobrazí detailní přehled o stavu účtu, držených pozicích a zisku/ztrátě."""
        self.print_banner()
        print("--- STAV ÚČTU A PORTFOLIA ---")
        try:
            summary = self.client.get_account_summary()
            cash = self.client.get_account_cash()
            currency = summary.get("currency", "CZK")

            print(f"Měna účtu:           {currency}")
            print(f"Celková hodnota:     {summary.get('totalValue', 0.0):,.2f} {currency}")
            print(f"Investováno:         {cash.get('invested', 0.0):,.2f} {currency}")
            print(f"Volná hotovost:      {cash.get('free', 0.0):,.2f} {currency}")
            print(f"Nerealizovaný PnL:   {cash.get('ppl', 0.0):+,.2f} {currency}")

            # Synchronizace a výpis pozic
            broker_positions = self.reconciler.sync_broker_portfolio()
            open_positions = self.tracker.get_open_positions()

            print(f"\n--- OTEVŘENÉ POZICE ({len(open_positions)} / {settings.max_positions}) ---")
            if not open_positions:
                print("Aktuálně nejsou otevřeny žádné pozice.")
            else:
                for ticker, pos in open_positions.items():
                    print(
                        f"• {ticker:<8} | Množství: {pos.quantity:<10} | "
                        f"Nákup: {pos.entry_price_usd:.2f} USD | "
                        f"Stop Loss: {pos.stop_loss_price:.2f} USD | "
                        f"Take Profit: {pos.take_profit_price:.2f} USD | "
                        f"Alokováno: {pos.allocated_eur:.2f} EUR"
                    )

            print(f"\nPoslední aktualizace perzistentního stavu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        except Exception as exc:
            logger.error(f"Chyba při zjišťování stavu účtu: {exc}")

    def run_daily_cycle(self, force: bool = False) -> Dict[str, Any]:
        """
        Provede kompletní denní analytický a exekuční cyklus.
        """
        now = datetime.now()
        day_of_week = now.weekday()  # 0=Po, 4=Pá, 5=So, 6=Ne

        # Ochrana před spouštěním o víkendu (trhy jsou zavřené)
        if day_of_week in (5, 6) and not force:
            logger.info("Dnes je víkend (US trhy jsou zavřené). Cyklus byl přeskočen.")
            return {"status": "SKIPPED_WEEKEND", "date": now.isoformat()}

        logger.info(f"=== ZAHÁJENÍ DENNÍHO CYKLU ({now.strftime('%Y-%m-%d %H:%M:%S')}) ===")

        # 0. Autonomní skenování čerstvých zpráv na nové AI/tech tituly a IPO
        try:
            discovered = self.discovery_engine.discover_and_register_opportunities(max_new_per_run=3)
            if discovered:
                logger.info(f"✨ [Autonomous Discovery] Do univerza nově zařazeny tituly z médií: {[d.ticker for d in discovered]}")
        except Exception as disc_err:
            logger.debug(f"Autonomní discovery zpráv: {disc_err}")

        # 1. Synchronizace s brokerem pro zjištění aktuálně otevřených pozic
        self.reconciler.sync_broker_portfolio()
        current_holdings = list(self.tracker.get_open_positions().keys())

        # 2. Autonomní skenování trhu a hloubková lokální AI analýza na NPU
        logger.info(
            f"Krok 1/3: Spouštím autonomní skener globálního trhu a AI predikci na NPU "
            f"(audituji {len(current_holdings)} držených pozic)..."
        )
        signals = self.picker.scan_opportunities(current_holdings=current_holdings, top_candidates_count=12)

        logger.info(f"Vygenerováno {len(signals)} investičních signálů:")
        for s in signals:
            curr_str = getattr(s, "currency", "USD")
            logger.info(
                f"  - {s.ticker:<7}: {s.action.value:<4} | Skóre: {s.total_score:5.1f}/100 "
                f"(Tech: {s.technical_score:.1f}, Sent: {s.sentiment_score:.1f}) | Cena: {s.current_price:.2f} {curr_str}"
            )

        # 2. Reconcile & Exekuce
        logger.info("Krok 2/3: Provádím kontrolu otevřených pozic a exekuci obchodů...")
        reconcile_result = self.reconciler.run_cycle(candidate_signals=signals)

        # 3. Vygenerování denního reportu
        logger.info("Krok 3/3: Sestavuji denní auditní report...")
        report = {
            "date": now.strftime("%Y-%m-%d"),
            "timestamp": now.isoformat(),
            "dry_run": self.dry_run,
            "signals_evaluated": [s.model_dump(mode="json") for s in signals],
            "execution_summary": reconcile_result,
        }

        # Uložení reportu do JSON souboru
        report_file = REPORTS_DIR / f"daily_report_{now.strftime('%Y%m%d')}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Denní auditní report uložen do: {report_file}")
        logger.info("=== DENNÍ CYKLUS ÚSPĚŠNĚ DOKONČEN ===")
        return report

    def start_scheduler(self, run_time: str = "21:00") -> None:
        """
        Nastartuje plánovač úloh pro běh v obchodních hodinách (výchozí: 21:00 CET).
        Podporuje jeden čas nebo čárkou oddělený seznam (např. '16:30,21:00').
        """
        self.print_banner()
        times = [t.strip() for t in run_time.split(",") if t.strip()]
        logger.info(f"Plánovač nastaven na spouštění v obchodních hodinách v: {', '.join(times)} CET.")
        logger.info("Při spuštění v obchodních hodinách probíhá okamžitá exekuce (instant fill).")
        logger.info("Stiskněte Ctrl+C pro bezpečné ukončení bota.")

        for t in times:
            schedule.every().day.at(t).do(self.run_daily_cycle)

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Trading Bot byl zastaven uživatelem (SIGINT / KeyboardInterrupt).")


    def show_research_status(self) -> None:
        """Zobrazí přehled všech zanalyzovaných akcií z perzistentní výzkumné cache."""
        self.print_banner()
        print("=== STAV NEPŘETRŽITÉHO VÝZKUMU (INTELLIGENCE CACHE) ===")
        entries = self.intelligence_cache.get_all_entries()
        market_stat = MarketHours.get_market_status()
        print(f"Stav trhu:          {market_stat['status_text']}")
        print(f"Lokální čas:        {market_stat['local_time']}")
        print(f"Zanalyzováno titulů: {len(entries)}\n")

        if not entries:
            print("Výzkumná cache je zatím prázdná. Spusťte bota s --daemon pro zahájení 24/7 výzkumu.")
            return

        print(f"{'Ticker':<7} | {'Doporučení':<10} | {'Sentiment':<9} | {'Jistota':<7} | {'Článků':<7} | {'Poslední analýza':<16} | {'Klíčový katalyzátor'}")
        print("-" * 105)
        for e in entries:
            analyzed_dt = datetime.fromtimestamp(e.last_analyzed_timestamp).strftime("%d.%m. %H:%M") if e.last_analyzed_timestamp else "N/A"
            driver = e.key_drivers[0] if e.key_drivers else e.latest_headline[:40]
            print(
                f"{e.ticker:<7} | {e.recommendation:<10} | {e.sentiment_score:+6.2f}    | "
                f"{e.confidence*100:4.0f} %  | {e.article_count:<7} | {analyzed_dt:<16} | {driver[:45]}"
            )
        print("\n")

    def discover_opportunities(self) -> None:
        """
        Vyhledá a ověří nové akcie, AI tituly a technologická IPO z globálních médií.
        Prověří tržní kapitalizaci, likviditu a cenu akcie a zařadí je do univerza bota.
        """
        self.print_banner()
        print("=== AUTONOMNÍ VYHLEDÁVÁNÍ NOVÝCH PŘÍLEŽITOSTÍ (BREAKING NEWS & IPO) ===")
        print("Skenuji breaking news, AI kontrakty, partnerství a nová technologická IPO...")
        new_opps = self.discovery_engine.discover_and_register_opportunities(max_new_per_run=10)
        if not new_opps:
            print("V aktuálním cyklu nebyly nalezeny žádné nové tituly splňující institucionální síto")
            print("(nebo již jsou v univerzu sledovaných společností).")
            print("Tip: Běžící daemon prohledává zprávy automaticky každých 30 minut.\n")
            return

        print(f"\nÚspěšně objeveno a kvalifikováno {len(new_opps)} nových společností ze světových médií:\n")
        print(f"{'Ticker':<7} | {'Společnost':<28} | {'Cena':<10} | {'Tržní kap.':<12} | {'Zdroj zprávy'}")
        print("-" * 105)
        for opp in new_opps:
            print(
                f"{opp.ticker:<7} | {opp.company_name[:26]:<28} | {opp.price:6.2f} USD  | "
                f"{opp.market_cap_usd/1e6:8.1f}M USD | {opp.headline[:45]}"
            )
        print("\n✨ Tyto tituly byly dynamicky zařazeny do univerza bota.")
        print("NPU model a kvantitativní skener je nyní automaticky analyzují a připravují k nákupu.\n")

    def scan_market_now(self, top_candidates: int = 12) -> None:
        """Spustí okamžitý hloubkový sken globálního trhu s Multi-LLM konsenzem."""
        self.print_banner()
        self.market_scanner.run_deep_scan(session_label="OKAMŽITÝ SKEN", top_quant_candidates=top_candidates)

    def schedule_market_scans(self) -> None:
        """Spustí plánovač pro dvoufázové denní skenování trhu (09:30 a 16:00 CET)."""
        self.print_banner()
        self.market_scanner.start_scheduler_loop()

    def show_scanner_status(self) -> None:
        """Zobrazí aktuální stav a schválené příležitosti z Rady AI v databázi."""
        self.print_banner()
        fpath = self.market_scanner.output_file
        if not fpath.exists():
            print("Databáze schválených příležitostí zatím neexistuje.")
            print("Spusťte skenování pomocí: python main.py --scan")
            return

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            opps = data.get("opportunities", [])
            last_iso = data.get("last_scan_iso", "N/A")
            label = data.get("session_label", "N/A")

            print(f"=== SCHVÁLENÉ PŘÍLEŽITOSTI S KONSENZEM RADY AI ===")
            print(f"Poslední sken:    {last_iso} [{label}]")
            print(f"Schváleno titulů: {len(opps)}\n")

            if not opps:
                print("V posledním skenu nezískal žádný titul potřebný konsenzus bez výhrad.")
                return

            print(f"{'Ticker':<7} | {'Verdikt':<8} | {'Shoda':<12} | {'Sentiment':<10} | {'Jistota':<8} | {'Syntéza a odůvodnění'}")
            print("-" * 95)
            for o in opps:
                dr = o.get("consolidated_catalysts", [""])[0] if o.get("consolidated_catalysts") else o.get("consensus_synthesis", "")[:40]
                shoda_str = f"{o.get('agreement_percentage', 0):.0f} % ({o.get('consensus_strength', '')[:8]})"
                print(
                    f"{o.get('ticker'):<7} | {o.get('final_recommendation'):<8} | {shoda_str:<12} | "
                    f"{o.get('average_sentiment', 0.0):+6.2f}     | {o.get('average_confidence', 0.5)*100:4.0f} %  | {dr[:40]}"
                )
            print("\n")
        except Exception as exc:
            print(f"Chyba při čtení schválených příležitostí: {exc}")

    def show_trade_memory(self) -> None:
        """Zobrazí auditovanou paměť obchodů, reálné účetní metriky a sebereflexi chyb z Trading 212 API."""
        self.print_banner()
        print("=== 🧠 OFICIÁLNÍ AUDIT PORTFOLIA & PAMĚŤ CHYB (API RECONCILIATION) ===")
        engine = TradeReflectionEngine(sentiment_engine=self.sentiment_engine)
        
        try:
            print("⏳ Ověřuji historii a stavy přes Trading 212 API...")
            metrics = engine.reconcile_with_broker_api(self.client)
        except Exception as exc:
            logger.warning(f"Chyba při živé rekonciliaci: {exc}. Používám lokální data.")
            metrics = engine.account_metrics

        curr = metrics.get("currency", "CZK")
        total_equity = metrics.get("total_account_equity", 0.0)
        free_cash = metrics.get("free_cash", 0.0)
        invested = metrics.get("invested_value", 0.0)
        realized_pnl = metrics.get("realized_pnl", 0.0)
        unrealized_pnl = metrics.get("unrealized_pnl", 0.0)
        fees = metrics.get("total_fees_paid", 0.0)
        total_closed = metrics.get("total_closed_trades", len(engine.trade_logs))
        wins = metrics.get("winning_trades", 0)
        losses = metrics.get("losing_trades", 0)
        win_rate = metrics.get("win_rate_percent", 0.0)
        open_count = metrics.get("open_positions_count", 0)

        # 1. ZÁKLADNÍ ÚČETNÍ METRIKY (PŘÍSNĚ ODDĚLENÉ)
        print("\n📊 [STAV ÚČTU A KAPITÁL - 100% VERIFIKOVÁNO Z API]:")
        print(f"  • Total Account Equity:     {total_equity:>12,.2f} {curr}")
        print(f"  • Volná hotovost (Cash):    {free_cash:>12,.2f} {curr}")
        print(f"  • Investovaná hodnota:      {invested:>12,.2f} {curr}")
        pnl_real_badge = f"+{realized_pnl:,.2f}" if realized_pnl >= 0 else f"{realized_pnl:,.2f}"
        pnl_unreal_badge = f"+{unrealized_pnl:,.2f}" if unrealized_pnl >= 0 else f"{unrealized_pnl:,.2f}"
        print(f"  • Realized PnL (Zisk/Ztr.): {pnl_real_badge:>12} {curr}")
        print(f"  • Unrealized PnL (Běžící):  {pnl_unreal_badge:>12} {curr}")
        print(f"  • Poplatky za konverzi FX:  {fees:>12,.2f} {curr}")
        print(f"  • Otevřené pozice:          {open_count:>12} ks")
        print(f"  • Statistika exekucí:       {wins} Ziskových / {losses} Ztrátových | Win Rate: {win_rate:.1f} %")

        # 2. TABULKA VERIFIKOVANÝCH OBCHODŮ
        print("\n📜 [OFICIÁLNÍ HISTORIE UZAVŘENÝCH OBCHODŮ (TRADING 212 API)]:")
        print(f"{'Order ID':<13} | {'Ticker':<7} | {'Směr':<5} | {'Cena':>9} | {'Množství':>9} | {'Hrubý objem':>12} | {'Poplatek':>9} | {'Realizovaný PnL':>17} | {'Ověřeno'}")
        print("-" * 115)
        for t in engine.trade_logs[-15:]:
            pnl_str = f"{t.realized_pnl:+,.2f} {curr} ({t.realized_pnl_percent:+.2f} %)"
            verified_tag = "✅ API ANO" if t.api_verified else "❌ NE"
            print(
                f"{t.order_id:<13} | {t.asset:<7} | {t.direction:<5} | {t.execution_price:>9.2f} | {t.quantity:>9.4f} | "
                f"{t.gross_amount:>9,.2f} {curr} | {t.fees:>6.2f} {curr} | {pnl_str:>17} | {verified_tag}"
            )

        # 3. KONTROLA CHYBNÝCH / INVALIDOVANÝCH ZÁZNAMŮ
        if engine.invalid_unverified_records:
            print(f"\n⚠️ [AUDITNÍ NÁLEZ - INVALIDOVANÉ SIMULOVANÉ ZÁZNAMY]:")
            print(f"  Detekováno a vyřazeno celkem {len(engine.invalid_unverified_records)} záznamů bez potvrzené exekuce na burze.")
            print(f"  Status: [INVALID/UNVERIFIED - SIMULATED LOOP ARTIFACT]. Fiktivní metriky byly z paměti kompletně vymazány.")

        # 4. AKTIVNÍ HEURISTIKY A PONAUČENÍ Z REÁLNÝCH ZTRÁT
        if engine.reflections:
            print("\n🧠 [AKTIVNÍ PREVENTIVNÍ PRAVIDLA & VETO Z REÁLNÝCH ZTRÁT]:")
            for r in engine.reflections[-6:]:
                print(f"  • {r.ticker:<7} [{r.mistake_category.value}]: {r.lesson_learned}")
        print("\n")

    def evaluate_committee_ticker(self, ticker: str) -> None:
        """Provede detailní oponenturu Rady AI pro jeden konkrétní titul."""
        self.print_banner()
        clean = ticker.strip().upper()
        print(f"=== ZASEDÁNÍ RADY AI ANALYTIKŮ PRO TITUL: {clean} ===\n")
        stock_data = self.data_provider.prepare_ticker_data(clean)
        if not stock_data:
            import yfinance as yf
            try:
                search_results = yf.Search(clean).quotes
                alts = [
                    f"{q['symbol']} ({q.get('longname') or q.get('shortname') or q.get('symbol')})"
                    for q in search_results if q.get("symbol") != clean
                ]
                if alts:
                    print(f"💡 Ticker '{clean}' nebyl na americkém trhu nalezen. Možná jste mysleli:")
                    for a in alts[:4]:
                        print(f"   • {a}")
                    print("\nTip: Evropské akcie vyžadují příponu burzy (např. .AS pro Amsterdam, .PR pro Prahu, .DE pro Frankfurt).")
            except Exception:
                pass
            print(f"Nepodařilo se načíst tržní data pro {clean}.\n")
            return

        consensus = self.committee.evaluate_consensus(stock_data)

        print(f"\nSPOLEČNOST: {clean} ({stock_data.company_name}) | Sektor: {stock_data.sector}")
        print(f"Aktuální cena: {stock_data.metrics.current_price:.2f} | Trend: {'BÝČÍ' if stock_data.metrics.trend_bullish else 'MEDVĚDÍ'} | RSI: {stock_data.metrics.rsi_14:.1f}\n")

        print("--- INDIVIDUÁLNÍ VERDIKTY RADY AI ---")
        for role_name, v in consensus.agent_verdicts.items():
            print(f"\n👤 [ROLE: {role_name}]")
            print(f"   Doporučení:     {v.recommendation}")
            print(f"   Sentiment:      {v.sentiment_score:+.2f} [-1.0 až +1.0] | Jistota: {v.confidence*100:.0f} %")
            print(f"   Argumenty:      {', '.join(v.key_arguments) if v.key_arguments else 'N/A'}")
            print(f"   Červené vlajky: {', '.join(v.red_flags) if v.red_flags else 'N/A'}")
            print(f"   Odůvodnění:     {v.reasoning}")

        print("\n" + "=" * 70)
        print(f"🏛️ VÝSLEDNÝ KONSENZUS: {consensus.consensus_strength} | Shoda: {consensus.agreement_percentage:.0f} %")
        print(f"   Finální doporučení:    {consensus.final_recommendation}")
        print(f"   Schváleno pro trading: {'🟢 ANO' if consensus.approved_for_trading else '🔴 NE'}")
        print(f"   Syntéza rady:          {consensus.consensus_synthesis}")
        print(f"   Ověření pravdivosti:   {consensus.truth_verification_note}")
        print("=" * 70 + "\n")

    def start_daemon(self, live_market_check_minutes: int = 15) -> None:
        """
        Spustí nepřetržitý 24/7 režim (Způsob 2):
        1. Vlákno na pozadí neustále prochází web, čte články a hloubkově analyzuje akcie na AMD NPU.
        2. Hlavní vlákno sleduje otevírací hodiny burzy (15:30 - 22:00 CET).
        3. Při otevřeném trhu bleskově exekuuje nákupy/rotace z připravené výzkumné cache (v řádu sekund!).
        """
        self.print_banner()
        print("=== SPUŠTĚNÍ NEPŘETRŽITÉHO REŽIMU (24/7 RESEARCH & LIVE TRADER) ===")
        print("• Výzkumný motor na NPU: AKTIVNÍ (průběžně analyzuje články na pozadí)")
        print("• Obchodní hodiny trhu:  Sledovány (Evropa 09:00 - 17:30 CET | USA 15:30 - 22:00 CET)")
        print("• Stiskněte Ctrl+C pro bezpečné zastavení bota.\n")

        # Spuštění výzkumného vlákna na NPU (pauza 20 s mezi analýzami pro klidný chod)
        self.researcher.start_background(pause_seconds=20.0)

        last_market_run: Optional[datetime] = None

        try:
            while True:
                market_stat = MarketHours.get_market_status()
                now = datetime.now()

                if market_stat["is_open"]:
                    # Trh je otevřen!
                    should_run = (
                        last_market_run is None or
                        (now - last_market_run).total_seconds() >= (live_market_check_minutes * 60)
                    )

                    if should_run:
                        logger.info(
                            f"🔔 [TRH JE OTEVŘEN] Spouštím live tržní exekuci z připravené výzkumné cache "
                            f"(čas v NY: {market_stat['ny_time']})..."
                        )
                        self.run_daily_cycle(force=True)
                        last_market_run = now
                    else:
                        elapsed = (now - last_market_run).total_seconds()
                        remaining = max(1, int((live_market_check_minutes * 60 - elapsed) / 60))
                        logger.debug(f"[Live trh] Příští kontrola pozic za {remaining} minut.")
                else:
                    logger.info(
                        f"🌙 [Trh je zavřen] {market_stat['status_text']}. "
                        "LLM na NPU nepřetržitě analyzuje web a zprávy na pozadí..."
                    )
                    last_market_run = None  # Reset pro okamžité spuštění při otevření trhu

                # Spánek před další kontrolou stavu
                time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Zastavuji nepřetržitého daemona a výzkumné vlákno...")
            self.researcher.stop()
            logger.info("Trading Bot byl bezpečně ukončen.")


def main():
    parser = argparse.ArgumentParser(
        description="AI Quantitative Trading Bot (Trading 212 & Lemonade Server na AMD NPU)"
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Provede jeden kompletní cyklus ihned a skončí.",
    )
    parser.add_argument(
        "--daemon",
        "--continuous",
        action="store_true",
        dest="daemon",
        help="Spustí nepřetržitý 24/7 režim: LLM na pozadí analyzuje web a zprávy, při otevření trhu okamžitě exekuuje nákupy.",
    )
    parser.add_argument(
        "--research-status",
        action="store_true",
        help="Zobrazí stav výzkumné paměti (všechny zanalyzované akcie a jejich sentiment z cache).",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Prohledá globální zprávy na nově oznámené AI/tech firmy, kontrakty a IPO a zařadí je do univerza bota.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simuluje rozhodování i exekuci bez odesílání skutečných příkazů na brokera.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Zobrazí aktuální stav účtu, otevřené pozice a úroveň rizik.",
    )
    parser.add_argument(
        "--time",
        default="21:00",
        help="Čas spouštění v plánovači (výchozí: 21:00 CET během otevřeného trhu, lze např. '16:30,21:00').",
    )
    parser.add_argument(
        "--daytrade",
        action="store_true",
        help="Spustí agresivní Day Trading & Leveraged Momentum režim (5m svíčky, VWAP, EMA Ribbon, páková ETF).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval skenování v sekundách pro Day Trading režim (výchozí: 60 s).",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help=f"Maximální počet otevřených pozic (výchozí: {settings.max_positions}).",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Spustí okamžitý hloubkový sken globálního trhu s Radou AI (Fundament + Skeptik + Katalyzátory) a uloží schválené příležitosti.",
    )
    parser.add_argument(
        "--schedule-scans",
        action="store_true",
        help="Spustí plánovač pro automatické dvoufázové denní skenování trhu (09:30 a 16:00 CET).",
    )
    parser.add_argument(
        "--scanner-status",
        action="store_true",
        help="Zobrazí aktuálně schválené příležitosti s konsenzem Rady AI v databázi (data/qualified_opportunities.json).",
    )
    parser.add_argument(
        "--committee",
        type=str,
        default=None,
        metavar="TICKER",
        help="Provede detailní oponenturu a ověření pravdivosti v Radě AI pro jeden konkrétní titul (např. --committee NVDA).",
    )
    parser.add_argument(
        "--trade-memory",
        "--lessons",
        action="store_true",
        dest="trade_memory",
        help="Zobrazí paměť sebereflexe chyb, win rate a ponaučení z minulých obchodů (data/trade_memory.json).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Vynutí spuštění cyklu i o víkendu při --run-once.",
    )

    args = parser.parse_args()

    if args.daytrade:
        daytrader = DayTraderEngine(max_active_positions=args.max_positions, dry_run=args.dry_run)
        daytrader.start_loop(scan_interval_seconds=args.interval)
        return

    app = TradingBotApp(dry_run=args.dry_run)

    if args.scan:
        app.scan_market_now()
    elif args.schedule_scans:
        app.schedule_market_scans()
    elif args.scanner_status:
        app.show_scanner_status()
    elif args.trade_memory:
        app.show_trade_memory()
    elif args.committee:
        app.evaluate_committee_ticker(args.committee)
    elif args.status:
        app.show_status()
    elif args.research_status:
        app.show_research_status()
    elif args.discover:
        app.discover_opportunities()
    elif args.run_once:
        app.print_banner()
        app.run_daily_cycle(force=args.force)
    elif args.daemon:
        app.start_daemon()
    else:
        app.start_scheduler(run_time=args.time)


if __name__ == "__main__":
    main()
