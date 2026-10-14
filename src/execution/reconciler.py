import logging
import time
from typing import Any, Dict, List, Optional
from config import settings
from src.broker.t212_client import Trading212Client
from src.data.market_data import MarketDataProvider
from src.execution.order_executor import OrderExecutor
from src.execution.portfolio_tracker import PortfolioTracker
from src.execution.ticker_mapper import TickerMapper
from src.llm.sentiment_engine import SentimentEngine
from src.strategy.models import SignalAction, TradingSignal
from src.strategy.risk_manager import RiskManager
from src.strategy.stock_picker import StockPicker

logger = logging.getLogger(__name__)


class PortfolioReconciler:
    """
    Sjednocuje a řídí kompletní exekuční cyklus:
    1. Synchronizuje stav účtu s brokerem
    2. Vyhodnocuje existující pozice na výstupní podmínky (Stop Loss / Take Profit / Trend)
    3. Realizuje prodeje pozic, které splnily výstupní podmínky
    4. Vyhodnocuje volnou kapacitu pro nové vstupy (max 3 pozice, 50 EUR limit)
    5. Z nových BUY signálů filtruje již držené tituly a otevírá nové pozice
    """

    def __init__(
        self,
        client: Optional[Trading212Client] = None,
        tracker: Optional[PortfolioTracker] = None,
        risk_manager: Optional[RiskManager] = None,
        executor: Optional[OrderExecutor] = None,
        data_provider: Optional[MarketDataProvider] = None,
        sentiment_engine: Optional[SentimentEngine] = None,
        dry_run: bool = False,
    ):
        self.client = client or Trading212Client()
        self.tracker = tracker or PortfolioTracker()
        self.risk_manager = risk_manager or RiskManager()
        self.executor = executor or OrderExecutor(
            client=self.client,
            tracker=self.tracker,
            risk_manager=self.risk_manager,
            dry_run=dry_run,
        )
        if dry_run:
            self.executor.dry_run = True
        self.data_provider = data_provider or MarketDataProvider()
        self.sentiment_engine = sentiment_engine or SentimentEngine()
        self.dry_run = dry_run

    def sync_broker_portfolio(self) -> List[Dict[str, Any]]:
        """Stáhne aktuální portfolio z Trading 212 a synchronizuje lokální stav."""
        broker_positions = self.client.get_portfolio()
        self.tracker.sync_with_broker(broker_positions)
        return broker_positions

    def evaluate_and_close_positions(self) -> List[Dict[str, Any]]:
        """
        Zkontroluje všechny otevřené pozice a v případě naplnění výstupních kritérií
        provede prodejní příkaz na brokerovi.
        """
        closed_reports: List[Dict[str, Any]] = []
        open_positions = self.tracker.get_open_positions()

        logger.info(f"Vyhodnocuji výstupní podmínky pro {len(open_positions)} otevřených pozic...")

        for ticker, pos in open_positions.items():
            stock_data = self.data_provider.prepare_ticker_data(ticker)
            if not stock_data:
                logger.warning(f"Nelze načíst aktuální tržní data pro otevřenou pozici {ticker}.")
                continue

            # Pokud je potřeba, ověříme i AI sentiment pro detekci zhoršení
            ai_verdict = None
            if stock_data.metrics.current_price < pos.entry_price_usd * 0.96:
                # Pokud pozice klesá, dotážeme se LLM na případné závažné negativní zprávy
                ai_verdict = self.sentiment_engine.analyze_stock(stock_data)

            exit_signal = self.risk_manager.evaluate_position_exit(
                ticker=ticker,
                current_price=stock_data.metrics.current_price,
                entry_price=pos.entry_price_usd,
                metrics=stock_data.metrics,
                ai_verdict=ai_verdict,
                custom_stop_loss=pos.stop_loss_price,
                custom_take_profit=pos.take_profit_price,
            )

            if exit_signal.should_exit:
                logger.info(f"Aktivován výstup pro {ticker}: {exit_signal.message}")
                order_res = self.executor.execute_sell(
                    ticker=ticker,
                    exit_signal=exit_signal,
                    quantity=pos.quantity
                )
                closed_reports.append({
                    "ticker": ticker,
                    "reason": exit_signal.reason.value if exit_signal.reason else "UNKNOWN",
                    "exit_price": exit_signal.exit_price,
                    "message": exit_signal.message,
                    "order_response": order_res,
                })
            else:
                logger.info(f"Pozice {ticker} drží trend (Cena: {stock_data.metrics.current_price:.2f} USD).")

        return closed_reports

    def evaluate_and_rotate_positions(
        self,
        candidate_signals: List[TradingSignal]
    ) -> List[Dict[str, Any]]:
        """
        Autonomní alfa rotace kapitálu:
        Pokud je portfolio plné (dosažen limit pozic), ale na trhu byla nalezena
        mimořádná příležitost (skóre >= 76.0 a BUY), porovná ji s nejslabší drženou pozicí.
        Pokud je stávající pozice zjevně oslabená (skóre < 56.0 nebo rozdíl skóre >= 18.0),
        provede prodej zaostávající pozice a nákup nové příležitosti.
        """
        from src.strategy.models import ExitReason, PositionExitSignal

        rotated_reports: List[Dict[str, Any]] = []
        open_positions = self.tracker.get_open_positions()
        if not open_positions:
            return rotated_reports

        # 1. Hledáme nejlepšího externího BUY kandidáta, kterého zatím nevlastníme
        external_buys = [
            s for s in candidate_signals
            if s.action == SignalAction.BUY and s.ticker not in open_positions and s.total_score >= 76.0
        ]
        if not external_buys:
            logger.info("Žádný externí kandidát nepřekročil práh vysoké konvikce (76.0) pro rotaci kapitálu.")
            return rotated_reports

        best_candidate = external_buys[0]

        # 2. Ohodnotíme držené pozice podle signálů z aktuálního auditu
        holding_signal_map = {s.ticker: s for s in candidate_signals if s.ticker in open_positions}

        scored_holdings = []
        for ticker, pos in open_positions.items():
            sig = holding_signal_map.get(ticker)
            score = sig.total_score if sig else 50.0
            is_sell_or_avoid = (sig.action == SignalAction.SELL) if sig else False
            scored_holdings.append((ticker, score, is_sell_or_avoid, pos))

        # Seřadíme od nejnižšího skóre k nejvyššímu
        scored_holdings.sort(key=lambda x: (not x[2], x[1]))
        weakest_ticker, weakest_score, is_sell_or_avoid, weakest_pos = scored_holdings[0]

        score_advantage = best_candidate.total_score - weakest_score

        # Podmínka pro rotaci:
        # A) Nejslabší titul má přímý signál SELL/AVOID
        # B) Nejslabší titul má skóre < 56.0 a nový kandidát má náskok alespoň 18 bodů
        should_rotate = is_sell_or_avoid or (weakest_score < 56.0 and score_advantage >= 18.0)

        if should_rotate:
            logger.info(
                f"🔄 ZAHÁJENA AUTONOMNÍ ALFA ROTACE: "
                f"Nový objev {best_candidate.ticker} (skóre: {best_candidate.total_score:.1f}) "
                f"výrazně překonává nejslabší drženou pozici {weakest_ticker} (skóre: {weakest_score:.1f}, "
                f"rozdíl: +{score_advantage:.1f} b)."
            )
            # 1. Prodej nejslabší pozice
            exit_sig = PositionExitSignal(
                ticker=weakest_ticker,
                should_exit=True,
                reason=ExitReason.ALPHA_ROTATION,
                exit_price=weakest_pos.entry_price_usd,
                message=f"Autonomní rotace kapitálu do vyššího potenciálu {best_candidate.ticker} (+{score_advantage:.1f} b)"
            )
            sell_res = self.executor.execute_sell(
                ticker=weakest_ticker,
                exit_signal=exit_sig,
                quantity=weakest_pos.quantity
            )
            time.sleep(2.0)

            # 2. Nákup nové příležitosti
            allocation_eur = self.risk_manager.calculate_allocation(
                current_open_positions_count=len(self.tracker.get_open_positions()),
                available_cash_eur=self.risk_manager.total_capital_eur,
                signal_score=best_candidate.total_score,
                atr_14=best_candidate.atr_14,
                current_price=best_candidate.current_price,
            )
            if allocation_eur > 0:
                best_candidate.suggested_allocation_eur = allocation_eur
                buy_res = self.executor.execute_buy(best_candidate)
                if buy_res:
                    rotated_reports.append({
                        "ticker": best_candidate.ticker,
                        "rotation_type": "ALPHA_ROTATION",
                        "sold_ticker": weakest_ticker,
                        "sold_score": weakest_score,
                        "bought_ticker": best_candidate.ticker,
                        "bought_score": best_candidate.total_score,
                        "allocated_eur": allocation_eur,
                        "price": best_candidate.current_price,
                        "currency": getattr(best_candidate, "currency", "USD"),
                        "score": best_candidate.total_score,
                        "order_response": buy_res,
                    })
                    logger.info(f"✅ Autonomní rotace dokončena: {weakest_ticker} -> {best_candidate.ticker}")
        else:
            logger.info(
                f"Stávající portfolio si drží solidní fundament (nejslabší {weakest_ticker} má skóre {weakest_score:.1f}). "
                f"Rotace do {best_candidate.ticker} (skóre {best_candidate.total_score:.1f}) není nutná."
            )

        return rotated_reports

    def allocate_new_positions(
        self,
        candidate_signals: List[TradingSignal]
    ) -> List[Dict[str, Any]]:
        """
        Vybere nejlépe hodnocené BUY kandidáty a otevře nové pozice až do limitu.
        Pokud je portfolio plné, provede vyhodnocení autonomní alfa rotace.
        """
        opened_reports: List[Dict[str, Any]] = []
        open_positions = self.tracker.get_open_positions()
        current_count = len(open_positions)
        available_slots = max(0, self.risk_manager.max_positions - current_count)

        # Zjištění reálné volné hotovosti z brokera pro přesnou alokaci
        try:
            cash_info = self.client.get_account_cash()
            free_czk = cash_info.get("free", 0.0)
            # Převod CZK -> EUR (orientační kurz 25.0 CZK/EUR)
            current_free_cash_eur = free_czk / 25.0
        except Exception as exc:
            logger.warning(f"Nelze načíst hotovost z brokera ({exc}), fallback na celkový kapitál.")
            current_free_cash_eur = self.risk_manager.total_capital_eur

        if available_slots <= 0 or current_free_cash_eur < 100.0:
            logger.info(
                f"Kapacita nebo hotovost vyčerpána ({current_count}/{self.risk_manager.max_positions} pozic, "
                f"volná hotovost: {current_free_cash_eur:,.2f} EUR). "
                "Spouštím vyhodnocení autonomní alfa rotace z oslabených pozic..."
            )
            rotation_reports = self.evaluate_and_rotate_positions(candidate_signals)
            opened_reports.extend(rotation_reports)
            return opened_reports

        logger.info(
            f"Dostupná kapacita: {available_slots} volných slotů | "
            f"Reálná volná hotovost: {current_free_cash_eur:,.2f} EUR"
        )

        # Filtrování signálů: pouze BUY akce a tituly, které již nevlastníme
        buy_candidates = [
            s for s in candidate_signals
            if s.action == SignalAction.BUY and s.ticker not in open_positions
        ]
        # Seřazení podle skóre
        buy_candidates.sort(key=lambda s: s.total_score, reverse=True)

        for signal in buy_candidates[:available_slots]:
            allocation_eur = self.risk_manager.calculate_allocation(
                current_open_positions_count=len(self.tracker.get_open_positions()),
                available_cash_eur=current_free_cash_eur,
                signal_score=signal.total_score,
                atr_14=signal.atr_14,
                current_price=signal.current_price,
            )

            if allocation_eur <= 0.0:
                logger.info("Nedostatek alokovatelného kapitálu pro další nákup.")
                break

            signal.suggested_allocation_eur = allocation_eur
            order_res = self.executor.execute_buy(signal)

            if order_res is not None:
                current_free_cash_eur = max(0.0, current_free_cash_eur - allocation_eur)
                opened_reports.append({
                    "ticker": signal.ticker,
                    "allocated_eur": allocation_eur,
                    "price": signal.current_price,
                    "currency": getattr(signal, "currency", "USD"),
                    "score": signal.total_score,
                    "order_response": order_res,
                })
                # Pauza mezi objednávkami pro ochranu před rate limitem brokera
                time.sleep(2.0)

        return opened_reports

    def run_cycle(self, candidate_signals: List[TradingSignal]) -> Dict[str, Any]:
        """
        Spustí kompletní vyhodnocovací a exekuční cyklus.
        """
        logger.info("=== SPOUŠTÍM EXEKUČNÍ CYKLUS RECONCILERU ===")
        
        # 1. Synchronizace s brokerem
        self.sync_broker_portfolio()
        initial_open = len(self.tracker.get_open_positions())

        # 2. Vyhodnocení a uzavření pozic (SL/TP/Trend)
        closed_trades = self.evaluate_and_close_positions()

        # 3. Alokace nových pozic
        opened_trades = self.allocate_new_positions(candidate_signals)

        final_open = len(self.tracker.get_open_positions())

        summary = {
            "initial_open_positions": initial_open,
            "closed_trades_count": len(closed_trades),
            "opened_trades_count": len(opened_trades),
            "final_open_positions": final_open,
            "closed_trades": closed_trades,
            "opened_trades": opened_trades,
            "current_positions": list(self.tracker.get_open_positions().keys()),
        }
        logger.info(f"=== EXEKUČNÍ CYKLUS DOKONČEN: Otevřeno {len(opened_trades)}, Uzavřeno {len(closed_trades)} ===")
        return summary
