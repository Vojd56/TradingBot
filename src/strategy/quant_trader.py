"""Orchestrátor kvantitativní strategie (QuantTraderEngine).

Sjednocuje 8modulovou kvantitativní architekturu dle research.md:
1. Feature Engineering (Causal & Dimensionless)
2. Regime Detector (SPY benchmark, vol percentil, CRISIS_PANIC)
3. Carver Orthogonal Signal Engine (Momentum + Mean-Reversion)
4. Prediction Model (Triple-Barrier Method & EV_net > eta * Costs)
5. Carver Risk Engine (Volatility Targeting, 10% Inertia Buffer, Correlation Filter)
6. Idempotent Execution Engine (SHA-256 client_order_id, Long-Only T212)
7. Monitoring & 6 Kill-Switches (Risk limits, desync audit, CUSUM)
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Set
import pandas as pd
from pydantic import BaseModel

from config import settings
from src.broker.t212_client import Trading212Client
from src.data.market_data import MarketDataProvider
from src.execution.portfolio_tracker import PortfolioTracker
from src.execution.ticker_mapper import TickerMapper
from src.quant.execution_engine import ExecutionEngine, ExecutionOrder
from src.quant.features import CausalFeatureEngine, StockFeatures
from src.quant.monitoring import KillSwitchStatus, QuantMonitoringSystem
from src.quant.prediction_model import ExpectedValueResult, PredictionModel, TripleBarriers
from src.quant.regime_detector import MarketRegime, MarketRegimeType, RegimeDetector
from src.quant.signal_engine import CarverSignalEngine, QuantSignal, SignalAction
from src.quant.risk_engine import CarverRiskEngine, PositionSizingResult
from src.strategy.opportunity_scanner import OpportunityScanner
from src.strategy.trade_reflection import TradeReflectionEngine

logger = logging.getLogger(__name__)


class QuantCycleReport(BaseModel):
    """Souhrnný report jednoho kvantitativního vyhodnocovacího cyklu."""
    timestamp: datetime
    benchmark_symbol: str
    regime: MarketRegime
    monitoring_status: KillSwitchStatus
    total_equity: float
    free_cash: float
    evaluated_tickers_count: int
    qualifying_buys_count: int
    orders_executed_count: int
    signals: Dict[str, QuantSignal] = {}
    ev_results: Dict[str, ExpectedValueResult] = {}
    sizing_results: Dict[str, PositionSizingResult] = {}
    executed_orders: List[ExecutionOrder] = []


class QuantTraderEngine:
    """Hlavní kvantitativní exekuční jádro."""

    def __init__(
        self,
        watchlist: Optional[List[str]] = None,
        benchmark_symbol: str = "SPY",
        dry_run: bool = True,
        client: Optional[Trading212Client] = None,
        data_provider: Optional[MarketDataProvider] = None,
        tracker: Optional[PortfolioTracker] = None,
    ):
        self.benchmark_symbol = benchmark_symbol
        self.dry_run = dry_run
        self.opportunity_scanner = OpportunityScanner()
        self.watchlist = watchlist or self.opportunity_scanner.get_all_universe_tickers()
        self.client = client or Trading212Client()
        self.data_provider = data_provider or MarketDataProvider()
        self.tracker = tracker or PortfolioTracker()
        self.reflection_engine = TradeReflectionEngine()
        self.ticker_mapper = TickerMapper()

        # Sledování denního PnL a maximálního propadu (ATH) pro jističe
        self.ath_equity: float = 0.0
        self.daily_start_equity: Optional[float] = None
        self.last_trading_day: Optional[str] = None

        # Inicializace 8 modulů kvantitativní architektury
        self.regime_detector = RegimeDetector()
        self.signal_engine = CarverSignalEngine()
        self.prediction_model = PredictionModel(fx_fee_pct=0.15)
        self.risk_engine = CarverRiskEngine(
            base_annual_target_vol=0.15,
            max_position_weight_pct=12.0,
            max_risk_per_trade_pct=1.5,
            target_positions_count=10,
            inertia_buffer_pct=10.0,
        )
        self.execution_engine = ExecutionEngine(
            strategy_id="QUANT_CARVER_V1",
            client=self.client,
            is_live=not self.dry_run,
        )
        self.monitoring = QuantMonitoringSystem()

    @staticmethod
    def get_usd_multiplier(currency: str) -> float:
        """Přepočítací koeficient pro převod měny účtu (např. CZK, EUR, GBP) do USD."""
        curr = currency.upper().strip()
        if curr in ("USD", "$"):
            return 1.0

        try:
            import yfinance as yf
            if curr == "CZK":
                ticker_df = yf.Ticker("CZK=X").history(period="5d")
                if not ticker_df.empty:
                    usd_czk = float(ticker_df["Close"].iloc[-1])
                    if usd_czk > 1.0:
                        return 1.0 / usd_czk
            elif curr == "EUR":
                ticker_df = yf.Ticker("EURUSD=X").history(period="5d")
                if not ticker_df.empty:
                    eur_usd = float(ticker_df["Close"].iloc[-1])
                    if eur_usd > 0.5:
                        return eur_usd
            elif curr == "GBP":
                ticker_df = yf.Ticker("GBPUSD=X").history(period="5d")
                if not ticker_df.empty:
                    gbp_usd = float(ticker_df["Close"].iloc[-1])
                    if gbp_usd > 0.5:
                        return gbp_usd
        except Exception:
            pass

        fallbacks = {"CZK": 1.0 / 23.5, "EUR": 1.08, "GBP": 1.28}
        return fallbacks.get(curr, 1.0)

    @staticmethod
    def get_market_fx_rates() -> Dict[str, float]:
        """Získá aktuální live FX kurzy do USD pro EUR, GBP, CHF."""
        rates = {"USD": 1.0, "EUR": 1.08, "GBP": 1.30, "CHF": 1.12}
        try:
            import yfinance as yf
            eur_usd = yf.Ticker("EURUSD=X").fast_info.get("last_price")
            if eur_usd and eur_usd > 0.5:
                rates["EUR"] = float(eur_usd)
            gbp_usd = yf.Ticker("GBPUSD=X").fast_info.get("last_price")
            if gbp_usd and gbp_usd > 0.5:
                rates["GBP"] = float(gbp_usd)
            usd_chf = yf.Ticker("USDCHF=X").fast_info.get("last_price")
            if usd_chf and usd_chf > 0.4:
                rates["CHF"] = 1.0 / float(usd_chf)
        except Exception as exc:
            logger.debug(f"Nelze stáhnout live FX kurzy: {exc}")
        return rates

    def get_portfolio_state(self) -> tuple[float, float, Dict[str, float]]:
        """
        Získá aktuální stav portfolia z Trading 212 API nebo lokálního trackeru.
        Vrací (total_equity, free_cash, positions_map) v USD.
        """
        equity = 100_000.0
        free_cash = 100_000.0
        positions: Dict[str, float] = {}

        if not self.dry_run:
            try:
                acc_cash = self.client.get_account_cash()
                summary = self.client.get_account_summary()
                curr = summary["currency"].upper()

                raw_equity = float(summary["totalValue"])
                raw_free_cash = float(acc_cash["free"])

                usd_mult = self.get_usd_multiplier(curr)
                equity = raw_equity * usd_mult
                free_cash = raw_free_cash * usd_mult

                logger.info(
                    f"Účet T212 ({curr}): Ekvita {raw_equity:.2f} {curr} -> {equity:.2f} USD | "
                    f"Volná hotovost: {raw_free_cash:.2f} {curr} -> {free_cash:.2f} USD"
                )

                open_pos = self.client.get_positions()
                if open_pos:
                    from src.execution.ticker_mapper import TickerMapper
                    for p in open_pos:
                        t = p.get("ticker", "")
                        qty = float(p.get("quantity", 0.0))
                        std_t = TickerMapper.to_standard_ticker(t)
                        positions[std_t] = qty
            except Exception as e:
                logger.error(f"Chyba při stahování stavu z T212 API: {e}", exc_info=True)
                raise RuntimeError("Nelze ověřit účet; cyklus nesmí obchodovat.") from e
        else:
            # V dry-run režimu použijeme tracker
            open_pos = self.tracker.get_open_positions()
            positions = {t: rec.quantity for t, rec in open_pos.items()}

        return equity, free_cash, positions

    def run_cycle(self) -> QuantCycleReport:
        """
        Provede jeden kompletní deterministický cyklus kvantitativního bota:
        1. Kontrola jističů (Kill-Switches)
        2. Analýza benchmarku a detekce tržního režimu (Regime Detector)
        3. Výpočet bezrozměrných příznaků pro univerzum (Causal Feature Engine)
        4. Průřezové řazení aktiv
        5. Carverovo signálové vyhodnocení (Momentum + Mean Reversion)
        6. Probabilistická Trojitá bariéra a výpočet EV_net
        7. Dimenzování pozic cílováním volatility (Carver Risk Engine)
        8. Idempotentní exekuce schválených objednávek
        """
        now = datetime.now()
        logger.info(f"--- Spouštím kvantitativní cyklus [{now.strftime('%Y-%m-%d %H:%M:%S')}] (Dry-Run: {self.dry_run}) ---")

        # 1. Zjištění stavu účtu a kontrola jističů
        equity, free_cash, current_positions = self.get_portfolio_state()

        today_str = now.strftime("%Y-%m-%d")
        if self.last_trading_day != today_str or self.daily_start_equity is None or self.daily_start_equity <= 0:
            self.last_trading_day = today_str
            self.daily_start_equity = equity
        if equity > self.ath_equity:
            self.ath_equity = equity

        daily_pnl_pct = ((equity - self.daily_start_equity) / self.daily_start_equity) * 100.0 if self.daily_start_equity > 0 else 0.0
        drawdown_pct = ((self.ath_equity - equity) / self.ath_equity) * 100.0 if self.ath_equity > 0 else 0.0

        monitoring_status = self.monitoring.evaluate_all(
            daily_pnl_pct=daily_pnl_pct,
            portfolio_drawdown_from_ath_pct=drawdown_pct,
        )

        # 2. Analýza tržního režimu z benchmarku
        logger.info(f"Stahuji historická data benchmarku ({self.benchmark_symbol})...")
        benchmark_df = self.data_provider.get_history(self.benchmark_symbol, period="1y", interval="1d")
        regime = self.regime_detector.detect_regime(benchmark_df, benchmark_symbol=self.benchmark_symbol)
        logger.info(f"Globální tržní režim: {regime.regime.value} | {regime.description}")

        # Pokud jističe nebo režim zastavily trading
        if monitoring_status.is_halted or not regime.allow_trading:
            logger.warning(f"Obchodování zablokováno bezpečnostními pravidly: {monitoring_status.reason or regime.description}")
            return QuantCycleReport(
                timestamp=now,
                benchmark_symbol=self.benchmark_symbol,
                regime=regime,
                monitoring_status=monitoring_status,
                total_equity=equity,
                free_cash=free_cash,
                evaluated_tickers_count=0,
                qualifying_buys_count=0,
                orders_executed_count=0,
            )

        # 3. Výpočet příznaků a denních výnosů pro sledované univerzum
        features_map: Dict[str, StockFeatures] = {}
        returns_map: Dict[str, pd.Series] = {}
        for ticker in self.watchlist:
            try:
                df = self.data_provider.get_history(ticker, period="1y", interval="1d")
                feat = CausalFeatureEngine.compute_single_ticker_features(df, ticker=ticker, min_bars_required=60)
                if feat:
                    features_map[ticker] = feat
                    if len(df) >= 2 and "Close" in df.columns:
                        returns_map[ticker] = df["Close"].pct_change()
            except Exception as e:
                logger.warning(f"{ticker}: Chyba při stahování dat/výpočtu příznaků: {e}")

        # Výpočet mezititulové korelační matice univerza
        corr_matrix: Dict[str, Dict[str, float]] = {}
        if len(returns_map) >= 2:
            try:
                corr_matrix = pd.DataFrame(returns_map).dropna(how="all").corr().to_dict()
            except Exception as e:
                logger.debug(f"Nelze spočítat korelační matici: {e}")

        # 4. Průřezové řazení (Cross-sectional ranking)
        features_map = CausalFeatureEngine.compute_cross_sectional_ranks(features_map)

        # 5. Carverovo signálové jádro
        signals = self.signal_engine.scan_universe(features_map, regime)

        # 6. Probabilistická Trojitá bariéra a výpočet EV_net
        ev_results: Dict[str, ExpectedValueResult] = {}
        qualifying_buys: List[str] = []

        for ticker, sig in signals.items():
            ev = self.prediction_model.evaluate(sig, regime)
            ev_results[ticker] = ev
            if ev.decision == SignalAction.BUY and ev.passes_hurdle:
                qualifying_buys.append(ticker)

        logger.info(f"Zpracováno {len(features_map)} aktiv. Kvalifikováno k nákupu (EV_net > Hurdle): {len(qualifying_buys)}")

        # 7. Dimenzování pozic a Carver Volatility Targeting
        sizing_results: Dict[str, PositionSizingResult] = {}
        executed_orders: List[ExecutionOrder] = []

        # Načtení schválených příležitostí z Rady AI (Qwen)
        ai_approved_tickers: Set[str] = set()
        qual_file = Path("data/qualified_opportunities.json")
        if qual_file.exists():
            try:
                with open(qual_file, "r", encoding="utf-8") as f:
                    q_data = json.load(f)
                    for opp in q_data.get("opportunities", []):
                        if opp.get("approved_for_trading", False):
                            ai_approved_tickers.add(opp.get("ticker", "").strip().upper())
            except Exception as e:
                logger.debug(f"Nelze načíst qualified_opportunities.json: {e}")

        # Řazení kandidátů: přednost mají tituly s konsenzem Rady AI (Qwen) a vyšším EV_net
        qualifying_buys.sort(
            key=lambda t: (1 if t.upper() in ai_approved_tickers else 0, ev_results[t].ev_net_pct),
            reverse=True
        )

        # 7.1 Zjištění čekajících objednávek na brokerovi (ochrana před duplicitami při zavřeném trhu)
        pending_tickers: Set[str] = set()
        if not self.dry_run and self.client:
            try:
                open_orders = self.client.get_open_orders()
                for o in open_orders:
                    raw_t = o.get("ticker", "")
                    if raw_t:
                        pending_tickers.add(self.ticker_mapper.to_standard_ticker(raw_t))
            except Exception as e:
                raise RuntimeError("Nelze ověřit čekající objednávky; nové pokyny zablokovány.") from e

        fx_rates = self.get_market_fx_rates()

        # 7.1.5 Správa a ochrana stávajících otevřených pozic (Stop Loss, Take Profit, Sell signál)
        contexts = self.client.get_submission_contexts() if (self.client and hasattr(self.client, "get_submission_contexts")) else {}
        for held_ticker, held_qty in list(current_positions.items()):
            if held_qty <= 0:
                continue
            feat = features_map.get(held_ticker)
            if not feat:
                continue
            raw_t212 = self.ticker_mapper.to_t212_ticker(held_ticker)
            ctx = contexts.get(raw_t212) or contexts.get(held_ticker) or {}
            sl_price = ctx.get("stop_loss")
            tp_price = ctx.get("take_profit")
            cur_price = feat.current_price

            exit_reason = None
            if sl_price and cur_price <= sl_price:
                exit_reason = f"STOP_LOSS (Cena {cur_price:.2f} <= SL {sl_price:.2f})"
            elif tp_price and cur_price >= tp_price:
                exit_reason = f"TAKE_PROFIT (Cena {cur_price:.2f} >= TP {tp_price:.2f})"
            elif held_ticker in signals and signals[held_ticker].action in (SignalAction.CLOSE, SignalAction.SELL):
                exit_reason = f"SIGNAL_SELL ({signals[held_ticker].rationale})"

            if exit_reason:
                logger.info(f"🛑 [QUANT EXIT] {held_ticker}: {exit_reason}. Odesílám prodej {held_qty} ks...")
                exit_sizing = PositionSizingResult(
                    ticker=held_ticker,
                    target_shares=0.0,
                    target_value=0.0,
                    portfolio_weight_pct=0.0,
                    risk_per_trade_cash=0.0,
                    risk_per_trade_pct=0.0,
                    current_shares=held_qty,
                    shares_diff=-held_qty,
                    is_rebalance_required=True,
                    approved_for_execution=True,
                    rationale=exit_reason,
                )
                exit_barriers = TripleBarriers(
                    entry_price=cur_price,
                    take_profit_price=cur_price * 1.04,
                    stop_loss_price=cur_price * 0.98,
                    max_holding_bars=1,
                    take_profit_pct=4.0,
                    stop_loss_pct=2.0,
                    rr_ratio=2.0,
                )
                exit_order = self.execution_engine.execute_sizing_order(
                    sizing=exit_sizing,
                    barriers=exit_barriers,
                    available_cash=0.0,
                    timestamp=now,
                    fx_rates=fx_rates,
                )
                if exit_order:
                    executed_orders.append(exit_order)
                    current_positions[held_ticker] = 0.0

        # 7.2 Předvýběr aktivních kandidátů (vynechání čekajících a korelační filtr)
        active_candidates: List[str] = []
        candidate_corr_multipliers: Dict[str, float] = {}

        for ticker in qualifying_buys:
            if ticker in pending_tickers:
                logger.info(f"{ticker}: Na brokerovi již čeká nevyřízený příkaz, vynechávám duplicitní objednávku.")
                continue

            # Korelační ochrana portfolia: vyloučení silně korelovaných aktiv s existujícími pozicemi
            held_tickers = [t for t, qty in current_positions.items() if qty > 0 and t != ticker]
            corr_mult = self.risk_engine.evaluate_correlation(ticker, held_tickers, corr_matrix)
            if corr_mult == 0.0:
                logger.info(f"🚫 [KORELACE VETO] {ticker}: Příliš vysoká korelace (> 0.80) s drženými pozicemi ({held_tickers}). Obchod zablokován.")
                continue

            active_candidates.append(ticker)
            candidate_corr_multipliers[ticker] = corr_mult

        # 7.3 Dimenzování a exekuce s férovým rozdělením hotovosti
        remaining_cash = free_cash
        remaining_candidates_count = len(active_candidates)

        for ticker in active_candidates:
            # Ponecháváme bezpečnostní rezervu alespoň 40 USD nebo 10 % z počáteční volné hotovosti pro maržové buffery brokera
            safety_reserve = max(40.0, free_cash * 0.10)
            usable_cash = max(0.0, remaining_cash - safety_reserve)

            if usable_cash < 15.0:
                logger.info(
                    f"Dostupná volná hotovost ({remaining_cash:.2f} USD, rezerva {safety_reserve:.2f} USD) "
                    f"nestačí na smysluplný nákup, ukončuji nákupy v tomto cyklu."
                )
                break

            ev = ev_results[ticker]
            feat = features_map[ticker]
            curr_shares = current_positions.get(ticker, 0.0)

            if ticker.upper() in ai_approved_tickers:
                logger.info(f"⭐ [Rada AI Konsenzus] {ticker} má ověřený nákupní konsenzus z hloubkového skenu Rady AI.")

            # Férové rozdělení dostupné hotovosti mezi zbývající schválené kandidáty
            budget_for_candidate = usable_cash / max(1, remaining_candidates_count)

            # Výpočet cílové velikosti pozice s normalizovanými cenami v USD a korelační penalizací
            corr_penalty = candidate_corr_multipliers.get(ticker, 1.0)
            sizing = self.risk_engine.calculate_position_size(
                ev_result=ev,
                features=feat,
                regime=regime,
                portfolio_equity=equity,
                current_shares=curr_shares,
                available_cash=budget_for_candidate,
                correlation_penalty_multiplier=corr_penalty,
                fx_rates=fx_rates,
            )
            sizing_results[ticker] = sizing

            # 8. Idempotentní exekuce
            if sizing.approved_for_execution and sizing.is_rebalance_required:
                order = self.execution_engine.execute_sizing_order(
                    sizing=sizing,
                    barriers=ev.barriers,
                    available_cash=budget_for_candidate,
                    timestamp=now,
                    fx_rates=fx_rates,
                )
                if order:
                    executed_orders.append(order)
                    if order.status in ("FILLED", "SUBMITTED", "UNKNOWN") and str(order.side).upper() in ("BUY", "ORDERSIDE.BUY"):
                        share_price_usd = CarverRiskEngine.get_share_price_usd(ticker, ev.barriers.entry_price, fx_rates)
                        cost_usd = order.quantity * share_price_usd

                        # Obnova skutečné volné hotovosti z brokera pro eliminaci odchylek měn a skluzu
                        if not self.dry_run and self.client:
                            try:
                                acc_cash = self.client.get_account_cash()
                                summary = self.client.get_account_summary()
                                curr = summary.get("currency", "USD").upper()
                                usd_mult = self.get_usd_multiplier(curr)
                                remaining_cash = min(max(0.0, remaining_cash - cost_usd), float(acc_cash["free"]) * usd_mult)
                                logger.debug(f"Aktualizována volná hotovost z API brokera: {remaining_cash:.2f} USD")
                            except Exception as e:
                                logger.debug(f"Nelze obnovit hotovost z API, použita kalkulace: {e}")
                                remaining_cash = max(0.0, remaining_cash - cost_usd)
                        else:
                            remaining_cash = max(0.0, remaining_cash - cost_usd)

            remaining_candidates_count -= 1

        report = QuantCycleReport(
            timestamp=now,
            benchmark_symbol=self.benchmark_symbol,
            regime=regime,
            monitoring_status=monitoring_status,
            total_equity=round(equity, 2),
            free_cash=round(free_cash, 2),
            evaluated_tickers_count=len(features_map),
            qualifying_buys_count=len(qualifying_buys),
            orders_executed_count=len(executed_orders),
            signals=signals,
            ev_results=ev_results,
            sizing_results=sizing_results,
            executed_orders=executed_orders,
        )

        logger.info(
            f"Cyklus dokončen: Režim={regime.regime.value}, Hodnoceno={len(features_map)}, "
            f"Kvalifikováno BUY={len(qualifying_buys)}, Exekuováno objednávek={len(executed_orders)}"
        )
        return report

    def start_loop(self, interval_seconds: int = 300) -> None:
        """Spustí nepřetržitou smyčku kvantitativního obchodování."""
        logger.info(f"Spouštím kontinuální kvantitativní smyčku (interval: {interval_seconds} s, dry_run: {self.dry_run})...")
        print("\nStiskněte Ctrl+C pro bezpečné zastavení kvantitativního tradera.\n")
        try:
            while True:
                try:
                    self.run_cycle()
                except Exception as e:
                    logger.error(f"Neočekávaná chyba v kvantitativním cyklu: {e}", exc_info=True)

                logger.info(f"Čekám {interval_seconds} sekund na další cyklus...")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Kvantitativní smyčka ukončena uživatelem (Ctrl+C).")
            print("\n🛑 Kvantitativní bot byl bezpečně ukončen uživatelem (Ctrl+C).")
