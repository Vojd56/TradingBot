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
import logging
import time
from typing import Dict, List, Optional
import pandas as pd
from pydantic import BaseModel

from config import settings
from src.broker.t212_client import Trading212Client
from src.data.market_data import MarketDataProvider
from src.execution.portfolio_tracker import PortfolioTracker
from src.quant.execution_engine import ExecutionEngine, ExecutionOrder
from src.quant.features import CausalFeatureEngine, StockFeatures
from src.quant.monitoring import KillSwitchStatus, QuantMonitoringSystem
from src.quant.prediction_model import ExpectedValueResult, PredictionModel
from src.quant.regime_detector import MarketRegime, MarketRegimeType, RegimeDetector
from src.quant.signal_engine import CarverSignalEngine, QuantSignal, SignalAction
from src.quant.risk_engine import CarverRiskEngine, PositionSizingResult

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
        self.watchlist = watchlist or settings.watched_tickers
        self.client = client or Trading212Client()
        self.data_provider = data_provider or MarketDataProvider()
        self.tracker = tracker or PortfolioTracker()

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

    def get_portfolio_state(self) -> tuple[float, float, Dict[str, float]]:
        """
        Získá aktuální stav portfolia z Trading 212 API nebo lokálního trackeru.
        Vrací (total_equity, free_cash, positions_map).
        """
        equity = 100_000.0
        free_cash = 100_000.0
        positions: Dict[str, float] = {}

        if not self.dry_run:
            try:
                acc_cash = self.client.get_account_cash()
                if acc_cash:
                    equity = float(acc_cash.get("total", 100_000.0))
                    free_cash = float(acc_cash.get("free", equity))

                open_pos = self.client.get_positions()
                if open_pos:
                    for p in open_pos:
                        t = p.get("ticker", "")
                        qty = float(p.get("quantity", 0.0))
                        positions[t] = qty
            except Exception as e:
                logger.error(f"Chyba při stahování stavu z T212 API: {e}", exc_info=True)
                # Použijeme tracker jako fallback
                open_pos = self.tracker.get_open_positions()
                positions = {t: rec.quantity for t, rec in open_pos.items()}
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
        monitoring_status = self.monitoring.evaluate_all(
            daily_pnl_pct=0.0,
            portfolio_drawdown_from_ath_pct=0.0,
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

        # 3. Výpočet příznaků pro sledované univerzum
        features_map: Dict[str, StockFeatures] = {}
        for ticker in self.watchlist:
            try:
                df = self.data_provider.get_history(ticker, period="1y", interval="1d")
                feat = CausalFeatureEngine.compute_single_ticker_features(df, ticker=ticker, min_bars_required=60)
                if feat:
                    features_map[ticker] = feat
            except Exception as e:
                logger.warning(f"{ticker}: Chyba při stahování dat/výpočtu příznaků: {e}")

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

        # Řazení kandidátů podle čistého EV sestupně
        qualifying_buys.sort(key=lambda t: ev_results[t].ev_net_pct, reverse=True)

        remaining_cash = free_cash
        for ticker in qualifying_buys:
            ev = ev_results[ticker]
            feat = features_map[ticker]
            curr_shares = current_positions.get(ticker, 0.0)

            # Výpočet cílové velikosti pozice
            sizing = self.risk_engine.calculate_position_size(
                ev_result=ev,
                features=feat,
                regime=regime,
                portfolio_equity=equity,
                current_shares=curr_shares,
                available_cash=remaining_cash,
            )
            sizing_results[ticker] = sizing

            # 8. Idempotentní exekuce
            if sizing.approved_for_execution and sizing.is_rebalance_required:
                order = self.execution_engine.execute_sizing_order(
                    sizing=sizing,
                    barriers=ev.barriers,
                    available_cash=remaining_cash,
                    timestamp=now,
                )
                if order:
                    executed_orders.append(order)
                    if order.side == "BUY":
                        remaining_cash -= (order.quantity * ev.barriers.entry_price)

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
        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Kvantitativní smyčka ukončena uživatelem (Ctrl+C).")
                break
            except Exception as e:
                logger.error(f"Neočekávaná chyba v kvantitativním cyklu: {e}", exc_info=True)

            logger.info(f"Čekám {interval_seconds} sekund na další cyklus...")
            time.sleep(interval_seconds)
