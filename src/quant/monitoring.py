"""Modul 7: Risk Monitoring, 6 Kill-Switches & CUSUM Drift Detection.

Implementuje přísný provozní dohled a bezpečnostní jističe dle Sekce 11 a 15 v research.md:
1. Jistič 1 (Neobvyklé ztráty): Denní ztráta > 3.0 % nebo Max Drawdown > 10.0 % -> HALT
2. Jistič 2 (Extrémní volatilita): Intradenní pohyb benchmarku > 4.0 % nebo Trading Halt -> HALT
3. Jistič 3 (Integrita dat): Zastaralá data, nulový objem, cenový skok > 15 % bez ohlášení -> HALT
4. Jistič 4 (Desynchronizace portfolia): Rozpor mezi interním stavem a broker API -> HALT
5. Jistič 5 (Neočekávané náklady): Realizovaný skluz > 3x modelovaný skluz -> HALT
6. Jistič 6 (API Loop Breaker): >= 3 chyby API (429, 400, timeout) za 60 sekund -> DISCONNECT
7. Statistická CUSUM detekce driftu (kumulativní odchylka realizovaných zisků od očekávaných EV)
"""

from collections import deque
from datetime import datetime, timedelta
import logging
from typing import Deque, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class KillSwitchStatus(BaseModel):
    """Výsledek vyhodnocení jističů."""
    is_halted: bool = False
    triggered_switch: Optional[str] = None
    reason: str = "Systém v normálním provozu."
    active_alerts: List[str] = []


class CusumDriftDetector:
    """Statistický detektor rozpadu strategie (Concept Drift) přes CUSUM."""

    def __init__(self, allowance_k: float = 0.5, threshold_h: float = 4.0):
        self.k = allowance_k          # Přípustná odchylka na krok
        self.h = threshold_h          # Rozhodovací interval pro vyhlášení driftu
        self.pos_sum = 0.0            # Kladná kumulativní suma (výrazně horší výsledky než model)
        self.neg_sum = 0.0            # Záporná kumulativní suma (výrazně lepší výsledky)
        self.observations_count = 0

    def update(self, realized_return_pct: float, expected_ev_pct: float) -> tuple[bool, str]:
        """
        Zaktualizuje CUSUM statistiku po uzavření obchodu.
        Vrací (is_drift_detected, detail_message).
        """
        # Reziduum: o kolik je skutečnost horší než predikce modelu
        residual = expected_ev_pct - realized_return_pct
        self.observations_count += 1

        self.pos_sum = max(0.0, self.pos_sum + (residual - self.k))
        self.neg_sum = max(0.0, self.neg_sum + (-residual - self.k))

        if self.pos_sum >= self.h:
            msg = (
                f"CUSUM DRIFT DETECTED: Strategie systematicky zaostává za modelem "
                f"(CUSUM statistic {self.pos_sum:.2f} >= práh {self.h:.2f} po {self.observations_count} obchodech)."
            )
            logger.warning(msg)
            return True, msg

        return False, f"CUSUM v normě (pos: {self.pos_sum:.2f}, neg: {self.neg_sum:.2f})."


class QuantMonitoringSystem:
    """Řídící centrum bezpečnostních jističů (Kill-Switches) a monitoringu."""

    def __init__(
        self,
        max_daily_loss_pct: float = 3.0,
        max_portfolio_drawdown_pct: float = 10.0,
        max_benchmark_daily_move_pct: float = 4.0,
        max_price_jump_pct: float = 15.0,
        max_data_staleness_minutes: int = 15,
        max_api_errors_per_minute: int = 3,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_portfolio_drawdown_pct = max_portfolio_drawdown_pct
        self.max_benchmark_daily_move_pct = max_benchmark_daily_move_pct
        self.max_price_jump_pct = max_price_jump_pct
        self.max_data_staleness_minutes = max_data_staleness_minutes
        self.max_api_errors_per_minute = max_api_errors_per_minute

        self.cusum = CusumDriftDetector()
        self.api_error_timestamps: Deque[datetime] = deque()
        self.recent_slippages: Deque[float] = deque(maxlen=5)

    def check_unusual_loss(
        self,
        daily_pnl_pct: float,
        portfolio_drawdown_from_ath_pct: float,
    ) -> Optional[tuple[str, str]]:
        """Jistič 1: Neobvyklé ztráty."""
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            return (
                "KILL_SWITCH_1_DAILY_LOSS",
                f"Denní ztráta {daily_pnl_pct:.2f} % překročila limit {-self.max_daily_loss_pct:.2f} %."
            )
        if portfolio_drawdown_from_ath_pct >= self.max_portfolio_drawdown_pct:
            return (
                "KILL_SWITCH_1_MAX_DRAWDOWN",
                f"Celkový propad portfolia {portfolio_drawdown_from_ath_pct:.2f} % překročil strop {self.max_portfolio_drawdown_pct:.2f} %."
            )
        return None

    def check_extreme_volatility(
        self,
        benchmark_daily_change_pct: float,
        is_trading_halted_by_exchange: bool = False,
    ) -> Optional[tuple[str, str]]:
        """Jistič 2: Extrémní volatilita nebo pozastavení trhu."""
        if is_trading_halted_by_exchange:
            return "KILL_SWITCH_2_EXCHANGE_HALT", "Burza vyhlásila oficiální Trading Halt."
        if abs(benchmark_daily_change_pct) >= self.max_benchmark_daily_move_pct:
            return (
                "KILL_SWITCH_2_EXTREME_VOLATILITY",
                f"Denní pohyb benchmarku ({benchmark_daily_change_pct:+.2f} %) překročil bezpečný limit {self.max_benchmark_daily_move_pct:.1f} %."
            )
        return None

    def check_data_integrity(
        self,
        last_data_timestamp: datetime,
        current_time: datetime,
        bar_volume: float,
        bar_price_jump_pct: float,
        is_market_open: bool = True,
    ) -> Optional[tuple[str, str]]:
        """Jistič 3: Integrita a čerstvost dat."""
        if is_market_open:
            staleness = (current_time - last_data_timestamp).total_seconds() / 60.0
            if staleness > self.max_data_staleness_minutes:
                return (
                    "KILL_SWITCH_3_STALE_DATA",
                    f"Výpadek datového toku: Poslední data před {staleness:.1f} min (limit {self.max_data_staleness_minutes} min)."
                )
            if bar_volume <= 0.0:
                return "KILL_SWITCH_3_ZERO_VOLUME", "Detekována svíčka s nulovým objemem během hlavní obchodní seance."

        if abs(bar_price_jump_pct) >= self.max_price_jump_pct:
            return (
                "KILL_SWITCH_3_ANOMALOUS_PRICE_JUMP",
                f"Podezřelý cenový skok {bar_price_jump_pct:+.2f} % bez potvrzené korporátní akce."
            )
        return None

    def check_portfolio_desync(
        self,
        internal_shares: Dict[str, float],
        broker_shares: Dict[str, float],
    ) -> Optional[tuple[str, str]]:
        """Jistič 4: Rozpor mezi interní evidencí a stavem u brokera."""
        all_tickers = set(internal_shares.keys()).union(broker_shares.keys())
        for ticker in all_tickers:
            qty_internal = internal_shares.get(ticker, 0.0)
            qty_broker = broker_shares.get(ticker, 0.0)
            diff = abs(qty_internal - qty_broker)
            if diff > 0.001:
                return (
                    "KILL_SWITCH_4_PORTFOLIO_DESYNC",
                    f"Kritický rozpor pro {ticker}: interně evidováno {qty_internal:.4f} ks, broker hlásí {qty_broker:.4f} ks (rozdíl {diff:.4f})."
                )
        return None

    def check_slippage(
        self,
        last_trade_slippage_pct: float,
        expected_slippage_pct: float,
    ) -> Optional[tuple[str, str]]:
        """Jistič 5: Neočekávané transakční náklady a skluz."""
        self.recent_slippages.append(last_trade_slippage_pct)
        if len(self.recent_slippages) >= 3:
            avg_slippage = sum(self.recent_slippages) / len(self.recent_slippages)
            if avg_slippage > 3.0 * expected_slippage_pct:
                return (
                    "KILL_SWITCH_5_EXCESSIVE_SLIPPAGE",
                    f"Průměrný skluz ({avg_slippage:.3f} %) překročil 3násobek očekávaného ({expected_slippage_pct:.3f} %)."
                )
        return None

    def record_api_error(self, timestamp: Optional[datetime] = None) -> Optional[tuple[str, str]]:
        """Jistič 6: API Loop Breaker."""
        now = timestamp or datetime.now()
        self.api_error_timestamps.append(now)

        # Odstranění chyb starších než 60 sekund
        cutoff = now - timedelta(seconds=60)
        while self.api_error_timestamps and self.api_error_timestamps[0] < cutoff:
            self.api_error_timestamps.popleft()

        if len(self.api_error_timestamps) >= self.max_api_errors_per_minute:
            return (
                "KILL_SWITCH_6_API_LOOP_BREAKER",
                f"Detekováno {len(self.api_error_timestamps)} chyb API během 60 sekund. Exekuce odpojena pro zabránění nekonečné smyčky."
            )
        return None

    def evaluate_all(
        self,
        daily_pnl_pct: float = 0.0,
        portfolio_drawdown_from_ath_pct: float = 0.0,
        benchmark_daily_change_pct: float = 0.0,
        is_trading_halted: bool = False,
        internal_shares: Optional[Dict[str, float]] = None,
        broker_shares: Optional[Dict[str, float]] = None,
    ) -> KillSwitchStatus:
        """Kompletní kontrola všech aktivních systémových jističů."""
        alerts: List[str] = []

        # 1. Neobvyklé ztráty
        k1 = self.check_unusual_loss(daily_pnl_pct, portfolio_drawdown_from_ath_pct)
        if k1:
            return KillSwitchStatus(is_halted=True, triggered_switch=k1[0], reason=k1[1], active_alerts=[k1[1]])

        # 2. Extrémní volatilita
        k2 = self.check_extreme_volatility(benchmark_daily_change_pct, is_trading_halted)
        if k2:
            return KillSwitchStatus(is_halted=True, triggered_switch=k2[0], reason=k2[1], active_alerts=[k2[1]])

        # 4. Desynchronizace portfolia (pokud byla předána data)
        if internal_shares is not None and broker_shares is not None:
            k4 = self.check_portfolio_desync(internal_shares, broker_shares)
            if k4:
                return KillSwitchStatus(is_halted=True, triggered_switch=k4[0], reason=k4[1], active_alerts=[k4[1]])

        return KillSwitchStatus(is_halted=False, reason="Všechny bezpečnostní jističe v normě.")
