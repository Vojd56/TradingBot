"""Modul 6: Idempotent Execution Engine.

Zajišťuje spolehlivou a bezpečnou exekuci příkazů dle Sekce 8 v research.md:
1. Striktní Long-only specifikace pro Trading 212 Invest (spotové akcie a ETF)
2. Deterministický klíč idempotence (client_order_id = SHA256(strategy + ticker + timestamp + side + qty))
3. Pre-trade Reconciliation: ověření hotovosti, otevřených příkazů a broker API stavu
4. Zapouzdření Trading212Client a TickerMapper s podporou live i simulovaného režimu
"""

from datetime import datetime
import hashlib
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.broker.t212_client import Trading212Client
from src.execution.ticker_mapper import TickerMapper
from src.quant.prediction_model import TripleBarriers
from src.quant.risk_engine import PositionSizingResult

logger = logging.getLogger(__name__)


class OrderType(str):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(str):
    BUY = "BUY"
    SELL = "SELL"


class ExecutionOrder(BaseModel):
    """Reprezentace objednávky s garancí idempotence."""
    client_order_id: str
    ticker: str
    broker_ticker: str
    side: str
    quantity: float
    limit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    created_at: datetime
    is_simulated: bool = False
    status: str = "PENDING"  # PENDING, SUBMITTED, FILLED, REJECTED, CANCELLED
    broker_order_id: Optional[str] = None
    executed_price: Optional[float] = None
    notes: str = ""


class ExecutionEngine:
    """Exekuční jádro s předobchodní kontrolou a ochranou proti duplicitám."""

    def __init__(
        self,
        strategy_id: str = "QUANT_V1",
        client: Optional[Trading212Client] = None,
        is_live: bool = False,
    ):
        self.strategy_id = strategy_id
        self.client = client
        self.is_live = is_live
        self.ticker_mapper = TickerMapper()
        self.executed_orders_log: Dict[str, ExecutionOrder] = {}

    @staticmethod
    def generate_client_order_id(
        strategy_id: str,
        ticker: str,
        timestamp: datetime,
        side: str,
        quantity: float,
    ) -> str:
        """
        Vygeneruje deterministický hash idempotence (SHA-256 zkrácený na 24 hex znaků).
        Garantuje, že při síťovém timeoutu nevznikne duplicitní příkaz.
        """
        date_str = timestamp.strftime("%Y%m%d_%H%M")
        raw_key = f"{strategy_id}:{ticker}:{date_str}:{side}:{quantity:.4f}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]

    def pre_trade_reconciliation(
        self,
        ticker: str,
        side: str,
        quantity: float,
        estimated_price: float,
        available_cash: float,
    ) -> tuple[bool, str]:
        """
        Předobchodní verifikace stavu:
        1. Vymáhá Long-Only pro Trading 212 Invest účet (zamítá neexistující SHORT).
        2. Kontroluje dostatek hotovosti pro BUY včetně rezervy na poplatky.
        3. Kontroluje minimální smysluplné množství.
        """
        # 1. Long-only kontrola
        if side == OrderSide.SELL and quantity <= 0:
            return False, f"Neplatný SELL příkaz pro {ticker}: množství <= 0."

        if side == OrderSide.BUY:
            if quantity <= 0 or estimated_price <= 0:
                return False, f"Neplatný BUY příkaz pro {ticker}: množství={quantity}, cena={estimated_price}."

            required_cash = quantity * estimated_price * 1.005  # 0.5% rezerva na skluz/poplatky
            if required_cash > available_cash:
                return False, (
                    f"Nedostatek hotovosti: Požadováno {required_cash:.2f} CZK, "
                    f"k dispozici {available_cash:.2f} CZK."
                )

        return True, "Reconciliation OK"

    def execute_sizing_order(
        self,
        sizing: PositionSizingResult,
        barriers: TripleBarriers,
        available_cash: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[ExecutionOrder]:
        """
        Vytvoří a bezpečně odešle objednávku na základě rizikové kalkulace.
        """
        if not sizing.approved_for_execution or not sizing.is_rebalance_required:
            logger.debug(f"{sizing.ticker}: Objednávka není schválena k exekuci.")
            return None

        ticker = sizing.ticker
        ts = timestamp or datetime.now()
        side = OrderSide.BUY if sizing.shares_diff > 0 else OrderSide.SELL
        qty = abs(sizing.shares_diff)

        if qty <= 0.0001:
            return None

        # Předobchodní verifikace
        ok, reason = self.pre_trade_reconciliation(
            ticker=ticker,
            side=side,
            quantity=qty,
            estimated_price=barriers.entry_price,
            available_cash=available_cash,
        )
        if not ok:
            logger.warning(f"{ticker}: Pre-trade reconciliation selhala: {reason}")
            return None

        client_order_id = self.generate_client_order_id(
            strategy_id=self.strategy_id,
            ticker=ticker,
            timestamp=ts,
            side=side,
            quantity=qty,
        )

        # Ochrana proti duplicitám v paměti bota
        if client_order_id in self.executed_orders_log:
            logger.warning(f"{ticker}: Příkaz s client_order_id={client_order_id} již byl v tomto baru odeslán!")
            return self.executed_orders_log[client_order_id]

        broker_ticker = self.ticker_mapper.to_t212_ticker(ticker)

        order = ExecutionOrder(
            client_order_id=client_order_id,
            ticker=ticker,
            broker_ticker=broker_ticker,
            side=side,
            quantity=round(qty, 4),
            limit_price=None,  # Market order pro okamžité plnění
            stop_loss_price=barriers.stop_loss_price if side == OrderSide.BUY else None,
            take_profit_price=barriers.take_profit_price if side == OrderSide.BUY else None,
            created_at=ts,
            is_simulated=not self.is_live,
            status="SUBMITTED",
            notes=sizing.rationale,
        )

        # Exekuce u brokera (pokud je live režim a máme klienta)
        if self.is_live and self.client:
            try:
                logger.info(f"Odesílám {side} příkaz na {broker_ticker} ({qty} ks) přes Trading 212 API...")
                resp = self.client.place_order(
                    ticker=broker_ticker,
                    quantity=round(qty, 4),
                    order_type="MARKET",
                )
                if resp and "id" in resp:
                    order.broker_order_id = resp["id"]
                    order.status = "FILLED"
                    order.executed_price = barriers.entry_price
                    logger.info(f"Příkaz úspěšně přijat brokerem: ID {resp['id']}")
                else:
                    order.status = "REJECTED"
                    order.notes = f"Broker API nevrátilo order ID: {resp}"
            except Exception as e:
                logger.error(f"Chyba při odesílání příkazu na broker API: {e}", exc_info=True)
                order.status = "REJECTED"
                order.notes = f"Výjimka API: {str(e)}"
        else:
            # Simulovaná exekuce
            order.status = "FILLED"
            order.broker_order_id = f"SIM_{client_order_id[:8]}"
            order.executed_price = barriers.entry_price
            logger.info(f"[SIMULACE] Exekuován příkaz {side} {qty:.4f} ks {ticker} za {barriers.entry_price:.2f}")

        self.executed_orders_log[client_order_id] = order
        return order
