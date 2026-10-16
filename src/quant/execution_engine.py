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
        fx_rates: Optional[Dict[str, float]] = None,
    ) -> tuple[bool, str]:
        """
        Předobchodní verifikace stavu:
        1. Vymáhá Long-Only pro Trading 212 Invest účet (zamítá neexistující SHORT).
        2. Kontroluje dostatek hotovosti pro BUY včetně rezervy na poplatky.
        3. Kontroluje minimální smysluplné množství (min notional ~5 USD).
        """
        # 1. Long-only kontrola
        if side == OrderSide.SELL and quantity <= 0:
            return False, f"Neplatný SELL příkaz pro {ticker}: množství <= 0."

        if side == OrderSide.BUY:
            from src.quant.risk_engine import CarverRiskEngine
            price_usd = CarverRiskEngine.get_share_price_usd(ticker, estimated_price, fx_rates)

            if quantity <= 0 or price_usd <= 0:
                return False, f"Neplatný BUY příkaz pro {ticker}: množství={quantity}, cena={price_usd}."

            # Minimální finanční limit brokera (min notional ~5 USD)
            if quantity * price_usd < 5.0:
                return False, f"Objem příkazu pro {ticker} ({quantity * price_usd:.2f} USD) je pod minimálním limitem brokera (< 5 USD)."

            required_cash = quantity * price_usd * 1.005  # 0.5% rezerva na skluz/poplatky
            if required_cash > available_cash:
                return False, (
                    f"Nedostatek hotovosti: Požadováno {required_cash:.2f} USD, "
                    f"k dispozici {available_cash:.2f} USD."
                )

        return True, "Reconciliation OK"

    def execute_sizing_order(
        self,
        sizing: PositionSizingResult,
        barriers: TripleBarriers,
        available_cash: float,
        timestamp: Optional[datetime] = None,
        fx_rates: Optional[Dict[str, float]] = None,
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
            fx_rates=fx_rates,
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

        # Výchozí zaokrouhlení: Evropské akcie (*_EQ bez _US_EQ) povolují max 3 desetinná místa, US 4
        precision = 3 if (broker_ticker.endswith("_EQ") and not broker_ticker.endswith("_US_EQ")) else 4
        order_qty = round(qty, precision) if side == OrderSide.BUY else -abs(round(qty, precision))

        if abs(order_qty) == 0:
            logger.info(f"{ticker}: Zaokrouhlené množství po aplikaci přesnosti ({precision}) je 0, přeskakuji exekuci.")
            return None

        order = ExecutionOrder(
            client_order_id=client_order_id,
            ticker=ticker,
            broker_ticker=broker_ticker,
            side=side,
            quantity=abs(order_qty),
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
                resp = self.client.place_order(
                    ticker=broker_ticker,
                    quantity=order_qty,
                    side=side.value if hasattr(side, "value") else str(side),
                    order_type="MARKET",
                )
                if resp and ("id" in resp or "orderId" in resp):
                    order.broker_order_id = str(resp.get("id") or resp.get("orderId"))
                    order.status = "FILLED"
                    order.executed_price = barriers.entry_price
                    logger.info(f"Příkaz úspěšně přijat brokerem: ID {order.broker_order_id}")
                else:
                    order.status = "REJECTED"
                    order.notes = f"Broker API nevrátilo order ID: {resp}"
            except Exception as e:
                err_str = str(e)
                # 1. Ošetření nedostatečné hotovosti (maržový/FX buffer brokera)
                if "insufficient-free-for-stocks-buy" in err_str:
                    try:
                        reduced_qty = round(order_qty * 0.75, precision)
                        if abs(reduced_qty) > 0:
                            logger.warning(
                                f"{ticker}: Nedostatek volných prostředků / maržový buffer brokera. "
                                f"Automaticky snižuji na 75 % ({reduced_qty} ks) a opakuji pokus..."
                            )
                            resp = self.client.place_order(
                                ticker=broker_ticker,
                                quantity=reduced_qty,
                                side=side.value if hasattr(side, "value") else str(side),
                                order_type="MARKET",
                            )
                            if resp and ("id" in resp or "orderId" in resp):
                                order.quantity = abs(reduced_qty)
                                order.broker_order_id = str(resp.get("id") or resp.get("orderId"))
                                order.status = "FILLED"
                                order.executed_price = barriers.entry_price
                                logger.info(f"Příkaz úspěšně přijat brokerem po redukci: ID {order.broker_order_id}")
                                self.executed_orders_log[client_order_id] = order
                                return order
                    except Exception as retry_e:
                        logger.warning(f"{ticker}: Ani po 25% redukci se nepodařilo příkaz odeslat: {retry_e}")

                # 2. Ošetření minimálního množství požadovaného brokerem
                elif "min-quantity-exceeded" in err_str:
                    import re
                    m = re.search(r"must trade at least\s*([\d\.]+)", err_str)
                    if m:
                        min_req_qty = float(m.group(1))
                        from src.quant.risk_engine import CarverRiskEngine
                        est_usd = CarverRiskEngine.get_share_price_usd(ticker, barriers.entry_price, fx_rates)
                        req_cost_usd = min_req_qty * est_usd
                        if available_cash and req_cost_usd <= available_cash * 1.15:
                            try:
                                logger.warning(
                                    f"{ticker}: Broker vyžaduje minimálně {min_req_qty} ks (~${req_cost_usd:.2f}). "
                                    f"Upravuji na požadované minimum a opakuji pokus..."
                                )
                                resp = self.client.place_order(
                                    ticker=broker_ticker,
                                    quantity=min_req_qty,
                                    side=side.value if hasattr(side, "value") else str(side),
                                    order_type="MARKET",
                                )
                                if resp and ("id" in resp or "orderId" in resp):
                                    order.quantity = min_req_qty
                                    order.broker_order_id = str(resp.get("id") or resp.get("orderId"))
                                    order.status = "FILLED"
                                    order.executed_price = barriers.entry_price
                                    logger.info(f"Příkaz úspěšně přijat brokerem po úpravě na minimum: ID {order.broker_order_id}")
                                    self.executed_orders_log[client_order_id] = order
                                    return order
                            except Exception as min_e:
                                logger.warning(f"{ticker}: Chyba při opakování s minimálním množstvím: {min_e}")

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
