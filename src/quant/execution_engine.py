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
import math
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from config import settings, TradingMode
from src.broker.t212_client import Trading212Client, Trading212APIError
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
        if is_live and client is None:
            raise ValueError("Broker execution requires a client.")
        if is_live and settings.trading_mode == TradingMode.BACKTEST:
            raise ValueError("Backtest cannot submit broker orders.")
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
        Lokální klíč; sám o sobě nezaručuje idempotenci brokera ani přežití restartu.
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
        if side not in (OrderSide.BUY, OrderSide.SELL) or not all(math.isfinite(v) for v in (quantity, estimated_price, available_cash)) or estimated_price <= 0:
            return False, "Neplatné vstupy pokynu."
        # 1. Long-only kontrola
        if side == OrderSide.SELL and quantity <= 0:
            return False, f"Neplatný SELL příkaz pro {ticker}: množství <= 0."

        if side == OrderSide.BUY:
            from src.quant.risk_engine import CarverRiskEngine
            price_usd = CarverRiskEngine.get_share_price_usd(ticker, estimated_price, fx_rates)

            if quantity <= 0 or not math.isfinite(price_usd) or price_usd <= 0:
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
        if side == OrderSide.SELL and (not math.isfinite(sizing.current_shares) or qty > sizing.current_shares):
            return None

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
        if any(o.ticker == ticker and o.status in ("SUBMITTED", "UNKNOWN", "PENDING") for o in self.executed_orders_log.values()):
            return None

        # Výchozí zaokrouhlení: Evropské akcie (*_EQ bez _US_EQ) povolují max 3 desetinná místa, US 4
        precision = 3 if (broker_ticker.endswith("_EQ") and not broker_ticker.endswith("_US_EQ")) else 4
        rounded_qty = float(Decimal(str(qty)).quantize(Decimal(10) ** -precision, rounding=ROUND_DOWN))
        order_qty = rounded_qty if side == OrderSide.BUY else -rounded_qty

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

        # Register before submission; a timeout does not prove rejection.
        self.executed_orders_log[client_order_id] = order
        if self.is_live:
            try:
                resp = self.client.place_order(
                    ticker=broker_ticker, quantity=order_qty, side=side, order_type="MARKET",
                    context={"stop_loss": barriers.stop_loss_price, "take_profit": barriers.take_profit_price,
                             "strategy_id": self.strategy_id},
                )
                broker_id = resp.get("id", resp.get("orderId")) if isinstance(resp, dict) else None
                if broker_id is not None:
                    order.broker_order_id = str(broker_id)
                    order.status = "SUBMITTED"
                    # Only reconciliation of broker fills may supply executed_price.
                else:
                    order.status = "UNKNOWN"
                    order.notes = "Missing broker order ID; reconcile before retry."
            except Trading212APIError as exc:
                order.status = "REJECTED" if 400 <= exc.status_code < 500 and exc.status_code != 408 else "UNKNOWN"
                order.notes = str(exc)
            except Exception as exc:
                order.status = "UNKNOWN"
                order.notes = str(exc)
        else:
            order.status = "FILLED"
            order.broker_order_id = f"SIM_{client_order_id}"
            order.executed_price = barriers.entry_price
        return order
