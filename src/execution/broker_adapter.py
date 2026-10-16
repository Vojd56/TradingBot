"""Modul rozhraní brokera (BrokerAdapter).

Definuje abstraktní vrstvu BrokerAdapter a konkrétní implementace:
1. PaperBrokerAdapter: Bezpečný lokální simulátor pro offline testy a backtest.
2. T212BrokerAdapter: Adaptér pro Trading 212 API.
3. AlpacaBrokerAdapter: Připravený slot pro DMA brokera Alpaca.
"""

from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Dict, Optional

from src.broker.t212_client import Trading212APIError, Trading212Client
from src.execution.ticker_mapper import TickerMapper
from src.quant.risk_engine import ApprovedOrder

logger = logging.getLogger(__name__)


class BrokerAdapter(ABC):
    """Abstraktní rozhraní pro exekuci u brokera."""

    @abstractmethod
    def get_account_cash(self) -> float:
        """Vrátí dostupnou hotovost na účtu."""
        pass

    @abstractmethod
    def get_open_positions(self) -> Dict[str, float]:
        """Vrátí slovník otevřených pozic ve formátu {standard_ticker: shares}."""
        pass

    @abstractmethod
    def place_order(self, order: ApprovedOrder) -> Dict[str, Any]:
        """Odešle schválený příkaz k exekuci u brokera."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Zruší čekající objednávku u brokera."""
        pass


class PaperBrokerAdapter(BrokerAdapter):
    """
    Simulovaný broker pro bezpečný papírový provoz a deterministické unit testy.
    Udržuje virtuální stav hotovosti a pozic v paměti bez síťových volání.
    """

    def __init__(self, initial_cash: float = 100_000.0):
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}
        self.executed_orders: Dict[str, ApprovedOrder] = {}

    def get_account_cash(self) -> float:
        return self.cash

    def get_open_positions(self) -> Dict[str, float]:
        return dict(self.positions)

    def place_order(self, order: ApprovedOrder) -> Dict[str, Any]:
        cost = order.shares * order.entry_price
        if order.side.upper() == "BUY":
            self.cash -= cost
            self.positions[order.ticker] = self.positions.get(order.ticker, 0.0) + order.shares
        elif order.side.upper() == "SELL":
            self.cash += cost
            cur_qty = self.positions.get(order.ticker, 0.0)
            new_qty = cur_qty - order.shares
            if new_qty <= 1e-6:
                self.positions.pop(order.ticker, None)
            else:
                self.positions[order.ticker] = new_qty

        self.executed_orders[order.client_order_id] = order
        logger.info(
            f"[PaperBroker] Exekuce {order.side} {order.shares} ks {order.ticker} "
            f"@ {order.entry_price:.2f} (client_order_id: {order.client_order_id})"
        )
        return {
            "id": order.client_order_id,
            "status": "FILLED",
            "ticker": order.ticker,
            "quantity": order.shares,
            "price": order.entry_price,
            "side": order.side,
            "is_paper": True,
        }

    def cancel_order(self, order_id: str) -> bool:
        return bool(order_id in self.executed_orders)


class T212BrokerAdapter(BrokerAdapter):
    """
    Adaptér pro Trading 212 API s automatickým mapováním tickerů
    a podporou extended hours.
    """

    def __init__(self, client: Optional[Trading212Client] = None):
        self.client = client or Trading212Client()

    def get_account_cash(self) -> float:
        try:
            cash_info = self.client.get_account_cash()
            return float(cash_info.get("free", cash_info.get("total", 0.0)))
        except Exception as exc:
            logger.error(f"T212: Chyba při načítání hotovosti: {exc}")
            return 0.0

    def get_open_positions(self) -> Dict[str, float]:
        try:
            positions_list = self.client.get_open_positions()
            res = {}
            for p in positions_list:
                t212_ticker = p.get("ticker", "")
                std_ticker = TickerMapper.to_standard_ticker(t212_ticker)
                qty = float(p.get("quantity", 0.0))
                res[std_ticker] = qty
            return res
        except Exception as exc:
            logger.error(f"T212: Chyba při načítání otevřených pozic: {exc}")
            return {}

    def place_order(self, order: ApprovedOrder) -> Dict[str, Any]:
        t212_ticker = TickerMapper.to_t212_ticker(order.ticker)
        qty = order.shares if order.side.upper() == "BUY" else -abs(order.shares)

        try:
            try:
                res = self.client.place_market_order(ticker=t212_ticker, quantity=qty, extended_hours=True)
            except Trading212APIError:
                res = self.client.place_market_order(ticker=t212_ticker, quantity=qty)

            res["client_order_id"] = order.client_order_id
            return res
        except Trading212APIError as exc:
            logger.error(f"T212 odmítlo objednávku pro {t212_ticker}: {exc}")
            raise RuntimeError(f"Trading212 order failed: {exc}") from exc

    def cancel_order(self, order_id: str) -> bool:
        try:
            return self.client.cancel_order(order_id)
        except Exception as exc:
            logger.error(f"T212: Chyba při rušení příkazu {order_id}: {exc}")
            return False


class AlpacaBrokerAdapter(BrokerAdapter):
    """
    Připravený slot pro Alpaca Trading API (DMA day trading).
    Aktivuje se, pokud jsou nastaveny ALPACA_API_KEY a ALPACA_SECRET_KEY.
    """

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, paper: bool = True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.paper = paper
        self._is_configured = bool(self.api_key and self.secret_key)

    def get_account_cash(self) -> float:
        if not self._is_configured:
            raise NotImplementedError("Alpaca API klíče nejsou nakonfigurovány v prostředí.")
        # Slot pro alpaca-py / requests volání
        return 0.0

    def get_open_positions(self) -> Dict[str, float]:
        if not self._is_configured:
            raise NotImplementedError("Alpaca API klíče nejsou nakonfigurovány v prostředí.")
        return {}

    def place_order(self, order: ApprovedOrder) -> Dict[str, Any]:
        if not self._is_configured:
            raise NotImplementedError("Alpaca API klíče nejsou nakonfigurovány v prostředí.")
        return {"id": order.client_order_id, "status": "SUBMITTED"}

    def cancel_order(self, order_id: str) -> bool:
        if not self._is_configured:
            raise NotImplementedError("Alpaca API klíče nejsou nakonfigurovány v prostředí.")
        return True
