import logging
import time
from typing import Any, Dict, List, Optional
import requests
from requests.auth import HTTPBasicAuth
from config import settings

logger = logging.getLogger(__name__)


class Trading212APIError(Exception):
    """Základní výjimka pro chyby Trading 212 API."""
    def __init__(self, status_code: int, message: str, raw_response: Optional[str] = None):
        self.status_code = status_code
        self.message = message
        self.raw_response = raw_response
        super().__init__(f"T212 API Error [{status_code}]: {message}")


class Trading212Client:
    """Klient pro Trading 212 Invest API (v0) s podporou HTTP Basic Authentication."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.api_key = (api_key or settings.t212_api_key or "").strip()
        self.api_secret = (api_secret or settings.t212_api_secret or "").strip()
        self.base_url = (base_url or settings.t212_base_url).rstrip("/")

        # Bezpečnostní pojistka proti záměně prostředí
        if settings.trading_mode == "demo" and "demo.trading212.com" not in self.base_url:
            raise ValueError("Kritická chyba: TRADING_MODE je 'demo', ale base_url nesměřuje na demo!")

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "TradingBot-Python/1.0"
        })

        # Autentizace: HTTP Basic Auth (pokud je dostupný secret) nebo přímá hlavička
        if self.api_secret:
            self.session.auth = HTTPBasicAuth(self.api_key, self.api_secret)
        else:
            self.session.headers["Authorization"] = self.api_key

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Any:
        url = f"{self.base_url}{endpoint}"

        for attempt in range(1, retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    timeout=15.0
                )

                if response.status_code == 429:
                    raw_retry = response.headers.get("Retry-After")
                    retry_after = max(2, int(raw_retry)) if raw_retry and raw_retry.isdigit() else (2 * attempt)
                    logger.warning(f"Rate limit hit (429). Čekám {retry_after}s... (pokus {attempt}/{retries})")
                    time.sleep(retry_after)
                    if attempt == retries:
                        raise Trading212APIError(status_code=429, message="Trading 212 rate limit překročen po všech pokusech.")
                    continue

                if response.status_code in (200, 201):
                    return response.json() if response.content else {}

                if response.status_code == 204:
                    return {}

                error_msg = response.text
                try:
                    err_json = response.json()
                    error_msg = err_json.get("errorMessage", err_json.get("message", response.text))
                except Exception:
                    pass

                raise Trading212APIError(
                    status_code=response.status_code,
                    message=error_msg,
                    raw_response=response.text
                )

            except requests.RequestException as exc:
                if attempt == retries:
                    raise Trading212APIError(status_code=0, message=f"Síťové selhání po {retries} pokusech: {exc}")
                time.sleep(1.5 * attempt)

        raise Trading212APIError(status_code=429, message="Požadavek selhal na rate limit po všech pokusech.")

    def get_account_summary(self) -> Dict[str, Any]:
        """Vrátí celkový přehled účtu."""
        res = self._request("GET", "/api/v0/equity/account/summary")
        return res if isinstance(res, dict) else {}

    def get_account_cash(self) -> Dict[str, float]:
        """Vrátí stav hotovosti účtu."""
        try:
            data = self._request("GET", "/api/v0/equity/account/cash")
            if isinstance(data, dict):
                return {
                    "free": float(data.get("free", 0.0)),
                    "total": float(data.get("total", 0.0)),
                    "ppl": float(data.get("ppl", 0.0)),
                    "result": float(data.get("result", 0.0)),
                    "invested": float(data.get("invested", 0.0))
                }
        except Trading212APIError:
            pass

        # Fallback na endpoint summary
        summary = self.get_account_summary()
        cash_info = summary.get("cash", {})
        invest_info = summary.get("investments", {})
        return {
            "free": float(cash_info.get("availableToTrade", 0.0)),
            "total": float(summary.get("totalValue", 0.0)),
            "ppl": float(invest_info.get("unrealizedProfitLoss", 0.0)),
            "result": float(invest_info.get("realizedProfitLoss", 0.0)),
            "invested": float(invest_info.get("currentValue", 0.0))
        }

    def get_portfolio(self) -> List[Dict[str, Any]]:
        """Vrátí všechny aktuálně otevřené pozice."""
        return self._request("GET", "/api/v0/equity/portfolio")

    def get_open_position(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Vrátí detail konkrétní pozice podle tickeru."""
        try:
            return self._request("GET", f"/api/v0/equity/portfolio/{ticker}")
        except Trading212APIError as exc:
            if exc.status_code == 404:
                return None
            raise

    def place_market_order(
        self,
        ticker: str,
        quantity: float,
        extended_hours: bool = False
    ) -> Dict[str, Any]:
        """
        Zadá nákupní nebo prodejní market příkaz.
        Kladná quantity = BUY, záporná quantity = SELL.
        """
        if quantity == 0:
            raise ValueError("Množství (quantity) nemůže být 0.")

        payload = {
            "quantity": round(quantity, 6),
            "ticker": ticker,
        }
        if extended_hours:
            payload["extendedHours"] = True

        logger.info(f"Odesílám market order: {ticker} | množství: {quantity}")
        return self._request("POST", "/api/v0/equity/orders/market", json_data=payload)

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Vrátí všechny čekající (nevyplněné) objednávky."""
        return self._request("GET", "/api/v0/equity/orders")

    def cancel_order(self, order_id: int | str) -> Dict[str, Any]:
        """Zruší čekající objednávku podle jejího ID."""
        logger.info(f"Ruším objednávku ID {order_id} na brokerovi...")
        return self._request("DELETE", f"/api/v0/equity/orders/{order_id}")

    def get_order_history(self, limit: int = 50, cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        Vrátí stránku historie exekuovaných a zrušených objednávek z Trading 212 API.
        """
        endpoint = f"/api/v0/equity/history/orders?limit={min(max(1, limit), 50)}"
        if cursor:
            endpoint += f"&cursor={cursor}"
        res = self._request("GET", endpoint)
        return res if isinstance(res, dict) else {}

    def get_all_order_history(self, max_pages: int = 20) -> List[Dict[str, Any]]:
        """
        Načte kompletní historii objednávek z brokera přes stránkovací kurzor.
        Každý prvek obsahuje { 'order': {...}, 'fill': {...} }.
        """
        import urllib.parse
        all_orders: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(max_pages):
            res = self.get_order_history(limit=50, cursor=cursor)
            items = res.get("items", [])
            if not items:
                break
            all_orders.extend(items)
            next_path = res.get("nextPagePath")
            if not next_path:
                break
            parsed = urllib.parse.urlparse(next_path)
            qs = urllib.parse.parse_qs(parsed.query)
            cursor = qs.get("cursor", [None])[0]
            if not cursor:
                break
        return all_orders