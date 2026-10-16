import logging
import time
import math
import hashlib
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional
import requests
from requests.auth import HTTPBasicAuth
from config import settings, TradingMode

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
        base_url: Optional[str] = None,
        journal_path: Optional[str] = None,
    ):
        self.api_key = (api_key or settings.t212_api_key or "").strip()
        self.api_secret = (api_secret or settings.t212_api_secret or "").strip()
        self.base_url = (base_url or settings.t212_base_url).rstrip("/")
        account_key = hashlib.sha256(f"{self.base_url}:{self.api_key}".encode()).hexdigest()[:16]
        self.journal_path = journal_path or f"data/order_submissions_{account_key}.sqlite"

        # Bezpečnostní pojistka proti záměně prostředí
        expected_host = "live.trading212.com" if settings.trading_mode == TradingMode.LIVE else "demo.trading212.com"
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or parsed.netloc != expected_host or parsed.path not in ("", "/"):
            raise ValueError("Broker URL neodpovídá zvolenému obchodnímu prostředí.")
        if settings.trading_mode == TradingMode.LIVE:
            settings.validate_live_mode()

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
        # A timeout on a write does not establish whether the broker accepted it.
        # Only reads may be retried automatically.
        retries = max(1, retries) if method.upper() == "GET" else 1

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
                    if attempt == retries:
                        raise Trading212APIError(status_code=429, message="Trading 212 rate limit překročen.")
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
        try:
            result = self._request("GET", "/api/v0/equity/positions")
        except Trading212APIError as exc:
            if exc.status_code != 404:
                raise
            result = self._request("GET", "/api/v0/equity/portfolio")
        if not isinstance(result, list):
            raise ValueError("Broker positions response must be a list.")
        normalized = []
        for item in result:
            p = dict(item)
            p["ticker"] = p.get("ticker") or p.get("instrument", {}).get("ticker")
            if not p["ticker"] or "quantity" not in p:
                raise ValueError("Incomplete broker position.")
            if "averagePricePaid" in p:
                p["averagePrice"] = p["averagePricePaid"]
            normalized.append(p)
        return normalized

    def get_positions(self) -> List[Dict[str, Any]]:
        """Alias pro get_portfolio() pro kompatibilitu napříč strategiemi."""
        return self.get_portfolio()

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Alias pro get_portfolio() pro kompatibilitu s BrokerAdapter."""
        return self.get_portfolio()

    def get_open_position(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Vrátí detail konkrétní pozice podle tickeru."""
        return next((p for p in self.get_portfolio() if p["ticker"] == ticker), None)

    def place_order(
        self,
        ticker: str,
        quantity: float,
        side: Optional[str] = None,
        order_type: str = "MARKET",
        extended_hours: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Univerzální metoda pro odeslání příkazu na Trading 212 API.
        Kladná quantity = BUY, záporná quantity = SELL (případně odvozeno ze side).
        """
        from src.execution.ticker_mapper import TickerMapper
        t212_ticker = TickerMapper.to_t212_ticker(ticker)

        order_qty = quantity
        if side:
            order_qty = abs(quantity) if str(side).upper() in ("BUY", "ORDERSIDE.BUY") else -abs(quantity)

        if order_type.upper() != "MARKET":
            raise ValueError("Tato metoda podporuje pouze MARKET pokyny.")
        if side and str(side).upper() not in ("BUY", "SELL", "ORDERSIDE.BUY", "ORDERSIDE.SELL"):
            raise ValueError("Neplatný směr pokynu.")
        extra = {"context": kwargs["context"]} if "context" in kwargs else {}
        return self.place_market_order(t212_ticker, order_qty, extended_hours=extended_hours, **extra)

    def place_market_order(
        self,
        ticker: str,
        quantity: float,
        extended_hours: bool = False,
        _precision_retry: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Zadá nákupní nebo prodejní market příkaz.
        Kladná quantity = BUY, záporná quantity = SELL.
        """
        if not math.isfinite(quantity) or quantity == 0:
            raise ValueError("Množství (quantity) nemůže být 0.")

        from decimal import Decimal, ROUND_DOWN
        quantity = float(Decimal(str(quantity)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))
        if quantity == 0:
            raise ValueError("Quantity is below supported precision.")
        payload = {
            "quantity": quantity,
            "ticker": ticker,
        }
        if extended_hours:
            payload["extendedHours"] = True

        if settings.trading_mode == TradingMode.BACKTEST:
            raise RuntimeError("Backtest nesmí odesílat broker pokyny.")
        from src.execution.submission_journal import SubmissionJournal
        journal = SubmissionJournal(self.journal_path)
        # Preserve uncertainty across crashes and concurrent client instances.
        journal.reserve(ticker, payload, context=context)

        logger.info(f"Odesílám market order: {ticker} | množství: {quantity}")
        try:
            result = self._request("POST", "/api/v0/equity/orders/market", json_data=payload)
        except Trading212APIError as exc:
            if 400 <= exc.status_code < 500 and exc.status_code != 408:
                journal.reject(ticker)
            raise
        broker_id = result.get("id", result.get("orderId")) if isinstance(result, dict) else None
        if broker_id is not None:
            journal.acknowledge(ticker, broker_id)
        return result

    def reconcile_submissions(self, history: Optional[List[Dict[str, Any]]] = None) -> int:
        """Release known terminal submissions using broker history, never local prices.

        UNKNOWN requests without a broker ID deliberately remain blocked.
        """
        from src.execution.submission_journal import SubmissionJournal
        journal = SubmissionJournal(self.journal_path)
        count = 0
        for item in history if history is not None else self.get_all_order_history():
            order = item.get("order", {})
            ticker = order.get("ticker") or order.get("instrument", {}).get("ticker")
            if ticker:
                count += int(journal.reconcile_terminal(ticker, order))
        return count

    def get_submission_contexts(self) -> Dict[str, Any]:
        from src.execution.submission_journal import SubmissionJournal
        return SubmissionJournal(self.journal_path).contexts()

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
