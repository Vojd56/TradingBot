import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import requests
from config import settings

logger = logging.getLogger(__name__)


class FMPAuditData(BaseModel):
    """Výsledek hloubkového institucionálního auditu z Financial Modeling Prep (FMP)."""
    dcf_fair_value: Optional[float] = None
    dcf_diff_pct: Optional[float] = None  # Kolik % je tržní cena pod/nad DCF hodnotou
    roic_ttm_pct: Optional[float] = None  # Návratnost investovaného kapitálu (ROIC) v %
    ev_to_ebitda: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    fcf_yield_pct: Optional[float] = None
    graham_number: Optional[float] = None
    insider_summary: Optional[str] = None


class FMPClient:
    """
    Klient pro komunikaci s Financial Modeling Prep (FMP) API v moderní verzi /stable/.
    Poskytuje auditovaná data z výkazů: DCF vnitřní hodnotu, ROIC, Net Debt / EBITDA a EV / EBITDA.
    Pokud není zadán API klíč nebo titul není na bezplatném plánu dostupný,
    klient bezpečně vrátí None a bot pokračuje s Yahoo Finance.
    """

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or settings.fmp_api_key).strip()

    @property
    def is_enabled(self) -> bool:
        """Vrátí True, pokud je zadán platný API klíč."""
        return len(self.api_key) > 8

    def _get(self, endpoint: str, symbol: str, timeout: float = 6.0) -> Optional[Any]:
        if not self.is_enabled:
            return None

        url = f"{self.BASE_URL}/{endpoint}"
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }

        try:
            res = requests.get(url, params=params, timeout=timeout)
            if res.status_code == 200 and res.text.strip():
                try:
                    return res.json()
                except Exception:
                    return None
            elif res.status_code in (402, 403):
                # 402/403 pro zahraniční burzy nebo placené endpoints na free plánu
                logger.debug(f"FMP endpoint {endpoint} pro {symbol} vyžaduje placený plán ({res.status_code}).")
            elif res.status_code != 404:
                logger.debug(f"FMP požadavek na {endpoint} vrátil kód {res.status_code}.")
        except Exception as exc:
            logger.debug(f"Chyba při volání FMP API ({endpoint}): {exc}")
        return None

    def get_dcf_valuation(self, symbol: str) -> Optional[float]:
        """Stáhne odhad vnitřní hodnoty metodou diskontovaného cash flow (DCF)."""
        data = self._get("discounted-cash-flow", symbol)
        if data and isinstance(data, list) and len(data) > 0:
            dcf = data[0].get("dcf")
            if dcf and isinstance(dcf, (int, float)) and dcf > 0:
                return round(float(dcf), 2)
        return None

    def get_key_metrics_ttm(self, symbol: str) -> Dict[str, Any]:
        """Stáhne klíčové TTM ukazatele (ROIC, Net Debt / EBITDA, EV/EBITDA, FCF Yield)."""
        data = self._get("key-metrics-ttm", symbol)
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return {}

    def audit_stock(self, ticker: str, current_price: float) -> Optional[FMPAuditData]:
        """
        Sestaví kompletní FMP audit pro danou akcii:
        - DCF Fair Value a rozdíl oproti tržní ceně
        - ROIC (kapitálová návratnost)
        - Net Debt / EBITDA
        - EV / EBITDA
        - FCF Yield
        """
        if not self.is_enabled:
            return None

        clean = ticker.strip().upper()
        # Pokud má ticker příponu zahraniční burzy (např. RHM.DE, RR.L), FMP free plán vrací 402,
        # v takovém případě bezpečně přeskočíme a spoléháme na Yahoo Finance
        if "." in clean and not clean.endswith(".US"):
            return None

        try:
            dcf = self.get_dcf_valuation(clean)
            metrics = self.get_key_metrics_ttm(clean)

            if not dcf and not metrics:
                return None

            dcf_diff = None
            if dcf and dcf > 0 and current_price > 0:
                # Kladné = DCF je nad tržní cenou (podhodnoceno / upside)
                # Záporné = tržní cena je nad DCF (předraženo / downside)
                dcf_diff = round(((dcf - current_price) / current_price) * 100.0, 1)

            roic = metrics.get("returnOnInvestedCapitalTTM")
            net_debt_ebitda = metrics.get("netDebtToEBITDATTM")
            ev_ebitda = metrics.get("evToEBITDATTM")
            fcf_yield = metrics.get("freeCashFlowYieldTTM")
            graham = metrics.get("grahamNumberTTM")

            return FMPAuditData(
                dcf_fair_value=dcf,
                dcf_diff_pct=dcf_diff,
                roic_ttm_pct=round(roic * 100.0, 1) if roic is not None else None,
                ev_to_ebitda=round(ev_ebitda, 1) if ev_ebitda is not None else None,
                net_debt_to_ebitda=round(net_debt_ebitda, 2) if net_debt_ebitda is not None else None,
                fcf_yield_pct=round(fcf_yield * 100.0, 1) if fcf_yield is not None else None,
                graham_number=round(graham, 2) if graham is not None else None,
            )
        except Exception as exc:
            logger.debug(f"FMP audit pro {ticker} selhal: {exc}")
            return None
