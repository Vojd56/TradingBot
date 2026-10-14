import logging
import re
from typing import Dict, List, Optional, Set, Tuple
import urllib.request
import xml.etree.ElementTree as ET
import yfinance as yf

from src.execution.ticker_mapper import TickerMapper
from src.strategy.opportunity_scanner import OpportunityScanner

logger = logging.getLogger(__name__)


# Seznam běžných zkratek a slov, která nejsou burzovními tickery
FALSE_POSITIVE_TICKERS: Set[str] = {
    "AI", "CEO", "CFO", "CTO", "COO", "IPO", "ETF", "ETFs", "US", "USA", "SEC",
    "GDP", "CPI", "USD", "EUR", "GBP", "CZK", "NEW", "TOP", "BUY", "SELL",
    "HOLD", "ALL", "FOR", "THE", "AND", "NOT", "ARE", "CAN", "MAY", "HAS",
    "WAS", "OUT", "DAY", "PER", "Q1", "Q2", "Q3", "Q4", "ESG", "EV", "PE",
    "PEG", "DCF", "ROI", "ROE", "ROC", "DOW", "LSE", "NYSE", "NASDAQ",
    "R&D", "PR", "IR", "IT", "TECH", "FED", "EST", "PST", "GMT", "CET",
}


class DiscoveredOpportunity:
    """Reprezentuje novou příležitost objevenou z globálních zpráv."""
    def __init__(
        self,
        ticker: str,
        company_name: str,
        price: float,
        market_cap_usd: float,
        headline: str,
        catalyst_snippet: str,
    ):
        self.ticker = ticker
        self.company_name = company_name
        self.price = price
        self.market_cap_usd = market_cap_usd
        self.headline = headline
        self.catalyst_snippet = catalyst_snippet

    def __repr__(self) -> str:
        return (
            f"<Discovered {self.ticker} ({self.company_name}): "
            f"Cena={self.price:.2f}, MCap={self.market_cap_usd/1e6:.1f}M USD | {self.headline[:40]}...>"
        )


class AutonomousDiscoveryEngine:
    """
    Autonomní vyhledávací motor pro objevování nových firem, čerstvých IPO
    a rostoucích technologických titulů přímo z globálních médií.
    
    1. Sleduje breaking news o AI zakázkách, fúzích, patentech a vstupech na burzu.
    2. Pomocí NLP/RegEx a LLM detekuje nově zmíněné společnosti.
    3. Provádí přísné institucionální bezpečnostní síto (likvidita, market cap > 250M USD).
    4. Dynamicky registruje ověřené objevy do univerza OpportunityScanneru.
    """

    RSS_FEEDS = [
        "https://news.google.com/rss/search?q=stocks+AI+contract+OR+partnership+OR+breakout&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=defense+tech+semiconductor+stocks+contract+award+OR+acquisition&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=tech+stock+IPO+OR+listing+AI&hl=en-US&gl=US&ceid=US:en",
    ]

    def __init__(
        self,
        scanner: Optional[OpportunityScanner] = None,
        min_market_cap_usd: float = 250_000_000.0,  # Min. 250 mil. USD (ochrana proti penny stocks)
        min_price_usd: float = 2.0,                  # Min. 2 USD za akcii
        min_daily_volume: int = 50_000,              # Min. 50 000 akcií denně
    ):
        self.scanner = scanner or OpportunityScanner()
        self.min_market_cap_usd = min_market_cap_usd
        self.min_price_usd = min_price_usd
        self.min_daily_volume = min_daily_volume
        self._seen_tickers: Set[str] = set()

    def fetch_market_breaking_news(self, max_articles_per_feed: int = 25) -> List[Tuple[str, str]]:
        """Stáhne aktuální články a tiskové zprávy z globálních RSS kanálů."""
        articles: List[Tuple[str, str]] = []
        for url in self.RSS_FEEDS:
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TradingBot/2.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    tree = ET.fromstring(resp.read())
                    items = tree.findall(".//item")[:max_articles_per_feed]
                    for it in items:
                        title = it.find("title").text if it.find("title") is not None else ""
                        desc = it.find("description").text if it.find("description") is not None else ""
                        if title:
                            articles.append((title.strip(), desc.strip()))
            except Exception as exc:
                logger.debug(f"Chyba při stahování RSS kanálu {url[:45]}...: {exc}")

        logger.info(f"🔎 [Autonomous Discovery] Načteno {len(articles)} čerstvých článků o nových AI/tech zakázkách a IPO.")
        return articles

    def extract_candidate_tickers(self, articles: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        Vyextrahuje potenciální burzovní tickery a jejich kontext z textů zpráv.
        Vrací slovník {ticker: titulek_článku}.
        """
        candidates: Dict[str, str] = {}

        for title, desc in articles:
            combined = f"{title} {desc}"

            # Vzor 1: (NASDAQ: BBAI), (NYSE: PLTR), (CRWD)
            matches1 = re.findall(r"\((?:NYSE|NASDAQ|AMEX|XETRA|LSE)?:?\s*([A-Za-z0-9\.\-_]{1,6})\)", combined)
            # Vzor 2: $BBAI, $RDDT, $TEM
            matches2 = re.findall(r"\$([A-Za-z]{1,5})\b", combined)
            # Vzor 3: Ticker XYZ shares surge / climbs / falls
            matches3 = re.findall(r"\b([A-Z]{2,5})\s+(?:shares|stock|climbs|gains|surges|falls|rallies|jumps)\b", combined)

            for t in matches1 + matches2 + matches3:
                clean = t.strip().upper()
                if (
                    len(clean) >= 2
                    and clean not in FALSE_POSITIVE_TICKERS
                    and not clean.isdigit()
                    and clean not in candidates
                ):
                    candidates[clean] = title

        return candidates

    def verify_and_qualify_ticker(self, ticker: str, headline: str) -> Optional[DiscoveredOpportunity]:
        """
        Provede přísné kvantitativní a institucionální síto:
        - Ověří existenci a data na burze
        - Ověří velikost tržní kapitalizace (min. 250 mil. USD)
        - Ověří cenu akcie (min. 2 USD)
        - Ověří likviditu (denní objem)
        """
        try:
            t_obj = yf.Ticker(ticker)
            fi = t_obj.fast_info

            price = fi.get("last_price")
            if not price or price < self.min_price_usd:
                # Zkusíme fallback na historii
                hist = t_obj.history(period="5d")
                if len(hist) > 0:
                    price = float(hist["Close"].iloc[-1])
                else:
                    return None

            if price < self.min_price_usd:
                return None

            mcap = fi.get("market_cap")
            if not mcap:
                info = t_obj.info or {}
                mcap = info.get("marketCap", 0.0)

            if mcap < self.min_market_cap_usd:
                logger.debug(f"{ticker} vyřazen: tržní kapitalizace {mcap/1e6:.1f}M USD je pod limitem {self.min_market_cap_usd/1e6:.1f}M USD.")
                return None

            # Kontrola likvidity
            ten_day_vol = fi.get("ten_day_average_volume", 0)
            if ten_day_vol and ten_day_vol < self.min_daily_volume:
                logger.debug(f"{ticker} vyřazen: nízká likvidita ({ten_day_vol} ks/den).")
                return None

            company_name = getattr(fi, "name", "")
            if not company_name:
                company_name = (t_obj.info or {}).get("shortName", ticker)

            return DiscoveredOpportunity(
                ticker=ticker,
                company_name=company_name,
                price=round(price, 2),
                market_cap_usd=float(mcap),
                headline=headline,
                catalyst_snippet=headline[:80],
            )

        except Exception as exc:
            logger.debug(f"Ověření kandidáta {ticker} selhalo: {exc}")
            return None

    def discover_and_register_opportunities(self, max_new_per_run: int = 5) -> List[DiscoveredOpportunity]:
        """
        Spustí kompletní cyklus autonomního objevování:
        1. Stáhne zprávy z médií
        2. Vytáhne kandidáty
        3. Kvalifikuje a ověří přes bezpečnostní filtry
        4. Zaregistruje do univerza bota
        """
        articles = self.fetch_market_breaking_news()
        if not articles:
            return []

        candidates = self.extract_candidate_tickers(articles)
        logger.info(f"🔎 [Autonomous Discovery] Detekováno {len(candidates)} potenciálních burzovních kandidátů v médiích.")

        existing_universe = set(self.scanner.get_all_universe_tickers())
        qualified: List[DiscoveredOpportunity] = []

        for ticker, headline in candidates.items():
            if ticker in existing_universe or ticker in self._seen_tickers:
                continue

            self._seen_tickers.add(ticker)
            opp = self.verify_and_qualify_ticker(ticker, headline)

            if opp:
                qualified.append(opp)
                # Zaregistrujeme nový objev do univerza OpportunityScanneru
                added = self.scanner.register_discovered_ticker(ticker, category="AUTONOMOUS_NEWS_DISCOVERIES")
                if added:
                    logger.info(
                        f"🚀 [AUTONOMOUS DISCOVERY] Nalezen nový titul z médií: "
                        f"{opp.ticker} ({opp.company_name}) | "
                        f"Cena: {opp.price:.2f} USD | "
                        f"Tržní kapitalizace: {opp.market_cap_usd/1e6:.1f}M USD | "
                        f"Článek: \"{opp.headline[:70]}...\""
                    )

            if len(qualified) >= max_new_per_run:
                break

        return qualified
