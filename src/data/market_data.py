import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import yfinance as yf

logger = logging.getLogger(__name__)


class TechnicalMetrics(BaseModel):
    ticker: str
    current_price: float
    currency: str = "USD"
    sma_50: float
    sma_200: float
    rsi_14: float
    atr_14: float
    trend_bullish: bool
    rsi_neutral_or_pullback: bool


class NewsItem(BaseModel):
    title: str
    publisher: str
    published_at: str
    link: str
    summary: str = ""


class FundamentalMetrics(BaseModel):
    """Klíčové fundamentální a valuační ukazatele pro profesionální stockpicking."""
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_sales: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    free_cashflow_b: Optional[float] = None
    debt_to_equity: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    distance_from_52w_high_pct: Optional[float] = None


class StockAnalysisData(BaseModel):
    ticker: str
    company_name: str
    sector: str
    industry: str = "N/A"
    business_summary: str = ""
    metrics: TechnicalMetrics
    fundamentals: Optional[FundamentalMetrics] = None
    fmp_audit: Optional[Any] = None
    recent_news: List[NewsItem]


class MarketDataProvider:
    """Zajišťuje stahování cenových dat, výpočet indikátorů, fundamenty a agregaci zpráv."""

    def __init__(self, fmp_client: Optional[Any] = None):
        from src.data.fmp_client import FMPClient
        self.fmp_client = fmp_client or FMPClient()

    def get_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Stáhne historická OHLCV data pro daný ticker z Yahoo Finance."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            return df
        except Exception as exc:
            logger.error(f"Chyba při stahování historie pro {ticker}: {exc}")
            return pd.DataFrame()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["High"]
        low = df["Low"]
        close_prev = df["Close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def get_technical_metrics(self, ticker: str) -> Optional[TechnicalMetrics]:
        """Stáhne roční historii a spočítá SMA, RSI a ATR."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y", interval="1d")

            # Vyžadujeme alespoň 50 dní pro spolehlivou kalkulaci SMA50, RSI14 a ATR14
            if df.empty or len(df) < 50:
                logger.warning(f"{ticker}: Nedostatek historických dat (vyžadováno min. 50 dní, nalezeno {len(df)}).")
                return None

            close = df["Close"]
            current_price = float(close.iloc[-1])

            currency = "USD"
            try:
                currency = stock.fast_info.get("currency", "USD") or "USD"
            except Exception:
                pass

            sma_50 = float(close.rolling(window=min(len(close), 50)).mean().iloc[-1])
            sma_200 = float(close.rolling(window=min(len(close), 200)).mean().iloc[-1])

            rsi_series = self.calculate_rsi(close, period=14)
            rsi_14 = float(rsi_series.iloc[-1])

            atr_series = self.calculate_atr(df, period=14)
            atr_14 = float(atr_series.iloc[-1])

            trend_bullish = current_price > sma_50 and (len(close) < 150 or sma_50 >= sma_200)
            rsi_neutral_or_pullback = 35.0 <= rsi_14 <= 68.0

            return TechnicalMetrics(
                ticker=ticker,
                current_price=round(current_price, 2),
                currency=currency,
                sma_50=round(sma_50, 2),
                sma_200=round(sma_200, 2),
                rsi_14=round(rsi_14, 2),
                atr_14=round(atr_14, 2),
                trend_bullish=trend_bullish,
                rsi_neutral_or_pullback=rsi_neutral_or_pullback,
            )
        except Exception as exc:
            logger.error(f"Chyba při výpočtu technických indikátorů pro {ticker}: {exc}")
            return None

    @staticmethod
    def fetch_article_summary(url: str, timeout: float = 3.5) -> str:
        """Stáhne a extrahuje klíčový text článku (odstavce o zakázkách, investicích a výsledcích)."""
        if not url or not url.startswith("http"):
            return ""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            res = requests.get(url, headers=headers, timeout=timeout)
            if res.status_code != 200:
                return ""
            soup = BeautifulSoup(res.text, "html.parser")
            paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 50]
            if not paragraphs:
                return ""
            full_text = " ".join(paragraphs[:3])
            return full_text[:450].strip()
        except Exception:
            return ""

    def get_recent_news(
        self,
        ticker: str,
        company_name: str = "",
        limit: int = 8,
        fetch_summaries_count: int = 3
    ) -> List[NewsItem]:
        """Stáhne a normalizuje nejnovější zprávy pro daný ticker s deduplikací, vyhledáváním dle názvu a čtením obsahu."""
        news_items: List[NewsItem] = []
        raw_news = []

        try:
            stock = yf.Ticker(ticker)
            raw_news = stock.news or []
        except Exception:
            pass

        # Pokud Ticker.news vrátí málo článků, zkusíme hledat ticker přes yf.Search
        if len(raw_news) < limit:
            try:
                search = yf.Search(ticker, news_count=limit)
                search_news = getattr(search, "news", []) or []
                raw_news.extend(search_news)
            except Exception as exc:
                logger.warning(f"Fallback vyhledávání zpráv pro {ticker} selhal: {exc}")

        # Pokud stále máme málo článků (typické pro evropské burzy jako RHM.DE či RR.L),
        # zkusíme vyhledávat podle čistého názvu společnosti (např. 'Rheinmetall' či 'Rolls Royce')
        if len(raw_news) < limit and company_name:
            import re
            clean_search_name = re.split(
                r'\s+(?:AG|SE|Inc|Corp|Holdings|Corporation|plc|PLC|GmbH|NV|N\.V\.|Co\.|Ltd\.)\b',
                company_name,
                flags=re.IGNORECASE
            )[0].strip()
            clean_search_name = re.sub(r'\s+', ' ', clean_search_name)
            if len(clean_search_name) >= 3 and clean_search_name.upper() != ticker.upper():
                try:
                    name_search = yf.Search(clean_search_name, news_count=limit)
                    name_news = getattr(name_search, "news", []) or []
                    raw_news.extend(name_news)
                except Exception as exc:
                    logger.debug(f"Vyhledávání zpráv podle názvu {clean_search_name} selhalo: {exc}")

        seen_titles = set()
        for item in raw_news:
            title = item.get("title") or item.get("content", {}).get("title", "")
            clean_title = title.strip() if title else ""
            if not clean_title or clean_title in seen_titles:
                continue

            seen_titles.add(clean_title)
            publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName", "Neznámý")
            link = item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url", "")
            pub_time = item.get("providerPublishTime")

            if pub_time:
                published_str = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")
            else:
                published_str = "Nedávno"

            # Pro prvních několik klíčových článků stáhneme skutečný obsah/výtah z webu
            summary = ""
            if len(news_items) < fetch_summaries_count and link:
                summary = self.fetch_article_summary(link)

            news_items.append(
                NewsItem(
                    title=clean_title,
                    publisher=publisher.strip(),
                    published_at=published_str,
                    link=link.strip(),
                    summary=summary,
                )
            )

            if len(news_items) >= limit:
                break

        return news_items

    def prepare_ticker_data(self, ticker: str, news_limit: int = 8) -> Optional[StockAnalysisData]:
        """Sestaví kompletní analytický profil akcie (ceny, metriky, zprávy i obsah článků)."""
        metrics = self.get_technical_metrics(ticker)
        if not metrics:
            return None

        # 1. Zjištění informací o společnosti a fundamentů
        try:
            info = yf.Ticker(ticker).info or {}
            company_name = info.get("shortName") or info.get("longName") or ticker
            sector = info.get("sector", "Neznámý sektor")
            industry = info.get("industry", "Neznámé odvětví")
            summary = (info.get("longBusinessSummary") or "")[:350]

            f_pe = info.get("forwardPE")
            t_pe = info.get("trailingPE")
            peg = info.get("pegRatio")
            ps = info.get("priceToSalesTrailing12Months")
            rev_g = info.get("revenueGrowth")
            earn_g = info.get("earningsGrowth")
            op_m = info.get("operatingMargins")
            gr_m = info.get("grossMargins")
            fcf = info.get("freeCashflow")
            fcf_b = round(fcf / 1e9, 2) if fcf else None
            dte = info.get("debtToEquity")
            w52_high = info.get("fiftyTwoWeekHigh")

            dist_high = None
            if w52_high and w52_high > 0:
                dist_high = round(((metrics.current_price - w52_high) / w52_high) * 100.0, 1)

            fundamentals = FundamentalMetrics(
                trailing_pe=round(t_pe, 1) if t_pe else None,
                forward_pe=round(f_pe, 1) if f_pe else None,
                peg_ratio=round(peg, 2) if peg else None,
                price_to_sales=round(ps, 2) if ps else None,
                revenue_growth_yoy=round(rev_g * 100.0, 1) if rev_g is not None else None,
                earnings_growth_yoy=round(earn_g * 100.0, 1) if earn_g is not None else None,
                operating_margin=round(op_m * 100.0, 1) if op_m is not None else None,
                gross_margin=round(gr_m * 100.0, 1) if gr_m is not None else None,
                free_cashflow_b=fcf_b,
                debt_to_equity=round(dte, 1) if dte else None,
                fifty_two_week_high=round(w52_high, 2) if w52_high else None,
                distance_from_52w_high_pct=dist_high,
            )
        except Exception as exc:
            logger.warning(f"Chyba při stahování fundamentálních dat pro {ticker}: {exc}")
            company_name = ticker
            sector = "N/A"
            industry = "N/A"
            summary = ""
            fundamentals = None

        # 2. Načtení zpráv a stažení obsahu článků o investicích/zakázkách
        news = self.get_recent_news(ticker, company_name=company_name, limit=news_limit, fetch_summaries_count=3)

        # 3. Volitelný institucionální audit z FMP (DCF vnitřní hodnota, ROIC, Insider obchody)
        fmp_audit = None
        if self.fmp_client.is_enabled:
            fmp_audit = self.fmp_client.audit_stock(ticker, metrics.current_price)

        return StockAnalysisData(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            industry=industry,
            business_summary=summary,
            metrics=metrics,
            fundamentals=fundamentals,
            fmp_audit=fmp_audit,
            recent_news=news,
        )