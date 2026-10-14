from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.llm.sentiment_engine import LLMStockVerdict

logger = logging.getLogger(__name__)

CACHE_FILE = Path("data") / "intelligence_cache.json"


class CachedIntelligence:
    def __init__(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        sentiment_score: float,
        recommendation: str,
        confidence: float,
        key_drivers: List[str],
        risks: List[str],
        reasoning: str,
        article_count: int,
        latest_headline: str,
        last_analyzed_iso: str,
        last_analyzed_timestamp: float,
    ):
        self.ticker = ticker
        self.company_name = company_name
        self.sector = sector
        self.sentiment_score = sentiment_score
        self.recommendation = recommendation
        self.confidence = confidence
        self.key_drivers = key_drivers
        self.risks = risks
        self.reasoning = reasoning
        self.article_count = article_count
        self.latest_headline = latest_headline
        self.last_analyzed_iso = last_analyzed_iso
        self.last_analyzed_timestamp = last_analyzed_timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "sentiment_score": self.sentiment_score,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "key_drivers": self.key_drivers,
            "risks": self.risks,
            "reasoning": self.reasoning,
            "article_count": self.article_count,
            "latest_headline": self.latest_headline,
            "last_analyzed_iso": self.last_analyzed_iso,
            "last_analyzed_timestamp": self.last_analyzed_timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedIntelligence":
        return cls(
            ticker=data.get("ticker", ""),
            company_name=data.get("company_name", ""),
            sector=data.get("sector", ""),
            sentiment_score=float(data.get("sentiment_score", 0.0)),
            recommendation=data.get("recommendation", "HOLD"),
            confidence=float(data.get("confidence", 0.5)),
            key_drivers=data.get("key_drivers", []),
            risks=data.get("risks", []),
            reasoning=data.get("reasoning", ""),
            article_count=int(data.get("article_count", 0)),
            latest_headline=data.get("latest_headline", ""),
            last_analyzed_iso=data.get("last_analyzed_iso", ""),
            last_analyzed_timestamp=float(data.get("last_analyzed_timestamp", 0.0)),
        )

    def to_llm_verdict(self) -> LLMStockVerdict:
        """Převede cache záznam na standardní LLMStockVerdict."""
        return LLMStockVerdict(
            ticker=self.ticker,
            sentiment_score=self.sentiment_score,
            recommendation=self.recommendation,  # type: ignore
            confidence=self.confidence,
            key_drivers=self.key_drivers,
            risks=self.risks,
            reasoning=self.reasoning,
        )


class IntelligenceCache:
    """
    Perzistentní znalostní databáze pro nepřetržitý sběr a analýzu informací z webu.
    Uchovává hloubkový výzkum zpracovaný lokálním modelem na AMD NPU.
    Umožňuje okamžitou exekuci nákupů při otevření trhu bez časového stresu.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path or CACHE_FILE
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, CachedIntelligence] = {}
        self.load()

    def load(self) -> None:
        """Načte data z disku."""
        if not self.cache_path.exists():
            self._entries = {}
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = {
                ticker: CachedIntelligence.from_dict(v)
                for ticker, v in data.items()
            }
            logger.debug(f"Načteno {len(self._entries)} výzkumných záznamů z {self.cache_path}.")
        except Exception as exc:
            logger.error(f"Chyba při načítání intelligence cache: {exc}")
            self._entries = {}

    def save(self) -> None:
        """Uloží data atomicky na disk."""
        tmp_path = self.cache_path.with_suffix(".tmp")
        try:
            data = {t: e.to_dict() for t, e in self._entries.items()}
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp_path.replace(self.cache_path)
            logger.debug(f"Uloženo {len(self._entries)} výzkumných záznamů do cache.")
        except Exception as exc:
            logger.error(f"Chyba při ukládání intelligence cache: {exc}")
            if tmp_path.exists():
                tmp_path.unlink()

    def get_verdict(self, ticker: str, max_age_hours: float = 36.0) -> Optional[LLMStockVerdict]:
        """
        Vrátí cached LLM verdikt, pokud je čerstvý (mladší než max_age_hours).
        V opačném případě vrátí None.
        """
        clean = ticker.strip().upper()
        entry = self._entries.get(clean)
        if not entry:
            return None

        age_seconds = datetime.now().timestamp() - entry.last_analyzed_timestamp
        age_hours = age_seconds / 3600.0

        if age_hours > max_age_hours:
            logger.debug(f"Výzkum pro {clean} je starší než {max_age_hours}h ({age_hours:.1f}h).")
            return None

        return entry.to_llm_verdict()

    def save_verdict(
        self,
        ticker: str,
        verdict: LLMStockVerdict,
        company_name: str = "",
        sector: str = "",
        article_count: int = 0,
        latest_headline: str = "",
    ) -> None:
        """Uloží nový hloubkový výzkum zpracovaný lokálním LLM."""
        now = datetime.now()
        clean = ticker.strip().upper()
        entry = CachedIntelligence(
            ticker=clean,
            company_name=company_name or clean,
            sector=sector or "N/A",
            sentiment_score=verdict.sentiment_score,
            recommendation=verdict.recommendation,
            confidence=verdict.confidence,
            key_drivers=verdict.key_drivers,
            risks=verdict.risks,
            reasoning=verdict.reasoning,
            article_count=article_count,
            latest_headline=latest_headline,
            last_analyzed_iso=now.isoformat(),
            last_analyzed_timestamp=now.timestamp(),
        )
        self._entries[clean] = entry
        self.save()
        logger.info(
            f"🧠 Uložen výzkum pro {clean}: {verdict.recommendation} | "
            f"Sentiment: {verdict.sentiment_score:+.2f} | Článků: {article_count}"
        )

    def get_next_tickers_to_research(
        self,
        universe_tickers: List[str],
        limit: int = 5
    ) -> List[str]:
        """
        Vrátí tickery z univerza seřazené podle priority výzkumu:
        1. Tituly, které v cache vůbec nejsou (nové neprozkoumané příležitosti)
        2. Tituly s nejstarším datem poslední analýzy (staleness rotation)
        """
        scored_tickers = []
        now_ts = datetime.now().timestamp()

        for t in universe_tickers:
            clean = t.strip().upper()
            entry = self._entries.get(clean)
            if not entry:
                # Nikdy neanalyzováno = nejvyšší priorita
                scored_tickers.append((clean, 0.0, 999999.0))
            else:
                age_hours = (now_ts - entry.last_analyzed_timestamp) / 3600.0
                scored_tickers.append((clean, entry.last_analyzed_timestamp, age_hours))

        # Seřadit podle timestampu vzestupně (nejstarší první)
        scored_tickers.sort(key=lambda x: x[1])
        return [t[0] for t in scored_tickers[:limit]]

    def get_all_entries(self) -> List[CachedIntelligence]:
        """Vrátí všechny záznamy seřazené podle sentiment skóre sestupně."""
        entries = list(self._entries.values())
        entries.sort(key=lambda e: e.sentiment_score, reverse=True)
        return entries
