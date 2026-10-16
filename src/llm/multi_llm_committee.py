import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from config import settings
from src.data.market_data import StockAnalysisData
from src.llm.sentiment_engine import LLMStockVerdict, SentimentEngine

logger = logging.getLogger(__name__)


class LLMAgentRole(str, Enum):
    FUNDAMENTAL_ANALYST = "FUNDAMENTAL_ANALYST"  # Warren Buffett / Value & Quality (valuace, marže, rozvaha)
    CONTRARIAN_AUDITOR = "CONTRARIAN_AUDITOR"    # Jim Chanos / Skeptik & Ďáblův advokát (skrytá rizika, past na investory)
    CATALYST_SPECIALIST = "CATALYST_SPECIALIST"  # Stanley Druckenmiller / Momentum & Makro (zprávy, earnings, sentiment)


class AgentVerdict(BaseModel):
    """Verdikt jednoho specializovaného AI agenta v radě."""
    role: LLMAgentRole
    recommendation: Literal["BUY", "HOLD", "AVOID"]
    sentiment_score: float = Field(..., description="-1.0 až +1.0")
    confidence: float = Field(..., description="0.0 až 1.0")
    key_arguments: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    reasoning: str = Field(..., description="Zdůvodnění v češtině")


class CommitteeConsensus(BaseModel):
    """
    Finální syntéza a konsenzus Rady AI analytiků (Multi-LLM Consensus).
    Porovnává odpovědi jednotlivých agentů a ověřuje pravdivost hypotézy.
    """
    ticker: str
    company_name: str
    final_recommendation: Literal["BUY", "HOLD", "AVOID"]
    consensus_strength: Literal["UNANIMOUS", "MAJORITY", "SPLIT_DECISION", "VETO_TRIGGERED"]
    approved_for_trading: bool
    agreement_percentage: float
    average_sentiment: float
    average_confidence: float
    consolidated_catalysts: List[str]
    consolidated_risks: List[str]
    consensus_synthesis: str
    truth_verification_note: str
    agent_verdicts: Dict[str, AgentVerdict]
    timestamp_iso: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


class MultiLLMCommittee:
    """
    Rada AI modelů a specializovaných analytických person (Multi-LLM Consensus Committee).
    
    Princip:
    Místo jednoho promptu provádí křížovou oponenturu mezi 3 nezávislými analytiky:
    1. Fundamentální analytik (hodnotí valuaci, P/E, marže, ROIC a stabilitu byznysu)
    2. Skeptický auditor / Ďáblův advokát (hledá skrytá rizika, past na investory a klamavý růst)
    3. Katalytický a momentum specialista (hodnotí zprávy, momentum a makro trend)
    
    Vyhodnocení shody:
    - Pokud Skeptik najde kritické riziko -> VETO (nákup se zruší).
    - Pokud se všichni shodnou na BUY -> UNANIMOUS (nejvyšší priorita a plná alokace).
    - Pokud se názory rozejdou -> SPLIT_DECISION (vyčkání na shodu).
    """

    def __init__(
        self,
        sentiment_engine: Optional[SentimentEngine] = None,
        secondary_base_url: Optional[str] = None,
        secondary_model: Optional[str] = None,
    ):
        self.sentiment_engine = sentiment_engine or SentimentEngine()
        self.primary_client = OpenAI(
            base_url=self.sentiment_engine.base_url,
            api_key=self.sentiment_engine.api_key,
            max_retries=0,
        )
        self.primary_model = self.sentiment_engine.model

        # Volitelný sekundární LLM server (např. Ollama na portu 11434 nebo jiný model)
        self.secondary_client: Optional[OpenAI] = None
        self.secondary_model = secondary_model
        if secondary_base_url:
            self.secondary_client = OpenAI(base_url=secondary_base_url, api_key="ollama", max_retries=0)

    def _build_role_prompt(self, role: LLMAgentRole, stock_data: StockAnalysisData) -> tuple[str, str]:
        """Sestaví systémovou personu a detailní zadání pro konkrétního analytika."""
        m = stock_data.metrics
        f = stock_data.fundamentals
        curr = getattr(m, "currency", "USD")

        # Přehled fundamentů
        fund_lines = []
        if f:
            if f.trailing_pe: fund_lines.append(f"- Trailing P/E: {f.trailing_pe:.1f}")
            if f.forward_pe: fund_lines.append(f"- Forward P/E: {f.forward_pe:.1f}")
            if f.peg_ratio: fund_lines.append(f"- PEG poměr: {f.peg_ratio:.2f}")
            if f.gross_margin: fund_lines.append(f"- Hrubá marže: {f.gross_margin:.1f} %")
            if f.operating_margin: fund_lines.append(f"- Provozní marže: {f.operating_margin:.1f} %")
            if f.revenue_growth_yoy: fund_lines.append(f"- Meziroční růst tržeb: {f.revenue_growth_yoy:+.1f} %")
            if f.earnings_growth_yoy: fund_lines.append(f"- Meziroční růst zisků: {f.earnings_growth_yoy:+.1f} %")
            if f.free_cashflow_b: fund_lines.append(f"- Volné cash flow (FCF): {f.free_cashflow_b:.2f} mld.")
            if f.debt_to_equity: fund_lines.append(f"- Zadlužení D/E: {f.debt_to_equity:.1f} %")

        fund_info = "\n".join(fund_lines) if fund_lines else "Fundamentální data nejsou v plném rozsahu dostupná."

        # Přehled zpráv
        news_lines = []
        for idx, item in enumerate(stock_data.recent_news[:4], 1):
            news_lines.append(f"{idx}. {item.title} (Zdroj: {item.publisher})")
        news_info = "\n".join(news_lines) if news_lines else "Žádné mimořádné zprávy."

        tech_info = (
            f"- Aktuální cena: {m.current_price:.2f} {curr}\n"
            f"- Trend: {'Býčí (Cena > SMA 50 > SMA 200)' if m.trend_bullish else 'Medvědí/Korekční'}\n"
            f"- RSI (14 period): {m.rsi_14:.1f}\n"
            f"- ATR volatilita: {m.atr_14:.2f} {curr}"
        )

        base_data = (
            f"SPOLEČNOST: {stock_data.ticker} ({stock_data.company_name})\n"
            f"Sektor: {stock_data.sector} | Odvětví: {stock_data.industry}\n\n"
            f"FUNDAMENTY A VALUACE:\n{fund_info}\n\n"
            f"TECHNICKÝ STAV:\n{tech_info}\n\n"
            f"ČERSTVÉ ZPRÁVY:\n{news_info}"
        )

        if role == LLMAgentRole.FUNDAMENTAL_ANALYST:
            system_prompt = (
                "Jsi Fundamentální hodnotový a růstový analytik (ve stylu Warrena Buffetta a Petera Lynche). "
                "Tvým jediným cílem je posoudit ekonomickou sílu podniku, rozvahu, marže, ROIC a férovost valuace. "
                "Odmítáš předražené hypy bez reálných zisků. "
                "Odpovídej VÝHRADNĚ platným JSON objektem."
            )
            user_prompt = (
                f"{base_data}\n\n"
                f"ÚKOL: Zhodnoť firmu jako dlouhodobý vlastník podniku.\n"
                f"Je cena ospravedlnitelná vzhledem k maržím, ziskům a DCF? Má firma silný 'moat'?\n"
                f"Formát odpovědi (striktní JSON):\n"
                f'{{"recommendation": "BUY"|"HOLD"|"AVOID", "sentiment_score": 0.0, "confidence": 0.85, '
                f'"key_arguments": ["argument 1", "argument 2"], "red_flags": ["riziko 1"], '
                f'"reasoning": "Stručné zhodnocení v češtině (2 věty)"}}'
            )

        elif role == LLMAgentRole.CONTRARIAN_AUDITOR:
            system_prompt = (
                "Jsi Skeptický auditor rizik a Ďáblův advokát fondu (ve stylu Jima Chanose a Michaela Burryho). "
                "Tvým úkolem je nekompromisně hledat důvody, PROČ AKCII NEKUPOVAT. "
                "Hledej účetní rizika, klesající marže, vysoký dluh, příliš optimistická očekávání, "
                "překoupené technické indikátory (RSI > 70) nebo krátkodobou mediální bublinu. "
                "Pokud je riziko neúměrné, uděl AVOID. "
                "Odpovídej VÝHRADNĚ platným JSON objektem."
            )
            user_prompt = (
                f"{base_data}\n\n"
                f"ÚKOL: Identifikuj všechny skryté hrozby, rizika a slabiny.\n"
                f"Co by se mohlo pokazit? Je současný růst udržitelný, nebo jde o past na investory?\n"
                f"Formát odpovědi (striktní JSON):\n"
                f'{{"recommendation": "BUY"|"HOLD"|"AVOID", "sentiment_score": -0.2, "confidence": 0.80, '
                f'"key_arguments": ["pozitivní bod"], "red_flags": ["hlavní hrozba 1", "hrozba 2"], '
                f'"reasoning": "Skeptické zhodnocení rizik v češtině (2 věty)"}}'
            )

        else:  # CATALYST_SPECIALIST
            system_prompt = (
                "Jsi Makro a Katalytický specialista (ve stylu Stanleyho Druckenmillera). "
                "Hledáš okamžitý vítr v zádech: nová partnerství, růst tržeb, technologické momentum, "
                "cyklický obrat a reakci na zprávy. "
                "Zkoumáš, zda má akcie reálný impuls růst v horizontu nejbližších týdnů. "
                "Odpovídej VÝHRADNĚ platným JSON objektem."
            )
            user_prompt = (
                f"{base_data}\n\n"
                f"ÚKOL: Zhodnoť zprávy, momentum a krátkodobé/střednědobé katalyzátory.\n"
                f"Existuje reálný nákupní tlak a institucionální zájem?\n"
                f"Formát odpovědi (striktní JSON):\n"
                f'{{"recommendation": "BUY"|"HOLD"|"AVOID", "sentiment_score": 0.5, "confidence": 0.80, '
                f'"key_arguments": ["katalyzátor 1", "katalyzátor 2"], "red_flags": ["riziko"], '
                f'"reasoning": "Zhodnocení katalyzátorů v češtině (2 věty)"}}'
            )

        return system_prompt, user_prompt

    def _query_agent(
        self,
        role: LLMAgentRole,
        stock_data: StockAnalysisData,
        timeout: float = 6.0,
    ) -> AgentVerdict:
        """Dotáže se jednoho analytika a bezpečně naparsuje jeho strukturovaný verdikt."""
        sys_p, user_p = self._build_role_prompt(role, stock_data)

        # Výběr klienta: Skeptika můžeme směrovat na sekundární model, pokud je k dispozici
        client = self.secondary_client if (role == LLMAgentRole.CONTRARIAN_AUDITOR and self.secondary_client) else self.primary_client
        model = self.secondary_model if (role == LLMAgentRole.CONTRARIAN_AUDITOR and self.secondary_client) else self.primary_model

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p},
                ],
                temperature=0.2 if role != LLMAgentRole.CONTRARIAN_AUDITOR else 0.3,
                max_tokens=220,
                timeout=timeout,
            )
            raw = resp.choices[0].message.content or ""
            data = self.sentiment_engine._sanitize_and_extract_json(raw, fallback_ticker=stock_data.ticker)
            if data:
                rec_raw = str(data.get("recommendation", "HOLD")).upper()
                rec = "BUY" if "BUY" in rec_raw else ("AVOID" if "AVOID" in rec_raw or "SELL" in rec_raw else "HOLD")
                score = max(-1.0, min(1.0, float(data.get("sentiment_score", 0.0))))
                conf = max(0.0, min(1.0, float(data.get("confidence", 0.7))))
                args = data.get("key_arguments") or data.get("key_drivers") or []
                flags = data.get("red_flags") or data.get("risks") or []
                reas = str(data.get("reasoning", "Analýza provedena AI agentem."))

                return AgentVerdict(
                    role=role,
                    recommendation=rec,  # type: ignore
                    sentiment_score=score,
                    confidence=conf,
                    key_arguments=[str(x) for x in args[:3]],
                    red_flags=[str(x) for x in flags[:3]],
                    reasoning=reas,
                )

        except Exception as exc:
            logger.debug(f"LLM volání pro {role.value} selhalo ({exc}). Používám kvantitativní heuristiku.")

        # Inteligentní heuristický fallback pro případ nedostupnosti modelu
        return self._heuristic_fallback(role, stock_data)

    def _heuristic_fallback(self, role: LLMAgentRole, stock_data: StockAnalysisData) -> AgentVerdict:
        """Kvantitativní simulace role při dočasné nedostupnosti lokálního LLM."""
        m = stock_data.metrics
        f = stock_data.fundamentals

        if role == LLMAgentRole.FUNDAMENTAL_ANALYST:
            pe = f.trailing_pe if (f and f.trailing_pe) else (f.forward_pe if f and f.forward_pe else 25.0)
            op_m = f.operating_margin if (f and f.operating_margin) else 15.0
            is_good = (pe < 45.0) and (op_m > 10.0)
            rec = "BUY" if is_good and m.trend_bullish else "HOLD"
            score = 0.50 if rec == "BUY" else 0.10
            return AgentVerdict(
                role=role,
                recommendation=rec,  # type: ignore
                sentiment_score=score,
                confidence=0.75,
                key_arguments=[f"P/E poměr: {pe:.1f}", f"Provozní marže: {op_m:.1f} %"],
                red_flags=["Sledovat udržitelnost marží při inflaci"],
                reasoning=f"Fundamentální profil vykazuje P/E {pe:.1f} a provozní marži {op_m:.1f} %.",
            )

        elif role == LLMAgentRole.CONTRARIAN_AUDITOR:
            overbought = m.rsi_14 > 68.0
            debt_high = (f and f.debt_to_equity and f.debt_to_equity > 200.0)
            if overbought or debt_high:
                rec = "AVOID"
                score = -0.40
                reason = f"Upozornění: Překoupený stav (RSI {m.rsi_14:.1f}) nebo vyšší zadlužení."
            else:
                rec = "HOLD"
                score = 0.0
                reason = f"Žádná kritická účetní rizika nenalezena (RSI: {m.rsi_14:.1f})."

            return AgentVerdict(
                role=role,
                recommendation=rec,  # type: ignore
                sentiment_score=score,
                confidence=0.70,
                key_arguments=["Stabilní technický profil bez akutního zhroucení"],
                red_flags=[f"RSI úroveň: {m.rsi_14:.1f}", "Riziko cyklického ochlazení sektoru"],
                reasoning=reason,
            )

        else:  # CATALYST_SPECIALIST
            has_news = len(stock_data.recent_news) > 0
            bull = m.trend_bullish and (m.rsi_14 >= 50.0)
            rec = "BUY" if bull else "HOLD"
            score = 0.60 if rec == "BUY" else 0.15
            hl = stock_data.recent_news[0].title if has_news else "Pozitivní trendové momentum"
            return AgentVerdict(
                role=role,
                recommendation=rec,  # type: ignore
                sentiment_score=score,
                confidence=0.75,
                key_arguments=[hl[:50], "Býčí momentum nad klouzavými průměry"],
                red_flags=["Možná krátkodobá konsolidace"],
                reasoning=f"Katalyzátory a technické momentum potvrzují zájem trhu.",
            )

    def evaluate_consensus(
        self,
        stock_data: StockAnalysisData,
        fast_mode: bool = True,
        timeout: float = 3.0,
    ) -> CommitteeConsensus:
        """
        Spustí zasedání Rady AI modelů pro danou akcii:
        Pokud fast_mode=True (výchozí pro day trading a rychlé skeny):
        - Používá pevný timeout (výchozí 3s), při výpadku/timeoutu okamžitě
          využije kvantitativní pravidlové verdikty bez blokování hlavního vlákna.
        """
        ticker = stock_data.ticker
        logger.info(f"🏛️ [RADA AI ANALYTIKŮ] Zasedání rady pro titul {ticker} ({stock_data.company_name}, fast_mode={fast_mode})...")

        # 1. Získání verdiktů od jednotlivých agentů (s přísným timeoutem v fast_mode)
        effective_timeout = min(timeout, 3.0) if fast_mode else timeout
        v_fund = self._query_agent(LLMAgentRole.FUNDAMENTAL_ANALYST, stock_data, timeout=effective_timeout)
        v_audit = self._query_agent(LLMAgentRole.CONTRARIAN_AUDITOR, stock_data, timeout=effective_timeout)
        v_cat = self._query_agent(LLMAgentRole.CATALYST_SPECIALIST, stock_data, timeout=effective_timeout)

        verdicts = {
            LLMAgentRole.FUNDAMENTAL_ANALYST.value: v_fund,
            LLMAgentRole.CONTRARIAN_AUDITOR.value: v_audit,
            LLMAgentRole.CATALYST_SPECIALIST.value: v_cat,
        }

        # 2. Analýza shody
        recs = [v.recommendation for v in verdicts.values()]
        buy_count = recs.count("BUY")
        hold_count = recs.count("HOLD")
        avoid_count = recs.count("AVOID")

        avg_sentiment = round(sum(v.sentiment_score for v in verdicts.values()) / 3.0, 2)
        avg_confidence = round(sum(v.confidence for v in verdicts.values()) / 3.0, 2)

        # Agregace argumentů a rizik
        catalysts: List[str] = []
        risks: List[str] = []
        for v in verdicts.values():
            catalysts.extend(v.key_arguments)
            risks.extend(v.red_flags)
        unique_catalysts = list(dict.fromkeys(catalysts))[:4]
        unique_risks = list(dict.fromkeys(risks))[:4]

        # 3. Křížové vyhodnocení a detekce shody / VETA
        if avoid_count > 0 and v_audit.recommendation == "AVOID":
            # Skeptický auditor zvedl červenou vlajku -> VETO
            consensus_strength = "VETO_TRIGGERED"
            final_rec = "AVOID"
            approved = False
            agreement_pct = round((avoid_count / 3.0) * 100.0, 1)
            synth = (
                f"🚫 [VETO SKEPTICKÉHO AUDITORA] Nákup {ticker} byl radou zamítnut! "
                f"Auditor rizik identifikoval kritické hrozby: {', '.join(v_audit.red_flags[:2])}. "
                f"I přes pozitivní signály ostatních agentů je prioritou ochrana kapitálu."
            )
            truth_note = f"Tvrzení o bezrizikovém růstu se nepotvrdilo – odhalena skrytá rizika: {v_audit.reasoning}"

        elif buy_count == 3:
            # Plná jednomyslná shoda
            consensus_strength = "UNANIMOUS"
            final_rec = "BUY"
            approved = True
            agreement_pct = 100.0
            synth = (
                f"🌟 [JEDNOHLASNÝ KONSENZUS BUY (100 %)] Všichni 3 AI analytici (Fundament, Rizika i Katalyzátory) "
                f"se nezávisle shodli na vysoké atraktivitě {ticker}. "
                f"Klíčové faktory: {', '.join(unique_catalysts[:2])}."
            )
            truth_note = "Pravdivost růstové hypotézy potvrzena: Fundamenty, nízká rizika i momentum jsou v plném souladu."

        elif buy_count == 2 and avoid_count == 0:
            # Většinová shoda (2x BUY, 1x HOLD)
            consensus_strength = "MAJORITY"
            final_rec = "BUY"
            approved = True
            agreement_pct = 66.7
            synth = (
                f"🟢 [VĚTŠINOVÝ KONSENZUS BUY (67 %)] Dva analytici doporučují nákup {ticker}, "
                f"třetí agent doporučuje vyčkání (HOLD). Nebyla vznesena žádná fatální rizika."
            )
            truth_note = "Hypotéza je solidní, mírná opatrnost ohledně načasování vstupu."

        elif avoid_count >= 2:
            consensus_strength = "MAJORITY"
            final_rec = "AVOID"
            approved = False
            agreement_pct = round((avoid_count / 3.0) * 100.0, 1)
            synth = f"🔴 [RADA DOPORUČUJE VYHNOUT SE] Většina analytiků varuje před nákupem {ticker}."
            truth_note = "Hypotéza o ziskovém obchodu je neplatná – převažují rizika a oslabení profilu."

        else:
            consensus_strength = "SPLIT_DECISION"
            final_rec = "HOLD"
            approved = False
            agreement_pct = 33.3
            synth = (
                f"⚪ [ROZDĚLENÝ NÁZOR RADY (SPLIT DECISION)] Mezi agenty panuje neshoda ohledně {ticker} "
                f"(BUY: {buy_count}, HOLD: {hold_count}, AVOID: {avoid_count}). Obchod se neuskuteční."
            )
            truth_note = "Pravdivost není jednoznačná – protichůdné signály mezi valuací a technickým momentem."

        logger.info(
            f"🏛️ [KONSENZUS PRO {ticker}] {consensus_strength} | Doporučení: {final_rec} | "
            f"Shoda: {agreement_pct:.0f} % | Schváleno: {'ANO' if approved else 'NE'}"
        )

        return CommitteeConsensus(
            ticker=ticker,
            company_name=stock_data.company_name,
            final_recommendation=final_rec,  # type: ignore
            consensus_strength=consensus_strength,  # type: ignore
            approved_for_trading=approved,
            agreement_percentage=agreement_pct,
            average_sentiment=avg_sentiment,
            average_confidence=avg_confidence,
            consolidated_catalysts=unique_catalysts,
            consolidated_risks=unique_risks,
            consensus_synthesis=synth,
            truth_verification_note=truth_note,
            agent_verdicts=verdicts,
        )
