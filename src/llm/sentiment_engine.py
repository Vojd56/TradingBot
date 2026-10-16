import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional
from openai import OpenAI
from pydantic import AliasChoices, BaseModel, Field, field_validator
from config import settings
from src.data import StockAnalysisData

logger = logging.getLogger(__name__)


class LLMStockVerdict(BaseModel):
    """Strukturované rozhodnutí a sentimentová analýza z LLM."""
    ticker: str
    sentiment_score: float = Field(
        ...,
        description="-1.0 (velmi negativní) až +1.0 (velmi pozitivní)"
    )
    recommendation: Literal["BUY", "HOLD", "AVOID"]
    confidence: float = Field(
        ...,
        description="Míra jistoty modelu 0.0 až 1.0"
    )
    key_drivers: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("key_drivers", "key_driver", "driver", "drivers", "factors", "positive_factors", "catalysts"),
        description="Klíčové fundamentální nebo sentimentové katalyzátory"
    )
    risks: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("risks", "risk", "risk_factors", "risk_factor", "negative_factors", "concerns"),
        description="Identifikovaná rizika pro střednědobý a dlouhodobý horizont"
    )
    reasoning: str = Field(
        ...,
        description="Syntéza a odůvodnění rozhodnutí v češtině"
    )

    @field_validator("sentiment_score", mode="before")
    @classmethod
    def clamp_sentiment(cls, v: Any) -> float:
        try:
            val = float(v)
            return max(-1.0, min(1.0, round(val, 2)))
        except (ValueError, TypeError):
            return 0.0

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            val = float(v)
            return max(0.0, min(1.0, round(val, 2)))
        except (ValueError, TypeError):
            return 0.5

    @field_validator("recommendation", mode="before")
    @classmethod
    def normalize_recommendation(cls, v: Any) -> str:
        if isinstance(v, str):
            clean = v.strip().upper()
            if "BUY" in clean or "NAKUP" in clean:
                return "BUY"
            if "AVOID" in clean or "SELL" in clean or "PRODEJ" in clean:
                return "AVOID"
            if "HOLD" in clean or "DRZ" in clean:
                return "HOLD"
        return "HOLD"


class SentimentEngine:
    """
    Komunikační rozhraní s lokálním Lemonade Serverem (OpenAI API specifikace).
    Zajišťuje formulaci promptu, volání modelu na NPU, sanitizaci a validaci JSON.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.base_url = base_url or settings.lemonade_base_url
        self.api_key = api_key or settings.lemonade_api_key
        self.model = model or settings.lemonade_model

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _build_prompt(self, stock_data: StockAnalysisData) -> str:
        m = stock_data.metrics
        f = stock_data.fundamentals
        curr = getattr(m, "currency", "USD")

        # Formátování fundamentálních ukazatelů
        fund_lines = []
        if f:
            t_pe_str = f"{f.trailing_pe:.1f}" if f.trailing_pe is not None else "N/A (ve ztrátě nebo nedostupné)"
            f_pe_str = f"{f.forward_pe:.1f}" if f.forward_pe is not None else "N/A"
            peg_str = f"{f.peg_ratio:.2f}" if f.peg_ratio is not None else "N/A"
            ps_str = f"{f.price_to_sales:.2f}" if f.price_to_sales is not None else "N/A"
            rev_g_str = f"{f.revenue_growth_yoy:+.1f} %" if f.revenue_growth_yoy is not None else "N/A"
            earn_g_str = f"{f.earnings_growth_yoy:+.1f} %" if f.earnings_growth_yoy is not None else "N/A"
            op_m_str = f"{f.operating_margin:.1f} %" if f.operating_margin is not None else "N/A"
            gr_m_str = f"{f.gross_margin:.1f} %" if f.gross_margin is not None else "N/A"
            fcf_str = f"{f.free_cashflow_b:.2f} mld." if f.free_cashflow_b is not None else "N/A"
            dte_str = f"{f.debt_to_equity:.1f} %" if f.debt_to_equity is not None else "N/A"
            dist_high_str = f"{f.distance_from_52w_high_pct:+.1f} %" if f.distance_from_52w_high_pct is not None else "N/A"

            fund_lines.append(f"- Trailing P/E: {t_pe_str} | Forward P/E: {f_pe_str}")
            fund_lines.append(f"- PEG Ratio: {peg_str} | Price/Sales (P/S): {ps_str}")
            fund_lines.append(f"- Růst tržeb (YoY): {rev_g_str} | Růst zisků (YoY): {earn_g_str}")
            fund_lines.append(f"- Hrubá marže: {gr_m_str} | Provozní marže: {op_m_str}")
            fund_lines.append(f"- Volné cash flow (FCF): {fcf_str} | Dluh vůči jmění (D/E): {dte_str}")
            fund_lines.append(f"- Vzdálenost od 52týdenního maxima: {dist_high_str}")
        else:
            fund_lines.append("- Detailní valuační násobky nejsou pro tento titul plně k dispozici.")

        fund_text = "\n".join(fund_lines)

        # Volitelný institucionální audit z FMP
        fmp = getattr(stock_data, "fmp_audit", None)
        fmp_lines = []
        if fmp:
            dcf_str = f"{fmp.dcf_fair_value:.2f} {curr}" if fmp.dcf_fair_value else "N/A"
            if fmp.dcf_diff_pct is not None:
                if fmp.dcf_diff_pct > 0:
                    dcf_diff_str = f"(+podhodnoceno o {fmp.dcf_diff_pct:.1f} % vůči vnitřní hodnotě DCF - příznivý upside)"
                else:
                    dcf_diff_str = f"(-předraženo o {abs(fmp.dcf_diff_pct):.1f} % nad vnitřní hodnotu DCF - varování)"
            else:
                dcf_diff_str = ""
            roic_str = f"{fmp.roic_ttm_pct:.1f} %" if fmp.roic_ttm_pct is not None else "N/A"
            ev_ebitda_str = f"{fmp.ev_to_ebitda:.1f}x" if fmp.ev_to_ebitda else "N/A"
            net_debt_str = f"{fmp.net_debt_to_ebitda:.2f}x" if fmp.net_debt_to_ebitda is not None else "N/A"
            fcf_yield_str = f"{fmp.fcf_yield_pct:.1f} %" if fmp.fcf_yield_pct is not None else "N/A"
            graham_str = f"{fmp.graham_number:.2f} {curr}" if fmp.graham_number else "N/A"

            fmp_lines.append(f"- Vnitřní hodnota (DCF Fair Value): {dcf_str} {dcf_diff_str}")
            fmp_lines.append(f"- Návratnost investovaného kapitálu (ROIC): {roic_str} (Buffettova kapitálová efektivita, benchmark > 15 %)")
            fmp_lines.append(f"- EV / EBITDA: {ev_ebitda_str} | Čistý dluh / EBITDA: {net_debt_str}")
            fmp_lines.append(f"- FCF Výnos (FCF Yield): {fcf_yield_str} | Grahamovo číslo: {graham_str}")

        fmp_text = "\nINSTITUCIONÁLNÍ AUDIT (FMP AUDITOVANÁ DATA):\n" + "\n".join(fmp_lines) if fmp_lines else ""

        news_lines = []
        if stock_data.recent_news:
            for idx, item in enumerate(stock_data.recent_news[:4], 1):
                line = f"{idx}. [{item.published_at}] {item.title} (Zdroj: {item.publisher})"
                if item.summary:
                    clean_sum = (item.summary[:200] + "...") if len(item.summary) > 200 else item.summary
                    line += f"\n   Shrnutí: {clean_sum}"
                news_lines.append(line)
            news_text = "\n".join(news_lines)
        else:
            news_text = "Žádné nové zprávy nejsou k dispozici. Rozhoduj dle fundamentu, valuace a technického profilu."

        summary_text = f"\n- Profil byznysu: {stock_data.business_summary[:200]}..." if stock_data.business_summary else ""

        prompt = f"""Jsi seniorní globální akciový analytik a portfolio manažer řídící koncentrované 30leté portfolio. Tvým úkolem je nekompromisně posoudit následující společnost z pohledu asymetrického poměru rizika a výnosu.

SPOLEČNOST K ANALÝZE:
- Ticker: {stock_data.ticker}
- Název: {stock_data.company_name}
- Sektor: {stock_data.sector} | Odvětví: {stock_data.industry}{summary_text}

FUNDAMENTÁLNÍ VALUACE A FINANČNÍ ZDRAVÍ:
{fund_text}{fmp_text}

TECHNICKÝ STAV A MOMENTUM:
- Aktuální cena: {m.current_price:.2f} {curr}
- 50denní průměr (SMA 50): {m.sma_50:.2f} {curr} | 200denní průměr (SMA 200): {m.sma_200:.2f} {curr}
- RSI (14 period): {m.rsi_14:.1f}
- ATR volatilita (14 period): {m.atr_14:.2f} {curr}
- Dlouhodobý trend: {'Býčí (Cena > SMA 50 > SMA 200)' if m.trend_bullish else 'Neutrální / Medvědí (pod klouzavými průměry)'}

AKTUÁLNÍ ZPRÁVY A TITULKY:
{news_text}

DŮRAZNÁ PRAVIDLA PRO INVESTIČNÍ ROZHODOVÁNÍ:
1. VÝCHOZÍ POSTOJ JE 'HOLD' NEBO 'AVOID': Většina akcií (70-80 %) na trhu NENÍ výhodnou investicí. Skutečný investor nenakupuje bez silné bezpečnostní marže.
2. ZÁKAZ 'BUY':
   - Pokud je akcie výrazně předražená nad vnitřní hodnotu DCF (-předraženo),
   - NEBO je firma ve ztrátě (P/E N/A) či má slabý/nulový ROIC (< 10 %),
   - NEBO je cena pod 200denním průměrem (SMA 200),
   - NIKDY nesmíš udělit 'BUY'! V takovém případě musíš udělit 'AVOID' nebo 'HOLD'.
3. PODMÍNKY PRO 'BUY' (Pouze výjimečné příležitosti):
   - Asymetrický poměr rizika a zisku, rozumná valuace vzhledem k růstu, silný moat, vysoký ROIC a RSI pod 68.

Odpověz VÝHRADNĚ platným JSON objektem v tomto přesném formátu (všechny klíče a texty v dvojitých uvozovkách):
{{
  "ticker": "{stock_data.ticker}",
  "sentiment_score": 0.0,
  "recommendation": "HOLD",
  "confidence": 0.70,
  "key_drivers": ["Konkrétní katalyzátor specifický pro {stock_data.ticker}", "Druhý katalyzátor"],
  "risks": ["Konkrétní riziko pro {stock_data.ticker}", "Druhé riziko"],
  "reasoning": "2-3 věty v češtině vysvětlující verdikt z pohledu valuace, marží, zpráv a RSI."
}}
Povolené hodnoty pro recommendation jsou výhradně: "BUY", "HOLD", "AVOID". Nahraď ukázkové hodnoty skutečnou analýzou pro {stock_data.ticker}."""
        return prompt

    def _sanitize_and_extract_json(
        self,
        raw_text: str,
        fallback_ticker: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Robustní extrakce a oprava JSONu z výstupu LLM:
        - odstraní markdown bloky
        - najde vnější složené závorky { ... }
        - opraví případné chyby tokenizeru (např. chybějící uvozovky nebo lomítka)
        - opraví opakované čárky a čárky před uzavírací závorkou
        - fallback na regex extrakci polí v případě poškozené syntaxe
        """
        if not raw_text or not raw_text.strip():
            return None

        # 1. Odstranění markdown bloků ```json ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE).strip()

        # 2. Hledání vnějších složených závorek
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1:
            if end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx : end_idx + 1]
            else:
                cleaned = cleaned[start_idx:] + '\n}'

        # 3. Přímý pokus o parsing čistého JSONu
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 4. Pokud přímý parsing selhal, zkusíme bezpečné opravy syntaxe
        repaired = cleaned
        # Odstranění zdvojených dvojteček (::)
        repaired = re.sub(r':\s*:\s*', ': ', repaired)
        # Sjednocení názvů klíčů (např. key_driverss -> key_drivers)
        repaired = re.sub(r'"key_drivers?s?"', '"key_drivers"', repaired)
        repaired = re.sub(r'"risks?s?"', '"risks"', repaired)
        # Odstranění zdvojených úvodních uvozovek (" ") typických pro ONNX tokenizéry
        repaired = re.sub(r'"\s+"', '"', repaired)
        # Oprava specifických chyb tokenizeru ONNX (např. \key": namísto "key":)
        repaired = re.sub(r'\\([a-zA-Z0-9_]+)":', r'"\1":', repaired)
        # Oprava chybějící uvozovky na konci řádku před čárkou nebo závorkou
        repaired = re.sub(r'([^"\s])\s*\n\s*([,\]\}])', r'\1"\n\2', repaired)
        # Odstranění opakovaných čárek
        repaired = re.sub(r",(\s*,)+", ",", repaired)
        # Odstranění čárky před uzavírací složenou závorkou
        repaired = re.sub(r",\s*\}", "}", repaired)
        # Odstranění čárky před uzavírací hranatou závorkou
        repaired = re.sub(r",\s*\]", "]", repaired)
        # Odstranění úvodních čárek za { nebo [
        repaired = re.sub(r"([{\[])\s*,+", r"\1", repaired)
        # Oprava lichého počtu uvozovek (např. neuzavřený řetězec před koncem JSONu)
        if repaired.count('"') % 2 != 0:
            repaired = re.sub(r'\}\s*$', '"}', repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            logger.warning(f"Standardní JSON parsing selhal ({exc}). Zkouším nouzovou regex rekonstrukci.")

        # 5. Nouzová regex extrakce klíčových polí (pokud model porušil uvozovky či syntax)
        try:
            extracted: Dict[str, Any] = {}

            # ticker
            m_ticker = re.search(r'"ticker"\s*:\s*["\s]*([A-Za-z0-9\.\-_]+)', raw_text)
            if m_ticker:
                extracted["ticker"] = m_ticker.group(1).strip()
            elif fallback_ticker:
                extracted["ticker"] = fallback_ticker

            # sentiment_score
            m_score = re.search(r'"sentiment_score"\s*:\s*["\s]*([+-]?\d+(?:\.\d+)?)', raw_text)
            if m_score:
                extracted["sentiment_score"] = float(m_score.group(1))
            else:
                extracted["sentiment_score"] = 0.0

            # recommendation
            m_rec = re.search(r'"recommendation"\s*:\s*["\s]*(BUY|HOLD|AVOID)', raw_text, re.IGNORECASE)
            if m_rec:
                extracted["recommendation"] = m_rec.group(1).upper()
            else:
                extracted["recommendation"] = "HOLD"

            # confidence
            m_conf = re.search(r'"confidence"\s*:\s*([+-]?\d+(?:\.\d+)?)', raw_text)
            if m_conf:
                extracted["confidence"] = float(m_conf.group(1))
            else:
                extracted["confidence"] = 0.60

            # reasoning
            m_reas = re.search(r'"reasoning"\s*:\s*(?:\[\s*"([^"]+)"|"([^"]+)")', raw_text)
            if m_reas:
                extracted["reasoning"] = (m_reas.group(1) or m_reas.group(2) or "").strip()
            else:
                m_reas_loose = re.search(r'"reasoning"\s*:\s*"?([^"\n\}]+)"?', raw_text)
                if m_reas_loose:
                    extracted["reasoning"] = m_reas_loose.group(1).strip().lstrip("[").strip('"')
                else:
                    extracted["reasoning"] = "Analýza rekonstruována z výstupu LLM."

            # key_drivers
            m_drivers = re.search(r'"(?:key_drivers?|drivers?|catalysts?)"\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
            if m_drivers:
                extracted["key_drivers"] = [
                    re.sub(r'^["\'\s]+|["\'\s]+$', '', item)
                    for item in m_drivers.group(1).split(",")
                    if re.sub(r'^["\'\s]+|["\'\s]+$', '', item)
                ]
            else:
                extracted["key_drivers"] = []

            # risks
            m_risks = re.search(r'"(?:risks?|risk_factors?)"\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
            if m_risks:
                extracted["risks"] = [
                    re.sub(r'^["\'\s]+|["\'\s]+$', '', item)
                    for item in m_risks.group(1).split(",")
                    if re.sub(r'^["\'\s]+|["\'\s]+$', '', item)
                ]
            else:
                extracted["risks"] = []

            if "ticker" in extracted and "recommendation" in extracted:
                logger.info(f"Nouzová regex rekonstrukce úspěšná pro {extracted.get('ticker')}.")
                return extracted

        except Exception as fallback_exc:
            logger.error(f"Nouzová rekonstrukce selhala: {fallback_exc}")

        return None

    def analyze_stock(
        self,
        stock_data: StockAnalysisData,
        retries: int = 2
    ) -> Optional[LLMStockVerdict]:
        """
        Zanalyzuje akcii přes Lemonade Server a vrátí validovaný LLMStockVerdict.
        V případě dočasné chyby provede opakovaný pokus.
        """
        prompt = self._build_prompt(stock_data)

        for attempt in range(1, retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Jsi senior kvantitativní analytik se zaměřením na výběr akcií a fundamentální zprávy. "
                                "Vždy odpovídáš VÝHRADNĚ platným JSON objektem bez vysvětlujícího textu."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=650,
                    timeout=120.0,
                )

                raw_content = response.choices[0].message.content or ""
                data = self._sanitize_and_extract_json(raw_content, fallback_ticker=stock_data.ticker)

                if not data:
                    logger.warning(
                        f"{stock_data.ticker} (pokus {attempt}/{retries}): "
                        f"Nepodařilo se extrahovat JSON z výstupu: {raw_content[:150]}"
                    )
                    continue

                # Zajištění přítomnosti tickeru
                if "ticker" not in data or not data["ticker"]:
                    data["ticker"] = stock_data.ticker

                verdict = LLMStockVerdict(**data)
                return verdict

            except Exception as exc:
                logger.warning(f"Chyba při komunikaci s Lemonade pro {stock_data.ticker} (pokus {attempt}): {exc}")

        # Bezpečný fallback, aby se nezastavila celá pipeline bota
        logger.error(f"Všechny pokusy o analýzu {stock_data.ticker} selhaly. Vracím neutrální HOLD fallback.")
        return LLMStockVerdict(
            ticker=stock_data.ticker,
            sentiment_score=0.0,
            recommendation="HOLD",
            confidence=0.30,
            key_drivers=["Technická a sentimentová data nebyla plně vyhodnocena."],
            risks=["Nedostupnost nebo nevalidní odpověď analytického LLM modulu."],
            reasoning=f"Automatický fallback pro {stock_data.ticker} z důvodu nedostupnosti LLM výstupu.",
        )

    def fast_daytrade_check(
        self,
        ticker: str,
        current_price: float,
        technical_score: float,
        headlines: List[str],
        timeout_seconds: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Bleskový AI sanity check pro Day Trader těsně před nákupem:
        - Zkontroluje čerstvé zprávy na přítomnost bezprostřední katastrofy, žaloby, účetního skandálu nebo krachu.
        - Zajišťuje, aby technické momentum nebylo 'pastí' před pádem.
        - Vygeneruje rychlé rozhodnutí: approved (True/False), sentiment (-1.0 až +1.0), důvod.
        - Využívá krátký timeout (default 4 s), aby nikdy nebrzdil intradenní cyklus.
        """
        if not headlines:
            return {
                "approved": True,
                "sentiment": 0.15,
                "reason": "Žádné bezprostřední negativní zprávy.",
                "llm_used": False,
            }

        news_summary = "\n".join([f"- {h}" for h in headlines[:3]])
        prompt = (
            f"Jsi bleskový AI risk manažer pro intradenní nákup akcie {ticker} (cena: {current_price:.2f}, technické skóre: {technical_score:.1f}/100).\n"
            f"Čerstvé zprávy:\n{news_summary}\n\n"
            f"Otázka: Hrozí podle těchto zpráv bezprostřední krach, vyšetřování SEC, účetní podvod nebo masivní propad?\n"
            f"Odpověz VÝHRADNĚ validním JSON objektem v tomto přesném tvaru:\n"
            f'{{"approved": true, "sentiment": 0.4, "reason": "Stručné shrnutí v češtině (1 věta)"}}\n'
            f'Pokud zprávy naznačují vážné riziko krachu nebo žaloby, nastav approved na false a sentiment na zápornou hodnotu.'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Jsi kvantitativní AI risk analytik. Odpovídáš VÝHRADNĚ jedním validním JSON objektem.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=120,
                timeout=timeout_seconds,
            )

            raw = response.choices[0].message.content or "{}"
            # Extrakce JSONu
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                approved = bool(data.get("approved", True))
                sentiment = float(data.get("sentiment", 0.0))
                reason = str(data.get("reason", "AI schválila nákup."))
                return {
                    "approved": approved,
                    "sentiment": max(-1.0, min(1.0, sentiment)),
                    "reason": reason,
                    "llm_used": True,
                }
        except Exception as exc:
            logger.debug(f"Blesková AI kontrola {ticker} nebyla dostupná přes LLM ({exc}). Pokračuji s kvantitativním schválením.")

        return {
            "approved": True,
            "sentiment": 0.0,
            "reason": "Kvantitativní schválení (Lemonade LLM offline nebo časový limit).",
            "llm_used": False,
        }