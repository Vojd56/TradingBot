from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
import yfinance as yf

from src.llm.sentiment_engine import SentimentEngine
from config import settings

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    RISK_ON_BULL = "RISK_ON_BULL"       # Býčí nálada, nízký strach, růst indexů -> agresivní TP (+2.8 %)
    NEUTRAL_CHOPPY = "NEUTRAL_CHOPPY"   # Trh do strany, smíšené makro -> rychlé skalpy (TP +1.6 %, BE +0.7 %)
    RISK_OFF_PANIC = "RISK_OFF_PANIC"   # Vysoký strach (VIX > 25), výprodeje -> zákaz nových long nákupů


@dataclass
class RegimeSnapshot:
    regime: MarketRegime
    vix: float
    spy_change_pct: float
    take_profit_pct: float
    stop_loss_pct: float
    breakeven_pct: float
    max_active_positions: int
    allow_new_buys: bool
    reasoning: str
    llm_used: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class MarketRegimeDetector:
    """
    Autonomní detektor globálního tržního režimu (Macro Market Regime Classifier).
    
    1. Sleduje index volatility CBOE VIX (index strachu Wall Street) a denní trend S&P 500 (SPY).
    2. Agreguje čerstvé makroekonomické zprávy (FED, sazby, geopolitika).
    3. Předává data lokálnímu LLM modelu na AMD NPU pro sémantickou klasifikaci trhu.
    4. Dynamicky přizpůsobuje parametry Day Tradera (Take Profit, Breakeven, Stop Loss, počet slotů).
    """

    def __init__(
        self,
        sentiment_engine: Optional[SentimentEngine] = None,
        cache_ttl_seconds: int = 1800,  # 30 minut cache pro klidný chod
        max_positions: Optional[int] = None,
    ):
        self.sentiment_engine = sentiment_engine or SentimentEngine()
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_positions = max_positions or settings.max_positions
        self._cached_snapshot: Optional[RegimeSnapshot] = None
        self._last_evaluated_ts: float = 0.0

    def fetch_macro_indicators(self) -> Dict[str, Any]:
        """Stáhne aktuální data pro VIX a SPY."""
        res = {"vix": 16.0, "spy_change_pct": 0.0, "headlines": []}
        try:
            df = yf.download(["SPY", "^VIX"], period="2d", interval="1d", progress=False)["Close"]
            if len(df) >= 2:
                spy_today = float(df["SPY"].iloc[-1])
                spy_prev = float(df["SPY"].iloc[-2])
                res["spy_change_pct"] = round(((spy_today - spy_prev) / spy_prev) * 100.0, 2)
                res["vix"] = round(float(df["^VIX"].iloc[-1]), 2)
            elif len(df) == 1:
                res["vix"] = round(float(df["^VIX"].iloc[-1]), 2)
        except Exception as exc:
            logger.debug(f"Chyba při stahování makro indikátorů: {exc}")

        # Zprávy pro S&P 500
        try:
            news = yf.Ticker("SPY").news or []
            for n in news[:3]:
                title = n.get("title") or n.get("content", {}).get("title", "")
                if title:
                    res["headlines"].append(title.strip())
        except Exception:
            pass

        return res

    def _rule_based_fallback(self, vix: float, spy_chg: float, reason_prefix: str = "") -> RegimeSnapshot:
        """Kvantitativní pravidlový fallback v případě nedostupnosti LLM."""
        if vix >= 26.0 or spy_chg <= -1.8:
            return RegimeSnapshot(
                regime=MarketRegime.RISK_OFF_PANIC,
                vix=vix,
                spy_change_pct=spy_chg,
                take_profit_pct=1.5,
                stop_loss_pct=1.0,
                breakeven_pct=0.7,
                max_active_positions=0,
                allow_new_buys=False,
                reasoning=f"{reason_prefix}Vysoká tržní volatilita (VIX: {vix}) a pokles SPY ({spy_chg:+.2f} %). Nové nákupy pozastaveny.",
                llm_used=False,
            )
        elif vix < 18.0 and spy_chg >= 0.1:
            return RegimeSnapshot(
                regime=MarketRegime.RISK_ON_BULL,
                vix=vix,
                spy_change_pct=spy_chg,
                take_profit_pct=2.8,
                stop_loss_pct=1.2,
                breakeven_pct=1.0,
                max_active_positions=self.max_positions,
                allow_new_buys=True,
                reasoning=f"{reason_prefix}Nízký VIX ({vix}) a růst indexu SPY ({spy_chg:+.2f} %). Optimální býčí podmínky pro trend.",
                llm_used=False,
            )
        else:
            return RegimeSnapshot(
                regime=MarketRegime.NEUTRAL_CHOPPY,
                vix=vix,
                spy_change_pct=spy_chg,
                take_profit_pct=1.6,
                stop_loss_pct=1.0,
                breakeven_pct=0.7,
                max_active_positions=self.max_positions,
                allow_new_buys=True,
                reasoning=f"{reason_prefix}Mírná volatilita (VIX: {vix}, SPY: {spy_chg:+.2f} %). Rychlé skalpy a včasný Breakeven.",
                llm_used=False,
            )

    def evaluate_regime(self, force: bool = False) -> RegimeSnapshot:
        """Vyhodnotí makro režim pomocí lokálního LLM (s rychlým pravidlovým fallbackem)."""
        now_ts = time.time()
        if not force and self._cached_snapshot and (now_ts - self._last_evaluated_ts) < self.cache_ttl_seconds:
            return self._cached_snapshot

        macro_data = self.fetch_macro_indicators()
        vix = macro_data["vix"]
        spy_chg = macro_data["spy_change_pct"]
        headlines = macro_data["headlines"]

        # Prompt pro LLM na NPU
        news_text = "\n".join([f"- {h}" for h in headlines[:3]]) if headlines else "Běžné tržní zprávy."
        prompt = (
            f"Jsi hlavní makroanalytik kvantitativního fondu. Zde jsou aktuální tržní data:\n"
            f"- CBOE Volatility Index (VIX - index strachu): {vix:.2f} (norma 12-20, panika > 25)\n"
            f"- Index S&P 500 (SPY) denní změna: {spy_chg:+.2f} %\n"
            f"- Klíčové makro zprávy:\n{news_text}\n\n"
            f"Úkol: Zařaď trh do JEDNOHO z těchto 3 režimů:\n"
            f"1. RISK_ON_BULL: Nízký strach, stabilní růst, příznivý sentiment pro nákupy.\n"
            f"2. NEUTRAL_CHOPPY: Smíšené zprávy, trh bez jasného směru, mírná volatilita.\n"
            f"3. RISK_OFF_PANIC: Vysoký VIX, prudký výprodej, panika z inflace nebo války.\n\n"
            f"Odpověz VÝHRADNĚ platným JSON objektem:\n"
            f'{{"regime": "RISK_ON_BULL", "reasoning": "Stručné odůvodnění v češtině (1 věta)"}}'
        )

        try:
            response = self.sentiment_engine.client.chat.completions.create(
                model=self.sentiment_engine.model,
                messages=[
                    {"role": "system", "content": "Jsi kvantitativní makroanalytik. Odpovídáš VÝHRADNĚ jedním JSON objektem."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=140,
                timeout=5.0,
            )
            raw = response.choices[0].message.content or "{}"
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                reg_str = str(data.get("regime", "NEUTRAL_CHOPPY")).upper()
                reasoning = str(data.get("reasoning", "Vyhodnoceno lokálním modelem na NPU."))

                if "BULL" in reg_str or "RISK_ON" in reg_str:
                    regime = MarketRegime.RISK_ON_BULL
                    tp, sl, be, slots, buys = 2.8, 1.2, 1.0, self.max_positions, True
                elif "PANIC" in reg_str or "RISK_OFF" in reg_str or vix >= 26.0:
                    regime = MarketRegime.RISK_OFF_PANIC
                    tp, sl, be, slots, buys = 1.5, 1.0, 0.7, 0, False
                else:
                    regime = MarketRegime.NEUTRAL_CHOPPY
                    tp, sl, be, slots, buys = 1.6, 1.0, 0.7, self.max_positions, True

                snapshot = RegimeSnapshot(
                    regime=regime,
                    vix=vix,
                    spy_change_pct=spy_chg,
                    take_profit_pct=tp,
                    stop_loss_pct=sl,
                    breakeven_pct=be,
                    max_active_positions=slots,
                    allow_new_buys=buys,
                    reasoning=reasoning,
                    llm_used=True,
                )
                self._cached_snapshot = snapshot
                self._last_evaluated_ts = now_ts
                logger.info(f"🧠 [AI MAKRO REŽIM] {snapshot.regime.value} | VIX: {vix} | SPY: {spy_chg:+.2f}% | {reasoning}")
                return snapshot

        except Exception as exc:
            logger.debug(f"LLM makro analýza nebyla dostupná ({exc}). Používám pravidlový model.")

        snapshot = self._rule_based_fallback(vix, spy_chg, reason_prefix="Kvantitativní pravidlo: ")
        self._cached_snapshot = snapshot
        self._last_evaluated_ts = now_ts
        return snapshot
