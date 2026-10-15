from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field

from src.llm.sentiment_engine import SentimentEngine
from src.strategy.intraday_data import IntradaySnapshot

logger = logging.getLogger(__name__)

DEFAULT_TRADE_MEMORY_FILE = Path("data") / "trade_memory.json"


class TradeMistakeCategory(str, Enum):
    OPENING_VOLATILITY_CHOP = "OPENING_VOLATILITY_CHOP"     # Ranní otevírací whip-saw v prvních 15 minutách
    OVERBOUGHT_RSI = "OVERBOUGHT_RSI"                       # Vstup při překoupeném RSI (> 68-70)
    VWAP_OVEREXTENSION = "VWAP_OVEREXTENSION"               # Nákup příliš daleko nad VWAP (přestřelená cena)
    LIQUIDITY_SWEEP_FAKEOUT = "LIQUIDITY_SWEEP_FAKEOUT"     # Falešný průraz High (Turtle Soup), kde retail vymetou instituce
    COUNTER_TREND_FIGHT = "COUNTER_TREND_FIGHT"             # Obchod proti trendu vyššího rámce nebo padajícímu indexu
    SLIPPAGE_SPREAD = "SLIPPAGE_SPREAD"                     # Ztráta způsobená širokým spreadem či nízkou likviditou
    PROFITABLE_EXECUTION = "PROFITABLE_EXECUTION"           # Vzorový ziskový obchod
    NORMAL_STOP_OUT = "NORMAL_STOP_OUT"                     # Běžný statistický stop-out v mezích normálního rozptylu


class TradeReflectionRecord(BaseModel):
    """Záznam o provedené post-mortem analýze obchodu a vytvořeném ponaučení."""
    trade_id: str
    ticker: str
    action: str = "BUY"
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_eur: float
    exit_reason: str
    entry_time: str
    exit_time: str
    duration_minutes: float
    entry_rsi: float = 50.0
    entry_vwap_dist_pct: float = 0.0
    entry_volume_surge: float = 1.0
    market_regime: str = "NEUTRAL"
    mistake_category: TradeMistakeCategory
    lesson_learned: str
    prevention_rule: Dict[str, Any] = Field(default_factory=dict)
    llm_used: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class TradeReflectionEngine:
    """
    Samoučící se modul reflexe chyb a tržní paměti (Trade Reflection & Self-Critique Engine).
    
    Inspirováno architekturou FinMem (Layered Memory & Self-Reflection) a metodikou Marka Minerviniho:
    1. Po uzavření obchodu (zejména ztráty na Stop Lossu) provede detailní Post-Mortem.
    2. Využívá lokální LLM model na AMD NPU pro nalezení pravé příčiny neúspěchu.
    3. Kodifikuje ponaučení do perzistentní paměti 'data/trade_memory.json'.
    4. Před každým novým obchodem prověří, zda kandidát neopakuje dřívější chybu, a případně udělí VETO.
    """

    def __init__(
        self,
        sentiment_engine: Optional[SentimentEngine] = None,
        memory_file: Optional[Path] = None,
    ):
        self.sentiment_engine = sentiment_engine or SentimentEngine()
        self.memory_file = memory_file or DEFAULT_TRADE_MEMORY_FILE
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.reflections: List[TradeReflectionRecord] = []
        self._load_memory()

    def _load_memory(self) -> None:
        if not self.memory_file.exists():
            self._save_memory()
            return
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_list = data.get("reflections", [])
            self.reflections = [TradeReflectionRecord(**r) for r in raw_list]
            logger.info(f"Načteno {len(self.reflections)} záznamů z paměti chyb ({self.memory_file}).")
        except Exception as exc:
            logger.error(f"Chyba při načítání paměti chyb: {exc}")
            self.reflections = []

    def _save_memory(self) -> None:
        try:
            total_trades = len(self.reflections)
            wins = sum(1 for r in self.reflections if r.pnl_pct > 0)
            losses = sum(1 for r in self.reflections if r.pnl_pct <= 0)
            win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
            total_pnl = sum(r.pnl_eur for r in self.reflections)

            # Sestavení aktivních prevencí pro rychlé vyhledávání
            active_rules = []
            for r in self.reflections:
                if r.pnl_pct < 0 and r.prevention_rule:
                    active_rules.append({
                        "ticker": r.ticker,
                        "category": r.mistake_category.value,
                        "rule": r.prevention_rule,
                        "lesson": r.lesson_learned,
                        "date": r.created_at[:10],
                    })

            payload = {
                "last_updated": datetime.now().isoformat(),
                "summary": {
                    "total_analyzed_trades": total_trades,
                    "winning_trades": wins,
                    "losing_trades": losses,
                    "win_rate_pct": round(win_rate, 1),
                    "total_realized_pnl_eur": round(total_pnl, 2),
                },
                "active_heuristics": active_rules[-20:],  # Posledních 20 aktivních pravidel
                "reflections": [r.model_dump() for r in self.reflections],
            }
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"Chyba při ukládání paměti chyb do {self.memory_file}: {exc}")

    def _rule_based_fallback_reflection(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        pnl_eur: float,
        exit_reason: str,
        entry_time: str,
        exit_time: str,
        duration_minutes: float,
        entry_rsi: float,
        entry_vwap_dist_pct: float,
        entry_volume_surge: float,
        regime: str,
    ) -> TradeReflectionRecord:
        """Kvantitativní analytické pravidlo v případě nedostupnosti lokálního LLM."""
        # 1. Ziskový obchod
        if pnl_pct > 0:
            category = TradeMistakeCategory.PROFITABLE_EXECUTION
            lesson = f"Ziskový obchod (+{pnl_pct:.2f} %). Strategie přesně zachytila momentum a výstup na Take Profit proběhl ukázkově."
            rule = {"min_score": 62.0}

        # 2. Ztráta v otevíracím okně (09:00-09:15 CET nebo 15:30-15:45 CET)
        elif self._is_time_in_opening_window(entry_time):
            category = TradeMistakeCategory.OPENING_VOLATILITY_CHOP
            lesson = f"Ztráta v otevíracím okně ({entry_time[-8:-3]} CET). Ranní institucionální whip-saw způsobil falešný signál. Vyžadovat přísnější filtr."
            rule = {"avoid_opening_minutes": 15, "min_score_in_open": 76.0}

        # 3. Ztráta při překoupeném RSI (> 68)
        elif entry_rsi >= 68.0:
            category = TradeMistakeCategory.OVERBOUGHT_RSI
            lesson = f"Nákup na vyčerpaném momentu (RSI {entry_rsi:.1f} >= 68.0). Následovala okamžitá korekce. Zpřísnit limit pro maximální RSI."
            rule = {"max_entry_rsi": 66.0}

        # 4. Ztráta při přepálené vzdálenosti od VWAP (> 1.4 %)
        elif entry_vwap_dist_pct >= 1.4:
            category = TradeMistakeCategory.VWAP_OVEREXTENSION
            lesson = f"Vstup proběhl příliš daleko od VWAP ({entry_vwap_dist_pct:+.2f} %). Trh předvedl mean-reversion propad zpět k férové ceně."
            rule = {"max_vwap_distance_pct": 1.0}

        # 5. Rychlý stop-out na lokálním maximu (do 12 minut)
        elif duration_minutes <= 12.0 and exit_reason == "STOP_LOSS":
            category = TradeMistakeCategory.LIQUIDITY_SWEEP_FAKEOUT
            lesson = f"Bleskový Stop Loss po {duration_minutes:.0f} minutách. Pravděpodobný Liquidity Sweep (vymetení stopů na High a obrat dolů)."
            rule = {"require_vwap_bounce": True, "min_volume_surge": 1.5}

        # 6. Běžný stop-out
        else:
            category = TradeMistakeCategory.NORMAL_STOP_OUT
            lesson = f"Běžný statistický Stop Loss ({pnl_pct:.2f} %) v mezích řízení rizik. Ztráta byla včas omezena."
            rule = {"maintain_stop_loss": True}

        trade_id = f"{ticker}_{int(time.time())}"
        return TradeReflectionRecord(
            trade_id=trade_id,
            ticker=ticker,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=round(pnl_pct, 2),
            pnl_eur=round(pnl_eur, 2),
            exit_reason=exit_reason,
            entry_time=entry_time,
            exit_time=exit_time,
            duration_minutes=round(duration_minutes, 1),
            entry_rsi=entry_rsi,
            entry_vwap_dist_pct=entry_vwap_dist_pct,
            entry_volume_surge=entry_volume_surge,
            market_regime=regime,
            mistake_category=category,
            lesson_learned=lesson,
            prevention_rule=rule,
            llm_used=False,
        )

    @staticmethod
    def _is_time_in_opening_window(iso_time_str: str) -> bool:
        """Ověří, zda čas spadá do prvních 15 minut otevření trhu."""
        try:
            dt = datetime.fromisoformat(iso_time_str)
            t = dt.time()
            # 09:00 - 09:15 nebo 15:30 - 15:45
            return (9, 0) <= (t.hour, t.minute) <= (9, 15) or (15, 30) <= (t.hour, t.minute) <= (15, 45)
        except Exception:
            return False

    def record_and_reflect(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        allocated_eur: float,
        exit_reason: str,
        entry_time: Optional[str] = None,
        exit_time: Optional[str] = None,
        entry_snapshot: Optional[IntradaySnapshot] = None,
        regime_name: str = "NEUTRAL",
    ) -> TradeReflectionRecord:
        """
        Provede hloubkový Post-Mortem rozbor uzavřené pozice a uloží ponaučení do paměti.
        """
        now = datetime.now()
        ent_str = entry_time or now.isoformat()
        ex_str = exit_time or now.isoformat()

        # Výpočet doby držení
        try:
            d_ent = datetime.fromisoformat(ent_str)
            d_ex = datetime.fromisoformat(ex_str)
            duration_min = max(1.0, (d_ex - d_ent).total_seconds() / 60.0)
        except Exception:
            duration_min = 10.0

        pnl_eur = allocated_eur * (pnl_pct / 100.0)
        rsi = entry_snapshot.rsi_5m if entry_snapshot else 50.0
        vwap_dist = 0.0
        if entry_snapshot and entry_snapshot.vwap > 0:
            vwap_dist = round(((entry_snapshot.current_price - entry_snapshot.vwap) / entry_snapshot.vwap) * 100.0, 2)
        vol_surge = entry_snapshot.volume_surge_ratio if entry_snapshot else 1.0

        # Pokud obchod skončil v zisku nebo na breakevenu, reflexe je pozitivní
        if pnl_pct >= 0.0:
            rec = self._rule_based_fallback_reflection(
                ticker, entry_price, exit_price, pnl_pct, pnl_eur, exit_reason,
                ent_str, ex_str, duration_min, rsi, vwap_dist, vol_surge, regime_name
            )
            self.reflections.append(rec)
            self._save_memory()
            return rec

        # Pro ztrátové obchody spouštíme AI Post-Mortem na AMD NPU
        prompt = (
            f"Jsi hlavní risk auditor kvantitativního fondu. Zde jsou reálná data uzavřeného ztrátového obchodu:\n"
            f"- Ticker: {ticker}\n"
            f"- Nákupní cena: {entry_price:.2f} | Výstupní cena: {exit_price:.2f} ({exit_reason})\n"
            f"- Výsledek: {pnl_pct:+.2f} % (Ztráta: {pnl_eur:.2f} EUR) | Doba držení: {duration_min:.0f} minut\n"
            f"- Stav při vstupu: 5m RSI = {rsi:.1f}, Vzdálenost od VWAP = {vwap_dist:+.2f} %, Objem = {vol_surge:.1f}x\n"
            f"- Tržní makro režim: {regime_name}\n\n"
            f"Úkol: Vyhodnoť, proč obchod selhal, a definuj konkrétní pravidlo, aby bot stejnou chybu neopakoval.\n"
            f"Kategorie na výběr: OPENING_VOLATILITY_CHOP, OVERBOUGHT_RSI, VWAP_OVEREXTENSION, LIQUIDITY_SWEEP_FAKEOUT, COUNTER_TREND_FIGHT, NORMAL_STOP_OUT.\n\n"
            f"Odpověz VÝHRADNĚ platným JSON objektem v tomto formátu:\n"
            f'{{\n'
            f'  "mistake_category": "OVERBOUGHT_RSI",\n'
            f'  "lesson_learned": "Věcné odůvodnění v češtině (max 2 věty).",\n'
            f'  "max_allowed_rsi": 66.0,\n'
            f'  "max_allowed_vwap_distance_pct": 1.0,\n'
            f'  "min_required_score": 75.0\n'
            f'}}'
        )

        try:
            response = self.sentiment_engine.client.chat.completions.create(
                model=self.sentiment_engine.model,
                messages=[
                    {"role": "system", "content": "Jsi nekompromisní kvantitativní risk auditor. Odpovídáš VÝHRADNĚ jedním JSON objektem."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=180,
                timeout=5.0,
            )
            raw = response.choices[0].message.content or "{}"
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                cat_str = str(data.get("mistake_category", "NORMAL_STOP_OUT")).upper()
                try:
                    category = TradeMistakeCategory(cat_str)
                except ValueError:
                    category = TradeMistakeCategory.NORMAL_STOP_OUT

                lesson = str(data.get("lesson_learned", "Ztráta vyhodnocena modelem na NPU."))
                rule = {
                    "max_entry_rsi": float(data.get("max_allowed_rsi", 66.0)),
                    "max_vwap_distance_pct": float(data.get("max_allowed_vwap_distance_pct", 1.0)),
                    "min_required_score": float(data.get("min_required_score", 70.0)),
                }

                rec = TradeReflectionRecord(
                    trade_id=f"{ticker}_{int(time.time())}",
                    ticker=ticker,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=round(pnl_pct, 2),
                    pnl_eur=round(pnl_eur, 2),
                    exit_reason=exit_reason,
                    entry_time=ent_str,
                    exit_time=ex_str,
                    duration_minutes=round(duration_min, 1),
                    entry_rsi=rsi,
                    entry_vwap_dist_pct=vwap_dist,
                    entry_volume_surge=vol_surge,
                    market_regime=regime_name,
                    mistake_category=category,
                    lesson_learned=lesson,
                    prevention_rule=rule,
                    llm_used=True,
                )
                self.reflections.append(rec)
                self._save_memory()
                logger.info(f"🧠 [AI SEBEREFLEXE ULOŽENA] {ticker} ({category.value}): {lesson}")
                return rec
        except Exception as exc:
            logger.debug(f"Chyba při volání LLM reflexe ({exc}), použiji pravidlový model.")

        rec = self._rule_based_fallback_reflection(
            ticker, entry_price, exit_price, pnl_pct, pnl_eur, exit_reason,
            ent_str, ex_str, duration_min, rsi, vwap_dist, vol_surge, regime_name
        )
        self.reflections.append(rec)
        self._save_memory()
        return rec

    def check_candidate_against_memory(
        self,
        ticker: str,
        current_price: float,
        rsi_5m: float,
        vwap: float,
        score: float,
    ) -> Dict[str, Any]:
        """
        Prověří nového nákupního kandidáta proti uložené paměti chyb.
        Pokud kandidát splňuje stejný vzorec, který vedl k nedávné ztrátě, udělí VETO.
        """
        clean_t = ticker.strip().upper()
        now_cet = datetime.now(ZoneInfo("Europe/Prague"))
        cur_time = now_cet.time()

        vwap_dist_pct = 0.0
        if vwap > 0:
            vwap_dist_pct = round(((current_price - vwap) / vwap) * 100.0, 2)

        # Hledáme nedávná ponaučení pro tento konkrétní ticker (posledních 5)
        ticker_losses = [r for r in self.reflections if r.ticker == clean_t and r.pnl_pct < 0][-5:]

        for r in ticker_losses:
            # 1. Kontrola ranního whip-saw okna
            if r.mistake_category == TradeMistakeCategory.OPENING_VOLATILITY_CHOP:
                if (9, 0) <= (cur_time.hour, cur_time.minute) <= (9, 20) or (15, 30) <= (cur_time.hour, cur_time.minute) <= (15, 45):
                    return {
                        "approved": False,
                        "category": r.mistake_category.value,
                        "reason": f"Paměť chyb: {clean_t} zde nedávno utrpěl ztrátu v otevíracím okně ({r.entry_time[:10]}). Vstup v ranním čase zablokován.",
                    }

            # 2. Kontrola překoupeného RSI
            if r.mistake_category == TradeMistakeCategory.OVERBOUGHT_RSI:
                max_rsi = r.prevention_rule.get("max_entry_rsi", 66.0)
                if rsi_5m > max_rsi:
                    return {
                        "approved": False,
                        "category": r.mistake_category.value,
                        "reason": f"Paměť chyb: {clean_t} selhal při RSI {rsi_5m:.1f} > povolených {max_rsi:.1f}. Čekám na zklidnění indikátoru.",
                    }

            # 3. Kontrola přestřelení od VWAP
            if r.mistake_category == TradeMistakeCategory.VWAP_OVEREXTENSION:
                max_dist = r.prevention_rule.get("max_vwap_distance_pct", 1.0)
                if vwap_dist_pct > max_dist:
                    return {
                        "approved": False,
                        "category": r.mistake_category.value,
                        "reason": f"Paměť chyb: {clean_t} je příliš daleko od VWAP ({vwap_dist_pct:+.2f} % > {max_dist:.1f} %). Riziko mean-reversion propadu.",
                    }

            # 4. Kontrola zvýšeného bodového prahu
            min_score = r.prevention_rule.get("min_required_score", 70.0)
            if score < min_score:
                return {
                    "approved": False,
                    "category": r.mistake_category.value,
                    "reason": f"Paměť chyb: Pro {clean_t} je po předchozí ztrátě vyžadováno zvýšené skóre {min_score:.0f} b (aktuálně {score:.1f} b).",
                }

        # Pokud prošel všemi kontrolami, je schválen
        return {"approved": True, "category": "CLEAR", "reason": "Kandidát splňuje všechny paměťové bezpečnostní filtry."}

    def bootstrap_from_portfolio_history(self, portfolio_state_file: Path) -> int:
        """
        Jednorázově projde existující historii uzavřených obchodů v 'portfolio_state.json'
        a vygeneruje počáteční reflexi pro obchody, které ještě nejsou v paměti.
        """
        if not portfolio_state_file.exists():
            return 0

        try:
            with open(portfolio_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            existing_trade_ids = {r.trade_id for r in self.reflections}
            positions = data.get("positions", {})
            history = data.get("trade_history", [])

            processed = 0
            for ticker, p in positions.items():
                if p.get("status") == "CLOSED":
                    t_id = f"{ticker}_{p.get('entry_date', '')[:19]}"
                    if t_id in existing_trade_ids:
                        continue

                    entry_p = float(p.get("entry_price_usd", 100.0))
                    exit_p = float(p.get("close_price_usd") or entry_p)
                    pnl_usd = float(p.get("realized_pnl_usd") or 0.0)
                    pnl_pct = ((exit_p - entry_p) / entry_p * 100.0) if entry_p > 0 else 0.0
                    reason = p.get("close_reason", "MANUAL_EXTERNAL_EXIT")

                    rec = self._rule_based_fallback_reflection(
                        ticker=ticker,
                        entry_price=entry_p,
                        exit_price=exit_p,
                        pnl_pct=pnl_pct,
                        pnl_eur=pnl_usd,
                        exit_reason=reason,
                        entry_time=p.get("entry_date", datetime.now().isoformat()),
                        exit_time=p.get("close_date", datetime.now().isoformat()),
                        duration_minutes=60.0,
                        entry_rsi=55.0,
                        entry_vwap_dist_pct=0.5,
                        entry_volume_surge=1.2,
                        regime="NEUTRAL",
                    )
                    rec.trade_id = t_id
                    self.reflections.append(rec)
                    existing_trade_ids.add(t_id)
                    processed += 1

            if processed > 0:
                self._save_memory()
                logger.info(f"✨ [Trade Reflection Engine] Úspěšně zaindexováno {processed} historických obchodů.")
            return processed
        except Exception as exc:
            logger.debug(f"Chyba při bootstrapu z historie portfolia: {exc}")
            return 0
