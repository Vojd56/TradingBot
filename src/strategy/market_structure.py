from dataclasses import dataclass, field
from datetime import datetime, time as dtime
import logging
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from src.strategy.intraday_data import IntradaySnapshot

logger = logging.getLogger(__name__)

# Páková ETF se specifickými vlastnostmi denního resetu a páky
LEVERAGED_TICKERS_MAP: Dict[str, float] = {
    "TQQQ": 3.0,
    "SOXL": 3.0,
    "UPRO": 3.0,
    "FNGU": 3.0,
    "TECL": 3.0,
    "DPST": 3.0,
    "LABU": 3.0,
    "NVDL": 2.0,
    "TSLL": 2.0,
    "CONL": 2.0,
    "USD": 2.0,
}


@dataclass
class MarketStructureSnapshot:
    """Výsledek analýzy tržní struktury, likvidity a páky."""
    ticker: str
    current_price: float
    vwap: float
    vwap_distance_pct: float
    is_overextended_from_vwap: bool
    is_liquidity_sweep_suspect: bool
    opening_range_risk: bool
    leverage_factor: float
    volatility_drag_est_pct: float
    score_adjustment: float
    reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class MarketStructureAnalyzer:
    """
    Kvantitativní analyzátor tržní struktury (Market Structure, Liquidity Sweeps a Leveraged Volatility).
    
    Vychází z reálných empirických tržních zákonitostí:
    1. Liquidity Sweeps (Turtle Soup): Falešné průrazy Highs/Lows vyvolané vymetáním stop-lossů.
    2. VWAP Mean-Reversion: Přepálená vzdálenost od VWAPu statisticky vede k návratu k průměru.
    3. Opening Range Imbalance (09:00-09:15 CET / 15:30-15:45 CET): Vysoké spready a ranní whip-saws.
    4. Pákový rozpad (Volatility Drag): Matematické opotřebení 2x/3x ETF v bočních fázích trhu.
    """

    def __init__(
        self,
        max_normal_vwap_dist_pct: float = 1.4,
        max_leveraged_vwap_dist_pct: float = 2.4,
    ):
        self.max_normal_vwap_dist_pct = max_normal_vwap_dist_pct
        self.max_leveraged_vwap_dist_pct = max_leveraged_vwap_dist_pct

    def is_opening_volatility_window(self, ticker: str) -> bool:
        """
        Zjišťuje, zda se akcie nachází v prvních 15 minutách po otevření své burzy.
        Během této doby institucionální objednávky způsobují falešné breakouty a zvýšený skluz (slippage).
        """
        now_cet = datetime.now(ZoneInfo("Europe/Prague"))
        cur_time = now_cet.time()
        t = ticker.upper()

        # Evropské burzy (09:00 - 09:15 CET)
        if t.endswith((".DE", ".PA", ".AS", ".MC", ".MI", ".L", ".SW")):
            if dtime(9, 0) <= cur_time < dtime(9, 15):
                return True

        # Americké burzy (15:30 - 15:45 CET)
        else:
            if dtime(15, 30) <= cur_time < dtime(15, 45):
                return True

        return False

    def get_leverage_factor(self, ticker: str) -> float:
        """Vrátí pákový koeficient instrumentu (1.0 = běžná akcie, 2.0 = 2x ETF, 3.0 = 3x ETF)."""
        clean = ticker.upper().split(".")[0]
        return LEVERAGED_TICKERS_MAP.get(clean, 1.0)

    def calculate_volatility_drag_pct(self, leverage: float, atr_5m: float, price: float) -> float:
        """
        Odhadne roční volatilitní rozpad (Volatility Drag) pákového ETF:
        Decay ≈ 0.5 * Leverage^2 * Variance
        """
        if leverage <= 1.0 or price <= 0:
            return 0.0
        # Normalizovaná intradenní volatilita přepočtená na roční bázi
        intraday_vol = (atr_5m / price)
        annual_vol = intraday_vol * (78 ** 0.5) * (252 ** 0.5)  # 78 5m svíček denně * 252 obchodních dní
        decay_annual = 0.5 * (leverage ** 2) * (annual_vol ** 2) * 100.0
        return round(min(100.0, decay_annual), 1)

    def analyze_snapshot(self, snap: IntradaySnapshot) -> MarketStructureSnapshot:
        """
        Provede hloubkový rozbor tržní struktury pro daný intradenní snímek.
        """
        reasons: List[str] = []
        score_adj = 0.0
        leverage = self.get_leverage_factor(snap.ticker)

        # 1. Výpočet vzdálenosti od VWAP
        vwap_dist_pct = 0.0
        if snap.vwap > 0:
            vwap_dist_pct = round(((snap.current_price - snap.vwap) / snap.vwap) * 100.0, 2)

        max_allowed_dist = self.max_leveraged_vwap_dist_pct if leverage > 1.0 else self.max_normal_vwap_dist_pct
        is_overextended = vwap_dist_pct > max_allowed_dist

        if is_overextended:
            score_adj -= 18.0
            reasons.append(
                f"Přepálená vzdálenost od VWAP ({vwap_dist_pct:+.2f} % > limit {max_allowed_dist:.1f} %) - vysoké riziko návratu k průměru"
            )
        elif 0.0 <= vwap_dist_pct <= 0.6:
            score_adj += 8.0
            reasons.append(f"Zdravý odraz od VWAP ({vwap_dist_pct:+.2f} %) s minimálním rizikem přestřelení")

        # 2. Detekce Liquidity Sweep / Turtle Soup (Falešného průrazu maxima)
        # Pokud cena lízla 30m maximum, ale 5m RSI je vysoce překoupené (> 72) a objem prudce vyletěl (> 3x),
        # je vysoká pravděpodobnost vymetení stopů a následného propadu.
        is_sweep = False
        if snap.is_breaking_high and snap.rsi_5m > 72.0 and snap.volume_surge_ratio >= 2.5:
            is_sweep = True
            score_adj -= 22.0
            reasons.append(
                f"Podezření na Liquidity Sweep / Fakeout (30m High + RSI {snap.rsi_5m:.1f} + Objem {snap.volume_surge_ratio:.1f}x) - nebezpečí nákupu na vrcholu"
            )

        # 3. Riziko otevíracího okna (Opening Range Volatility)
        opening_risk = self.is_opening_volatility_window(snap.ticker)
        if opening_risk:
            score_adj -= 12.0
            reasons.append("Ranní otevírací okno (prvních 15 minut) - zvýšený spread a volatilní falešné průrazy")

        # 4. Volatilitní rozpad pákových ETF
        drag_pct = self.calculate_volatility_drag_pct(leverage, snap.atr_5m, snap.current_price)
        if leverage > 1.0 and drag_pct > 25.0:
            reasons.append(f"Vysoký odhadovaný pákový rozpad ({drag_pct:.0f} % p.a. při páce {leverage:.0f}x) - striktně vyžadován rychlý skalp")

        return MarketStructureSnapshot(
            ticker=snap.ticker,
            current_price=snap.current_price,
            vwap=snap.vwap,
            vwap_distance_pct=vwap_dist_pct,
            is_overextended_from_vwap=is_overextended,
            is_liquidity_sweep_suspect=is_sweep,
            opening_range_risk=opening_risk,
            leverage_factor=leverage,
            volatility_drag_est_pct=drag_pct,
            score_adjustment=round(score_adj, 1),
            reasons=reasons,
        )
