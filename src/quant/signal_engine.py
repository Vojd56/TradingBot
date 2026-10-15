"""Modul 3: Carver Orthogonal Signal Engine.

Implementuje ortogonální kombinaci střednědobého Trend-Followingu / Momenta
a krátkodobého návratu k průměru (Mean Reversion) dle Sekce 1 a Sekce 8 v research.md:
1. Momentum složka: volatilitně normalizovaná relativní síla, průřezové pořadí, trendový filtr
2. Mean-Reversion složka: Z-skóre od SMA 20 s ochranou proti propadajícím se nožům (falling knife filter)
3. Ortogonální Carverovo sloučení s dynamickými vahami z RegimeDetectoru
4. Long-only omezení pro Trading 212 Invest účet (zamítnutí SHORT větví do NO TRADE)
"""

from enum import Enum
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from src.quant.features import StockFeatures
from src.quant.regime_detector import MarketRegime, MarketRegimeType

logger = logging.getLogger(__name__)


class SignalAction(str, Enum):
    """Přípustné akce signálového jádra."""
    BUY = "BUY"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"
    CLOSE = "CLOSE"


class QuantSignal(BaseModel):
    """Výstup signálového jádra pro jedno aktivum v čase t."""
    ticker: str
    action: SignalAction
    composite_score: float = Field(..., ge=-1.0, le=1.0)      # Finální Carverovo skóre (-1.0 až +1.0)
    momentum_score: float = Field(..., ge=-1.0, le=1.0)       # Normalizované skóre momenta (-1.0 až +1.0)
    mean_reversion_score: float = Field(..., ge=-1.0, le=1.0) # Normalizované skóre návratu k průměru (-1.0 až +1.0)
    features: StockFeatures
    regime: MarketRegimeType
    is_long_only_compliant: bool = True
    rationale: str = ""


class CarverSignalEngine:
    """Ortogonální Carverovo signálové jádro."""

    def __init__(
        self,
        min_buy_composite_threshold: float = 0.35,
        mr_z_oversold_threshold: float = -1.2,
        mr_z_extreme_meltdown: float = -3.5,
    ):
        self.min_buy_composite_threshold = min_buy_composite_threshold
        self.mr_z_oversold_threshold = mr_z_oversold_threshold
        self.mr_z_extreme_meltdown = mr_z_extreme_meltdown

    def evaluate_momentum(self, feat: StockFeatures) -> float:
        """
        Vypočte normalizované skóre momenta v intervalu [-1.0, 1.0].
        
        Kombinuje průřezové pořadí (cross-sectional rank), volatilitně normalizované 3M momentum
        a trendový filtr (SMA 200).
        """
        # 1. Průřezové momentum (rank 0.0 až 1.0 -> škálováno na -1.0 až +1.0)
        cs_score = 2.0 * (feat.cross_sectional_rank - 0.5)

        # 2. Časové volatilitně normalizované momentum (ořez na [-2.0, 2.0] a normalizace)
        ts_3m_clipped = max(-2.0, min(2.0, feat.vol_norm_momentum_3m))
        ts_score = ts_3m_clipped / 2.0

        # 3. Dlouhodobý trendový filtr (pokud je cena pod SMA200 nebo má klesající sklon, momentum penalizujeme)
        trend_penalty = 1.0
        if feat.price_to_sma200_ratio < 1.0:
            trend_penalty *= 0.5
        if feat.sma200_slope_20d_pct < 0.0:
            trend_penalty *= 0.5

        raw_mom = (0.6 * cs_score + 0.4 * ts_score) * trend_penalty
        return float(max(-1.0, min(1.0, raw_mom)))

    def evaluate_mean_reversion(self, feat: StockFeatures) -> float:
        """
        Vypočte normalizované skóre návratu k průměru v intervalu [-1.0, 1.0].
        
        Klíčový prvek ze Sekce 3.B v research.md:
        Nákup přeprodaných aktiv pouze tehdy, pokud nedošlo k fundamentálnímu kolapsu!
        """
        z = feat.mean_reversion_z_20

        # Ochrana proti padajícímu noži (Falling knife filter):
        # Pokud je Z-skóre katastrofálně propadlé (pod -3.5) a objem je extrémní (RVOL > 2.5),
        # jde s vysokou pravděpodobností o fundamentální implozi / earnings blow-up.
        if z < self.mr_z_extreme_meltdown and feat.rvol_20 > 2.5:
            logger.debug(f"{feat.ticker}: Detekován padající nůž (Z={z}, RVOL={feat.rvol_20}) -> MR nulováno.")
            return -0.5  # Záporné skóre / varování

        # Pokud je cena přeprodaná v kontrolovaném pásmu (např. -3.0 <= Z <= -1.2)
        if z <= self.mr_z_oversold_threshold:
            # Čím hlouběji v rozumném pásmu, tím silnější nákupní odrazový potenciál
            oversold_intensity = min(1.0, abs(z - self.mr_z_oversold_threshold) / 1.5)
            # Potvrzení kompresí volatility (pokud je stlačená volatilita, odraz je bezpečnější)
            squeeze_bonus = 0.2 if feat.is_volatility_squeezed else 0.0
            return float(min(1.0, 0.5 + 0.3 * oversold_intensity + squeeze_bonus))

        # Pokud je cena překoupená (Z >= 1.5), signál k výstupu nebo ochlazení
        elif z >= 1.5:
            overbought_intensity = min(1.0, (z - 1.5) / 1.5)
            return float(max(-1.0, -0.4 - 0.6 * overbought_intensity))

        # Neutrální zóna okolo SMA 20
        return 0.0

    def generate_signal(
        self,
        feat: StockFeatures,
        regime: MarketRegime,
    ) -> QuantSignal:
        """
        Vygeneruje finální ortogonální signál pro dané aktivum s ohledem na globální režim trhu.
        """
        # Pokud režim zakazuje obchodování (CRISIS_PANIC), okamžitý tvrdý NO_TRADE
        if not regime.allow_trading:
            return QuantSignal(
                ticker=feat.ticker,
                action=SignalAction.NO_TRADE,
                composite_score=0.0,
                momentum_score=0.0,
                mean_reversion_score=0.0,
                features=feat,
                regime=regime.regime,
                rationale=f"Globální tržní režim {regime.regime} blokuje veškeré obchody (CRISIS_PANIC / High Volatility).",
            )

        # 1. Výpočet ortogonálních komponent
        s_mom = self.evaluate_momentum(feat)
        s_mr = self.evaluate_mean_reversion(feat)

        # 2. Vážené sloučení podle tržního režimu (Carverův přístup)
        composite = regime.momentum_weight * s_mom + regime.mean_reversion_weight * s_mr
        composite = float(max(-1.0, min(1.0, composite)))

        # 3. Určení akce
        # Long-only specifikace pro Trading 212 Invest:
        # Pouze výrazně kladné kompozitní skóre zakládá nákupní signál
        if composite >= self.min_buy_composite_threshold:
            action = SignalAction.BUY
            rationale = (
                f"Kvalifikovaný BUY signál: Kompozitní skóre {composite:+.2f} "
                f"(Momentum: {s_mom:+.2f} [váha {regime.momentum_weight:.2f}], "
                f"Mean-Reversion: {s_mr:+.2f} [váha {regime.mean_reversion_weight:.2f}])."
            )
        elif composite <= -0.40:
            action = SignalAction.CLOSE
            rationale = f"Medvědí vyčerpání: Kompozitní skóre {composite:+.2f} indikuje slabost."
        elif composite > 0.0:
            action = SignalAction.HOLD
            rationale = f"Mírně pozitivní skóre ({composite:+.2f}), ale pod prahem {self.min_buy_composite_threshold}."
        else:
            action = SignalAction.NO_TRADE
            rationale = f"Neutrální/záporné skóre ({composite:+.2f}). Výchozí stav NO_TRADE."

        return QuantSignal(
            ticker=feat.ticker,
            action=action,
            composite_score=round(composite, 4),
            momentum_score=round(s_mom, 4),
            mean_reversion_score=round(s_mr, 4),
            features=feat,
            regime=regime.regime,
            is_long_only_compliant=True,
            rationale=rationale,
        )

    def scan_universe(
        self,
        features_map: Dict[str, StockFeatures],
        regime: MarketRegime,
    ) -> Dict[str, QuantSignal]:
        """Projde celé univerzum a vrátí signály pro všechny tituly."""
        results = {}
        for ticker, feat in features_map.items():
            results[ticker] = self.generate_signal(feat, regime)
        return results
