"""Modul 4: Triple-Barrier Method & Net Expected Value (EV_net) Engine.

Implementuje probabilistické rozhodovací jádro dle Sekce 5 a 6 v research.md:
1. Dynamická trojitá bariéra (Triple-Barrier Method):
   - Horní bariéra (Take Profit): m_TP * ATR_20
   - Dolní bariéra (Stop Loss): m_SL * ATR_20 s garantovaným R:R >= 1:2
   - Časová bariéra: H barů (např. 20 obchodních dnů)
2. Kalibrace pravděpodobnosti výhry (p_win) přes ohraničenou logistickou funkci bez iluzí (max 65 %)
3. Výpočet čisté očekávané hodnoty (EV_net):
   EV_net = p_win * E[R|TP] - p_loss * E[|L||SL] - C_total
   kde C_total = spread + broker_fees + FX_conversion + slippage(ATR).
4. Rozhodovací brána s výchozím stavem NO TRADE:
   Nákupní příkaz BUY povolen POUZE při EV_net > eta * C_total (bezpečnostní polštář).
"""

import logging
import math
from typing import Optional
from pydantic import BaseModel, Field

from src.quant.features import StockFeatures
from src.quant.regime_detector import MarketRegime, MarketRegimeType
from src.quant.signal_engine import QuantSignal, SignalAction

logger = logging.getLogger(__name__)


class TripleBarriers(BaseModel):
    """Dynamické bariéry obchodu odvozené od realizované volatility aktiva."""
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    max_holding_bars: int = 20
    take_profit_pct: float
    stop_loss_pct: float
    rr_ratio: float = Field(..., ge=2.0)  # Striktní vymáhání R:R >= 1:2 dle AGENTS.md


class ExpectedValueResult(BaseModel):
    """Pravděpodobnostní kalkulace a čistá očekávaná hodnota obchodu."""
    ticker: str
    decision: SignalAction
    p_win: float = Field(..., ge=0.0, le=1.0)
    p_loss: float = Field(..., ge=0.0, le=1.0)
    expected_gain_pct: float
    expected_loss_pct: float
    total_friction_cost_pct: float
    ev_net_pct: float
    hurdle_threshold_pct: float
    passes_hurdle: bool
    barriers: TripleBarriers
    rationale: str


class PredictionModel:
    """Probabilistické rozhodovací jádro a kalkulátor čisté očekávané hodnoty."""

    def __init__(
        self,
        min_m_sl: float = 1.2,
        min_m_tp: float = 2.5,
        default_holding_bars: int = 20,
        cost_hurdle_multiplier: float = 1.5,
        min_ev_net_absolute_pct: float = 0.50,  # Minimálně +0.50 % čisté EV po nákladech
        fx_fee_pct: float = 0.15,               # Trading 212 FX poplatek 0.15 %
    ):
        self.min_m_sl = min_m_sl
        self.min_m_tp = max(min_m_tp, min_m_sl * 2.0)  # Vynucení R:R >= 2.0
        self.default_holding_bars = default_holding_bars
        self.cost_hurdle_multiplier = cost_hurdle_multiplier
        self.min_ev_net_absolute_pct = min_ev_net_absolute_pct
        self.fx_fee_pct = fx_fee_pct

    def generate_barriers(
        self,
        features: StockFeatures,
        regime: MarketRegime,
    ) -> TripleBarriers:
        """
        Vypočte dynamické bariéry trojité bariéry na základě ATR a režimu trhu.
        Garantuje R:R ratio >= 1:2.
        """
        curr_price = features.current_price
        atr = features.atr_20

        # Ve volatilnějším režimu lehce rozšíříme násobky pro zabránění vyklepání ze šumu
        vol_scale = 1.15 if regime.regime == MarketRegimeType.BULL_VOLATILE else 1.0

        m_sl = self.min_m_sl * vol_scale
        m_tp = max(self.min_m_tp * vol_scale, m_sl * 2.0)  # Vždy dodržen minimální poměr 1:2

        sl_distance = m_sl * atr
        tp_distance = m_tp * atr

        sl_price = round(curr_price - sl_distance, 4)
        tp_price = round(curr_price + tp_distance, 4)

        sl_pct = (sl_distance / curr_price) * 100.0
        tp_pct = (tp_distance / curr_price) * 100.0
        rr_ratio = tp_pct / max(1e-6, sl_pct)

        return TripleBarriers(
            entry_price=round(curr_price, 4),
            take_profit_price=tp_price,
            stop_loss_price=sl_price,
            max_holding_bars=self.default_holding_bars,
            take_profit_pct=round(tp_pct, 2),
            stop_loss_pct=round(sl_pct, 2),
            rr_ratio=round(rr_ratio, 2),
        )

    def calibrate_win_probability(
        self,
        signal: QuantSignal,
        barriers: TripleBarriers,
    ) -> float:
        """
        Realisticky kalibruje pravděpodobnost zasažení Take Profitu před Stop Lossem.
        
        Používá logistickou transformaci kompozitního skóre, ohraničenou na racionální
        interval [35 %, 63 %] bez přeučených a naivních předpokladů (dle Sekce 7 v research.md).
        """
        score = signal.composite_score  # v intervalu [-1.0, 1.0]

        # Logistická sigmoid funkce: p = 1 / (1 + exp(-k * (score - 0.35)))
        # Pro skóre 0.35 -> p = 0.50 (vyrovnaná mince)
        # Pro skóre 0.70+ -> p konverguje k hornímu limitu 0.62 (62% úspěšnost)
        k = 3.5
        sigmoid_val = 1.0 / (1.0 + math.exp(-k * (score - 0.35)))

        # Ohraničení na realistický interval
        p_calibrated = 0.38 + 0.24 * sigmoid_val

        # Korekce na nepříznivý poměr R:R (čím vyšší TP vůči SL, tím přirozeně nižší geometrická pravděpodobnost)
        rr_damping = math.sqrt(2.0 / max(2.0, barriers.rr_ratio))
        p_final = p_calibrated * rr_damping

        return round(float(max(0.30, min(0.65, p_final))), 4)

    def estimate_friction_costs(
        self,
        features: StockFeatures,
    ) -> float:
        """
        Odhadne realistické frikční náklady obchodu v procentech:
        Spread + poplatky brokera (0) + FX konverze (0.15 %) + skluz úměrný volatilitě.
        """
        # Odhad bid-ask spreadu v % (u likvidních akcií 0.05-0.10 %, u volatilních až 0.20 %)
        spread_pct = max(0.06, min(0.25, 0.04 * features.atr_20_pct))

        # Modelovaný skluz exekuce: 0.05 * ATR%
        slippage_pct = max(0.04, 0.05 * features.atr_20_pct)

        total_friction = spread_pct + self.fx_fee_pct + slippage_pct
        return round(float(total_friction), 3)

    def evaluate(
        self,
        signal: QuantSignal,
        regime: MarketRegime,
    ) -> ExpectedValueResult:
        """
        Kompletní probabilistické vyhodnocení.
        
        Vypočte EV_net a aplikuje rozhodovací bránu. Výchozí stav je vždy NO TRADE.
        """
        # Pokud signálové jádro nevrátilo BUY nebo režim nepovoluje trading -> rovnou NO TRADE
        if signal.action != SignalAction.BUY or not regime.allow_trading:
            dummy_barriers = self.generate_barriers(signal.features, regime)
            return ExpectedValueResult(
                ticker=signal.ticker,
                decision=SignalAction.NO_TRADE,
                p_win=0.50,
                p_loss=0.50,
                expected_gain_pct=dummy_barriers.take_profit_pct,
                expected_loss_pct=dummy_barriers.stop_loss_pct,
                total_friction_cost_pct=0.25,
                ev_net_pct=0.0,
                hurdle_threshold_pct=0.50,
                passes_hurdle=False,
                barriers=dummy_barriers,
                rationale=f"Vstup zamítnut: Signál {signal.action} nebo režim {regime.regime} neumožňuje nákup.",
            )

        barriers = self.generate_barriers(signal.features, regime)
        p_win = self.calibrate_win_probability(signal, barriers)
        p_loss = round(1.0 - p_win, 4)

        costs = self.estimate_friction_costs(signal.features)

        # Čistá očekávaná hodnota: EV_net = p_win * TP% - p_loss * SL% - Costs%
        ev_gain = p_win * barriers.take_profit_pct
        ev_loss = p_loss * barriers.stop_loss_pct
        ev_net = ev_gain - ev_loss - costs

        # Bezpečnostní práh (hurdle): EV musí překonat náklady s násobkem eta a dosáhnout min absolutní hodnoty
        hurdle = max(self.cost_hurdle_multiplier * costs, self.min_ev_net_absolute_pct)
        passes = bool(ev_net >= hurdle)

        if passes:
            decision = SignalAction.BUY
            rationale = (
                f"Kvalifikovaný nákup SCHVÁLEN: EV_net = {ev_net:+.2f} % "
                f"(p_win: {p_win*100:.1f} %, TP: +{barriers.take_profit_pct:.2f} %, "
                f"SL: -{barriers.stop_loss_pct:.2f} %, R:R: 1:{barriers.rr_ratio:.1f}, "
                f"náklady: {costs:.2f} %, práh: {hurdle:.2f} %)."
            )
        else:
            decision = SignalAction.NO_TRADE
            rationale = (
                f"Nedostatečná hrana obchodu: EV_net ({ev_net:+.2f} %) "
                f"nepřekonala bezpečnostní práh {hurdle:.2f} %. "
                f"Aktivována ochrana kapitálu (výchozí stav NO TRADE)."
            )

        return ExpectedValueResult(
            ticker=signal.ticker,
            decision=decision,
            p_win=p_win,
            p_loss=p_loss,
            expected_gain_pct=round(barriers.take_profit_pct, 2),
            expected_loss_pct=round(barriers.stop_loss_pct, 2),
            total_friction_cost_pct=costs,
            ev_net_pct=round(ev_net, 3),
            hurdle_threshold_pct=round(hurdle, 3),
            passes_hurdle=passes,
            barriers=barriers,
            rationale=rationale,
        )
