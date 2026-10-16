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
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
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


def create_triple_barrier_labels(
    df: pd.DataFrame,
    signals: pd.Series,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 1.0,
    max_bars: int = 12,
) -> pd.DataFrame:
    """
    Vygeneruje Triple-Barrier labely a realizovaný čistý výnos v násobcích R (Sekce 5 v research.md).
    
    Bariéry:
    - Horní bariéra (TP): Entry + tp_atr_mult * ATR
    - Dolní bariéra (SL): Entry - sl_atr_mult * ATR
    - Časová bariéra: max_bars
    
    Kolizní pravidlo: Pokud je na jedné svíčce zasažen SL i TP, pesimisticky vyhrává SL (Y=0, R=-1.0).
    """
    records = []
    n = len(df)
    close_vals = df["Close"].values
    high_vals = df["High"].values
    low_vals = df["Low"].values
    open_vals = df["Open"].values

    if "atr_14" in df.columns:
        atr_vals = df["atr_14"].values
    elif "atr_20" in df.columns:
        atr_vals = df["atr_20"].values
    else:
        atr_vals = pd.Series(high_vals - low_vals).rolling(14, min_periods=1).mean().values

    sig_indices = np.where(signals.values == 1)[0]

    for t in sig_indices:
        if t >= n - 1:
            continue

        entry_idx = t + 1
        entry_price = float(open_vals[entry_idx])
        atr = float(atr_vals[t])
        if atr <= 0 or np.isnan(atr):
            atr = max(0.01, entry_price * 0.01)

        tp_price = entry_price + (tp_atr_mult * atr)
        sl_price = entry_price - (sl_atr_mult * atr)
        risk_per_share = entry_price - sl_price

        exit_idx = entry_idx
        exit_reason = "TIME_STOP"
        realized_r = 0.0
        label = 0

        end_horizon = min(n, entry_idx + max_bars)

        for k in range(entry_idx, end_horizon):
            k_low = float(low_vals[k])
            k_high = float(high_vals[k])

            if k_low <= sl_price and k_high >= tp_price:
                exit_idx = k
                exit_reason = "STOP_LOSS"
                realized_r = -1.0
                label = 0
                break
            elif k_low <= sl_price:
                exit_idx = k
                exit_reason = "STOP_LOSS"
                realized_r = -1.0
                label = 0
                break
            elif k_high >= tp_price:
                exit_idx = k
                exit_reason = "TAKE_PROFIT"
                realized_r = float(tp_atr_mult / sl_atr_mult)
                label = 1
                break
        else:
            exit_idx = end_horizon - 1
            exit_price = float(close_vals[exit_idx])
            price_change = exit_price - entry_price
            realized_r = float(price_change / max(1e-6, risk_per_share))
            label = 1 if realized_r > 0 else 0
            exit_reason = "TIME_STOP"

        records.append({
            "signal_idx": t,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry_price": round(entry_price, 4),
            "tp_price": round(tp_price, 4),
            "sl_price": round(sl_price, 4),
            "label": label,
            "realized_r": round(realized_r, 4),
            "exit_reason": exit_reason,
        })

    if not records:
        return pd.DataFrame(columns=[
            "signal_idx", "entry_idx", "exit_idx", "entry_price",
            "tp_price", "sl_price", "label", "realized_r", "exit_reason"
        ])

    return pd.DataFrame(records)


class PurgedTimeSeriesSplit:
    """
    Purged & Embargoed Cross-Validation splitter (De Prado, 2018).
    
    Odstraňuje překrývající se labely mezi trénovací a validační množinou (Purging)
    a vkládá ochranné okno po testovací sadě (Embargo) proti auto-korelaci.
    """
    def __init__(self, n_splits: int = 5, embargo_bars: int = 12):
        if n_splits < 2:
            raise ValueError("n_splits musí být alespoň 2.")
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars

    def split(self, X: Any, events: Optional[pd.DataFrame] = None):
        """
        Generuje dvojice (train_indices, test_indices).
        
        Pokud je events DataFrame s 'entry_idx' a 'exit_idx', každý obchod v trénovací
        sadě, jehož časový interval [entry, exit] zasahuje do testovacího okna [test_start, test_end],
        je z trénovací sady odstraněn (Purging).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False

            # Embargo bezprostředně za testovacím oknem
            embargo_end = min(n_samples, test_end + self.embargo_bars)
            train_mask[test_end:embargo_end] = False

            # Purging překryvů obchodů
            if events is not None and "entry_idx" in events.columns and "exit_idx" in events.columns:
                for _, row in events.iterrows():
                    e_in = int(row["entry_idx"])
                    e_out = int(row["exit_idx"])
                    # Pokud obchod protíná testovací okno [test_start, test_end)
                    if not (e_out < test_start or e_in >= test_end):
                        train_mask[max(0, e_in):min(n_samples, e_out + 1)] = False

            train_indices = indices[train_mask]
            yield train_indices, test_indices


class MetaLabelingClassifier:
    """
    Kvantitativní meta-model pro filtraci falešných průrazů a signálů.
    Využívá L2-regularizovanou logistickou regresi s Plattovým škálováním (čistý NumPy).
    """
    def __init__(self, l2_reg: float = 1.0, learning_rate: float = 0.05, max_iter: int = 200):
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.platt_a: float = 1.0
        self.platt_b: float = 0.0
        self.is_fitted: bool = False

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MetaLabelingClassifier":
        """Nafituje regularizovanou logistickou regresi na matici příznaků X a targetu y."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        if n_samples == 0:
            return self

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std < 1e-8] = 1.0
        X_norm = (X - self.mean) / self.std

        w = np.zeros(n_features)
        b = 0.0

        for _ in range(self.max_iter):
            z = np.dot(X_norm, w) + b
            preds = self._sigmoid(z)
            error = preds - y

            grad_w = (np.dot(X_norm.T, error) / n_samples) + (self.l2_reg * w / n_samples)
            grad_b = float(np.mean(error))

            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

        self.weights = w
        self.bias = b
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Vrátí kalibrovanou pravděpodobnost úspěchu P(Y=1) pro každý vzorek."""
        if not self.is_fitted or self.weights is None or self.mean is None or self.std is None:
            return np.full(len(X), 0.50, dtype=float)

        X = np.asarray(X, dtype=float)
        X_norm = (X - self.mean) / self.std
        z = np.dot(X_norm, self.weights) + self.bias
        z_cal = self.platt_a * z + self.platt_b
        return self._sigmoid(z_cal)

    def evaluate_oos_profit_factor(
        self,
        y_true: np.ndarray,
        realized_r: np.ndarray,
        probabilities: np.ndarray,
        threshold: float = 0.50,
    ) -> float:
        """
        Vypočte Profit Factor na Out-of-Sample vzorku při aplikaci meta-filtru.
        PF = Součet kladných R / Součet záporných R.
        """
        selected = probabilities >= threshold
        if np.sum(selected) == 0:
            return 0.0

        oos_r = realized_r[selected]
        gains = np.sum(oos_r[oos_r > 0])
        losses = np.abs(np.sum(oos_r[oos_r < 0]))
        if losses == 0:
            return float(gains) if gains > 0 else 1.0
        return float(gains / losses)


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
        use_ml_meta_filter: bool = False,
        meta_classifier: Optional[MetaLabelingClassifier] = None,
    ):
        self.min_m_sl = min_m_sl
        self.min_m_tp = max(min_m_tp, min_m_sl * 2.0)  # Vynucení R:R >= 2.0
        self.default_holding_bars = default_holding_bars
        self.cost_hurdle_multiplier = cost_hurdle_multiplier
        self.min_ev_net_absolute_pct = min_ev_net_absolute_pct
        self.fx_fee_pct = fx_fee_pct
        self.use_ml_meta_filter = use_ml_meta_filter
        self.meta_classifier = meta_classifier

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

        # Pokud je aktivní ML meta-model a je natrénovaný, zohledníme predikci meta-modelu
        if self.use_ml_meta_filter and self.meta_classifier and self.meta_classifier.is_fitted:
            feat_vec = np.array([[
                signal.features.momentum_3m,
                signal.features.vol_norm_momentum_3m,
                signal.features.mean_reversion_z_20,
                signal.features.atr_20_pct,
                signal.features.rvol_20,
                signal.composite_score,
            ]])
            p_ml = float(self.meta_classifier.predict_proba(feat_vec)[0])
            p_final = 0.5 * p_final + 0.5 * p_ml

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
