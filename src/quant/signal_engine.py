"""Modul 3: Carver Orthogonal Signal Engine.

Implementuje ortogonální kombinaci střednědobého Trend-Followingu / Momenta
a krátkodobého návratu k průměru (Mean Reversion) dle Sekce 1 a Sekce 8 v research.md:
1. Momentum složka: volatilitně normalizovaná relativní síla, průřezové pořadí, trendový filtr
2. Mean-Reversion složka: Z-skóre od SMA 20 s ochranou proti propadajícím se nožům (falling knife filter)
3. Ortogonální Carverovo sloučení s dynamickými vahami z RegimeDetectoru
4. Long-only omezení pro Trading 212 Invest účet (zamítnutí SHORT větví do NO TRADE)
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.quant.features import CausalFeatureEngine, StockFeatures
from src.quant.regime_detector import MarketRegime, MarketRegimeType

logger = logging.getLogger(__name__)


class SignalAction(str, Enum):
    """Přípustné akce signálového jádra."""
    BUY = "BUY"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"
    CLOSE = "CLOSE"
    SELL = "SELL"


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


class IntradaySignal(BaseModel):
    """Kvantifikovaný intradenní signál pro exekuci v denním obchodování."""
    ticker: str
    strategy_name: str
    action: SignalAction
    entry_price: float
    stop_loss: float
    take_profit: float
    time_stop_bars: int = 8
    r_multiple_target: float = 2.0
    reasons: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)


class ORB15mStrategy:
    """
    Strategie 1: ORB 15m Momentum (Opening Range Breakout).
    - Vstupní okno: 09:45 – 11:00 EST (minutes_from_open 15 až 90)
    - Podmínka: RVOL >= 1.5, ORB šířka <= 0.75 * ATR_14, Close > OR_High a Close > VWAP, RS_SPY >= 0.
    - SL: max(OR_Low, Close - 1.2 * ATR_14), TP: Close + 2.0 * R.
    - Max holding: 8 svíček (40 min).
    """

    def __init__(self, rvol_threshold: float = 1.5, max_orb_width_atr: float = 0.75):
        self.rvol_threshold = rvol_threshold
        self.max_orb_width_atr = max_orb_width_atr

    def evaluate_bar(
        self,
        df: pd.DataFrame,
        regime: MarketRegime,
        idx: int = -1,
        ticker: str = "STOCK",
    ) -> Optional[IntradaySignal]:
        if not regime.allow_trading or regime.regime in (
            MarketRegimeType.CRISIS_PANIC,
            MarketRegimeType.HIGH_VOL_PANIC,
            MarketRegimeType.BEAR_QUIET,
            MarketRegimeType.TRENDING_BEAR,
        ):
            return None

        pos = idx if idx >= 0 else len(df) + idx
        if pos < 1:
            return None

        row = df.iloc[pos]
        min_open = int(row.get("minutes_from_open", -1))
        # Vstupní časové okno 09:45 – 11:00 EST
        if min_open < 15 or min_open > 90:
            return None

        orb_high = float(row.get("orb_high", np.nan))
        orb_low = float(row.get("orb_low", np.nan))
        orb_width = float(row.get("orb_width", np.nan))
        vwap = float(row.get("vwap", np.nan))
        rvol = float(row.get("rvol_at_time", 1.0))
        atr = float(row.get("atr_14", row.get("atr_20", 0.0)))
        rs_z = float(row.get("intraday_rs_zscore", 0.0))

        if np.isnan(orb_high) or np.isnan(orb_low) or np.isnan(vwap) or atr <= 0:
            return None

        cur_close = float(row["Close"])
        cur_open = float(row["Open"])
        prev_close = float(df["Close"].iloc[pos - 1])

        # 1. Podmínka průrazu ORB High a VWAP
        if not (cur_close > orb_high and cur_close > vwap):
            return None

        # 2. RVOL filtr (zvýšený institucionální objem)
        if rvol < self.rvol_threshold:
            return None

        # 3. Šířka ORB nesmí být přepálená (max 75 % denního ATR)
        if orb_width > self.max_orb_width_atr * atr:
            return None

        # 4. Relativní síla vůči SPY nezáporná
        if rs_z < -0.2:
            return None

        sl = round(max(orb_low, cur_close - 1.2 * atr), 2)
        price_risk = cur_close - sl
        if price_risk <= 0.01:
            return None

        tp = round(cur_close + (2.0 * price_risk), 2)

        return IntradaySignal(
            ticker=ticker,
            strategy_name="ORB_15M_MOMENTUM",
            action=SignalAction.BUY,
            entry_price=round(cur_close, 2),
            stop_loss=sl,
            take_profit=tp,
            time_stop_bars=8,
            r_multiple_target=2.0,
            reasons=[
                f"ORB 15m průraz nad {orb_high:.2f}",
                f"Close {cur_close:.2f} > VWAP {vwap:.2f}",
                f"RVOL {rvol:.2f} >= {self.rvol_threshold:.1f}",
            ],
            timestamp=df.index[pos].to_pydatetime() if hasattr(df.index[pos], "to_pydatetime") else datetime.now(),
        )


class VWAPPullbackStrategy:
    """
    Strategie 2: VWAP Trend Pullback.
    - Vstupní okno: 10:00 – 14:30 EST (minutes_from_open 30 až 300)
    - Režim: TRENDING_BULL / BULL_QUIET
    - Podmínka: Akcie většinu dne nad VWAPem, pullback otestuje pásmo VWAP + 0.3 * ATR,
      svíčka se odrazí a uzavře zelená.
    - SL: min(Low, VWAP) - 0.5 * ATR, TP: Close + 2.0 * R.
    - Max holding: 12 svíček (60 min).
    """

    def evaluate_bar(
        self,
        df: pd.DataFrame,
        regime: MarketRegime,
        idx: int = -1,
        ticker: str = "STOCK",
    ) -> Optional[IntradaySignal]:
        if not regime.allow_trading or regime.regime not in (
            MarketRegimeType.TRENDING_BULL,
            MarketRegimeType.BULL_QUIET,
        ):
            return None

        pos = idx if idx >= 0 else len(df) + idx
        if pos < 5:
            return None

        row = df.iloc[pos]
        min_open = int(row.get("minutes_from_open", -1))
        # Vstupní okno 10:00 až 14:30 EST
        if min_open < 30 or min_open > 300:
            return None

        vwap = float(row.get("vwap", np.nan))
        atr = float(row.get("atr_14", row.get("atr_20", 0.0)))
        rs_z = float(row.get("intraday_rs_zscore", 0.0))

        if np.isnan(vwap) or atr <= 0:
            return None

        cur_close = float(row["Close"])
        cur_open = float(row["Open"])
        cur_low = float(row["Low"])
        prev_close = float(df["Close"].iloc[pos - 1])

        # 1. Low otestovalo pásmo VWAP (max +0.3 ATR nad VWAP) a Close zůstává nad VWAP
        pullback_tested = (cur_low <= vwap + 0.3 * atr) and (cur_close >= vwap)
        if not pullback_tested:
            return None

        # 2. Býčí obratová svíčka (zelená svíčka)
        is_bullish_bounce = (cur_close > cur_open) and (cur_close > prev_close)
        if not is_bullish_bounce:
            return None

        # 3. Relativní síla vůči SPY
        if rs_z < 0.2:
            return None

        sl = round(max(0.01, min(cur_low, vwap) - 0.5 * atr), 2)
        price_risk = cur_close - sl
        if price_risk <= 0.01 or price_risk > 1.5 * atr:
            sl = round(cur_close - 1.2 * atr, 2)
            price_risk = cur_close - sl

        tp = round(cur_close + (2.0 * price_risk), 2)

        return IntradaySignal(
            ticker=ticker,
            strategy_name="VWAP_TREND_PULLBACK",
            action=SignalAction.BUY,
            entry_price=round(cur_close, 2),
            stop_loss=sl,
            take_profit=tp,
            time_stop_bars=12,
            r_multiple_target=2.0,
            reasons=[
                f"Úspěšný test a odraz od VWAP ({vwap:.2f})",
                f"Býčí obratová svíčka nad VWAP",
                f"RS vs SPY {rs_z:.2f} > 0.2",
            ],
            timestamp=df.index[pos].to_pydatetime() if hasattr(df.index[pos], "to_pydatetime") else datetime.now(),
        )


class VWAPMeanReversionStrategy:
    """
    Strategie 3: VWAP Band Reversion (odraz z přeprodaného pásma v bočním trhu).
    - Vstupní okno: 10:30 – 14:30 EST (minutes_from_open 60 až 300)
    - Režim: CHOPPY_RANGE / SIDEWAYS
    - Podmínka: RVOL < 2.0 (není to krizový fundamentální propad),
      cena klesla pod spodní pásmo (VWAP - 2.0 * sigma) a odrazila se zpět nahoru.
    - SL: Close - 1.0 * ATR, TP: návrat k Session VWAP.
    - Max holding: 6 svíček (30 min).
    """

    def evaluate_bar(
        self,
        df: pd.DataFrame,
        regime: MarketRegime,
        idx: int = -1,
        ticker: str = "STOCK",
    ) -> Optional[IntradaySignal]:
        if not regime.allow_trading or regime.regime not in (
            MarketRegimeType.CHOPPY_RANGE,
            MarketRegimeType.SIDEWAYS,
        ):
            return None

        pos = idx if idx >= 0 else len(df) + idx
        if pos < 5:
            return None

        row = df.iloc[pos]
        min_open = int(row.get("minutes_from_open", -1))
        # Vstupní okno 10:30 až 14:30 EST
        if min_open < 60 or min_open > 300:
            return None

        vwap = float(row.get("vwap", np.nan))
        vwap_lower_2 = float(row.get("vwap_lower_2", np.nan))
        atr = float(row.get("atr_14", row.get("atr_20", 0.0)))
        rvol = float(row.get("rvol_at_time", 1.0))

        if np.isnan(vwap) or np.isnan(vwap_lower_2) or atr <= 0:
            return None

        # 1. Zákaz nákupu při panickém objemu (katalytický propad)
        if rvol >= 2.0:
            return None

        cur_close = float(row["Close"])
        cur_open = float(row["Open"])
        cur_low = float(row["Low"])

        # 2. Cena lízla nebo prorazila pod pásmo -2 sigma a odrazila se zelenou svíčkou
        stretched = (cur_low <= vwap_lower_2) and (cur_close > cur_open) and (cur_close < vwap)
        if not stretched:
            return None

        sl = round(max(0.01, cur_close - 1.0 * atr), 2)
        price_risk = cur_close - sl
        reward = vwap - cur_close

        # RRR musí být alespoň 1:1.2 zpět k VWAP
        if reward < 1.2 * price_risk or price_risk <= 0.01:
            return None

        return IntradaySignal(
            ticker=ticker,
            strategy_name="VWAP_BAND_REVERSION",
            action=SignalAction.BUY,
            entry_price=round(cur_close, 2),
            stop_loss=sl,
            take_profit=round(vwap, 2),
            time_stop_bars=6,
            r_multiple_target=round(reward / price_risk, 2),
            reasons=[
                f"Přeprodanost pod -2.0 sigma pásmem ({vwap_lower_2:.2f})",
                f"Návratový cíl k Session VWAP ({vwap:.2f})",
                f"Klidný RVOL {rvol:.2f} < 2.0",
            ],
            timestamp=df.index[pos].to_pydatetime() if hasattr(df.index[pos], "to_pydatetime") else datetime.now(),
        )


class IntradaySignalEngine:
    """
    Sjednocený intradenní signálový modul kombinující 3 modulární strategie:
    1. ORB15mStrategy (Momentum v trendovém/normálním trhu 09:45–11:00 EST)
    2. VWAPPullbackStrategy (Pullback v býčím trendu 10:00–14:30 EST)
    3. VWAPMeanReversionStrategy (Odraz z pásma v bočním trhu 10:30–14:30 EST)
    """

    def __init__(self):
        self.orb_strategy = ORB15mStrategy()
        self.pullback_strategy = VWAPPullbackStrategy()
        self.mean_reversion_strategy = VWAPMeanReversionStrategy()

    def evaluate(
        self,
        df: pd.DataFrame,
        regime: MarketRegime,
        current_idx: int = -1,
        ticker: str = "STOCK",
    ) -> Optional[IntradaySignal]:
        """Vyhodnotí všechny strategie pro daný bar a vrátí první kvalifikovaný signál."""
        # 1. ORB Breakout
        sig = self.orb_strategy.evaluate_bar(df, regime, idx=current_idx, ticker=ticker)
        if sig is not None:
            return sig

        # 2. VWAP Pullback
        sig = self.pullback_strategy.evaluate_bar(df, regime, idx=current_idx, ticker=ticker)
        if sig is not None:
            return sig

        # 3. VWAP Mean Reversion
        sig = self.mean_reversion_strategy.evaluate_bar(df, regime, idx=current_idx, ticker=ticker)
        if sig is not None:
            return sig

        return None

    def generate_signals_for_dataframe(
        self,
        df: pd.DataFrame,
        regime: MarketRegime,
        ticker: str = "STOCK",
    ) -> pd.DataFrame:
        """
        Vygeneruje signály a cenové bariéry pro celý DataFrame, připravené pro IntradayBacktester.
        """
        n = len(df)
        signals = np.zeros(n, dtype=int)
        stop_losses = np.full(n, np.nan, dtype=float)
        take_profits = np.full(n, np.nan, dtype=float)
        strat_names = np.full(n, "", dtype=object)

        for i in range(n):
            sig = self.evaluate(df, regime, current_idx=i, ticker=ticker)
            if sig is not None and sig.action == SignalAction.BUY:
                signals[i] = 1
                stop_losses[i] = sig.stop_loss
                take_profits[i] = sig.take_profit
                strat_names[i] = sig.strategy_name

        return pd.DataFrame(
            {
                "signal": signals,
                "stop_loss": stop_losses,
                "take_profit": take_profits,
                "strategy_name": strat_names,
            },
            index=df.index,
        )

