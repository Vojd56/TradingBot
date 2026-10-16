"""Modul 2: Quantitative Regime Detector.

Deterministický a vícerozměrný detektor globálního tržního režimu (Sekce 8 v research.md).
Vyhodnocuje benchmark trhu (SPY / QQQ / primární index) a portfolio:
1. Sklon 200denního klouzavého průměru (SMA 200 trend slope)
2. Poměr ceny vůči SMA 200 (Long-term regime filter)
3. Percentil 21denní realizované volatility oproti 252d/500d historii
4. Volatilitní komprese a expanze (poměr ATR_14 / ATR_100)
5. Detekce krizových stavů (CRISIS_PANIC) s nekompromisním tvrdým příznakem NO TRADE
"""

from enum import Enum
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.quant.features import CausalFeatureEngine

logger = logging.getLogger(__name__)


class MarketRegimeType(str, Enum):
    """Typy globálního tržního režimu."""
    BULL_QUIET = "BULL_QUIET"          # Klidný růstový trend: ideální pro Momentum a Trend Following
    BULL_VOLATILE = "BULL_VOLATILE"    # Růstový, ale rozkolísaný trh: širší stop-lossy, redukce velikosti pozic
    SIDEWAYS = "SIDEWAYS"              # Boční trh bez jasného trendu: dominance Mean Reversion
    BEAR_QUIET = "BEAR_QUIET"          # Klidný pokles / korekce: defenzivní režim, jen selektivní nákupy
    CRISIS_PANIC = "CRISIS_PANIC"      # Panický výprodej / likviditní šok: TVRDÝ NO TRADE / HALT

    # Intradenní sjednocené režimy dle AG-05
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    CHOPPY_RANGE = "CHOPPY_RANGE"
    HIGH_VOL_PANIC = "HIGH_VOL_PANIC"


class MarketRegime(BaseModel):
    """Kompletní stav globálního tržního režimu v čase t."""
    regime: MarketRegimeType
    allow_trading: bool                # False při CRISIS_PANIC nebo extrémní volatilitě > 95. percentil
    risk_multiplier: float = Field(default=1.0, ge=0.0, le=1.0)  # Škálování celkového rizika (0.0 až 1.0)
    momentum_weight: float = Field(default=0.7, ge=0.0, le=1.0)  # Relativní váha složky Momentum
    mean_reversion_weight: float = Field(default=0.3, ge=0.0, le=1.0)  # Relativní váha složky Mean Reversion

    # Kvantitativní metriky benchmarku
    benchmark_symbol: str = "SPY"
    current_price: float = 0.0
    sma_200: float = 0.0
    price_to_sma200: float = 1.0
    sma200_slope_20d_pct: float = 0.0
    realized_vol_annualized: float = 0.15
    vol_percentile: float = 50.0       # 0 až 100 %
    volatility_ratio_atr: float = 1.0  # ATR_14 / ATR_100
    recent_drawdown_5d_pct: float = 0.0
    description: str = ""


class RegimeDetector:
    """Detektor tržních režimů z kauzálních časových řad benchmarku."""

    def __init__(self, panic_vol_percentile_threshold: float = 95.0, panic_drawdown_5d_threshold: float = -5.0):
        self.panic_vol_percentile_threshold = panic_vol_percentile_threshold
        self.panic_drawdown_5d_threshold = panic_drawdown_5d_threshold

    def detect_intraday_regime(
        self,
        spy_df: Optional[pd.DataFrame] = None,
        vix: Optional[float] = None,
    ) -> MarketRegime:
        """
        Vyhodnotí intradenní tržní režim (TRENDING_BULL, TRENDING_BEAR, CHOPPY_RANGE, HIGH_VOL_PANIC).
        Využívá intradenní odchylku SPY od otevírací ceny/VWAP a volatilitu VIX.
        """
        # 1. Krizový filtr volatility (VIX >= 26.0 nebo chybějící data)
        if vix is not None and vix >= 26.0:
            return MarketRegime(
                regime=MarketRegimeType.HIGH_VOL_PANIC,
                allow_trading=False,
                risk_multiplier=0.0,
                momentum_weight=0.0,
                mean_reversion_weight=0.0,
                description=f"HIGH_VOL_PANIC: VIX na úrovni {vix:.1f} překračuje panický práh 26.0. Zákaz intradenních nákupů.",
            )

        if spy_df is None or len(spy_df) < 5:
            return MarketRegime(
                regime=MarketRegimeType.CHOPPY_RANGE,
                allow_trading=True,
                risk_multiplier=0.5,
                momentum_weight=0.3,
                mean_reversion_weight=0.7,
                description="Nedostatek intradenních dat pro SPY - výchozí neutrální CHOPPY_RANGE.",
            )

        close = spy_df["Close"]
        curr_price = float(close.iloc[-1])
        open_price = float(spy_df["Open"].iloc[0])
        ret_today_pct = ((curr_price - open_price) / open_price) * 100.0

        # Silný intradenní propad benchmarku (> -1.5 %)
        if ret_today_pct <= -1.5:
            return MarketRegime(
                regime=MarketRegimeType.HIGH_VOL_PANIC,
                allow_trading=False,
                risk_multiplier=0.0,
                momentum_weight=0.0,
                mean_reversion_weight=0.0,
                current_price=curr_price,
                description=f"HIGH_VOL_PANIC: Intradenní propad SPY {ret_today_pct:.2f} % překračuje bezpečnostní limit.",
            )

        # Sklon krátkodobého trendu (SMA 20 na 5m baru)
        sma20 = close.rolling(min(20, len(close))).mean()
        curr_sma = float(sma20.iloc[-1])

        if curr_price > curr_sma and ret_today_pct >= 0.2:
            return MarketRegime(
                regime=MarketRegimeType.TRENDING_BULL,
                allow_trading=True,
                risk_multiplier=1.0,
                momentum_weight=0.8,
                mean_reversion_weight=0.2,
                current_price=curr_price,
                description=f"TRENDING_BULL: SPY roste (+{ret_today_pct:.2f} %) nad SMA 20.",
            )
        elif curr_price < curr_sma and ret_today_pct <= -0.4:
            return MarketRegime(
                regime=MarketRegimeType.TRENDING_BEAR,
                allow_trading=False,
                risk_multiplier=0.2,
                momentum_weight=0.3,
                mean_reversion_weight=0.7,
                current_price=curr_price,
                description=f"TRENDING_BEAR: SPY klesá ({ret_today_pct:.2f} %) pod SMA 20.",
            )
        else:
            return MarketRegime(
                regime=MarketRegimeType.CHOPPY_RANGE,
                allow_trading=True,
                risk_multiplier=0.6,
                momentum_weight=0.2,
                mean_reversion_weight=0.8,
                current_price=curr_price,
                description=f"CHOPPY_RANGE: Boční pohyb trhu ({ret_today_pct:+.2f} %).",
            )


    def detect_regime(
        self,
        benchmark_df: pd.DataFrame,
        benchmark_symbol: str = "SPY",
        min_bars: int = 100,
    ) -> MarketRegime:
        """
        Vyhodnotí globální tržní režim z historických dat benchmarku.
        
        Garantuje nulový look-ahead bias (data <= t).
        """
        if benchmark_df is None or len(benchmark_df) < min_bars:
            logger.warning(f"Nedostatek dat pro RegimeDetector ({len(benchmark_df) if benchmark_df is not None else 0} barů). Výchozí defenzivní režim.")
            return MarketRegime(
                regime=MarketRegimeType.SIDEWAYS,
                allow_trading=True,
                risk_multiplier=0.5,
                momentum_weight=0.5,
                mean_reversion_weight=0.5,
                benchmark_symbol=benchmark_symbol,
                description="Nedostatek dat pro plnou detekci - aktivován defenzivní režim.",
            )

        df = benchmark_df.sort_index().copy()
        close = df["Close"]
        curr_price = float(close.iloc[-1])

        # 1. Realizovaná volatilita (21d anualizovaná)
        log_ret = np.log(close / close.shift(1))
        vol_21d = log_ret.rolling(window=21).std() * np.sqrt(252)
        curr_vol = float(vol_21d.iloc[-1]) if not np.isnan(vol_21d.iloc[-1]) else 0.16

        # 2. Percentil realizované volatility v historii (posledních min(500, len) barů)
        hist_window = min(len(vol_21d.dropna()), 500)
        valid_vols = vol_21d.dropna().iloc[-hist_window:]
        if len(valid_vols) > 20:
            vol_pct = float((valid_vols < curr_vol).mean() * 100.0)
        else:
            vol_pct = 50.0

        # 3. SMA 200 a její sklon za 20 dní
        len_sma = min(200, len(close))
        sma_200_series = close.rolling(window=len_sma).mean()
        curr_sma_200 = float(sma_200_series.iloc[-1])
        price_to_sma200 = curr_price / max(1e-6, curr_sma_200)

        if len(sma_200_series) >= 20 and not np.isnan(sma_200_series.iloc[-20]):
            past_sma = float(sma_200_series.iloc[-20])
            sma200_slope_pct = ((curr_sma_200 - past_sma) / past_sma) * 100.0
        else:
            sma200_slope_pct = 0.0

        # 4. Volatilitní poměr ATR_14 / ATR_100
        atr_14 = CausalFeatureEngine.calculate_atr(df, period=14)
        atr_100 = CausalFeatureEngine.calculate_atr(df, period=min(100, max(20, len(df) - 5)))
        curr_atr_14 = float(atr_14.iloc[-1]) if not np.isnan(atr_14.iloc[-1]) else 1.0
        curr_atr_100 = float(atr_100.iloc[-1]) if not np.isnan(atr_100.iloc[-1]) else 1.0
        vol_ratio = curr_atr_14 / max(1e-6, curr_atr_100)

        # 5. 5denní krátkodobý drawdown benchmarku
        close_5d_ago = float(close.iloc[-5]) if len(close) >= 5 else curr_price
        ret_5d_pct = ((curr_price - close_5d_ago) / close_5d_ago) * 100.0

        # --- KLASIFIKAČNÍ ROZHODOVACÍ STROM ---
        # A) Krizový panický stav (CRISIS_PANIC)
        is_panic = (
            vol_pct >= self.panic_vol_percentile_threshold
            or ret_5d_pct <= self.panic_drawdown_5d_threshold
            or (price_to_sma200 < 0.90 and vol_pct > 80.0)
        )

        if is_panic:
            regime = MarketRegimeType.CRISIS_PANIC
            allow_trading = False
            risk_mult = 0.0
            w_mom = 0.0
            w_mr = 0.0
            desc = f"CRISIS_PANIC: Volatilita {vol_pct:.1f}. percentil, 5d propad {ret_5d_pct:.2f}%. Obchodování zmrazeno (HARD NO TRADE)."

        # B) Klidný růstový trend (BULL_QUIET)
        elif curr_price >= curr_sma_200 and sma200_slope_pct >= 0.0 and vol_pct < 60.0:
            regime = MarketRegimeType.BULL_QUIET
            allow_trading = True
            risk_mult = 1.0
            w_mom = 0.75
            w_mr = 0.25
            desc = f"BULL_QUIET: Trh nad SMA200 (sklon +{sma200_slope_pct:.2f}%), nízká volatilita ({vol_pct:.1f}%). Plná alokace na Momentum."

        # C) Růstový, ale volatilní trh (BULL_VOLATILE)
        elif curr_price >= curr_sma_200 and vol_pct >= 60.0:
            regime = MarketRegimeType.BULL_VOLATILE
            allow_trading = True
            risk_mult = 0.65
            w_mom = 0.55
            w_mr = 0.45
            desc = f"BULL_VOLATILE: Trh nad SMA200 s expanzí volatility ({vol_pct:.1f}%). Redukce rizika na 65%, posílení Mean Reversion."

        # D) Klidný medvědí / korekční trh (BEAR_QUIET)
        elif curr_price < curr_sma_200 and sma200_slope_pct < 0.0 and vol_pct < 80.0:
            regime = MarketRegimeType.BEAR_QUIET
            allow_trading = True
            risk_mult = 0.40
            w_mom = 0.20
            w_mr = 0.80
            desc = f"BEAR_QUIET: Trh pod SMA200 (sklon {sma200_slope_pct:.2f}%). Defenzivní alokace 40%, dominance Mean Reversion na kvalitních aktivech."

        # E) Boční netrendující trh (SIDEWAYS)
        else:
            regime = MarketRegimeType.SIDEWAYS
            allow_trading = True
            risk_mult = 0.70
            w_mom = 0.35
            w_mr = 0.65
            desc = f"SIDEWAYS: Boční trh poblíž SMA200. Střední riziko (70%), vyšší váha Mean Reversion (65%)."

        return MarketRegime(
            regime=regime,
            allow_trading=allow_trading,
            risk_multiplier=round(risk_mult, 2),
            momentum_weight=round(w_mom, 2),
            mean_reversion_weight=round(w_mr, 2),
            benchmark_symbol=benchmark_symbol,
            current_price=round(curr_price, 2),
            sma_200=round(curr_sma_200, 2),
            price_to_sma200=round(price_to_sma200, 4),
            sma200_slope_20d_pct=round(sma200_slope_pct, 2),
            realized_vol_annualized=round(curr_vol, 4),
            vol_percentile=round(vol_pct, 1),
            volatility_ratio_atr=round(vol_ratio, 2),
            recent_drawdown_5d_pct=round(ret_5d_pct, 2),
            description=desc,
        )
