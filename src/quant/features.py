"""Modul 1: Causal & Dimensionless Feature Engineering.

Zajišťuje striktně kauzální výpočet stacionárních a bezrozměrných příznaků:
1. Volatilitně normalizované momentum (Time-Series & Cross-Sectional)
2. Mean-Reversion Z-skóre (odchylka ceny od rovnováhy / SMA 20)
3. Volatilitní filtry (poměr ATR_14 / ATR_100 pro detekci stlačení/expanze)
4. Objemové filtry (RVOL - relativní objem)

Striktní pravidlo: Veškeré výpočty probíhají na datech <= t bez look-ahead biasu.
"""

from datetime import datetime
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StockFeatures(BaseModel):
    """Kauzální stacionární a bezrozměrné příznaky pro jedno aktivum v čase t."""
    ticker: str
    timestamp: datetime
    current_price: float

    # Volatilitně normalizované časové momentum
    momentum_1m: float = 0.0                      # Výnos za 21 barů
    momentum_3m: float = 0.0                      # Výnos za 63 barů
    vol_norm_momentum_3m: float = 0.0             # 63d výnos / annualized vol
    vol_norm_momentum_12m_1m: float = 0.0         # 252d - 21d výnos / vol (klasické Fama-French-Carhart okno)

    # Volatilita
    realized_vol_annualized: float = 0.20         # 21d realizovaná volatilita (anualizovaná)
    atr_20: float = 1.0
    atr_20_pct: float = 2.0                       # (ATR_20 / Price) * 100

    # Mean-Reversion odchylky od rovnováhy
    mean_reversion_z_20: float = 0.0              # Z-skóre od SMA 20: (Price - SMA20) / Std20
    mean_reversion_z_50: float = 0.0              # Z-skóre od SMA 50: (Price - SMA50) / Std50

    # Volatilitní komprese a expanze (ATR poměry)
    volatility_compression_ratio: float = 1.0     # ATR_14 / ATR_100 (< 0.85 = squeeze, > 1.25 = expanze)
    is_volatility_squeezed: bool = False
    is_volatility_expanding: bool = False

    # Objemové anomálie
    rvol_20: float = 1.0                          # Volume / SMA20(Volume)

    # Dlouhodobý trendový kontext
    price_to_sma200_ratio: float = 1.0            # Price / SMA 200
    sma200_slope_20d_pct: float = 0.0             # 20d změna SMA 200 v % (sklon trendu)

    # Průřezové pořadí (vyplněno po srovnání univerza)
    cross_sectional_rank: float = 0.5             # 0.0 (nejslabší) až 1.0 (nejsilnější v univerzu)


class CausalFeatureEngine:
    """Kauzální kalkulátor stacionárních a bezrozměrných příznaků."""

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Kauzální výpočet Average True Range (Wilderova metoda / EMA)."""
        high = df["High"]
        low = df["Low"]
        close_prev = df["Close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Exponenciální vyhlazení ATR
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        return atr

    @classmethod
    def compute_single_ticker_features(
        cls,
        df: pd.DataFrame,
        ticker: str,
        min_bars_required: int = 60,
    ) -> Optional[StockFeatures]:
        """
        Vypočte všechny stacionární a bezrozměrné příznaky pro daný ticker v čase t (poslední řádek df).
        
        Garantuje nulový look-ahead bias: používá výhradně kauzální klouzavá okna.
        """
        if df is None or len(df) < min_bars_required:
            logger.debug(f"{ticker}: Nedostatek dat pro výpočet příznaků ({len(df) if df is not None else 0} barů).")
            return None

        # Ověření přítomnosti základních sloupců
        cols = {"Open", "High", "Low", "Close", "Volume"}
        if not cols.issubset(df.columns):
            logger.warning(f"{ticker}: V datech chybí povinné sloupce {cols - set(df.columns)}")
            return None

        # Pracujeme s kopií seřazenou podle času
        data = df.sort_index().copy()

        close = data["Close"]
        volume = data["Volume"].replace(0, np.nan).ffill().fillna(1.0)
        curr_price = float(close.iloc[-1])
        if curr_price <= 0:
            return None

        timestamp = data.index[-1]
        if isinstance(timestamp, pd.Timestamp):
            ts_dt = timestamp.to_pydatetime()
        else:
            ts_dt = datetime.now()

        # 1. Realizovaná volatilita (anualizovaná ze směrodatné odchylky denních log-výnosů za 21 barů)
        log_ret = np.log(close / close.shift(1))
        vol_21d = log_ret.rolling(window=21).std() * np.sqrt(252)
        realized_vol = float(vol_21d.iloc[-1]) if not np.isnan(vol_21d.iloc[-1]) and vol_21d.iloc[-1] > 0.01 else 0.20

        # 2. Volatilitně normalizované časové momentum
        ret_1m = float((close.iloc[-1] / close.iloc[-21] - 1.0)) if len(close) >= 21 else 0.0
        ret_3m = float((close.iloc[-1] / close.iloc[-63] - 1.0)) if len(close) >= 63 else ret_1m
        
        # 12M - 1M momentum (Fama-French / Carhart specifikace vylučující reverzi posledního měsíce)
        if len(close) >= 252:
            ret_12m_1m = float((close.iloc[-21] / close.iloc[-252] - 1.0))
        elif len(close) >= 126:
            ret_12m_1m = float((close.iloc[-21] / close.iloc[-126] - 1.0))
        else:
            ret_12m_1m = ret_3m

        vol_norm_3m = ret_3m / max(0.05, realized_vol)
        vol_norm_12m_1m = ret_12m_1m / max(0.05, realized_vol)

        # 3. ATR a volatilitní filtry
        atr_14 = cls.calculate_atr(data, period=14)
        atr_20 = cls.calculate_atr(data, period=20)
        atr_100 = cls.calculate_atr(data, period=min(100, max(30, len(data) - 5)))

        curr_atr_20 = float(atr_20.iloc[-1]) if not np.isnan(atr_20.iloc[-1]) else curr_price * 0.02
        atr_20_pct = (curr_atr_20 / curr_price) * 100.0

        val_atr_14 = float(atr_14.iloc[-1]) if not np.isnan(atr_14.iloc[-1]) else curr_atr_20
        val_atr_100 = float(atr_100.iloc[-1]) if not np.isnan(atr_100.iloc[-1]) else curr_atr_20
        vol_ratio = val_atr_14 / max(1e-6, val_atr_100)

        is_squeezed = bool(vol_ratio < 0.85)
        is_expanding = bool(vol_ratio > 1.25)

        # 4. Mean-Reversion Z-skóre ceny od rovnováhy
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        curr_sma_20 = float(sma_20.iloc[-1])
        curr_std_20 = float(std_20.iloc[-1]) if not np.isnan(std_20.iloc[-1]) and std_20.iloc[-1] > 0 else (curr_price * 0.01)
        z_20 = (curr_price - curr_sma_20) / curr_std_20

        sma_50 = close.rolling(window=min(50, len(close))).mean()
        std_50 = close.rolling(window=min(50, len(close))).std()
        curr_sma_50 = float(sma_50.iloc[-1])
        curr_std_50 = float(std_50.iloc[-1]) if not np.isnan(std_50.iloc[-1]) and std_50.iloc[-1] > 0 else (curr_price * 0.02)
        z_50 = (curr_price - curr_sma_50) / curr_std_50

        # 5. Relativní objem (RVOL)
        vol_sma_20 = volume.rolling(window=20).mean()
        curr_vol = float(volume.iloc[-1])
        curr_vol_sma = float(vol_sma_20.iloc[-1]) if not np.isnan(vol_sma_20.iloc[-1]) and vol_sma_20.iloc[-1] > 0 else curr_vol
        rvol = float(curr_vol / curr_vol_sma)

        # 6. Dlouhodobý trendový kontext (SMA 200)
        len_for_200 = min(200, len(close))
        sma_200_series = close.rolling(window=len_for_200).mean()
        curr_sma_200 = float(sma_200_series.iloc[-1])
        price_to_sma200 = curr_price / max(1e-6, curr_sma_200)

        # Sklon SMA 200 za 20 dní v %
        if len(sma_200_series) >= 20 and not np.isnan(sma_200_series.iloc[-20]):
            sma_200_past = float(sma_200_series.iloc[-20])
            slope_200_pct = ((curr_sma_200 - sma_200_past) / sma_200_past) * 100.0
        else:
            slope_200_pct = 0.0

        return StockFeatures(
            ticker=ticker,
            timestamp=ts_dt,
            current_price=round(curr_price, 4),
            momentum_1m=round(ret_1m, 4),
            momentum_3m=round(ret_3m, 4),
            vol_norm_momentum_3m=round(vol_norm_3m, 4),
            vol_norm_momentum_12m_1m=round(vol_norm_12m_1m, 4),
            realized_vol_annualized=round(realized_vol, 4),
            atr_20=round(curr_atr_20, 4),
            atr_20_pct=round(atr_20_pct, 2),
            mean_reversion_z_20=round(z_20, 2),
            mean_reversion_z_50=round(z_50, 2),
            volatility_compression_ratio=round(vol_ratio, 2),
            is_volatility_squeezed=is_squeezed,
            is_volatility_expanding=is_expanding,
            rvol_20=round(rvol, 2),
            price_to_sma200_ratio=round(price_to_sma200, 4),
            sma200_slope_20d_pct=round(slope_200_pct, 2),
            cross_sectional_rank=0.5,
        )

    @classmethod
    def compute_cross_sectional_ranks(
        cls,
        features_map: Dict[str, StockFeatures]
    ) -> Dict[str, StockFeatures]:
        """
        Vypočte průřezové percentilní pořadí (0.0 až 1.0) pro každé aktivum v univerzu.
        
        Řadí aktiva podle kompozitního volatilitně normalizovaného momenta.
        """
        if not features_map:
            return {}

        tickers = list(features_map.keys())
        # Kompozitní skóre pro průřezové řazení: 50% 3M momentum + 50% 12M-1M momentum
        scores = [
            0.5 * features_map[t].vol_norm_momentum_3m + 0.5 * features_map[t].vol_norm_momentum_12m_1m
            for t in tickers
        ]

        # Výpočet percentilního pořadí
        ranked_indices = np.argsort(np.argsort(scores))
        n = len(tickers)
        ranks = [(ranked_indices[i] + 0.5) / n for i in range(n)]

        for i, t in enumerate(tickers):
            features_map[t].cross_sectional_rank = round(float(ranks[i]), 4)

        return features_map
