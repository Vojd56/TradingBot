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

from src.strategy.market_hours import tag_market_session

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

    # Intradenní indikátory (Session VWAP, ORB 15m, Intradenní RVOL a RS)
    vwap: float = 0.0
    vwap_dist_pct: float = 0.0                    # ((Price - VWAP) / VWAP) * 100
    vwap_dist_atr: float = 0.0                    # (Price - VWAP) / ATR
    vwap_upper_2: float = 0.0
    vwap_lower_2: float = 0.0
    orb_high_15m: Optional[float] = None
    orb_low_15m: Optional[float] = None
    orb_width_atr: float = 0.0
    rvol_at_time: float = 1.0
    intraday_rs_zscore: float = 0.0

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

    @classmethod
    def compute_session_vwap(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kauzální Session VWAP a směrodatné odchylky resetované přesně v 09:30 EST každého dne.
        Počítáno výhradně z REGULAR session svíček (09:30–16:00 EST).
        
        Vrací DataFrame se sloupci:
        - vwap: Volume-Weighted Average Price od 09:30
        - vwap_std: směrodatná odchylka typické ceny vážená objemem
        - vwap_upper_1, vwap_lower_1: ±1.0 sigma pásmo
        - vwap_upper_2, vwap_lower_2: ±2.0 sigma pásmo
        - vwap_upper_2_5, vwap_lower_2_5: ±2.5 sigma pásmo
        - vwap_dist_pct: odchylka Close od VWAP v %
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # Zajištění session značek
        if "is_regular_session" not in df.columns or "trading_day" not in df.columns:
            df_tagged = tag_market_session(df)
        else:
            df_tagged = df

        reg = df_tagged["is_regular_session"]
        tp = (df_tagged["High"] + df_tagged["Low"] + df_tagged["Close"]) / 3.0
        vol = df_tagged["Volume"].clip(lower=0.0)

        cols = [
            "vwap", "vwap_std", "vwap_upper_1", "vwap_lower_1",
            "vwap_upper_2", "vwap_lower_2", "vwap_upper_2_5", "vwap_lower_2_5", "vwap_dist_pct"
        ]
        res = pd.DataFrame(index=df_tagged.index, columns=cols, dtype=float)

        if not reg.any():
            return res

        reg_df = df_tagged.loc[reg]
        reg_vol = vol.loc[reg]
        reg_tp = tp.loc[reg]
        reg_days = reg_df["trading_day"]

        # Kumulativní objem a kumulativní součin P * V od 09:30 pro každý den
        cum_vol = reg_vol.groupby(reg_days).cumsum()
        cum_pv = (reg_vol * reg_tp).groupby(reg_days).cumsum()
        vwap = cum_pv / cum_vol.replace(0, np.nan)

        # Vážená směrodatná odchylka typické ceny od VWAP
        # Center on the session's first price to reduce cancellation error.
        origin = reg_tp.groupby(reg_days).transform("first")
        shifted = reg_tp - origin
        mean_shift = (reg_vol * shifted).groupby(reg_days).cumsum() / cum_vol.replace(0, np.nan)
        second_moment = (reg_vol * shifted.pow(2)).groupby(reg_days).cumsum() / cum_vol.replace(0, np.nan)
        vwap_var = second_moment - mean_shift.pow(2)
        vwap_std = np.sqrt(np.maximum(0.0, vwap_var.fillna(0.0)))

        res.loc[reg, "vwap"] = vwap
        res.loc[reg, "vwap_std"] = vwap_std
        res.loc[reg, "vwap_upper_1"] = vwap + vwap_std
        res.loc[reg, "vwap_lower_1"] = vwap - vwap_std
        res.loc[reg, "vwap_upper_2"] = vwap + 2.0 * vwap_std
        res.loc[reg, "vwap_lower_2"] = vwap - 2.0 * vwap_std
        res.loc[reg, "vwap_upper_2_5"] = vwap + 2.5 * vwap_std
        res.loc[reg, "vwap_lower_2_5"] = vwap - 2.5 * vwap_std
        res.loc[reg, "vwap_dist_pct"] = ((reg_df["Close"] - vwap) / vwap.replace(0, np.nan)) * 100.0

        return res

    @classmethod
    def compute_opening_range(cls, df: pd.DataFrame, orb_minutes: int = 15) -> pd.DataFrame:
        """
        Kauzální výpočet Opening Range (ORB) z prvních `orb_minutes` minut (09:30–09:45 EST).
        
        KRITICKÉ: Před časem 09:45 EST (minutes_from_open < orb_minutes) jsou hodnoty přísně NaN
        pro vyloučení jakéhokoliv look-ahead biasu!
        
        Vrací DataFrame se sloupci:
        - orb_high: maximum prvních orb_minutes
        - orb_low: minimum prvních orb_minutes
        - orb_width: šířka (orb_high - orb_low)
        """
        if df is None or df.empty:
            return pd.DataFrame()

        if "is_regular_session" not in df.columns or "minutes_from_open" not in df.columns or "trading_day" not in df.columns:
            df_tagged = tag_market_session(df)
        else:
            df_tagged = df

        cols = ["orb_high", "orb_low", "orb_width"]
        res = pd.DataFrame(index=df_tagged.index, columns=cols, dtype=float)

        orb_mask = (
            df_tagged["is_regular_session"]
            & (df_tagged["minutes_from_open"] >= 0)
            & (df_tagged["minutes_from_open"] < orb_minutes)
        )
        orb_bars = df_tagged[orb_mask]

        if orb_bars.empty:
            return res

        day_highs = orb_bars.groupby("trading_day")["High"].max()
        day_lows = orb_bars.groupby("trading_day")["Low"].min()

        mapped_high = df_tagged["trading_day"].map(day_highs)
        mapped_low = df_tagged["trading_day"].map(day_lows)
        mapped_width = mapped_high - mapped_low

        # Přísné kauzální pravidlo: platné POUZE od 09:45 EST (minutes_from_open >= orb_minutes)
        valid_mask = df_tagged["is_regular_session"] & (df_tagged["minutes_from_open"] >= orb_minutes)
        res["orb_high"] = mapped_high.where(valid_mask, np.nan)
        res["orb_low"] = mapped_low.where(valid_mask, np.nan)
        res["orb_width"] = mapped_width.where(valid_mask, np.nan)

        return res

    @classmethod
    def compute_rvol_at_time(cls, df: pd.DataFrame, lookback_days: int = 20) -> pd.Series:
        """
        Kauzální výpočet relativního objemu v danou minutu dne (RVOL at time).
        Poměr kumulativního objemu od 09:30 do minuty m vůči 20dennímu klouzavému průměru
        kumulativního objemu v téže minutě m za předchozí dny.
        
        Používá přísný jednodenní posun (.shift(1)), aby byl vyloučen pohled do budoucna.
        """
        if df is None or df.empty:
            return pd.Series(dtype=float)

        if "is_regular_session" not in df.columns or "minutes_from_open" not in df.columns or "trading_day" not in df.columns:
            df_tagged = tag_market_session(df)
        else:
            df_tagged = df

        res = pd.Series(1.0, index=df_tagged.index, dtype=float)
        reg = df_tagged["is_regular_session"]
        if not reg.any():
            return res

        reg_df = df_tagged.loc[reg].copy()
        reg_df["_cum_vol"] = reg_df.groupby("trading_day")["Volume"].cumsum()

        pivot = reg_df.pivot_table(
            index="trading_day",
            columns="minutes_from_open",
            values="_cum_vol",
            aggfunc="last",
        ).ffill(axis=1)

        # Přísný shift(1) po řádcích (dnech) a klouzavý průměr
        hist_avg = pivot.shift(1).rolling(window=lookback_days, min_periods=1).mean()
        stacked = hist_avg.stack()

        idx = pd.MultiIndex.from_arrays([reg_df["trading_day"], reg_df["minutes_from_open"]])
        expected_vol = idx.map(stacked)

        # Pro první den nebo chybějící historii použijeme 1.0 jako neutrální fallback
        rvol = (reg_df["_cum_vol"] / expected_vol).fillna(1.0).replace([np.inf, -np.inf], 1.0)
        res.loc[reg] = rvol
        return res

    @classmethod
    def compute_intraday_rs(
        cls,
        stock_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        window: int = 6,
    ) -> pd.Series:
        """
        Kauzální relativní síla akcie vůči benchmarku (SPY) očištěná o volatilitu (z-skóre).
        RS_z = clip((Ret_stock - beta * Ret_bench) / sigma_diff, -3.0, 3.0)
        """
        if stock_df is None or stock_df.empty:
            return pd.Series(dtype=float)

        if benchmark_df is None or benchmark_df.empty or "Close" not in benchmark_df.columns:
            return pd.Series(0.0, index=stock_df.index, dtype=float)

        bench_close = benchmark_df["Close"].reindex(stock_df.index).ffill()
        stock_ret = stock_df["Close"].pct_change(window)
        bench_ret = bench_close.pct_change(window)

        cov = stock_ret.rolling(30, min_periods=window).cov(bench_ret)
        var_b = bench_ret.rolling(30, min_periods=window).var()
        beta = (cov / var_b.replace(0, np.nan)).fillna(1.0).clip(0.1, 3.0)

        excess_ret = stock_ret - beta * bench_ret
        diff_std = excess_ret.rolling(30, min_periods=window).std().replace(0, np.nan)
        rs_z = (excess_ret / diff_std).fillna(0.0).clip(-3.0, 3.0)
        return rs_z

    @classmethod
    def compute_intraday_features(
        cls,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Komplexní obohacení intradenních OHLCV dat o kauzální příznaky (VWAP, ORB, RVOL, RS, ATR).
        Garantuje nulový look-ahead bias a kompatibilitu s backtesterem i skenerem.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df_out = tag_market_session(df)

        # 1. Session VWAP
        vwap_df = cls.compute_session_vwap(df_out)
        for col in vwap_df.columns:
            df_out[col] = vwap_df[col]

        # 2. Opening Range (15m)
        orb_df = cls.compute_opening_range(df_out, orb_minutes=15)
        for col in orb_df.columns:
            df_out[col] = orb_df[col]

        # 3. RVOL at time
        df_out["rvol_at_time"] = cls.compute_rvol_at_time(df_out)

        # 4. Intradenní ATR (14 a 20 barů)
        df_out["atr_14"] = cls.calculate_atr(df_out, period=14)
        df_out["atr_20"] = cls.calculate_atr(df_out, period=20)
        df_out["vwap_dist_atr"] = (df_out["Close"] - df_out["vwap"]) / df_out["atr_14"].replace(0, np.nan)
        df_out["orb_width_atr"] = df_out["orb_width"] / df_out["atr_14"].replace(0, np.nan)

        # 5. Relative Strength vs Benchmark
        df_out["intraday_rs_zscore"] = cls.compute_intraday_rs(df_out, benchmark_df=benchmark_df)

        return df_out


# Modulové funkce pro přímý import
compute_session_vwap = CausalFeatureEngine.compute_session_vwap
compute_opening_range = CausalFeatureEngine.compute_opening_range
compute_rvol_at_time = CausalFeatureEngine.compute_rvol_at_time
compute_intraday_rs = CausalFeatureEngine.compute_intraday_rs
compute_intraday_features = CausalFeatureEngine.compute_intraday_features

