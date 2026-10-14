from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class IntradaySnapshot:
    """Reprezentuje technický profil akcie z intradenních 5minutových svíček."""
    ticker: str
    current_price: float
    vwap: float
    is_above_vwap: bool
    ema_9: float
    ema_21: float
    is_bullish_ema_cross: bool  # EMA 9 > EMA 21
    is_fresh_cross: bool        # EMA 9 překřížila EMA 21 na poslední svíčce
    rsi_5m: float
    volume_surge_ratio: float   # Aktuální objem vůči 10periodovému průměru
    atr_5m: float
    high_30m: float
    low_30m: float
    is_breaking_high: bool      # Proražení 30minutového maxima
    change_pct: float           # Procentuální změna od začátku seance
    timestamp: datetime


class IntradayDataProvider:
    """
    Stahuje a analyzuje intradenní svíčky (1m a 5m) pro day trading a scalping.
    Počítá VWAP, EMA Ribbon (9/21), 5m RSI, Volume Surge a ATR v reálném čase.
    """

    def __init__(self, interval: str = "5m", period: str = "2d"):
        self.interval = interval
        self.period = period

    def _compute_from_df(self, ticker: str, df: pd.DataFrame) -> Optional[IntradaySnapshot]:
        """Vypočítá technický profil (VWAP, EMA 9/21, RSI, Volume Surge, ATR) z DataFrame svíček."""
        if df.empty or len(df) < 5:
            return None

        try:
            # 1. Výpočet VWAP (Volume-Weighted Average Price)
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
            cum_vol = df['Volume'].cumsum()
            cum_vol_price = (typical_price * df['Volume']).cumsum()
            vwap_series = cum_vol_price / cum_vol.replace(0, np.nan)
            current_vwap = float(vwap_series.iloc[-1])

            # 2. EMA 9 a EMA 21 (Scalp Ribbon)
            ema_9_series = df['Close'].ewm(span=9, adjust=False).mean()
            ema_21_series = df['Close'].ewm(span=21, adjust=False).mean()
            curr_ema_9 = float(ema_9_series.iloc[-1])
            curr_ema_21 = float(ema_21_series.iloc[-1])
            prev_ema_9 = float(ema_9_series.iloc[-2]) if len(df) >= 2 else curr_ema_9
            prev_ema_21 = float(ema_21_series.iloc[-2]) if len(df) >= 2 else curr_ema_21

            is_bullish_ema = curr_ema_9 > curr_ema_21
            is_fresh_cross = (prev_ema_9 <= prev_ema_21) and (curr_ema_9 > curr_ema_21)

            # 3. Intradenní RSI (14 period na 5m grafu)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=3).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=3).mean()
            rs = gain / loss.replace(0, 0.001)
            rsi_series = 100 - (100 / (1 + rs))
            curr_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

            # 4. Volume Surge Ratio
            vol_ma = df['Volume'].rolling(window=10, min_periods=2).mean()
            curr_vol = float(df['Volume'].iloc[-1])
            avg_vol = float(vol_ma.iloc[-1]) if float(vol_ma.iloc[-1]) > 0 else 1.0
            volume_surge = round(curr_vol / avg_vol, 2)

            # 5. ATR (Average True Range) na 5m svíčkách
            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift()).abs()
            tr3 = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_series = tr.rolling(window=14, min_periods=3).mean()
            curr_atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else float(df['Close'].iloc[-1]) * 0.01

            # 6. Breakout 30minutového pásma
            last_6 = df.iloc[-6:] if len(df) >= 6 else df
            high_30m = float(last_6['High'].max())
            low_30m = float(last_6['Low'].min())
            curr_close = float(df['Close'].iloc[-1])
            is_breaking_high = curr_close >= (high_30m * 0.998)

            first_open = float(df['Open'].iloc[0])
            change_pct = ((curr_close - first_open) / first_open) * 100.0

            return IntradaySnapshot(
                ticker=ticker,
                current_price=round(curr_close, 2),
                vwap=round(current_vwap, 2),
                is_above_vwap=(curr_close > current_vwap),
                ema_9=round(curr_ema_9, 2),
                ema_21=round(curr_ema_21, 2),
                is_bullish_ema_cross=is_bullish_ema,
                is_fresh_cross=is_fresh_cross,
                rsi_5m=round(curr_rsi, 1),
                volume_surge_ratio=volume_surge,
                atr_5m=round(curr_atr, 2),
                high_30m=round(high_30m, 2),
                low_30m=round(low_30m, 2),
                is_breaking_high=is_breaking_high,
                change_pct=round(change_pct, 2),
                timestamp=datetime.now(),
            )
        except Exception as exc:
            logger.debug(f"[Intraday Data] Chyba výpočtu indikátorů pro {ticker}: {exc}")
            return None

    def fetch_intraday_snapshot(self, ticker: str) -> Optional[IntradaySnapshot]:
        """Stáhne a spočítá kompletní intradenní technický profil pro jeden ticker."""
        clean = ticker.strip().upper()
        try:
            df = yf.download(clean, period=self.period, interval=self.interval, progress=False)
            if df.empty or len(df) < 5:
                return None

            # Zploštění multi-indexu u yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            return self._compute_from_df(clean, df)
        except Exception as exc:
            logger.debug(f"[Intraday Data] Chyba při načítání 5m svíček pro {clean}: {exc}")
            return None

    def fetch_batch_snapshots(self, tickers: List[str]) -> Dict[str, IntradaySnapshot]:
        """Rychlé paralelní stažení intradenních profilů pro celé široké univerzum (50-100 tickerů)."""
        clean_tickers = [t.strip().upper() for t in tickers if t.strip()]
        if not clean_tickers:
            return {}

        results: Dict[str, IntradaySnapshot] = {}
        if len(clean_tickers) < 3:
            for t in clean_tickers:
                s = self.fetch_intraday_snapshot(t)
                if s:
                    results[s.ticker] = s
            return results

        try:
            raw_data = yf.download(clean_tickers, period=self.period, interval=self.interval, progress=False)
            if raw_data.empty:
                return {}

            for t in clean_tickers:
                try:
                    if len(clean_tickers) == 1:
                        sub_df = raw_data
                    else:
                        sub_df = pd.DataFrame({
                            'Open': raw_data['Open'][t],
                            'High': raw_data['High'][t],
                            'Low': raw_data['Low'][t],
                            'Close': raw_data['Close'][t],
                            'Volume': raw_data['Volume'][t],
                        }).dropna()

                    snap = self._compute_from_df(t, sub_df)
                    if snap:
                        results[snap.ticker] = snap
                except Exception as sub_exc:
                    logger.debug(f"Chyba při zpracování batch tickeru {t}: {sub_exc}")

        except Exception as exc:
            logger.warning(f"Batch stahování intradenních dat selhalo ({exc}), fallback na sekvenční.")
            for t in clean_tickers:
                s = self.fetch_intraday_snapshot(t)
                if s:
                    results[s.ticker] = s

        return results
