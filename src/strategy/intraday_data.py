from dataclasses import dataclass
from datetime import date, datetime
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def validate_ohlcv_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Reject corrupt market data instead of manufacturing prices or volume.

    Sorting is lossless. Missing fields, ambiguous timestamps and invalid candle
    geometry are errors; the caller must skip/quarantine the affected dataset.
    """
    required = ["Open", "High", "Low", "Close", "Volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=required)
    clean = df.copy()
    names = {name.lower(): name for name in required}
    names["vol"] = "Volume"
    clean = clean.rename(columns={c: names.get(str(c).lower(), c) for c in clean.columns})
    if clean.columns.duplicated().any() or any(c not in clean for c in required):
        raise ValueError("Missing or duplicated OHLCV columns.")
    if not isinstance(clean.index, pd.DatetimeIndex):
        date_cols = [c for c in clean if str(c).lower() in ("datetime", "date", "timestamp", "time")]
        if not date_cols:
            raise ValueError("OHLCV requires explicit timestamps, not a numeric index.")
        clean = clean.set_index(pd.to_datetime(clean[date_cols[0]], errors="raise"))
    if clean.index.hasnans or clean.index.has_duplicates:
        raise ValueError("Invalid or duplicated OHLCV timestamps.")
    values = clean[required].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Non-finite OHLCV values.")
    if (values[["Open", "High", "Low", "Close"]] <= 0).any().any() or (values["Volume"] < 0).any():
        raise ValueError("Non-positive prices or negative volume.")
    if ((values["High"] < values[["Open", "Close", "Low"]].max(axis=1)) |
            (values["Low"] > values[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError("Invalid OHLCV candle geometry.")
    clean[required] = values
    return clean.sort_index()

class ParquetBarStore:
    """
    Lokální sloupcové úložiště pro intradenní a denní OHLCV data v Parquet formátu.
    Cesta: data/bars/{timeframe}/{symbol}.parquet
    
    Zajišťuje bleskové načtení (< 50 ms) a inkrementální ukládání stažených svíček.
    """

    def __init__(self, base_dir: Union[str, Path] = "data/bars"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, symbol: str, timeframe: str) -> Path:
        clean_sym = symbol.strip().upper().replace("/", "_").replace("^", "_").replace(".", "_")
        clean_tf = timeframe.strip().lower()
        dir_path = self.base_dir / clean_tf
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{clean_sym}.parquet"

    def has_bars(self, symbol: str, timeframe: str) -> bool:
        """Zjistí, zda existují uložená data pro daný symbol a timeframe."""
        return self._get_file_path(symbol, timeframe).exists()

    def save_bars(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
        """
        Uloží nebo inkrementálně sloučí svíčky pro daný symbol.
        Automaticky provádí validaci integrity svíček.
        """
        if df is None or df.empty:
            raise ValueError(f"Nelze uložit prázdný DataFrame pro {symbol}.")

        validated_df = validate_ohlcv_integrity(df)
        file_path = self._get_file_path(symbol, timeframe)

        if file_path.exists():
            try:
                existing_df = pd.read_parquet(file_path)
                if not isinstance(existing_df.index, pd.DatetimeIndex):
                    existing_df.index = pd.to_datetime(existing_df.index)

                # Sloučení a deduplikace indexu (novější data přepisují starší při shodě)
                merged_df = pd.concat([existing_df, validated_df])
                merged_df = merged_df[~merged_df.index.duplicated(keep="last")].sort_index()
                self._atomic_write(file_path, validate_ohlcv_integrity(merged_df))
                return file_path
            except Exception as e:
                raise ValueError(f"Existující data {file_path} nelze bezpečně sloučit; soubor zachován.") from e

        self._atomic_write(file_path, validated_df)
        return file_path

    @staticmethod
    def _atomic_write(file_path: Path, df: pd.DataFrame) -> None:
        fd, temporary = tempfile.mkstemp(dir=file_path.parent, suffix=".parquet.tmp")
        os.close(fd)
        try:
            df.to_parquet(temporary, engine="pyarrow", compression="snappy")
            os.replace(temporary, file_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Načte svíčky z lokálního Parquet úložiště.
        Volitelně filtruje podle start_date a end_date.
        """
        file_path = self._get_file_path(symbol, timeframe)
        if not file_path.exists():
            return None

        try:
            df = pd.read_parquet(file_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            df = df.sort_index()

            if start_date is not None:
                ts_start = pd.to_datetime(start_date)
                if df.index.tz is not None and ts_start.tzinfo is None:
                    ts_start = ts_start.tz_localize(df.index.tz)
                df = df[df.index >= ts_start]

            if end_date is not None:
                ts_end = pd.to_datetime(end_date)
                if df.index.tz is not None and ts_end.tzinfo is None:
                    ts_end = ts_end.tz_localize(df.index.tz)
                df = df[df.index <= ts_end]

            return df
        except Exception as exc:
            logger.error(f"Chyba při načítání Parquet dat z {file_path}: {exc}")
            return None

    def get_cached_symbols(self, timeframe: str) -> List[str]:
        """Vrátí seznam všech symbolů uložených pro daný timeframe."""
        dir_path = self.base_dir / timeframe.strip().lower()
        if not dir_path.exists():
            return []
        return sorted([f.stem for f in dir_path.glob("*.parquet")])


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
    Integruje ParquetBarStore pro lokální caching.
    """

    def __init__(self, interval: str = "5m", period: str = "2d", bar_store: Optional[ParquetBarStore] = None):
        self.interval = interval
        self.period = period
        self.bar_store = bar_store or ParquetBarStore()

    def _compute_from_df(self, ticker: str, df: pd.DataFrame) -> Optional[IntradaySnapshot]:
        """Vypočítá technický profil (VWAP, EMA 9/21, RSI, Volume Surge, ATR) z DataFrame svíček."""
        if df.empty or len(df) < 5:
            return None

        try:
            df = validate_ohlcv_integrity(df)
            # Yahoo intraday timestamps denote bar starts. Only complete bars.
            if df.index.tz is None:
                raise ValueError("Intraday data requires a timezone.")
            interval_minutes = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "90m": 90, "1h": 60}[self.interval]
            df = df[df.index + pd.Timedelta(minutes=interval_minutes) <= pd.Timestamp.now(tz="UTC")]
            if len(df) < 5:
                return None
            # 1. Výpočet VWAP (Volume-Weighted Average Price)
            from src.quant.features import CausalFeatureEngine
            current_vwap = float(CausalFeatureEngine.compute_session_vwap(df)["vwap"].iloc[-1])
            if not np.isfinite(current_vwap):
                return None

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
                timestamp=(df.index[-1] + pd.Timedelta(minutes=interval_minutes)).to_pydatetime(),
            )
        except Exception as exc:
            logger.debug(f"[Intraday Data] Chyba výpočtu indikátorů pro {ticker}: {exc}")
            return None

    def fetch_intraday_snapshot(self, ticker: str, use_cache: bool = False) -> Optional[IntradaySnapshot]:
        """Stáhne a spočítá kompletní intradenní technický profil pro jeden ticker."""
        clean = ticker.strip().upper()
        df = None

        if use_cache and self.bar_store.has_bars(clean, self.interval):
            df = self.bar_store.load_bars(clean, self.interval)

        if df is None or df.empty or len(df) < 5:
            try:
                df = yf.download(clean, period=self.period, interval=self.interval, progress=False)
                if df.empty or len(df) < 5:
                    return None

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Uložení do Parquet úložiště
                try:
                    self.bar_store.save_bars(clean, self.interval, df)
                except Exception as save_exc:
                    logger.debug(f"Nepodařilo se uložit data do cache pro {clean}: {save_exc}")

            except Exception as exc:
                logger.debug(f"[Intraday Data] Chyba při načítání 5m svíček pro {clean}: {exc}")
                return None

        return self._compute_from_df(clean, df)

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
                        # Uložení do cache
                        try:
                            self.bar_store.save_bars(t, self.interval, sub_df)
                        except Exception:
                            pass
                except Exception as sub_exc:
                    logger.debug(f"Chyba při zpracování batch tickeru {t}: {sub_exc}")

        except Exception as exc:
            logger.warning(f"Batch stahování intradenních dat selhalo ({exc}), fallback na sekvenční.")
            for t in clean_tickers:
                s = self.fetch_intraday_snapshot(t)
                if s:
                    results[s.ticker] = s

        return results
