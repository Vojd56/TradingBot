from datetime import datetime, time as dtime, timedelta
import logging
from typing import Any, Dict, Optional, Tuple, Union
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")
LOCAL_TZ = ZoneInfo("Europe/Prague")


def tag_market_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministicky označí obchodní seance v časovém pásmu America/New_York.
    
    Přidává sloupce:
    - session: 'PRE' (04:00–09:30), 'REGULAR' (09:30–16:00), 'POST' (16:00–20:00), 'CLOSED'
    - is_regular_session: bool (True výhradně pro REGULAR seanci v Po–Pá)
    - trading_day: str (datum seance ve formátu YYYY-MM-DD v America/New_York)
    - minutes_from_open: int (0 při 09:30, 5 při 09:35, ..., 390 při 16:00; -1 mimo regular session)
    
    Korektně ošetřuje přechody letního/zimního času (DST) díky ZoneInfo('America/New_York').
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Zajištění DatetimeIndexu
    if not isinstance(out.index, pd.DatetimeIndex):
        date_cols = [c for c in out.columns if str(c).lower() in ("datetime", "date", "timestamp", "time")]
        if date_cols:
            out[date_cols[0]] = pd.to_datetime(out[date_cols[0]])
            out = out.set_index(date_cols[0])
        else:
            out.index = pd.to_datetime(out.index)

    # Převod/lokalizace do America/New_York
    if out.index.tz is None:
        out.index = out.index.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
    else:
        out.index = out.index.tz_convert("America/New_York")

    weekdays = out.index.weekday  # 0=Po, 4=Pá, 5=So, 6=Ne
    hours = out.index.hour
    minutes = out.index.minute
    total_minutes = hours * 60 + minutes

    # 04:00 = 240, 09:30 = 570, 16:00 = 960, 20:00 = 1200
    is_weekday = weekdays < 5
    is_pre = is_weekday & (total_minutes >= 240) & (total_minutes < 570)
    is_reg = is_weekday & (total_minutes >= 570) & (total_minutes <= 960)
    is_post = is_weekday & (total_minutes > 960) & (total_minutes <= 1200)

    session_arr = np.full(len(out), "CLOSED", dtype=object)
    session_arr[is_pre] = "PRE"
    session_arr[is_reg] = "REGULAR"
    session_arr[is_post] = "POST"

    out["session"] = session_arr
    out["is_regular_session"] = is_reg
    out["trading_day"] = out.index.strftime("%Y-%m-%d")

    # minutes_from_open: 0 při 09:30, 390 při 16:00, -1 mimo regular
    mins_from_open = np.full(len(out), -1, dtype=int)
    mins_from_open[is_reg] = (total_minutes[is_reg] - 570)
    out["minutes_from_open"] = mins_from_open

    return out


class MarketHours:
    """
    Hlídač otevíracích hodin amerických a evropských trhů.
    Standardní hodiny NYSE/NASDAQ: Po–Pá, 09:30 až 16:00 Eastern Time.
    V CET/CEST to odpovídá 15:30 až 22:00.
    """

    US_OPEN_TIME = dtime(9, 30)
    US_CLOSE_TIME = dtime(16, 0)
    EU_OPEN_TIME = dtime(9, 0)
    EU_CLOSE_TIME = dtime(17, 30)

    @classmethod
    def is_us_market_open(cls) -> bool:
        """Vrátí True, pokud je americký akciový trh (NYSE/NASDAQ) právě otevřen (15:30 - 22:00 CET)."""
        now_ny = datetime.now(NY_TZ)
        if now_ny.weekday() >= 5:  # Sobota (5) nebo Neděle (6)
            return False

        current_time = now_ny.time()
        return cls.US_OPEN_TIME <= current_time <= cls.US_CLOSE_TIME

    @classmethod
    def is_eu_market_open(cls) -> bool:
        """Vrátí True, pokud je evropský akciový trh (LSE, XETRA, Euronext) právě otevřen (09:00 - 17:30 CET)."""
        now_local = datetime.now(LOCAL_TZ)
        if now_local.weekday() >= 5:
            return False

        current_time = now_local.time()
        return cls.EU_OPEN_TIME <= current_time <= cls.EU_CLOSE_TIME

    @classmethod
    def is_market_open(cls) -> bool:
        """Vrátí True, pokud je otevřen alespoň jeden z hlavních trhů (Evropa 09:00-17:30 nebo USA 15:30-22:00 CET)."""
        return cls.is_eu_market_open() or cls.is_us_market_open()

    @classmethod
    def is_ticker_market_open(cls, ticker: str) -> bool:
        """
        Zjistí, zda je burza pro daný ticker v tuto chvíli otevřená.
        Evropské burzy (.DE, .PA, .AS, .L, .MC, .MI, .SW): 09:00 - 17:30 CET
        Americké burzy (NYSE, NASDAQ, ETF): 15:30 - 22:00 CET
        """
        upper = ticker.upper()
        if any(upper.endswith(sfx) for sfx in [".DE", ".PA", ".AS", ".L", ".MC", ".MI", ".SW"]):
            return cls.is_eu_market_open()
        return cls.is_us_market_open()

    @classmethod
    def get_time_until_open(cls) -> timedelta:
        """Vrátí čas zbývající do příštího otevření kteréhokoliv trhu."""
        now_local = datetime.now(LOCAL_TZ)
        today_eu_open = datetime.combine(now_local.date(), cls.EU_OPEN_TIME, tzinfo=LOCAL_TZ)

        if now_local.weekday() < 5 and now_local < today_eu_open:
            return today_eu_open - now_local

        # Hledáme příští všední den v 09:00 CET
        next_day = now_local + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)

        next_open = datetime.combine(next_day.date(), cls.EU_OPEN_TIME, tzinfo=LOCAL_TZ)
        return next_open - now_local

    @classmethod
    def get_market_status(cls) -> Dict[str, Any]:
        """Vrátí detailní slovník o stavu obou trhů (Evropa i USA) pro logování a UI."""
        eu_open = cls.is_eu_market_open()
        us_open = cls.is_us_market_open()
        any_open = eu_open or us_open

        now_ny = datetime.now(NY_TZ)
        now_local = datetime.now(LOCAL_TZ)

        status_parts = []
        if eu_open:
            status_parts.append("EVROPA OTEVŘENA (Xetra/LSE do 17:30)")
        if us_open:
            status_parts.append("USA OTEVŘENO (NYSE/NASDAQ do 22:00)")

        if any_open:
            status_text = " | ".join(status_parts)
        else:
            time_to_open = cls.get_time_until_open()
            hours, remainder = divmod(int(time_to_open.total_seconds()), 3600)
            mins, _ = divmod(remainder, 60)
            status_text = f"ZAVŘENO (příští otevření Evropy za {hours}h {mins}m v 09:00 CET)"

        return {
            "is_open": any_open,
            "eu_open": eu_open,
            "us_open": us_open,
            "status_text": status_text,
            "ny_time": now_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "local_time": now_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "seconds_until_open": cls.get_time_until_open().total_seconds() if not any_open else 0.0,
        }

    @staticmethod
    def tag_market_session(df: pd.DataFrame) -> pd.DataFrame:
        """Alias metody pro označení obchodních seancí v DataFrame."""
        return tag_market_session(df)
