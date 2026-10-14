from datetime import datetime, time as dtime, timedelta
import logging
from typing import Dict, Any, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")
LOCAL_TZ = ZoneInfo("Europe/Prague")


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
