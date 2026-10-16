"""Unit testy pro Etapu 1 (Tasky AG-00 a AG-01):
1. Zabezpečení provozních režimů (config.py & validate_live_mode).
2. Model transakčních nákladů (TransactionCostConfig).
3. Deterministický session kalendář a DST handling (market_hours.py).
4. Validace integrity OHLCV a ParquetBarStore (intraday_data.py).
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from config import AppSettings, TradingMode, TransactionCostConfig
from src.strategy.intraday_data import ParquetBarStore, validate_ohlcv_integrity
from src.strategy.market_hours import MarketHours, tag_market_session


class TestConfigSafety(unittest.TestCase):
    def test_default_trading_mode_is_safe(self):
        """Ověření, že výchozí režim je vždy bezpečný PAPER."""
        cfg = AppSettings(
            TRADING_MODE="paper",
            T212_API_KEY="test_key",
        )
        self.assertEqual(cfg.trading_mode, TradingMode.PAPER)
        self.assertFalse(cfg.trading_mode == TradingMode.LIVE)

    def test_demo_alias_maps_to_paper(self):
        """Ověření zpětné kompatibility TRADING_MODE='demo' -> TradingMode.PAPER."""
        cfg = AppSettings(
            TRADING_MODE="demo",
            T212_API_KEY="test_key",
        )
        self.assertEqual(cfg.trading_mode, TradingMode.PAPER)

    def test_live_mode_blocked_without_confirm_flag(self):
        """Pokus o spuštění v LIVE režimu bez CONFIRM_LIVE_TRADING=true musí vyhodit RuntimeError."""
        old_val = os.environ.get("CONFIRM_LIVE_TRADING")
        try:
            if "CONFIRM_LIVE_TRADING" in os.environ:
                del os.environ["CONFIRM_LIVE_TRADING"]

            cfg = AppSettings(
                TRADING_MODE="live",
                CONFIRM_LIVE_TRADING="false",
                I_UNDERSTAND_LIVE_RISK="no",
                T212_API_KEY="test_key",
            )
            with self.assertRaises(RuntimeError) as ctx:
                cfg.validate_live_mode()
            self.assertIn("CONFIRM_LIVE_TRADING=true", str(ctx.exception))
        finally:
            if old_val is not None:
                os.environ["CONFIRM_LIVE_TRADING"] = old_val

    def test_live_mode_allowed_with_explicit_confirm(self):
        """Režim LIVE je povolen pouze při explicitním nastavení obou pojistek."""
        cfg = AppSettings(
            TRADING_MODE="live",
            CONFIRM_LIVE_TRADING="true",
            I_UNDERSTAND_LIVE_RISK="yes",
            T212_API_KEY="test_key",
        )
        # Nesmí vyhodit výjimku
        cfg.validate_live_mode()
        self.assertEqual(cfg.trading_mode, TradingMode.LIVE)

    def test_transaction_cost_config(self):
        """Ověření výpočtu transakčních nákladů včetně FX poplatku."""
        cost = TransactionCostConfig(
            spread_bps=5.0,
            slippage_bps=5.0,
            commission_bps=0.0,
            fx_conversion_bps=15.0,
        )
        # S měnovou konverzí (např. T212): 5 + 5 + 0 + 15 = 25 bps = 0.25 %
        self.assertEqual(cost.total_cost_bps(is_cross_currency=True), 25.0)
        self.assertEqual(cost.total_cost_pct(is_cross_currency=True), 0.25)
        self.assertEqual(cost.roundtrip_cost_pct(is_cross_currency=True), 0.50)

        # Bez měnové konverze (čistý USD účet): 5 + 5 + 0 = 10 bps = 0.10 %
        self.assertEqual(cost.total_cost_bps(is_cross_currency=False), 10.0)
        self.assertEqual(cost.total_cost_pct(is_cross_currency=False), 0.10)
        self.assertEqual(cost.roundtrip_cost_pct(is_cross_currency=False), 0.20)


class TestMarketSessionTagging(unittest.TestCase):
    def test_tag_market_session_regular_hours(self):
        """Ověření značkování REGULAR seance (09:30–16:00 America/New_York) a minutes_from_open."""
        ny_tz = ZoneInfo("America/New_York")
        # Pondělí 2025-06-02
        dates = [
            datetime(2025, 6, 2, 8, 30, tzinfo=ny_tz),   # PRE
            datetime(2025, 6, 2, 9, 30, tzinfo=ny_tz),   # REGULAR start
            datetime(2025, 6, 2, 9, 45, tzinfo=ny_tz),   # REGULAR +15m
            datetime(2025, 6, 2, 12, 0, tzinfo=ny_tz),   # REGULAR poledne
            datetime(2025, 6, 2, 16, 0, tzinfo=ny_tz),   # REGULAR end
            datetime(2025, 6, 2, 16, 30, tzinfo=ny_tz),  # POST
            datetime(2025, 6, 2, 22, 0, tzinfo=ny_tz),   # CLOSED
            datetime(2025, 6, 7, 12, 0, tzinfo=ny_tz),   # Sobota -> CLOSED
        ]
        df = pd.DataFrame(
            {"Close": [100.0] * len(dates)},
            index=pd.DatetimeIndex(dates),
        )

        tagged = tag_market_session(df)

        self.assertEqual(tagged["session"].iloc[0], "PRE")
        self.assertFalse(tagged["is_regular_session"].iloc[0])
        self.assertEqual(tagged["minutes_from_open"].iloc[0], -1)

        # 09:30
        self.assertEqual(tagged["session"].iloc[1], "REGULAR")
        self.assertTrue(tagged["is_regular_session"].iloc[1])
        self.assertEqual(tagged["minutes_from_open"].iloc[1], 0)

        # 09:45
        self.assertEqual(tagged["minutes_from_open"].iloc[2], 15)

        # 12:00 -> 2.5h = 150 min
        self.assertEqual(tagged["minutes_from_open"].iloc[3], 150)

        # 16:00 -> 390 min
        self.assertEqual(tagged["session"].iloc[4], "REGULAR")
        self.assertEqual(tagged["minutes_from_open"].iloc[4], 390)

        # 16:30 -> POST
        self.assertEqual(tagged["session"].iloc[5], "POST")
        self.assertFalse(tagged["is_regular_session"].iloc[5])

        # Sobota -> CLOSED
        self.assertEqual(tagged["session"].iloc[7], "CLOSED")

    def test_dst_transition_handling(self):
        """Ověření, že US DST přechod v březnu nezpůsobí posun 09:30–16:00 okna."""
        # V pátek 7. března 2025 je New York v EST (UTC-5)
        # V pondělí 10. března 2025 je New York v EDT (UTC-4)
        t_est = datetime(2025, 3, 7, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        t_edt = datetime(2025, 3, 10, 9, 30, tzinfo=ZoneInfo("America/New_York"))

        df = pd.DataFrame({"Close": [100.0, 101.0]}, index=pd.DatetimeIndex([t_est, t_edt]))
        tagged = tag_market_session(df)

        self.assertTrue(tagged["is_regular_session"].iloc[0])
        self.assertEqual(tagged["minutes_from_open"].iloc[0], 0)

        self.assertTrue(tagged["is_regular_session"].iloc[1])
        self.assertEqual(tagged["minutes_from_open"].iloc[1], 0)


class TestIntradayDataAndParquet(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = ParquetBarStore(base_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_ohlcv_integrity(self):
        """Ověření opravy poškozených svíček (High < Low, záporný objem, duplicitní časy)."""
        dates = [
            datetime(2025, 1, 1, 10, 0),
            datetime(2025, 1, 1, 10, 0),  # Duplikát
            datetime(2025, 1, 1, 10, 5),
        ]
        df_corrupt = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 105.0],
                "High": [95.0, 95.0, 104.0],   # Poškozeno: High < Open!
                "Low": [110.0, 110.0, 106.0],  # Poškozeno: Low > High!
                "Close": [102.0, 102.0, 105.0],
                "Volume": [-500.0, 100.0, 200.0],  # Záporný objem
            },
            index=pd.DatetimeIndex(dates),
        )

        with self.assertRaises(ValueError):
            validate_ohlcv_integrity(df_corrupt)

    def test_parquet_store_save_and_load(self):
        """Ověření uložení do Parquetu a rychlého opětovného načtení."""
        dates = [datetime(2025, 1, 1, 10, 0) + timedelta(minutes=5 * i) for i in range(10)]
        df = pd.DataFrame(
            {
                "Open": [100.0 + i for i in range(10)],
                "High": [102.0 + i for i in range(10)],
                "Low": [99.0 + i for i in range(10)],
                "Close": [101.0 + i for i in range(10)],
                "Volume": [1000.0 + 100 * i for i in range(10)],
            },
            index=pd.DatetimeIndex(dates),
        )

        saved_path = self.store.save_bars("NVDA", "5m", df)
        self.assertTrue(saved_path.exists())
        self.assertTrue(self.store.has_bars("NVDA", "5m"))

        loaded = self.store.load_bars("NVDA", "5m")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 10)
        pd.testing.assert_series_equal(loaded["Close"], df["Close"], check_names=False)

    def test_parquet_store_incremental_merge(self):
        """Ověření inkrementálního přidání nových svíček bez duplicit."""
        dates1 = [datetime(2025, 1, 1, 10, 0) + timedelta(minutes=5 * i) for i in range(5)]
        dates2 = [datetime(2025, 1, 1, 10, 15) + timedelta(minutes=5 * i) for i in range(5)]  # Překryv

        df1 = pd.DataFrame(
            {"Open": [100.0] * 5, "High": [101.0] * 5, "Low": [99.0] * 5, "Close": [100.5] * 5, "Volume": [100.0] * 5},
            index=pd.DatetimeIndex(dates1),
        )
        df2 = pd.DataFrame(
            {"Open": [110.0] * 5, "High": [111.0] * 5, "Low": [109.0] * 5, "Close": [110.5] * 5, "Volume": [200.0] * 5},
            index=pd.DatetimeIndex(dates2),
        )

        self.store.save_bars("AAPL", "5m", df1)
        self.store.save_bars("AAPL", "5m", df2)

        merged = self.store.load_bars("AAPL", "5m")
        # Původně 5 svíček (10:00..10:20), druhé 5 svíček (10:15..10:35) -> celkem 8 unikátních svíček
        self.assertEqual(len(merged), 8)
        self.assertEqual(len(merged[~merged.index.duplicated()]), 8)


if __name__ == "__main__":
    unittest.main()
