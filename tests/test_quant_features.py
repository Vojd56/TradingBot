"""Unit testy pro Modul 1: Causal & Dimensionless Feature Engineering."""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.quant.features import CausalFeatureEngine, StockFeatures


class TestQuantFeatures(unittest.TestCase):
    def setUp(self):
        """Vygeneruje syntetická OHLCV data pro testování."""
        np.random.seed(42)
        n_bars = 300
        start_date = datetime(2025, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_bars)]

        # Simulace rostoucího trendu s volatilitou
        steps = np.random.normal(loc=0.001, scale=0.015, size=n_bars)
        price_path = 100.0 * np.exp(np.cumsum(steps))

        highs = price_path * (1.0 + np.abs(np.random.normal(0, 0.008, n_bars)))
        lows = price_path * (1.0 - np.abs(np.random.normal(0, 0.008, n_bars)))
        opens = price_path * (1.0 + np.random.normal(0, 0.003, n_bars))
        volumes = np.random.lognormal(mean=14, sigma=0.5, size=n_bars)

        self.df_trend_up = pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": price_path,
                "Volume": volumes,
            },
            index=pd.DatetimeIndex(dates),
        )

        # Druhý ticker s klesajícím trendem
        steps_down = np.random.normal(loc=-0.002, scale=0.02, size=n_bars)
        price_path_down = 100.0 * np.exp(np.cumsum(steps_down))
        self.df_trend_down = pd.DataFrame(
            {
                "Open": price_path_down * 1.002,
                "High": price_path_down * 1.01,
                "Low": price_path_down * 0.99,
                "Close": price_path_down,
                "Volume": volumes,
            },
            index=pd.DatetimeIndex(dates),
        )

    def test_atr_calculation(self):
        """Ověření Wilderovy ATR kalkulace."""
        atr = CausalFeatureEngine.calculate_atr(self.df_trend_up, period=14)
        self.assertEqual(len(atr), len(self.df_trend_up))
        self.assertFalse(np.isnan(atr.iloc[-1]))
        self.assertGreater(atr.iloc[-1], 0)

    def test_feature_extraction_trend_up(self):
        """Ověření extrakce bezrozměrných příznaků pro rostoucí aktivum."""
        features = CausalFeatureEngine.compute_single_ticker_features(
            self.df_trend_up, ticker="AAPL", min_bars_required=60
        )
        self.assertIsNotNone(features)
        self.assertIsInstance(features, StockFeatures)
        self.assertEqual(features.ticker, "AAPL")
        self.assertGreater(features.current_price, 0)
        self.assertGreater(features.realized_vol_annualized, 0)
        self.assertGreater(features.atr_20, 0)
        self.assertGreater(features.atr_20_pct, 0)
        self.assertGreater(features.rvol_20, 0)
        self.assertGreater(features.volatility_compression_ratio, 0)

    def test_insufficient_data_rejection(self):
        """Ověření, že při méně než min_bars_required vrátí None."""
        df_short = self.df_trend_up.iloc[:40].copy()
        features = CausalFeatureEngine.compute_single_ticker_features(
            df_short, ticker="SHORT_TICKER", min_bars_required=60
        )
        self.assertIsNone(features)

    def test_cross_sectional_ranking(self):
        """Ověření průřezového řazení (silnější aktivum musí mít vyšší percentil)."""
        f_up = CausalFeatureEngine.compute_single_ticker_features(self.df_trend_up, "UP")
        f_down = CausalFeatureEngine.compute_single_ticker_features(self.df_trend_down, "DOWN")

        features_map = {"UP": f_up, "DOWN": f_down}
        ranked = CausalFeatureEngine.compute_cross_sectional_ranks(features_map)

        self.assertGreater(ranked["UP"].cross_sectional_rank, ranked["DOWN"].cross_sectional_rank)
        self.assertGreaterEqual(ranked["DOWN"].cross_sectional_rank, 0.0)
        self.assertLessEqual(ranked["UP"].cross_sectional_rank, 1.0)

    def _generate_synthetic_intraday_data(self, n_days: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Pomocná metoda pro vygenerování realistických 5m dat pro US seanci."""
        start_date = pd.Timestamp("2025-01-06 09:30", tz="America/New_York")
        all_bars = []
        all_spy_bars = []

        np.random.seed(123)
        current_price = 100.0
        spy_price = 450.0

        for day in range(n_days):
            day_start = start_date + pd.Timedelta(days=day)
            # 78 pětiminutových svíček na regular seanci (09:30 až 15:55)
            times = pd.date_range(day_start, periods=78, freq="5min")
            for t in times:
                ret = np.random.normal(0.0002, 0.003)
                current_price *= np.exp(ret)
                high = current_price * (1.0 + abs(np.random.normal(0, 0.0015)))
                low = current_price * (1.0 - abs(np.random.normal(0, 0.0015)))
                open_p = current_price * (1.0 + np.random.normal(0, 0.0005))
                vol = float(np.random.lognormal(mean=9.0, sigma=0.4))
                all_bars.append({
                    "Open": open_p, "High": high, "Low": low, "Close": current_price, "Volume": vol
                })

                spy_ret = 0.7 * ret + np.random.normal(0.0001, 0.0015)
                spy_price *= np.exp(spy_ret)
                all_spy_bars.append({
                    "Open": spy_price, "High": spy_price * 1.0008, "Low": spy_price * 0.9992,
                    "Close": spy_price, "Volume": vol * 5.0
                })

        idx = pd.DatetimeIndex([b for day in range(n_days) for b in pd.date_range(start_date + pd.Timedelta(days=day), periods=78, freq="5min")])
        df_stock = pd.DataFrame(all_bars, index=idx)
        df_spy = pd.DataFrame(all_spy_bars, index=idx)
        return df_stock, df_spy

    def test_session_vwap_reset_and_bands(self):
        """Ověření, že Session VWAP se resetuje každé ráno v 09:30 a pásma jsou správně seřazena."""
        df_stock, _ = self._generate_synthetic_intraday_data(n_days=3)
        vwap_df = CausalFeatureEngine.compute_session_vwap(df_stock)

        self.assertIn("vwap", vwap_df.columns)
        self.assertIn("vwap_upper_1", vwap_df.columns)
        self.assertIn("vwap_lower_1", vwap_df.columns)

        # Na začátku každého dne (09:30, minutes_from_open == 0) je VWAP roven první typické ceně dne
        day1_0930 = vwap_df.index[0]
        day2_0930 = vwap_df.index[78]
        tp_day2_0930 = (df_stock.loc[day2_0930, "High"] + df_stock.loc[day2_0930, "Low"] + df_stock.loc[day2_0930, "Close"]) / 3.0

        self.assertAlmostEqual(vwap_df.loc[day2_0930, "vwap"], tp_day2_0930, places=4)
        # Pásma musí dodržovat relaci: lower_2 <= lower_1 <= vwap <= upper_1 <= upper_2
        valid = vwap_df.dropna()
        self.assertTrue((valid["vwap_upper_2"] >= valid["vwap_upper_1"]).all())
        self.assertTrue((valid["vwap_upper_1"] >= valid["vwap"]).all())
        self.assertTrue((valid["vwap"] >= valid["vwap_lower_1"]).all())
        self.assertTrue((valid["vwap_lower_1"] >= valid["vwap_lower_2"]).all())

    def test_opening_range_no_lookahead(self):
        """Ověření, že ORB 15m vrací NaN před 09:45 EST (žádný look-ahead leakage!)."""
        df_stock, _ = self._generate_synthetic_intraday_data(n_days=2)
        orb_df = CausalFeatureEngine.compute_opening_range(df_stock, orb_minutes=15)

        # První 3 svíčky dne (09:30, 09:35, 09:40) MUSÍ mít hodnoty NaN
        self.assertTrue(np.isnan(orb_df["orb_high"].iloc[0]))
        self.assertTrue(np.isnan(orb_df["orb_high"].iloc[1]))
        self.assertTrue(np.isnan(orb_df["orb_high"].iloc[2]))
        self.assertTrue(np.isnan(orb_df["orb_low"].iloc[0]))

        # 4. svíčka (09:45 EST, index 3) už musí mít pevné hodnoty prvních 15 minut
        expected_high = df_stock["High"].iloc[:3].max()
        expected_low = df_stock["Low"].iloc[:3].min()
        self.assertFalse(np.isnan(orb_df["orb_high"].iloc[3]))
        self.assertAlmostEqual(orb_df["orb_high"].iloc[3], expected_high, places=5)
        self.assertAlmostEqual(orb_df["orb_low"].iloc[3], expected_low, places=5)
        self.assertAlmostEqual(orb_df["orb_width"].iloc[3], expected_high - expected_low, places=5)

    def test_causal_leakage_prefix_invariance(self):
        """
        Klíčový Causal Leakage Test:
        Příznaky spočítané na prefixu dat df[:k] na indexu k-1 se MUSÍ exaktně shodovat
        s příznaky spočítanými na celém datasetu df.
        """
        df_stock, df_spy = self._generate_synthetic_intraday_data(n_days=4)
        full_features = CausalFeatureEngine.compute_intraday_features(df_stock, benchmark_df=df_spy)

        # Zvolíme několik bodů k:
        # k=20 (uprostřed dne 1), k=85 (v 10:05 dne 2 po ORB), k=180 (odpoledne dne 3)
        check_points = [20, 85, 180]
        test_columns = ["vwap", "vwap_std", "orb_high", "orb_low", "rvol_at_time", "atr_14", "intraday_rs_zscore"]

        for k in check_points:
            prefix_stock = df_stock.iloc[:k].copy()
            prefix_spy = df_spy.iloc[:k].copy()
            prefix_features = CausalFeatureEngine.compute_intraday_features(prefix_stock, benchmark_df=prefix_spy)

            for col in test_columns:
                val_prefix = prefix_features[col].iloc[-1]
                val_full = full_features[col].iloc[k - 1]

                if np.isnan(val_full):
                    self.assertTrue(
                        np.isnan(val_prefix),
                        f"Sloupec {col} na k={k} měl být NaN v obou případech, ale prefix={val_prefix}"
                    )
                else:
                    self.assertAlmostEqual(
                        val_prefix,
                        val_full,
                        places=6,
                        msg=f"Causal leakage detekován pro sloupec {col} na k={k}: prefix={val_prefix} vs full={val_full}",
                    )


if __name__ == "__main__":
    unittest.main()

