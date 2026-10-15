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


if __name__ == "__main__":
    unittest.main()
