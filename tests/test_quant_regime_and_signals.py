"""Unit testy pro Modul 2 (Regime Detector) a Modul 3 (Carver Signal Engine)."""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.quant.features import CausalFeatureEngine, StockFeatures
from src.quant.regime_detector import MarketRegime, MarketRegimeType, RegimeDetector
from src.quant.signal_engine import (
    CarverSignalEngine,
    IntradaySignal,
    IntradaySignalEngine,
    ORB15mStrategy,
    SignalAction,
    VWAPMeanReversionStrategy,
    VWAPPullbackStrategy,
)
from src.strategy.opportunity_scanner import OpportunityScanner


class TestQuantRegimeAndSignals(unittest.TestCase):
    def setUp(self):
        """Připraví syntetická data pro benchmark a akcie."""
        np.random.seed(42)
        n_bars = 300
        start_date = datetime(2025, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_bars)]

        # 1. Býčí klidný benchmark (SPY simulace)
        steps_bull = np.random.normal(loc=0.0008, scale=0.008, size=n_bars)
        price_bull = 400.0 * np.exp(np.cumsum(steps_bull))
        self.spy_bull_df = pd.DataFrame(
            {
                "Open": price_bull * 0.999,
                "High": price_bull * 1.005,
                "Low": price_bull * 0.995,
                "Close": price_bull,
                "Volume": np.random.uniform(5e7, 1e8, n_bars),
            },
            index=pd.DatetimeIndex(dates),
        )

        # 2. Panický krizový benchmark (prudký propad za posledních 5 dní)
        price_panic = price_bull.copy()
        price_panic[-5:] = price_panic[-6] * np.array([0.97, 0.94, 0.92, 0.90, 0.88])
        self.spy_panic_df = pd.DataFrame(
            {
                "Open": price_panic * 1.01,
                "High": price_panic * 1.02,
                "Low": price_panic * 0.97,
                "Close": price_panic,
                "Volume": np.random.uniform(1.5e8, 3e8, n_bars),
            },
            index=pd.DatetimeIndex(dates),
        )

        # 3. Akciové příznaky pro testování signálů
        self.strong_momentum_feat = StockFeatures(
            ticker="NVDA",
            timestamp=datetime.now(),
            current_price=120.0,
            momentum_1m=0.08,
            momentum_3m=0.25,
            vol_norm_momentum_3m=1.2,
            vol_norm_momentum_12m_1m=1.5,
            realized_vol_annualized=0.25,
            atr_20=3.0,
            atr_20_pct=2.5,
            mean_reversion_z_20=0.5,
            mean_reversion_z_50=0.8,
            volatility_compression_ratio=1.0,
            is_volatility_squeezed=False,
            is_volatility_expanding=False,
            rvol_20=1.2,
            price_to_sma200_ratio=1.15,
            sma200_slope_20d_pct=2.5,
            cross_sectional_rank=0.95,
        )

        self.oversold_rebound_feat = StockFeatures(
            ticker="AAPL",
            timestamp=datetime.now(),
            current_price=200.0,
            momentum_1m=-0.04,
            momentum_3m=0.05,
            vol_norm_momentum_3m=0.2,
            vol_norm_momentum_12m_1m=0.5,
            realized_vol_annualized=0.18,
            atr_20=3.5,
            atr_20_pct=1.75,
            mean_reversion_z_20=-1.8,  # Zdravé přeprodání
            mean_reversion_z_50=-1.2,
            volatility_compression_ratio=0.80,  # Squeeze
            is_volatility_squeezed=True,
            is_volatility_expanding=False,
            rvol_20=1.1,
            price_to_sma200_ratio=1.02,
            sma200_slope_20d_pct=0.8,
            cross_sectional_rank=0.50,
        )

        self.falling_knife_feat = StockFeatures(
            ticker="BAD_EARNINGS",
            timestamp=datetime.now(),
            current_price=50.0,
            momentum_1m=-0.35,
            momentum_3m=-0.40,
            vol_norm_momentum_3m=-2.0,
            vol_norm_momentum_12m_1m=-2.0,
            realized_vol_annualized=0.60,
            atr_20=5.0,
            atr_20_pct=10.0,
            mean_reversion_z_20=-4.2,  # Katastrofální pád
            mean_reversion_z_50=-3.8,
            volatility_compression_ratio=2.2,
            is_volatility_squeezed=False,
            is_volatility_expanding=True,
            rvol_20=4.5,  # Masivní panický objem
            price_to_sma200_ratio=0.65,
            sma200_slope_20d_pct=-8.0,
            cross_sectional_rank=0.02,
        )

    def test_regime_bull_quiet(self):
        """Ověření klasifikace BULL_QUIET a plného povolení rizika."""
        detector = RegimeDetector()
        regime = detector.detect_regime(self.spy_bull_df, benchmark_symbol="SPY")

        self.assertEqual(regime.regime, MarketRegimeType.BULL_QUIET)
        self.assertTrue(regime.allow_trading)
        self.assertEqual(regime.risk_multiplier, 1.0)
        self.assertGreater(regime.momentum_weight, regime.mean_reversion_weight)

    def test_regime_crisis_panic(self):
        """Ověření aktivace CRISIS_PANIC a tvrdého NO TRADE jističe."""
        detector = RegimeDetector()
        regime = detector.detect_regime(self.spy_panic_df, benchmark_symbol="SPY")

        self.assertEqual(regime.regime, MarketRegimeType.CRISIS_PANIC)
        self.assertFalse(regime.allow_trading)
        self.assertEqual(regime.risk_multiplier, 0.0)

    def test_signal_strong_momentum(self):
        """Ověření nákupního signálu pro silné momentum v klidném býčím trhu."""
        detector = RegimeDetector()
        regime = detector.detect_regime(self.spy_bull_df)
        engine = CarverSignalEngine()

        signal = engine.generate_signal(self.strong_momentum_feat, regime)
        self.assertEqual(signal.action, SignalAction.BUY)
        self.assertGreater(signal.composite_score, 0.35)
        self.assertGreater(signal.momentum_score, 0.5)

    def test_signal_oversold_rebound(self):
        """Ověření mean-reversion odrazu u zdravého aktiva."""
        # V bočním nebo mírně defenzivním trhu
        regime = MarketRegime(
            regime=MarketRegimeType.SIDEWAYS,
            allow_trading=True,
            risk_multiplier=0.7,
            momentum_weight=0.3,
            mean_reversion_weight=0.7,
        )
        engine = CarverSignalEngine()
        signal = engine.generate_signal(self.oversold_rebound_feat, regime)

        self.assertGreater(signal.mean_reversion_score, 0.5)
        self.assertIn(signal.action, [SignalAction.BUY, SignalAction.HOLD])

    def test_falling_knife_filter(self):
        """Ověření ochrany proti padajícímu noži (extrémní propad s obřím objemem)."""
        regime = MarketRegime(
            regime=MarketRegimeType.BULL_QUIET,
            allow_trading=True,
            risk_multiplier=1.0,
            momentum_weight=0.5,
            mean_reversion_weight=0.5,
        )
        engine = CarverSignalEngine()
        signal = engine.generate_signal(self.falling_knife_feat, regime)

        # Padající nůž NESMÍ vyvolat nákup!
        self.assertNotEqual(signal.action, SignalAction.BUY)
        self.assertLess(signal.mean_reversion_score, 0.0)
        self.assertIn(signal.action, [SignalAction.CLOSE, SignalAction.NO_TRADE])

    def test_crisis_panic_hard_veto(self):
        """V CRISIS_PANIC musí být všechny signály bez výjimky NO_TRADE."""
        detector = RegimeDetector()
        regime = detector.detect_regime(self.spy_panic_df)
        engine = CarverSignalEngine()

        signal = engine.generate_signal(self.strong_momentum_feat, regime)
        self.assertEqual(signal.action, SignalAction.NO_TRADE)
        self.assertEqual(signal.composite_score, 0.0)

    def test_orb_15m_strategy_breakout(self):
        """Ověření ORB breakout nákupu se splněním RVOL, VWAP a časového okna."""
        regime = MarketRegime(
            regime=MarketRegimeType.TRENDING_BULL,
            allow_trading=True,
            risk_multiplier=1.0,
        )
        dates = pd.date_range("2026-03-10 09:30", periods=10, freq="5min", tz="America/New_York")
        df = pd.DataFrame(
            {
                "Open": [100.0] * 10,
                "High": [101.0, 101.5, 101.8, 101.2, 103.5, 104.0, 104.2, 104.5, 104.8, 105.0],
                "Low": [99.5, 99.8, 99.7, 100.0, 101.0, 102.0, 102.5, 103.0, 103.5, 104.0],
                "Close": [100.5, 101.2, 101.5, 101.0, 103.0, 103.8, 104.0, 104.2, 104.5, 104.8],
                "Volume": [10000] * 10,
                "session": ["REGULAR"] * 10,
                "minutes_from_open": [i * 5 for i in range(10)],
                "vwap": [100.8] * 10,
                "orb_high": [np.nan, np.nan, np.nan, 101.8, 101.8, 101.8, 101.8, 101.8, 101.8, 101.8],
                "orb_low": [np.nan, np.nan, np.nan, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
                "orb_width": [np.nan, np.nan, np.nan, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3],
                "atr_14": [4.0] * 10,
                "rvol_at_time": [1.0, 1.0, 1.0, 1.2, 1.9, 2.0, 1.8, 1.5, 1.4, 1.3],
                "intraday_rs_zscore": [0.5] * 10,
            },
            index=dates,
        )
        strat = ORB15mStrategy()
        # Bar at idx 4 is at 09:50 (minutes_from_open = 20), Close = 103.0 > orb_high (101.8) and > vwap (100.8), RVOL = 1.9 >= 1.5
        sig = strat.evaluate_bar(df, regime, idx=4, ticker="TSLA")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.action, SignalAction.BUY)
        self.assertEqual(sig.strategy_name, "ORB_15M_MOMENTUM")
        self.assertEqual(sig.entry_price, 103.0)
        self.assertLess(sig.stop_loss, sig.entry_price)
        self.assertGreater(sig.take_profit, sig.entry_price)

        # Mimo povolené okno (> 90 minut, např. 11:30) musí vrátit None
        df_late = df.copy()
        df_late["minutes_from_open"] = 120
        self.assertIsNone(strat.evaluate_bar(df_late, regime, idx=4, ticker="TSLA"))

    def test_vwap_pullback_strategy(self):
        """Ověření nákupního signálu při pullbacku a odrazu od VWAP v býčím trendu."""
        regime = MarketRegime(
            regime=MarketRegimeType.TRENDING_BULL,
            allow_trading=True,
            risk_multiplier=1.0,
        )
        dates = pd.date_range("2026-03-10 10:00", periods=10, freq="5min", tz="America/New_York")
        df = pd.DataFrame(
            {
                "Open": [102.0, 101.8, 101.5, 101.2, 100.8, 100.2, 101.0, 101.5, 102.0, 102.2],
                "High": [102.5, 102.0, 101.8, 101.5, 101.0, 101.2, 101.5, 102.0, 102.5, 102.8],
                "Low": [101.5, 101.2, 101.0, 100.5, 100.2, 100.1, 100.8, 101.2, 101.5, 101.8],
                "Close": [101.8, 101.5, 101.2, 100.8, 100.2, 101.0, 101.4, 101.8, 102.2, 102.5],
                "Volume": [15000] * 10,
                "minutes_from_open": [30 + i * 5 for i in range(10)],
                "vwap": [100.0] * 10,
                "atr_14": [2.0] * 10,
                "intraday_rs_zscore": [0.8] * 10,
            },
            index=dates,
        )
        strat = VWAPPullbackStrategy()
        # Bar at idx=5: minutes_from_open=55, Low=100.1 <= 100.6 (vwap + 0.3*atr), Close=101.0 >= 100.0, Open=100.2
        sig = strat.evaluate_bar(df, regime, idx=5, ticker="NVDA")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.action, SignalAction.BUY)
        self.assertEqual(sig.strategy_name, "VWAP_TREND_PULLBACK")
        self.assertEqual(sig.entry_price, 101.0)
        self.assertLess(sig.stop_loss, 101.0)
        self.assertGreater(sig.take_profit, 101.0)

    def test_vwap_mean_reversion_strategy(self):
        """Ověření odrazu z přeprodaného -2 sigma pásma v bočním režimu CHOPPY_RANGE."""
        regime = MarketRegime(
            regime=MarketRegimeType.CHOPPY_RANGE,
            allow_trading=True,
            risk_multiplier=0.8,
        )
        dates = pd.date_range("2026-03-10 11:00", periods=10, freq="5min", tz="America/New_York")
        df = pd.DataFrame(
            {
                "Open": [97.0, 96.5, 96.0, 95.5, 94.2, 94.5, 95.5, 97.0, 98.0, 99.0],
                "High": [97.5, 97.0, 96.5, 95.8, 94.8, 96.0, 97.2, 98.2, 99.2, 100.2],
                "Low": [96.5, 96.0, 95.2, 94.0, 93.5, 93.8, 95.0, 96.5, 97.5, 98.5],
                "Close": [96.8, 96.2, 95.4, 94.2, 94.0, 95.8, 97.0, 98.0, 99.0, 100.0],
                "Volume": [8000] * 10,
                "minutes_from_open": [90 + i * 5 for i in range(10)],
                "vwap": [100.0] * 10,
                "vwap_lower_2": [94.5] * 10,
                "atr_14": [2.0] * 10,
                "rvol_at_time": [1.1] * 10,
            },
            index=dates,
        )
        strat = VWAPMeanReversionStrategy()
        # At idx=5: minutes_from_open=115, Low=93.8 <= 94.5, Close=95.8 > Open (94.5), Close < VWAP (100)
        sig = strat.evaluate_bar(df, regime, idx=5, ticker="AAPL")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.action, SignalAction.BUY)
        self.assertEqual(sig.strategy_name, "VWAP_BAND_REVERSION")
        self.assertEqual(sig.entry_price, 95.8)
        self.assertEqual(sig.take_profit, 100.0)

    def test_intraday_signal_engine_integration(self):
        """Ověření integrace všech strategií v IntradaySignalEngine a generování signálového DF."""
        engine = IntradaySignalEngine()
        regime = MarketRegime(
            regime=MarketRegimeType.TRENDING_BULL,
            allow_trading=True,
            risk_multiplier=1.0,
        )
        dates = pd.date_range("2026-03-10 09:30", periods=10, freq="5min", tz="America/New_York")
        df = pd.DataFrame(
            {
                "Open": [100.0] * 10,
                "High": [101.0, 101.5, 101.8, 101.2, 103.5, 104.0, 104.2, 104.5, 104.8, 105.0],
                "Low": [99.5, 99.8, 99.7, 100.0, 101.0, 102.0, 102.5, 103.0, 103.5, 104.0],
                "Close": [100.5, 101.2, 101.5, 101.0, 103.0, 103.8, 104.0, 104.2, 104.5, 104.8],
                "Volume": [10000] * 10,
                "session": ["REGULAR"] * 10,
                "minutes_from_open": [i * 5 for i in range(10)],
                "vwap": [100.8] * 10,
                "orb_high": [np.nan, np.nan, np.nan, 101.8, 101.8, 101.8, 101.8, 101.8, 101.8, 101.8],
                "orb_low": [np.nan, np.nan, np.nan, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
                "orb_width": [np.nan, np.nan, np.nan, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3],
                "atr_14": [4.0] * 10,
                "rvol_at_time": [1.0, 1.0, 1.0, 1.2, 1.9, 2.0, 1.8, 1.5, 1.4, 1.3],
                "intraday_rs_zscore": [0.5] * 10,
            },
            index=dates,
        )
        sig_df = engine.generate_signals_for_dataframe(df, regime, ticker="TSLA")
        self.assertEqual(len(sig_df), 10)
        self.assertIn("signal", sig_df.columns)
        self.assertIn("stop_loss", sig_df.columns)
        self.assertIn("take_profit", sig_df.columns)
        self.assertIn("strategy_name", sig_df.columns)
        # Bar 4 měl nákupní signál
        self.assertEqual(sig_df["signal"].iloc[4], 1)
        self.assertEqual(sig_df["strategy_name"].iloc[4], "ORB_15M_MOMENTUM")

    def test_opportunity_scanner_3step_pipeline(self):
        """Ověření 3stupňového výběru Stocks-In-Play skeneru."""
        scanner = OpportunityScanner()

        # 1. Test filtru likvidity a volatility
        daily_good = pd.DataFrame(
            {
                "Close": [150.0] * 25,
                "Volume": [1_000_000] * 25,
                "High": [155.0] * 25,
                "Low": [145.0] * 25,
            }
        )
        self.assertTrue(scanner.filter_liquidity_and_volatility(daily_good, "GOOD_STOCK"))

        # Akcie s příliš nízkou cenou (< 10 USD)
        daily_cheap = daily_good.copy()
        daily_cheap["Close"] = 5.0
        self.assertFalse(scanner.filter_liquidity_and_volatility(daily_cheap, "PENNY_STOCK"))

        # 2. Test Directional Bias
        df_bull = pd.DataFrame(
            {
                "Close": [100.0, 102.0, 104.0, 106.0, 107.0],
                "vwap": [100.0, 101.0, 102.0, 103.0, 104.0],
                "intraday_rs_zscore": [0.5, 0.8, 1.2, 1.5, 1.6],
            }
        )
        score_bull = scanner.compute_directional_bias(df_bull)
        self.assertGreater(score_bull, 0.4)

        df_bear = pd.DataFrame(
            {
                "Close": [106.0, 104.0, 102.0, 99.0, 98.0],
                "vwap": [105.0, 104.0, 103.0, 102.0, 101.0],
                "intraday_rs_zscore": [-0.5, -0.8, -1.2, -1.5, -1.6],
            }
        )
        score_bear = scanner.compute_directional_bias(df_bear)
        self.assertLess(score_bear, -0.4)

        # 3. Test kompletního scan_stocks_in_play
        candidates = scanner.scan_stocks_in_play(
            intraday_data_map={"BULL": df_bull, "BEAR": df_bear},
            daily_data_map={"BULL": daily_good, "BEAR": daily_good},
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["ticker"], "BULL")
        self.assertEqual(candidates[0]["direction"], "LONG")
        self.assertEqual(candidates[1]["ticker"], "BEAR")
        self.assertEqual(candidates[1]["direction"], "SHORT")


if __name__ == "__main__":
    unittest.main()

