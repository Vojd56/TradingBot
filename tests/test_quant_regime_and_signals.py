"""Unit testy pro Modul 2 (Regime Detector) a Modul 3 (Carver Signal Engine)."""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.quant.features import CausalFeatureEngine, StockFeatures
from src.quant.regime_detector import MarketRegime, MarketRegimeType, RegimeDetector
from src.quant.signal_engine import CarverSignalEngine, SignalAction


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


if __name__ == "__main__":
    unittest.main()
