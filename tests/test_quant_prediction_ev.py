"""Unit testy pro Modul 4: Triple-Barrier Method & EV_net Prediction Model."""

import unittest
from datetime import datetime

from src.quant.features import StockFeatures
from src.quant.prediction_model import PredictionModel, TripleBarriers
from src.quant.regime_detector import MarketRegime, MarketRegimeType
from src.quant.signal_engine import QuantSignal, SignalAction


class TestQuantPredictionEV(unittest.TestCase):
    def setUp(self):
        """Příprava instancí a dat pro testy."""
        self.bull_regime = MarketRegime(
            regime=MarketRegimeType.BULL_QUIET,
            allow_trading=True,
            risk_multiplier=1.0,
            momentum_weight=0.75,
            mean_reversion_weight=0.25,
        )

        self.panic_regime = MarketRegime(
            regime=MarketRegimeType.CRISIS_PANIC,
            allow_trading=False,
            risk_multiplier=0.0,
            momentum_weight=0.0,
            mean_reversion_weight=0.0,
        )

        self.stock_features = StockFeatures(
            ticker="MSFT",
            timestamp=datetime.now(),
            current_price=400.0,
            momentum_1m=0.05,
            momentum_3m=0.15,
            vol_norm_momentum_3m=1.0,
            vol_norm_momentum_12m_1m=1.2,
            realized_vol_annualized=0.20,
            atr_20=8.0,
            atr_20_pct=2.0,
            mean_reversion_z_20=0.2,
            mean_reversion_z_50=0.5,
            volatility_compression_ratio=1.0,
            is_volatility_squeezed=False,
            is_volatility_expanding=False,
            rvol_20=1.1,
            price_to_sma200_ratio=1.08,
            sma200_slope_20d_pct=1.5,
            cross_sectional_rank=0.85,
        )

        self.strong_buy_signal = QuantSignal(
            ticker="MSFT",
            action=SignalAction.BUY,
            composite_score=0.65,
            momentum_score=0.75,
            mean_reversion_score=0.35,
            features=self.stock_features,
            regime=MarketRegimeType.BULL_QUIET,
            is_long_only_compliant=True,
            rationale="Silný signál",
        )

        self.weak_buy_signal = QuantSignal(
            ticker="MSFT",
            action=SignalAction.BUY,
            composite_score=0.36,  # Těsně nad prahem, slabší hrana
            momentum_score=0.40,
            mean_reversion_score=0.20,
            features=self.stock_features,
            regime=MarketRegimeType.BULL_QUIET,
            is_long_only_compliant=True,
            rationale="Slabý nákup",
        )

    def test_triple_barriers_rr_ratio(self):
        """Ověření, že generované bariéry vždy striktně dodržují R:R >= 2.0 dle pravidel."""
        model = PredictionModel(min_m_sl=1.2, min_m_tp=2.5)
        barriers = model.generate_barriers(self.stock_features, self.bull_regime)

        self.assertGreaterEqual(barriers.rr_ratio, 2.0)
        self.assertLess(barriers.stop_loss_price, barriers.entry_price)
        self.assertGreater(barriers.take_profit_price, barriers.entry_price)
        self.assertEqual(barriers.max_holding_bars, 20)

    def test_friction_cost_estimation(self):
        """Ověření odhadu frikčních nákladů (spread + FX + slippage)."""
        model = PredictionModel(fx_fee_pct=0.15)
        costs = model.estimate_friction_costs(self.stock_features)

        self.assertGreater(costs, 0.15)  # Musí být minimálně vyšší než samotný FX poplatek
        self.assertLess(costs, 1.0)      # U likvidní akcie by nemělo přesáhnout 1 %

    def test_calibrated_win_probability_bounds(self):
        """Ověření, že kalibrovaná pravděpodobnost zůstává v realistických mezích (žádné halucinace 90%+)."""
        model = PredictionModel()
        barriers = model.generate_barriers(self.stock_features, self.bull_regime)
        p_win = model.calibrate_win_probability(self.strong_buy_signal, barriers)

        self.assertGreaterEqual(p_win, 0.35)
        self.assertLessEqual(p_win, 0.65)

    def test_strong_signal_passes_hurdle(self):
        """Silný nákupní signál s vysokým R:R a EV_net musí projít rozhodovací bránou."""
        model = PredictionModel()
        res = model.evaluate(self.strong_buy_signal, self.bull_regime)

        self.assertEqual(res.decision, SignalAction.BUY)
        self.assertTrue(res.passes_hurdle)
        self.assertGreater(res.ev_net_pct, res.hurdle_threshold_pct)

    def test_panic_regime_rejects_to_no_trade(self):
        """V panickém režimu musí evaluate vrátit výchozí stav NO_TRADE."""
        model = PredictionModel()
        res = model.evaluate(self.strong_buy_signal, self.panic_regime)

        self.assertEqual(res.decision, SignalAction.NO_TRADE)
        self.assertFalse(res.passes_hurdle)


if __name__ == "__main__":
    unittest.main()
