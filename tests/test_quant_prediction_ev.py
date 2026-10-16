"""Unit testy pro Modul 4: Triple-Barrier Method & EV_net Prediction Model."""

import unittest
from datetime import datetime
import numpy as np
import pandas as pd

from src.quant.features import StockFeatures
from src.quant.prediction_model import (
    MetaLabelingClassifier,
    PredictionModel,
    PurgedTimeSeriesSplit,
    TripleBarriers,
    create_triple_barrier_labels,
)
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

    def test_create_triple_barrier_labels(self):
        """Ověření generování Triple-Barrier labelů včetně pesimistické SL kolize."""
        # Vytvoříme 20 barů
        df = pd.DataFrame(
            {
                "Open": [100.0] * 20,
                "High": [101.0] * 20,
                "Low": [99.0] * 20,
                "Close": [100.0] * 20,
                "atr_14": [2.0] * 20,
            }
        )
        # Signál na baru 0 (exekuce na baru 1 za Open=100.0)
        # TP = 100 + 2*2 = 104.0, SL = 100 - 1*2 = 98.0
        signals = pd.Series(0, index=range(20))
        signals.iloc[0] = 1

        # Případ A: Bar 3 zasáhne TP (High=105.0)
        df_tp = df.copy()
        df_tp.loc[3, "High"] = 105.0
        labels_tp = create_triple_barrier_labels(df_tp, signals, tp_atr_mult=2.0, sl_atr_mult=1.0, max_bars=10)
        self.assertEqual(len(labels_tp), 1)
        self.assertEqual(labels_tp["label"].iloc[0], 1)
        self.assertEqual(labels_tp["exit_reason"].iloc[0], "TAKE_PROFIT")
        self.assertEqual(labels_tp["realized_r"].iloc[0], 2.0)

        # Případ B: Bar 3 zasáhne SL (Low=97.0)
        df_sl = df.copy()
        df_sl.loc[3, "Low"] = 97.0
        labels_sl = create_triple_barrier_labels(df_sl, signals, tp_atr_mult=2.0, sl_atr_mult=1.0, max_bars=10)
        self.assertEqual(labels_sl["label"].iloc[0], 0)
        self.assertEqual(labels_sl["exit_reason"].iloc[0], "STOP_LOSS")
        self.assertEqual(labels_sl["realized_r"].iloc[0], -1.0)

        # Případ C: Kolize na baru 3 (Low=97.0 a High=105.0 současně) -> pesimisticky SL
        df_col = df.copy()
        df_col.loc[3, "Low"] = 97.0
        df_col.loc[3, "High"] = 105.0
        labels_col = create_triple_barrier_labels(df_col, signals, tp_atr_mult=2.0, sl_atr_mult=1.0, max_bars=10)
        self.assertEqual(labels_col["label"].iloc[0], 0)
        self.assertEqual(labels_col["exit_reason"].iloc[0], "STOP_LOSS")
        self.assertEqual(labels_col["realized_r"].iloc[0], -1.0)

    def test_purged_time_series_split(self):
        """Ověření nulového překryvu, purging obchodů a aplikace embarga."""
        n_samples = 100
        X = pd.DataFrame(np.zeros((n_samples, 4)))
        splitter = PurgedTimeSeriesSplit(n_splits=5, embargo_bars=10)

        # Vytvoříme událost obchodu zasahující z trainu do prvního test splitu (např. split 1: test 20..40)
        events = pd.DataFrame(
            [
                {"entry_idx": 15, "exit_idx": 25},  # Zasahuje do testovacího okna [20..40]
            ]
        )

        for train_idx, test_idx in splitter.split(X, events=events):
            # 1. Striktně disjunktní množiny
            overlap = set(train_idx).intersection(set(test_idx))
            self.assertEqual(len(overlap), 0)

            # 2. Embargo kontrola (žádný train index těsně za test_idx v rozmezí embargo_bars)
            test_max = max(test_idx)
            embargoed_range = set(range(test_max + 1, min(n_samples, test_max + 1 + 10)))
            embargo_violation = embargoed_range.intersection(set(train_idx))
            self.assertEqual(len(embargo_violation), 0)

    def test_meta_labeling_classifier(self):
        """Ověření fitování regularizovaného meta-modelu a predikce pravděpodobností."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 4)
        # Lineární target
        y = (X[:, 0] * 1.5 + X[:, 1] * 0.8 > 0).astype(int)

        clf = MetaLabelingClassifier(l2_reg=0.5, max_iter=150)
        clf.fit(X, y)
        self.assertTrue(clf.is_fitted)

        probs = clf.predict_proba(X)
        self.assertEqual(len(probs), n)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

        # Test výpočtu OOS profit factoru
        realized_r = np.where(y == 1, 2.0, -1.0)
        pf = clf.evaluate_oos_profit_factor(y, realized_r, probs, threshold=0.5)
        self.assertGreater(pf, 0.0)

    def test_prediction_model_with_ml_filter(self):
        """Ověření přepínače use_ml_meta_filter v PredictionModel."""
        # Natrénujeme meta-model preferující pozitivní příznaky
        clf = MetaLabelingClassifier()
        X_mock = np.array([
            [1.0, 1.2, 0.5, 2.0, 1.5, 0.7],
            [-1.0, -1.2, -2.0, 4.0, 2.5, -0.5],
        ])
        y_mock = np.array([1, 0])
        clf.fit(X_mock, y_mock)

        # 1. S vypnutým ML filtrem
        model_rule = PredictionModel(use_ml_meta_filter=False)
        barriers = model_rule.generate_barriers(self.stock_features, self.bull_regime)
        p_rule = model_rule.calibrate_win_probability(self.strong_buy_signal, barriers)

        # 2. Se zapnutým ML filtrem
        model_ml = PredictionModel(use_ml_meta_filter=True, meta_classifier=clf)
        p_ml = model_ml.calibrate_win_probability(self.strong_buy_signal, barriers)

        self.assertGreaterEqual(p_ml, 0.30)
        self.assertLessEqual(p_ml, 0.65)


if __name__ == "__main__":
    unittest.main()

