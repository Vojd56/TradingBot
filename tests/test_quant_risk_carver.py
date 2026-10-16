"""Unit testy pro Modul 5: Carver Volatility Targeting Risk Engine."""

import unittest
from datetime import datetime

from src.quant.features import StockFeatures
from src.quant.prediction_model import ExpectedValueResult, TripleBarriers
from src.quant.regime_detector import MarketRegime, MarketRegimeType
from src.quant.risk_engine import CarverRiskEngine, PositionSizingResult
from src.quant.signal_engine import SignalAction


class TestQuantRiskCarver(unittest.TestCase):
    def setUp(self):
        """Příprava modelů a dat pro testy rizikového modulu."""
        self.risk_engine = CarverRiskEngine(
            base_annual_target_vol=0.15,
            max_position_weight_pct=12.0,
            max_risk_per_trade_pct=1.5,
            target_positions_count=10,
            inertia_buffer_pct=10.0,
        )

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

        # Klidné aktivum s nízkou volatilitou (ATR = 1.0)
        self.features_low_vol = StockFeatures(
            ticker="LOW_VOL",
            timestamp=datetime.now(),
            current_price=100.0,
            atr_20=1.0,
            atr_20_pct=1.0,
            realized_vol_annualized=0.12,
        )

        # Velmi volatilní aktivum (ATR = 5.0)
        self.features_high_vol = StockFeatures(
            ticker="HIGH_VOL",
            timestamp=datetime.now(),
            current_price=100.0,
            atr_20=5.0,
            atr_20_pct=5.0,
            realized_vol_annualized=0.45,
        )

        self.ev_approved = ExpectedValueResult(
            ticker="TEST",
            decision=SignalAction.BUY,
            p_win=0.55,
            p_loss=0.45,
            expected_gain_pct=4.0,
            expected_loss_pct=2.0,
            total_friction_cost_pct=0.25,
            ev_net_pct=1.05,
            hurdle_threshold_pct=0.50,
            passes_hurdle=True,
            barriers=TripleBarriers(
                entry_price=100.0,
                take_profit_price=104.0,
                stop_loss_price=98.0,
                max_holding_bars=20,
                take_profit_pct=4.0,
                stop_loss_pct=2.0,
                rr_ratio=2.0,
            ),
            rationale="Schváleno",
        )

    def test_volatility_targeting_inverse_scaling(self):
        """Aktivum s vyšší volatilitou musí dostat menší počet kusů než klidné aktivum."""
        equity = 100_000.0  # 100 000 CZK portfolio

        res_low_vol = self.risk_engine.calculate_position_size(
            self.ev_approved, self.features_low_vol, self.bull_regime, portfolio_equity=equity
        )
        res_high_vol = self.risk_engine.calculate_position_size(
            self.ev_approved, self.features_high_vol, self.bull_regime, portfolio_equity=equity
        )

        self.assertTrue(res_low_vol.approved_for_execution)
        self.assertTrue(res_high_vol.approved_for_execution)

        # Vyšší volatilita = méně kusů / nižší alokace
        self.assertGreater(res_low_vol.target_shares, res_high_vol.target_shares)

        # Obě pozice musí dodržet strop koncentrace (<= 12 %) a riziko SL (<= 1.5 %)
        self.assertLessEqual(res_low_vol.portfolio_weight_pct, 12.0)
        self.assertLessEqual(res_high_vol.portfolio_weight_pct, 12.0)
        self.assertLessEqual(res_low_vol.risk_per_trade_pct, 1.5)
        self.assertLessEqual(res_high_vol.risk_per_trade_pct, 1.5)

    def test_carver_inertia_buffer(self):
        """Pokud je změna stávající pozice menší než 10 %, rebalancování je potlačeno."""
        equity = 100_000.0
        # Zjistíme teoretický cíl
        initial_res = self.risk_engine.calculate_position_size(
            self.ev_approved, self.features_low_vol, self.bull_regime, portfolio_equity=equity
        )
        target = initial_res.target_shares

        # Pokud už držíme 96 % cíle (odchylka jen 4 % < 10 %)
        curr_shares = target * 0.96
        res_buffer = self.risk_engine.calculate_position_size(
            self.ev_approved, self.features_low_vol, self.bull_regime,
            portfolio_equity=equity, current_shares=curr_shares
        )

        self.assertFalse(res_buffer.is_rebalance_required)
        self.assertEqual(res_buffer.target_shares, round(curr_shares, 4))

        # Pokud držíme jen 80 % cíle (odchylka 20 % >= 10 %)
        curr_shares_far = target * 0.80
        res_rebalance = self.risk_engine.calculate_position_size(
            self.ev_approved, self.features_low_vol, self.bull_regime,
            portfolio_equity=equity, current_shares=curr_shares_far
        )
        self.assertTrue(res_rebalance.is_rebalance_required)

    def test_correlation_filtering(self):
        """Ověření penalizace a veta na základě korelace s existujícími pozicemi."""
        corr_matrix = {
            "CANDIDATE": {
                "HOLDING_A": 0.85,  # Příliš vysoká korelace (> 0.80) -> VETO
                "HOLDING_B": 0.40,
            },
            "MODERATE_CORR": {
                "HOLDING_A": 0.70,  # Zvýšená korelace (> 0.65) -> krácení na 50 %
            },
            "CLEAN": {
                "HOLDING_A": 0.20,  # Nízká korelace -> 100 %
            }
        }

        holdings = ["HOLDING_A", "HOLDING_B"]

        mult_veto = self.risk_engine.evaluate_correlation("CANDIDATE", holdings, corr_matrix)
        self.assertEqual(mult_veto, 0.0)

        mult_half = self.risk_engine.evaluate_correlation("MODERATE_CORR", holdings, corr_matrix)
        self.assertEqual(mult_half, 0.5)

        mult_clean = self.risk_engine.evaluate_correlation("CLEAN", holdings, corr_matrix)
        self.assertEqual(mult_clean, 1.0)

    def test_multicurrency_fx_position_sizing(self):
        """Ověření, že akcie kótované v GBp (pence) a EUR jsou v Carver sizingu správně přepočteny do USD."""
        equity = 57_000.0  # 57 000 USD
        fx_rates = {"EUR": 1.10, "GBP": 1.30, "CHF": 1.12, "USD": 1.0}

        # 1. Shell na LSE (SHEL.L): cena 3880 pence = 38.80 GBP = 50.44 USD
        feat_shell = StockFeatures(
            ticker="SHEL.L",
            timestamp=datetime.now(),
            current_price=3880.0,  # GBp (pence)
            atr_20=50.0,           # 50 pence
            atr_20_pct=1.3,
            realized_vol_annualized=0.20,
        )
        ev_shell = ExpectedValueResult(
            ticker="SHEL.L",
            decision=SignalAction.BUY,
            p_win=0.55,
            p_loss=0.45,
            expected_gain_pct=3.0,
            expected_loss_pct=1.5,
            total_friction_cost_pct=0.25,
            ev_net_pct=0.8,
            hurdle_threshold_pct=0.5,
            passes_hurdle=True,
            barriers=TripleBarriers(
                entry_price=3880.0,
                take_profit_price=3996.0,
                stop_loss_price=3821.0,
                max_holding_bars=20,
                take_profit_pct=3.0,
                stop_loss_pct=1.5,
                rr_ratio=2.0,
            ),
            rationale="OK",
        )

        res_shell = self.risk_engine.calculate_position_size(
            ev_shell, feat_shell, self.bull_regime, portfolio_equity=equity,
            available_cash=1000.0, fx_rates=fx_rates
        )
        # Cena akcie v USD je ~50.44 USD. Při rozpočtu 1000 USD * 0.95 = 950 USD musí koupit ~18.8 ks (nikoliv 0.013 ks!)
        self.assertGreater(res_shell.target_shares, 10.0)
        self.assertLess(res_shell.target_shares, 30.0)
        self.assertAlmostEqual(res_shell.target_value, res_shell.target_shares * (38.80 * 1.30), delta=5.0)

        # 2. TotalEnergies (TTE.PA): cena 80.0 EUR = 88.0 USD
        feat_tte = StockFeatures(
            ticker="TTE.PA",
            timestamp=datetime.now(),
            current_price=80.0,  # EUR
            atr_20=1.5,          # EUR
            atr_20_pct=1.9,
            realized_vol_annualized=0.22,
        )
        ev_tte = ExpectedValueResult(
            ticker="TTE.PA",
            decision=SignalAction.BUY,
            p_win=0.55,
            p_loss=0.45,
            expected_gain_pct=3.0,
            expected_loss_pct=1.5,
            total_friction_cost_pct=0.25,
            ev_net_pct=0.8,
            hurdle_threshold_pct=0.5,
            passes_hurdle=True,
            barriers=TripleBarriers(
                entry_price=80.0,
                take_profit_price=82.4,
                stop_loss_price=78.8,
                max_holding_bars=20,
                take_profit_pct=3.0,
                stop_loss_pct=1.5,
                rr_ratio=2.0,
            ),
            rationale="OK",
        )

        res_tte = self.risk_engine.calculate_position_size(
            ev_tte, feat_tte, self.bull_regime, portfolio_equity=equity,
            available_cash=1000.0, fx_rates=fx_rates
        )
        # Při rozpočtu 950 USD a ceně 88 USD musí koupit ~10.8 ks
        self.assertGreater(res_tte.target_shares, 8.0)
        self.assertLess(res_tte.target_shares, 15.0)


if __name__ == "__main__":
    unittest.main()

