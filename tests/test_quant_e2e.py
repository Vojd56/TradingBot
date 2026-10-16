"""End-to-End integrační test pro celou 8modulovou kvantitativní architekturu."""

from datetime import datetime, timedelta
import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from src.quant.signal_engine import SignalAction
from src.strategy.quant_trader import QuantTraderEngine


class TestQuantE2E(unittest.TestCase):
    def setUp(self):
        """Příprava deterministických syntetických tržních dat."""
        np.random.seed(42)
        n_bars = 260
        start_date = datetime(2025, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_bars)]

        def make_df(trend: float, vol: float, start_price: float):
            steps = np.random.normal(loc=trend, scale=vol, size=n_bars)
            path = start_price * np.exp(np.cumsum(steps))
            return pd.DataFrame(
                {
                    "Open": path * 0.998,
                    "High": path * 1.008,
                    "Low": path * 0.992,
                    "Close": path,
                    "Volume": np.random.uniform(1e6, 5e6, n_bars),
                },
                index=pd.DatetimeIndex(dates),
            )

        # Benchmark SPY (stabilní růstový trend)
        self.spy_df = make_df(trend=0.0006, vol=0.009, start_price=500.0)

        # Silný lídr (NVDA)
        self.nvda_df = make_df(trend=0.0015, vol=0.015, start_price=100.0)

        # Stabilní růst (AAPL)
        self.aapl_df = make_df(trend=0.0005, vol=0.010, start_price=200.0)

        # Klesající aktivum (DOWN)
        self.down_df = make_df(trend=-0.001, vol=0.015, start_price=80.0)

        self.mock_data = MagicMock()

        def mock_get_history(ticker, period="1y", interval="1d"):
            if ticker == "SPY":
                return self.spy_df
            elif ticker == "NVDA":
                return self.nvda_df
            elif ticker == "AAPL":
                return self.aapl_df
            elif ticker == "DOWN":
                return self.down_df
            return self.aapl_df

        self.mock_data.get_history.side_effect = mock_get_history

    def test_full_quant_cycle_execution(self):
        """Ověření kompletního běhu kvantitativního cyklu od benchmarku po exekuci."""
        engine = QuantTraderEngine(
            watchlist=["NVDA", "AAPL", "DOWN"],
            benchmark_symbol="SPY",
            dry_run=True,
            data_provider=self.mock_data,
        )

        engine.get_market_fx_rates = MagicMock(return_value={"USD": 1.0, "EUR": 1.08, "GBP": 1.3, "CHF": 1.12})
        report = engine.run_cycle()

        self.assertIsNotNone(report)
        self.assertEqual(report.benchmark_symbol, "SPY")
        self.assertIn(report.regime.regime.value, ["BULL_QUIET", "BULL_VOLATILE", "SIDEWAYS"])
        self.assertFalse(report.monitoring_status.is_halted)
        self.assertEqual(report.evaluated_tickers_count, 3)

        # Ověření, že pro všechny tituly existují signály a EV výsledky
        for ticker in ["NVDA", "AAPL", "DOWN"]:
            self.assertIn(ticker, report.signals)
            self.assertIn(ticker, report.ev_results)

        # Klesající titul DOWN nesmí mít BUY signál
        self.assertNotEqual(report.signals["DOWN"].action, SignalAction.BUY)

        # Pokud byla exekuována jakákoliv objednávka, musí být BUY s R:R >= 2.0
        for order in report.executed_orders:
            self.assertEqual(order.side, "BUY")
            self.assertGreater(order.quantity, 0)
            self.assertIsNotNone(order.stop_loss_price)
            self.assertIsNotNone(order.take_profit_price)
            # Ověření R:R
            sl_dist = order.executed_price - order.stop_loss_price
            tp_dist = order.take_profit_price - order.executed_price
            rr = tp_dist / sl_dist
            self.assertGreaterEqual(rr, 1.99)

    def test_quant_cycle_stop_loss_exit(self):
        """Ověření, že otevřená pozice pod Stop Lossem vygeneruje SELL příkaz."""
        engine = QuantTraderEngine(
            watchlist=["NVDA", "AAPL"],
            benchmark_symbol="SPY",
            dry_run=True,
            data_provider=self.mock_data,
        )
        engine.get_market_fx_rates = MagicMock(return_value={"USD": 1.0})
        # AAPL má nákupní kontext se stop-lossem 320 USD, ale aktuální cena je 308.29 USD -> SL zasažen!
        engine.get_portfolio_state = MagicMock(return_value=(100000.0, 50000.0, {"AAPL": 10.0}))
        engine.client.get_submission_contexts = MagicMock(return_value={"AAPL_US_EQ": {"stop_loss": 320.0, "take_profit": 400.0}})

        report = engine.run_cycle()

        exit_orders = [o for o in report.executed_orders if o.side == "SELL"]
        self.assertTrue(len(exit_orders) > 0, "Měl být exekuován alespoň jeden SELL příkaz pro Stop Loss")
        self.assertEqual(exit_orders[0].ticker, "AAPL")
        self.assertEqual(exit_orders[0].quantity, 10.0)

    def test_quant_cycle_correlation_veto(self):
        """Kandidát s korelací > 0.80 vůči existující pozici musí být vetován."""
        engine = QuantTraderEngine(
            watchlist=["NVDA", "AAPL"],
            benchmark_symbol="SPY",
            dry_run=True,
            data_provider=self.mock_data,
        )
        engine.get_market_fx_rates = MagicMock(return_value={"USD": 1.0})
        # Držíme již pozici v NVDA
        engine.get_portfolio_state = MagicMock(return_value=(100000.0, 50000.0, {"NVDA": 20.0}))
        # Nastavíme, aby korelační filtr hlásil 0.0 (veto při korelaci > 0.80) pro AAPL
        engine.risk_engine.evaluate_correlation = MagicMock(return_value=0.0)

        report = engine.run_cycle()

        # Žádný nákup AAPL nesmí projít
        aapl_buys = [o for o in report.executed_orders if o.ticker == "AAPL" and o.side == "BUY"]
        self.assertEqual(len(aapl_buys), 0, "AAPL měl být zablokován korelačním filtrem (veto)")

    def test_quant_cycle_daily_drawdown_kill_switch(self):
        """Při denním propadu ekvity >= 3 % musí jistič okamžitě zastavit cyklus."""
        engine = QuantTraderEngine(
            watchlist=["NVDA", "AAPL"],
            benchmark_symbol="SPY",
            dry_run=True,
            data_provider=self.mock_data,
        )
        engine.get_market_fx_rates = MagicMock(return_value={"USD": 1.0})
        # Nastavíme počáteční ekvitu na 100 000 USD
        engine.daily_start_equity = 100000.0
        engine.ath_equity = 100000.0
        engine.last_trading_day = datetime.now().strftime("%Y-%m-%d")
        # Aktuální stav hlásí 96 000 USD (-4 % propad)
        engine.get_portfolio_state = MagicMock(return_value=(96000.0, 96000.0, {}))

        report = engine.run_cycle()

        self.assertTrue(report.monitoring_status.is_halted)
        self.assertEqual(report.monitoring_status.triggered_switch, "KILL_SWITCH_1_DAILY_LOSS")
        self.assertEqual(report.orders_executed_count, 0)


if __name__ == "__main__":
    unittest.main()
