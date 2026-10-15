"""Unit testy pro Modul 6 (Execution Engine) a Modul 7 (Monitoring & Kill-Switches)."""

import unittest
from datetime import datetime, timedelta

from src.quant.execution_engine import ExecutionEngine, OrderSide
from src.quant.monitoring import CusumDriftDetector, QuantMonitoringSystem
from src.quant.prediction_model import TripleBarriers
from src.quant.risk_engine import PositionSizingResult


class TestQuantExecutionAndMonitoring(unittest.TestCase):
    def setUp(self):
        """Příprava instancí."""
        self.exec_engine = ExecutionEngine(strategy_id="TEST_STRAT", is_live=False)
        self.monitoring = QuantMonitoringSystem()

        self.barriers = TripleBarriers(
            entry_price=100.0,
            take_profit_price=105.0,
            stop_loss_price=97.5,
            max_holding_bars=20,
            take_profit_pct=5.0,
            stop_loss_pct=2.5,
            rr_ratio=2.0,
        )

        self.sizing = PositionSizingResult(
            ticker="AAPL",
            target_shares=10.0,
            target_value=1000.0,
            portfolio_weight_pct=5.0,
            risk_per_trade_cash=25.0,
            risk_per_trade_pct=0.125,
            current_shares=0.0,
            shares_diff=10.0,
            is_rebalance_required=True,
            approved_for_execution=True,
            rationale="Test nákup",
        )

    def test_idempotent_order_id_generation(self):
        """Ověření deterministického generování client_order_id."""
        ts = datetime(2025, 6, 1, 10, 30)
        id1 = ExecutionEngine.generate_client_order_id("STRAT", "AAPL", ts, "BUY", 10.0)
        id2 = ExecutionEngine.generate_client_order_id("STRAT", "AAPL", ts, "BUY", 10.0)
        id_diff_qty = ExecutionEngine.generate_client_order_id("STRAT", "AAPL", ts, "BUY", 10.5)

        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id_diff_qty)
        self.assertEqual(len(id1), 24)

    def test_pre_trade_reconciliation_cash_check(self):
        """Pre-trade kontrola musí zamítnout nákup při nedostatku hotovosti."""
        # Máme jen 500 CZK, ale nákup vyžaduje ~1000 CZK
        ok, reason = self.exec_engine.pre_trade_reconciliation(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10.0,
            estimated_price=100.0,
            available_cash=500.0,
        )
        self.assertFalse(ok)
        self.assertIn("Nedostatek hotovosti", reason)

        # Máme 2000 CZK -> OK
        ok_cash, _ = self.exec_engine.pre_trade_reconciliation(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10.0,
            estimated_price=100.0,
            available_cash=2000.0,
        )
        self.assertTrue(ok_cash)

    def test_idempotent_order_execution_deduplication(self):
        """Opakované odeslání stejného příkazu v témže baru nesmí vytvořit duplicitu."""
        ts = datetime(2025, 6, 1, 10, 30)
        order1 = self.exec_engine.execute_sizing_order(self.sizing, self.barriers, available_cash=5000.0, timestamp=ts)
        self.assertIsNotNone(order1)
        self.assertEqual(order1.status, "FILLED")

        # Druhé volání se stejnými parametry
        order2 = self.exec_engine.execute_sizing_order(self.sizing, self.barriers, available_cash=5000.0, timestamp=ts)
        self.assertEqual(order1.client_order_id, order2.client_order_id)
        # V logu musí být pouze jeden záznam
        self.assertEqual(len(self.exec_engine.executed_orders_log), 1)

    def test_kill_switch_1_unusual_loss(self):
        """Jistič 1: Denní ztráta přes 3 % aktivuje HALT."""
        status = self.monitoring.evaluate_all(daily_pnl_pct=-3.5)
        self.assertTrue(status.is_halted)
        self.assertEqual(status.triggered_switch, "KILL_SWITCH_1_DAILY_LOSS")

    def test_kill_switch_2_extreme_volatility(self):
        """Jistič 2: Denní pohyb indexu přes 4 % aktivuje HALT."""
        status = self.monitoring.evaluate_all(benchmark_daily_change_pct=-4.5)
        self.assertTrue(status.is_halted)
        self.assertEqual(status.triggered_switch, "KILL_SWITCH_2_EXTREME_VOLATILITY")

    def test_kill_switch_4_portfolio_desync(self):
        """Jistič 4: Nesoulad držených kusů mezi interním trackerem a brokerem."""
        internal = {"AAPL": 10.0, "MSFT": 5.0}
        broker = {"AAPL": 10.0, "MSFT": 0.0}  # Broker hlásí 0 kusů MSFT!

        status = self.monitoring.evaluate_all(internal_shares=internal, broker_shares=broker)
        self.assertTrue(status.is_halted)
        self.assertEqual(status.triggered_switch, "KILL_SWITCH_4_PORTFOLIO_DESYNC")

    def test_kill_switch_6_api_loop_breaker(self):
        """Jistič 6: 3 chyby API za 60 sekund odpojí exekuci."""
        now = datetime.now()
        self.monitoring.record_api_error(now)
        self.monitoring.record_api_error(now + timedelta(seconds=10))
        k6 = self.monitoring.record_api_error(now + timedelta(seconds=20))

        self.assertIsNotNone(k6)
        self.assertEqual(k6[0], "KILL_SWITCH_6_API_LOOP_BREAKER")

    def test_cusum_drift_detection(self):
        """CUSUM detektor odhalí soustavné zaostávání skutečnosti za očekávaným EV."""
        cusum = CusumDriftDetector(allowance_k=0.2, threshold_h=2.0)
        # Série 6 obchodů se ztrátou -2 % při očekávání +1 %
        for _ in range(5):
            is_drift, _ = cusum.update(realized_return_pct=-2.0, expected_ev_pct=1.0)
        self.assertTrue(is_drift)


if __name__ == "__main__":
    unittest.main()
