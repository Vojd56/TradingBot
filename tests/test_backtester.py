"""Unit testy pro Modul AG-03: Event-Driven Backtester a Walk-Forward Evaluator."""

import unittest
from datetime import datetime
import numpy as np
import pandas as pd

from config import TransactionCostConfig
from src.quant.backtester import (
    ExitReason,
    IntradayBacktester,
    TradeSide,
    WalkForwardEvaluator,
)


class TestIntradayBacktester(unittest.TestCase):
    def setUp(self):
        """Vygeneruje syntetická 5m intradenní data pro 3 obchodní dny."""
        np.random.seed(42)
        start_date = pd.Timestamp("2025-01-06 09:30", tz="America/New_York")
        n_days = 3
        bars = []

        price = 100.0
        for day in range(n_days):
            day_start = start_date + pd.Timedelta(days=day)
            times = pd.date_range(day_start, periods=78, freq="5min")
            for t in times:
                price *= np.exp(np.random.normal(0.0001, 0.002))
                high = price * 1.002
                low = price * 0.998
                open_p = price * 1.0005
                close_p = price
                vol = 5000.0
                bars.append({
                    "Open": open_p, "High": high, "Low": low, "Close": close_p, "Volume": vol
                })

        idx = pd.DatetimeIndex([
            b for d in range(n_days) for b in pd.date_range(start_date + pd.Timedelta(days=d), periods=78, freq="5min")
        ])
        self.df = pd.DataFrame(bars, index=idx)

    def test_execution_on_next_bar_open_with_costs(self):
        """
        Pravidlo 1: Signál na svíčce t se exekuuje na Open svíčky t+1 + spread + skluz,
        NIKOLIV na Close svíčky t.
        """
        cost_cfg = TransactionCostConfig(spread_bps=10.0, slippage_bps=10.0, commission_bps=0.0, fx_conversion_bps=0.0)
        # Celkové náklady = 20 bps = 0.20 %
        backtester = IntradayBacktester(cost_config=cost_cfg, initial_capital=100_000.0)

        # Vygenerujeme signál přesně na svíčce index=5 (09:55 dne 1)
        signals = pd.Series(0, index=self.df.index)
        signals.iloc[5] = 1

        res = backtester.run(self.df, signals=signals, symbol="TEST")
        self.assertEqual(res.total_trades, 1)

        trade = res.trades[0]
        # Vstup musel proběhnout na svíčce index=6 (10:00)
        self.assertEqual(trade.entry_bar_idx, 6)
        expected_raw_open = float(self.df["Open"].iloc[6])
        self.assertAlmostEqual(trade.raw_entry_price, expected_raw_open, places=4)

        # Skutečná nákupní cena musela být zhoršena o 20 bps (+0.20 %)
        expected_net_entry = expected_raw_open * (1.0 + 0.0020)
        self.assertAlmostEqual(trade.entry_price, expected_net_entry, places=4)

        # Ověření, že nákupní cena NENÍ rovna Close svíčky 5
        self.assertNotEqual(trade.raw_entry_price, self.df["Close"].iloc[5])

    def test_pessimistic_intra_bar_collision_triggers_stop_loss(self):
        """
        Pravidlo 2: Pokud ve stejné svíčce dojde k zasažení Low <= SL i High >= TP,
        backtester VŽDY vyhodnotí STOP LOSS (pesimistická zásada ochrany kapitálu).
        """
        df_collision = self.df.copy()
        # Signál na baru 10, vstup na baru 11
        signals = pd.Series(0, index=df_collision.index)
        signals.iloc[10] = 1

        entry_open = df_collision["Open"].iloc[11]
        sl = entry_open * 0.98  # -2.0 % SL
        tp = entry_open * 1.04  # +4.0 % TP

        # Na baru 12 vytvoříme extrémní svíčku, která zasáhne jak SL tak TP
        df_collision.loc[df_collision.index[12], "High"] = tp * 1.01  # proráží TP
        df_collision.loc[df_collision.index[12], "Low"] = sl * 0.99   # proráží SL

        sl_series = pd.Series(sl, index=df_collision.index)
        tp_series = pd.Series(tp, index=df_collision.index)

        backtester = IntradayBacktester(initial_capital=100_000.0)
        res = backtester.run(df_collision, signals=signals, sl_series=sl_series, tp_series=tp_series)

        self.assertEqual(res.total_trades, 1)
        trade = res.trades[0]
        self.assertEqual(trade.exit_bar_idx, 12)
        # Výsledek MUSÍ být STOP_LOSS
        self.assertEqual(trade.exit_reason, ExitReason.STOP_LOSS)
        self.assertAlmostEqual(trade.raw_exit_price, sl, places=3)
        self.assertLess(trade.net_pnl, 0.0)

    def test_eod_flatten_leaves_zero_overnight_positions(self):
        """
        Pravidlo 3: Všechny pozice musí být v 15:55 EST automaticky uzavřeny na Close.
        Žádná pozice nesmí zůstat otevřena přes noc.
        """
        signals = pd.Series(0, index=self.df.index)
        # Otevřeme pozici v 15:00 dne 1 (bar 66)
        signals.iloc[66] = 1

        backtester = IntradayBacktester(
            initial_capital=100_000.0,
            eod_flatten_minute=385,  # 15:55 EST
            max_holding_bars=50,     # Dlouhý time-stop, aby nezasáhl před EOD
        )
        res = backtester.run(self.df, signals=signals)

        self.assertEqual(res.total_trades, 1)
        trade = res.trades[0]

        # Poslední svíčka dne 1 je index 77 (15:55 EST)
        self.assertEqual(trade.exit_bar_idx, 77)
        self.assertEqual(trade.exit_reason, ExitReason.EOD_FLATTEN)
        # Cena výstupu musí odpovídat Close 15:55 svíčky
        expected_exit_raw = float(self.df["Close"].iloc[77])
        self.assertAlmostEqual(trade.raw_exit_price, expected_exit_raw, places=4)

    def test_time_stop_execution(self):
        """
        Pravidlo 4: Pokud pozice po max_holding_bars nezasáhne SL ani TP,
        ukončí se s důvodem TIME_STOP.
        """
        signals = pd.Series(0, index=self.df.index)
        signals.iloc[10] = 1  # Vstup na 11

        backtester = IntradayBacktester(
            initial_capital=100_000.0,
            max_holding_bars=5,
        )
        res = backtester.run(self.df, signals=signals)
        self.assertEqual(res.total_trades, 1)
        trade = res.trades[0]
        self.assertEqual(trade.holding_bars, 5)
        self.assertEqual(trade.exit_reason, ExitReason.TIME_STOP)

    def test_cost_deductions_lower_net_pnl(self):
        """Ověření, že model transakčních nákladů správně snižuje čistý výnos oproti hrubému."""
        cost_cfg = TransactionCostConfig(spread_bps=10.0, slippage_bps=10.0, commission_bps=5.0, fx_conversion_bps=15.0)
        # 40 bps jednosměrně = 80 bps roundtrip (0.80 %)
        backtester = IntradayBacktester(cost_config=cost_cfg, is_cross_currency=True)

        signals = pd.Series(0, index=self.df.index)
        signals.iloc[10] = 1

        res = backtester.run(self.df, signals=signals)
        self.assertEqual(res.total_trades, 1)
        trade = res.trades[0]

        self.assertGreater(trade.total_cost_paid, 0.0)
        self.assertLess(trade.net_pnl, trade.gross_pnl)
        self.assertAlmostEqual(trade.gross_pnl - trade.total_cost_paid, trade.net_pnl, delta=0.5)

    def test_walk_forward_evaluator_rolling_splits(self):
        """
        Ověření Walk-Forward validačního enginu s rozdělením oken, embargem
        a Bootstrap intervaly spolehlivosti.
        """
        evaluator = WalkForwardEvaluator(
            backtester=IntradayBacktester(initial_capital=100_000.0),
            in_sample_days=1,
            embargo_days=1,
            out_of_sample_days=1,
            n_bootstrap=200,
        )

        def simple_signal_gen(sub_df):
            # Signál na každém 10. baru
            s = pd.Series(0, index=sub_df.index)
            s.iloc[::15] = 1
            return s

        wf_res = evaluator.evaluate_walk_forward(self.df, simple_signal_gen, symbol="WFO_STOCK")

        self.assertIn("summary", wf_res)
        self.assertIn("folds", wf_res)
        self.assertIn("bootstrap_95_ci", wf_res)
        self.assertGreater(wf_res["folds_count"], 0)

        ci = wf_res["bootstrap_95_ci"]
        self.assertIn("mean", ci)
        self.assertIn("ci_lower", ci)
        self.assertIn("ci_upper", ci)


if __name__ == "__main__":
    unittest.main()
