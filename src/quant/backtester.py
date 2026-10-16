"""Modul: Realistický Event-Driven Backtester a Walk-Forward Evaluator.

Implementuje striktně kauzální simulaci intradenního obchodování dle Sekce 3 a 11 v ArgonReview:
1. Exekuce na následující svíčce (Next-Bar Open Fill):
   Signál na Close[t] je vyplněn na Open[t+1] s přičtením spreadu, skluzu a poplatků.
2. Pesimistické řešení intra-bar kolize (Collision Rule):
   Pokud na svíčce t+k platí Low <= StopLoss i High >= TakeProfit, obchod VŽDY končí na StopLossu.
3. Vynucený EOD Flatten:
   Všechny otevřené pozice se v 15:55 EST (nebo před koncem regular seance) uzavírají na Close.
   Žádná pozice nezůstává otevřena přes noc (nulové overnight gap riziko).
4. Time-Stop:
   Ukončení pozice po vypršení max_holding_bars, pokud nebyl zasažen SL ani TP.
5. Walk-Forward Validace:
   Rolling okna In-Sample -> Embargo -> Out-of-Sample s Bootstrap 95% intervaly spolehlivosti.
"""

from datetime import datetime
from enum import Enum
import logging
import math
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from config import TransactionCostConfig
from src.strategy.market_hours import tag_market_session

logger = logging.getLogger(__name__)


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TIME_STOP = "TIME_STOP"
    EOD_FLATTEN = "EOD_FLATTEN"


class TradeRecord(BaseModel):
    """Záznam o dokončeném simulovaném obchodu."""
    symbol: str
    side: TradeSide = TradeSide.LONG
    entry_bar_idx: int
    entry_time: datetime
    raw_entry_price: float
    entry_price: float
    shares: float
    exit_bar_idx: int
    exit_time: datetime
    raw_exit_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    initial_risk_cash: float
    exit_reason: ExitReason
    gross_pnl: float
    net_pnl: float
    net_return_pct: float
    pnl_r: float
    holding_bars: int
    total_cost_paid: float


class BacktestResult(BaseModel):
    """Souhrnné metriky a výsledky backtestu."""
    symbol: str = "PORTFOLIO"
    initial_capital: float
    final_capital: float
    net_pnl: float
    net_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    avg_trade_bps: float
    avg_r_multiple: float
    max_drawdown_pct: float
    max_drawdown_cash: float
    sharpe_ratio: float
    sortino_ratio: float
    trades: List[TradeRecord] = []
    equity_curve: List[float] = []

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["trades"] = [t.model_dump() for t in self.trades]
        return d


class ActivePosition:
    """Interní stav otevřené pozice během bar-by-bar simulace."""
    def __init__(
        self,
        symbol: str,
        side: TradeSide,
        entry_bar_idx: int,
        entry_time: datetime,
        raw_entry_price: float,
        entry_price: float,
        shares: float,
        stop_loss: float,
        take_profit: float,
        initial_risk_cash: float,
        entry_cost_paid: float,
    ):
        self.symbol = symbol
        self.side = side
        self.entry_bar_idx = entry_bar_idx
        self.entry_time = entry_time
        self.raw_entry_price = raw_entry_price
        self.entry_price = entry_price
        self.shares = shares
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.initial_risk_cash = max(1e-6, initial_risk_cash)
        self.entry_cost_paid = entry_cost_paid
        self.holding_bars = 0


class IntradayBacktester:
    """
    Realistický bar-by-bar event-driven backtester pro intradenní strategie.
    """

    def __init__(
        self,
        cost_config: Optional[TransactionCostConfig] = None,
        initial_capital: float = 100_000.0,
        max_risk_per_trade_pct: float = 0.005,
        max_position_notional_pct: float = 0.20,
        max_holding_bars: int = 8,
        eod_flatten_minute: int = 385,
        is_cross_currency: bool = False,
    ):
        self.cost_config = cost_config or TransactionCostConfig()
        self.initial_capital = initial_capital
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_position_notional_pct = max_position_notional_pct
        self.max_holding_bars = max_holding_bars
        self.eod_flatten_minute = eod_flatten_minute
        self.is_cross_currency = is_cross_currency

    def run(
        self,
        df: pd.DataFrame,
        signals: Optional[pd.Series] = None,
        sl_series: Optional[pd.Series] = None,
        tp_series: Optional[pd.Series] = None,
        symbol: str = "STOCK",
    ) -> BacktestResult:
        """
        Spustí deterministickou bar-by-bar simulaci.
        """
        if df is None or len(df) < 2:
            return self._empty_result(symbol)

        df_sim = tag_market_session(df)
        n_bars = len(df_sim)

        # Příprava signálů
        if signals is not None:
            sig_s = signals.reindex(df_sim.index).fillna(0)
        elif "signal" in df_sim.columns:
            sig_s = df_sim["signal"].fillna(0)
        else:
            sig_s = pd.Series(0, index=df_sim.index)

        # Příprava SL a TP sérií
        sl_s = sl_series.reindex(df_sim.index) if sl_series is not None else (
            df_sim["stop_loss"] if "stop_loss" in df_sim.columns else None
        )
        tp_s = tp_series.reindex(df_sim.index) if tp_series is not None else (
            df_sim["take_profit"] if "take_profit" in df_sim.columns else None
        )

        cost_pct = self.cost_config.total_cost_pct(self.is_cross_currency) / 100.0

        current_cash = self.initial_capital
        current_equity = self.initial_capital
        active_pos: Optional[ActivePosition] = None
        pending_order: Optional[Dict[str, Any]] = None
        trades: List[TradeRecord] = []
        equity_curve: List[float] = [self.initial_capital]

        opens = df_sim["Open"].values
        highs = df_sim["High"].values
        lows = df_sim["Low"].values
        closes = df_sim["Close"].values
        minutes_from_open = df_sim["minutes_from_open"].values if "minutes_from_open" in df_sim.columns else np.full(n_bars, 100)
        is_regular = df_sim["is_regular_session"].values if "is_regular_session" in df_sim.columns else np.full(n_bars, True)
        trading_days = df_sim["trading_day"].values if "trading_day" in df_sim.columns else np.full(n_bars, "DAY")
        indices = df_sim.index

        for t in range(n_bars):
            cur_time = indices[t]
            cur_open = float(opens[t])
            cur_high = float(highs[t])
            cur_low = float(lows[t])
            cur_close = float(closes[t])
            cur_min = int(minutes_from_open[t])
            is_reg = bool(is_regular[t])
            is_eod_bar = (cur_min >= self.eod_flatten_minute) or (t == n_bars - 1) or (
                t < n_bars - 1 and trading_days[t] != trading_days[t + 1]
            )

            # --- KROK 1: EXEKUCE ČEKAJÍCÍHO POKYNU Z BARU t-1 NA CENĚ Open[t] ---
            if pending_order is not None and active_pos is None:
                if is_reg and not is_eod_bar:
                    raw_entry = cur_open
                    entry_price = raw_entry * (1.0 + cost_pct)
                    sl_val = float(pending_order["stop_loss"])
                    tp_val = float(pending_order["take_profit"])

                    # Výpočet velikosti pozice podle rizika
                    risk_per_share = entry_price - sl_val
                    if risk_per_share > 0:
                        dollar_risk = current_equity * self.max_risk_per_trade_pct
                        target_shares = math.floor(dollar_risk / risk_per_share)
                        # Omezení maximální alokace na pozici
                        max_notional = current_equity * self.max_position_notional_pct
                        max_shares_cap = math.floor(max_notional / entry_price)
                        shares = min(target_shares, max_shares_cap)

                        # Omezení hotovostí
                        if shares * entry_price > current_cash:
                            shares = math.floor(current_cash / entry_price)

                        if shares > 0:
                            entry_cost = shares * entry_price * cost_pct
                            current_cash -= shares * entry_price
                            active_pos = ActivePosition(
                                symbol=symbol,
                                side=TradeSide.LONG,
                                entry_bar_idx=t,
                                entry_time=cur_time.to_pydatetime() if hasattr(cur_time, "to_pydatetime") else cur_time,
                                raw_entry_price=raw_entry,
                                entry_price=entry_price,
                                shares=shares,
                                stop_loss=sl_val,
                                take_profit=tp_val,
                                initial_risk_cash=shares * risk_per_share,
                                entry_cost_paid=entry_cost,
                            )
                pending_order = None

            # --- KROK 2: KONTROLA A VYHODNOCENÍ OTEVŘENÉ POZICE NA BARU t ---
            if active_pos is not None:
                active_pos.holding_bars += 1
                exit_triggered = False
                exit_reason = ExitReason.STOP_LOSS
                raw_exit_price = 0.0

                # 2A: Vynucený EOD Flatten (15:55 EST nebo konec dne)
                if is_eod_bar:
                    exit_triggered = True
                    exit_reason = ExitReason.EOD_FLATTEN
                    raw_exit_price = cur_close

                # 2B: Pesimistická kontrola SL a TP kolize
                elif cur_low <= active_pos.stop_loss and cur_high >= active_pos.take_profit:
                    # Obě bariéry zasaženy v téže svíčce -> pesimisticky STOP LOSS
                    exit_triggered = True
                    exit_reason = ExitReason.STOP_LOSS
                    raw_exit_price = active_pos.stop_loss

                # 2C: Zasažení Stop Lossu
                elif cur_low <= active_pos.stop_loss:
                    exit_triggered = True
                    exit_reason = ExitReason.STOP_LOSS
                    raw_exit_price = active_pos.stop_loss

                # 2D: Zasažení Take Profitu
                elif cur_high >= active_pos.take_profit:
                    exit_triggered = True
                    exit_reason = ExitReason.TAKE_PROFIT
                    raw_exit_price = active_pos.take_profit

                # 2E: Time-Stop po vypršení max holding bars
                elif active_pos.holding_bars >= self.max_holding_bars:
                    exit_triggered = True
                    exit_reason = ExitReason.TIME_STOP
                    raw_exit_price = cur_close

                # Provedení výstupu
                if exit_triggered:
                    exit_price = raw_exit_price * (1.0 - cost_pct)
                    exit_cost = active_pos.shares * raw_exit_price * cost_pct
                    total_costs = active_pos.entry_cost_paid + exit_cost
                    gross_pnl = (raw_exit_price - active_pos.raw_entry_price) * active_pos.shares
                    net_pnl = (exit_price - active_pos.entry_price) * active_pos.shares
                    net_ret_pct = (exit_price / active_pos.entry_price - 1.0) * 100.0
                    pnl_r = net_pnl / active_pos.initial_risk_cash

                    current_cash += active_pos.shares * exit_price

                    rec = TradeRecord(
                        symbol=symbol,
                        side=active_pos.side,
                        entry_bar_idx=active_pos.entry_bar_idx,
                        entry_time=active_pos.entry_time,
                        raw_entry_price=round(active_pos.raw_entry_price, 4),
                        entry_price=round(active_pos.entry_price, 4),
                        shares=active_pos.shares,
                        exit_bar_idx=t,
                        exit_time=cur_time.to_pydatetime() if hasattr(cur_time, "to_pydatetime") else cur_time,
                        raw_exit_price=round(raw_exit_price, 4),
                        exit_price=round(exit_price, 4),
                        stop_loss=round(active_pos.stop_loss, 4),
                        take_profit=round(active_pos.take_profit, 4),
                        initial_risk_cash=round(active_pos.initial_risk_cash, 2),
                        exit_reason=exit_reason,
                        gross_pnl=round(gross_pnl, 2),
                        net_pnl=round(net_pnl, 2),
                        net_return_pct=round(net_ret_pct, 4),
                        pnl_r=round(pnl_r, 4),
                        holding_bars=active_pos.holding_bars,
                        total_cost_paid=round(total_costs, 2),
                    )
                    trades.append(rec)
                    active_pos = None

            # Aktualizace průběžné equity
            if active_pos is not None:
                current_equity = current_cash + (active_pos.shares * cur_close * (1.0 - cost_pct))
            else:
                current_equity = current_cash
            equity_curve.append(round(current_equity, 2))

            # --- KROK 3: PŘÍJEM NOVÉHO SIGNÁLU NA KONCI BARU t (Close[t]) ---
            if active_pos is None and pending_order is None and is_reg and not is_eod_bar:
                sig_val = int(sig_s.iloc[t])
                if sig_val == 1:  # Long signál
                    # Určení SL a TP
                    if sl_s is not None and not np.isnan(sl_s.iloc[t]):
                        sl_target = float(sl_s.iloc[t])
                    else:
                        sl_target = cur_close * 0.985  # Výchozí 1.5% SL

                    if tp_s is not None and not np.isnan(tp_s.iloc[t]):
                        tp_target = float(tp_s.iloc[t])
                    else:
                        tp_target = cur_close * 1.030  # Výchozí 3.0% TP (R:R 1:2)

                    if sl_target < cur_close < tp_target:
                        pending_order = {
                            "signal_bar_idx": t,
                            "stop_loss": sl_target,
                            "take_profit": tp_target,
                        }

        # --- ZÁVĚREČNÉ VYHODNOCENÍ METRIK ---
        return self._calculate_metrics(
            symbol=symbol,
            initial_cap=self.initial_capital,
            final_cap=current_equity,
            trades=trades,
            equity_curve=equity_curve,
        )

    def _empty_result(self, symbol: str) -> BacktestResult:
        return BacktestResult(
            symbol=symbol,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            net_pnl=0.0,
            net_return_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_pnl=0.0,
            avg_trade_bps=0.0,
            avg_r_multiple=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_cash=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            trades=[],
            equity_curve=[self.initial_capital],
        )

    def _calculate_metrics(
        self,
        symbol: str,
        initial_cap: float,
        final_cap: float,
        trades: List[TradeRecord],
        equity_curve: List[float],
    ) -> BacktestResult:
        net_pnl = final_cap - initial_cap
        net_return_pct = (net_pnl / initial_cap) * 100.0
        n_trades = len(trades)

        if n_trades == 0:
            return self._empty_result(symbol)

        winning = [t for t in trades if t.net_pnl > 0]
        losing = [t for t in trades if t.net_pnl <= 0]
        win_rate = (len(winning) / n_trades) * 100.0

        gross_profit = sum(t.net_pnl for t in winning)
        gross_loss = abs(sum(t.net_pnl for t in losing))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        avg_trade_pnl = net_pnl / n_trades
        avg_trade_bps = float(np.mean([t.net_return_pct * 100.0 for t in trades]))
        avg_r = float(np.mean([t.pnl_r for t in trades]))

        # Max Drawdown z průběžné equity křivky
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        dd_cash = peak - eq_arr
        dd_pct = np.where(peak > 0, dd_cash / peak * 100.0, 0.0)
        max_dd_pct = float(np.max(dd_pct)) if len(dd_pct) > 0 else 0.0
        max_dd_cash = float(np.max(dd_cash)) if len(dd_cash) > 0 else 0.0

        # Sharpe & Sortino Ratio z procentuálních výnosů jednotlivých obchodů
        rets = np.array([t.net_return_pct / 100.0 for t in trades])
        std_ret = float(np.std(rets)) if len(rets) > 1 else 0.0
        mean_ret = float(np.mean(rets)) if len(rets) > 0 else 0.0

        # Odhad annualizovaného Sharpe (předpoklad cca 4 obchody denně * 252 dní = 1000 obchodů/rok)
        annual_factor = np.sqrt(min(1000, max(252, n_trades * 10)))
        sharpe = (mean_ret / std_ret * annual_factor) if std_ret > 1e-6 else 0.0

        # Downside deviation pro Sortino
        neg_rets = rets[rets < 0]
        downside_std = float(np.std(neg_rets)) if len(neg_rets) > 1 else (std_ret if std_ret > 0 else 0.0)
        sortino = (mean_ret / downside_std * annual_factor) if downside_std > 1e-6 else sharpe

        return BacktestResult(
            symbol=symbol,
            initial_capital=round(initial_cap, 2),
            final_capital=round(final_cap, 2),
            net_pnl=round(net_pnl, 2),
            net_return_pct=round(net_return_pct, 2),
            total_trades=n_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(win_rate, 2),
            profit_factor=round(min(99.0, profit_factor), 2),
            avg_trade_pnl=round(avg_trade_pnl, 2),
            avg_trade_bps=round(avg_trade_bps, 2),
            avg_r_multiple=round(avg_r, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            max_drawdown_cash=round(max_dd_cash, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            trades=trades,
            equity_curve=equity_curve,
        )


class WalkForwardEvaluator:
    """
    Klouzavý Walk-Forward validační engine s embargem a Bootstrap konfidenčními intervaly.
    """

    def __init__(
        self,
        backtester: Optional[IntradayBacktester] = None,
        in_sample_days: int = 60,
        embargo_days: int = 2,
        out_of_sample_days: int = 20,
        n_bootstrap: int = 1000,
    ):
        self.backtester = backtester or IntradayBacktester()
        self.in_sample_days = in_sample_days
        self.embargo_days = embargo_days
        self.out_of_sample_days = out_of_sample_days
        self.n_bootstrap = n_bootstrap

    def evaluate_walk_forward(
        self,
        df: pd.DataFrame,
        signal_generator_fn: Callable[[pd.DataFrame], pd.Series],
        symbol: str = "STOCK",
    ) -> Dict[str, Any]:
        """
        Provede rolling walk-forward rozdělení a agreguje výsledky výhradně z Out-of-Sample oken.
        """
        df_tagged = tag_market_session(df)
        unique_days = sorted(df_tagged["trading_day"].unique().tolist())
        total_days = len(unique_days)

        window_size = self.in_sample_days + self.embargo_days + self.out_of_sample_days

        # Pokud je dat méně než plná délka oken, dynamicky adaptujeme velikost pro test
        is_days = self.in_sample_days
        emb_days = self.embargo_days
        oos_days = self.out_of_sample_days

        if total_days < window_size:
            if total_days >= 4:
                is_days = max(2, total_days // 2)
                emb_days = 1 if total_days >= 5 else 0
                oos_days = max(1, total_days - is_days - emb_days)
            else:
                logger.warning(f"Nedostatek dnů ({total_days}) pro Walk-Forward evaluaci.")
                # Provedeme alespoň přímý backtest na dostupných datech
                signals = signal_generator_fn(df_tagged)
                res = self.backtester.run(df_tagged, signals=signals, symbol=symbol)
                return {
                    "summary": res.to_dict(),
                    "oos_trades_count": res.total_trades,
                    "bootstrap_95_ci": self._bootstrap_ci([t.net_return_pct for t in res.trades]),
                    "folds_count": 1,
                }

        step = oos_days
        start_idx = 0
        all_oos_trades: List[TradeRecord] = []
        folds_details = []

        while start_idx + is_days + emb_days + oos_days <= total_days:
            is_end = start_idx + is_days
            emb_end = is_end + emb_days
            oos_end = emb_end + oos_days

            in_sample_day_list = unique_days[start_idx:is_end]
            oos_day_list = unique_days[emb_end:oos_end]

            # Slice pro Out-of-Sample okno
            oos_df = df_tagged[df_tagged["trading_day"].isin(oos_day_list)].copy()
            if not oos_df.empty:
                # Generování signálu čistě na OOS datech
                oos_signals = signal_generator_fn(oos_df)
                fold_result = self.backtester.run(oos_df, signals=oos_signals, symbol=symbol)
                all_oos_trades.extend(fold_result.trades)
                folds_details.append({
                    "fold_index": len(folds_details) + 1,
                    "train_start": in_sample_day_list[0],
                    "train_end": in_sample_day_list[-1],
                    "oos_start": oos_day_list[0],
                    "oos_end": oos_day_list[-1],
                    "oos_trades": fold_result.total_trades,
                    "oos_win_rate": fold_result.win_rate,
                    "oos_net_pnl": fold_result.net_pnl,
                })

            start_idx += step

        # Agregace všech Out-of-Sample obchodů
        oos_returns = [t.net_return_pct for t in all_oos_trades]
        bootstrap_stats = self._bootstrap_ci(oos_returns)

        # Výpočet kombinovaných OOS metrik
        combined_result = self.backtester._calculate_metrics(
            symbol=symbol,
            initial_cap=self.backtester.initial_capital,
            final_cap=self.backtester.initial_capital + sum(t.net_pnl for t in all_oos_trades),
            trades=all_oos_trades,
            equity_curve=[self.backtester.initial_capital + sum(t.net_pnl for t in all_oos_trades[:i+1]) for i in range(len(all_oos_trades))] or [self.backtester.initial_capital],
        )

        return {
            "summary": combined_result.to_dict(),
            "folds_count": len(folds_details),
            "folds": folds_details,
            "oos_trades_count": len(all_oos_trades),
            "bootstrap_95_ci": bootstrap_stats,
        }

    def _bootstrap_ci(self, returns: List[float]) -> Dict[str, float]:
        """Výpočet 95% konfidenčního intervalu průměrného výnosu přes parametrický/neparametrický bootstrap."""
        if len(returns) < 5:
            return {"mean": round(float(np.mean(returns)), 4) if returns else 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        arr = np.array(returns)
        np.random.seed(42)
        boot_means = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            boot_means.append(float(np.mean(sample)))

        lower = float(np.percentile(boot_means, 2.5))
        upper = float(np.percentile(boot_means, 97.5))
        return {
            "mean": round(float(np.mean(arr)), 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
        }
