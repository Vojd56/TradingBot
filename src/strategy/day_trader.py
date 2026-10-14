from dataclasses import dataclass, field
from datetime import datetime, time as dtime
import logging
import time
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from src.broker.t212_client import Trading212Client
from src.execution.order_executor import OrderExecutor
from src.execution.portfolio_tracker import PortfolioTracker, PositionRecord
from src.execution.ticker_mapper import TickerMapper
from src.strategy.intraday_data import IntradayDataProvider, IntradaySnapshot
from src.strategy.market_hours import MarketHours
from src.strategy.models import ExitReason, PositionExitSignal, SignalAction, TradingSignal

logger = logging.getLogger(__name__)

# Výchozí agresivní intradenní watchlist: Páková ETF + High-Beta volatilní giganti
DEFAULT_DAYTRADE_WATCHLIST = [
    # 1. Páková ETF (vestavěná denní páka 2x až 3x)
    "TQQQ",  # 3x Leveraged Nasdaq 100
    "SOXL",  # 3x Leveraged Semiconductor & AI
    "NVDL",  # 2x Leveraged Nvidia
    "TSLL",  # 2x Leveraged Tesla
    "UPRO",  # 3x Leveraged S&P 500
    "CONL",  # 2x Leveraged Coinbase & Crypto
    # 2. Ultra-volatilní tech & krypto lídři
    "NVDA",
    "TSLA",
    "MSTR",  # Bitcoin proxy beta
    "COIN",
    "PLTR",
    "AMD",
    "ARM",
    "SMCI",
]


@dataclass
class DayTradeSignal:
    """Reprezentuje intradenní momentum signál pro rychlý obchod."""
    ticker: str
    action: SignalAction
    score: float
    current_price: float
    vwap: float
    ema_9: float
    ema_21: float
    rsi_5m: float
    volume_surge: float
    atr_5m: float
    stop_loss_price: float
    take_profit_price: float
    reasons: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class DayTraderEngine:
    """
    Agresivní experimentální Day Trading & Leveraged Momentum Bot.
    
    Klíčové vlastnosti:
    1. Sleduje 5minutové svíčky v reálném čase (VWAP, EMA Ribbon 9/21, Volume Surge, 5m RSI).
    2. Obchoduje páková ETF (TQQQ, SOXL, NVDL, TSLL) a ultra-volatilní akcie.
    3. Exekuce každých 60-90 sekund.
    4. Rychlý Take Profit (+2.2 %), těsný Stop Loss (-1.2 %) a Breakeven Stop po dosažení +1.0 % zisku.
    5. EOD ochrana: Možnost uzavřít všechny pozice před koncem dne (Flat na noc).
    """

    def __init__(
        self,
        client: Optional[Trading212Client] = None,
        tracker: Optional[PortfolioTracker] = None,
        executor: Optional[OrderExecutor] = None,
        data_provider: Optional[IntradayDataProvider] = None,
        watchlist: Optional[List[str]] = None,
        max_active_positions: int = 5,
        allocation_per_trade_eur: float = 5000.0,
        take_profit_pct: float = 2.2,     # +2.2 % rychlý skalp zisku
        stop_loss_pct: float = 1.2,       # -1.2 % těsný stop loss
        breakeven_trigger_pct: float = 1.0, # Při +1.0 % zisku posun stop-lossu na nákup
        dry_run: bool = False,
    ):
        self.client = client or Trading212Client()
        self.tracker = tracker or PortfolioTracker()
        self.dry_run = dry_run
        self.executor = executor or OrderExecutor(client=self.client, tracker=self.tracker, dry_run=self.dry_run)
        self.data_provider = data_provider or IntradayDataProvider(interval="5m")
        self.watchlist = watchlist or DEFAULT_DAYTRADE_WATCHLIST
        self.max_active_positions = max_active_positions
        self.allocation_per_trade_eur = allocation_per_trade_eur
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.breakeven_trigger_pct = breakeven_trigger_pct

        # Sledování nejvyšší dosažené ceny pro dynamický trailing stop
        self._peak_prices: Dict[str, float] = {}
        self._breakeven_applied: Set[str] = set()

    def evaluate_ticker_setup(self, ticker: str) -> Optional[DayTradeSignal]:
        """Vyhodnotí intradenní 5m setup na grafu a spočítá DayTradeScore (0 až 100)."""
        snap = self.data_provider.fetch_intraday_snapshot(ticker)
        if not snap:
            return None

        score = 0.0
        reasons = []

        # 1. Poloha vůči VWAP (+25 b)
        if snap.is_above_vwap:
            dist_vwap_pct = ((snap.current_price - snap.vwap) / snap.vwap) * 100.0
            if 0.0 <= dist_vwap_pct <= 0.8:
                score += 25.0
                reasons.append("Odraz od VWAP do růstu (ideální vstup)")
            elif dist_vwap_pct > 0.8:
                score += 18.0
                reasons.append("Cena nad VWAP (býčí teritorium)")
        else:
            # Cena pod VWAP je v day tradingu medvědí
            score -= 10.0

        # 2. EMA 9 / 21 Scalp Ribbon (+25 b)
        if snap.is_fresh_cross:
            score += 25.0
            reasons.append("ČERSTVÝ EMA 9/21 Golden Cross na 5m grafu!")
        elif snap.is_bullish_ema_cross:
            score += 15.0
            reasons.append("EMA 9 je nad EMA 21 (rostoucí momentum)")

        # 3. Volume Surge Ratio (+20 b)
        if snap.volume_surge_ratio >= 1.75:
            score += 20.0
            reasons.append(f"Objemový výbuch {snap.volume_surge_ratio:.1f}x nad průměrem")
        elif snap.volume_surge_ratio >= 1.25:
            score += 12.0
            reasons.append(f"Zvýšený nákupní objem {snap.volume_surge_ratio:.1f}x")

        # 4. Intradenní 5m RSI (+15 b)
        if 42.0 <= snap.rsi_5m <= 65.0:
            score += 15.0
            reasons.append(f"5m RSI v optimálním momentum pásmu ({snap.rsi_5m:.1f})")
        elif snap.rsi_5m < 35.0:
            score += 12.0
            reasons.append(f"5m RSI přeprodané ({snap.rsi_5m:.1f}) - možný rychlý odraz")
        elif snap.rsi_5m > 72.0:
            score -= 12.0
            reasons.append(f"5m RSI překoupené ({snap.rsi_5m:.1f}) - pozor na korekci")

        # 5. Proražení 30minutového maxima (+15 b)
        if snap.is_breaking_high:
            score += 15.0
            reasons.append("Proražení 30minutového lokálního maxima")

        # Zaokrouhlení a mantinely skóre
        score = max(0.0, min(100.0, round(score, 1)))

        # Výpočet dynamického Stop Lossu a Take Profitu
        sl_dist = max(snap.current_price * (self.stop_loss_pct / 100.0), snap.atr_5m * 1.2)
        tp_dist = max(snap.current_price * (self.take_profit_pct / 100.0), snap.atr_5m * 2.0)

        sl_price = round(snap.current_price - sl_dist, 2)
        tp_price = round(snap.current_price + tp_dist, 2)

        action = SignalAction.BUY if score >= 62.0 else SignalAction.HOLD

        return DayTradeSignal(
            ticker=snap.ticker,
            action=action,
            score=score,
            current_price=snap.current_price,
            vwap=snap.vwap,
            ema_9=snap.ema_9,
            ema_21=snap.ema_21,
            rsi_5m=snap.rsi_5m,
            volume_surge=snap.volume_surge_ratio,
            atr_5m=snap.atr_5m,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            reasons=reasons,
        )

    def audit_active_positions(self) -> List[Dict[str, Any]]:
        """
        Zkontroluje otevřené day trade pozice v reálném čase:
        - Take Profit (+2.2 %)
        - Stop Loss (-1.2 %)
        - Breakeven Stop (při +1.0 % posun na nákup)
        - Trailing Stop pro běžící zisky
        """
        closed_reports: List[Dict[str, Any]] = []
        open_positions = self.tracker.get_open_positions()

        for ticker, pos in open_positions.items():
            snap = self.data_provider.fetch_intraday_snapshot(ticker)
            if not snap:
                continue

            current_price = snap.current_price
            entry_price = pos.entry_price_usd
            if entry_price <= 0:
                continue

            pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

            # Evidence nejvyšší dosažené ceny
            peak = self._peak_prices.get(ticker, entry_price)
            if current_price > peak:
                self._peak_prices[ticker] = current_price
                peak = current_price

            # 1. Kontrola Breakeven posunu při zisku >= breakeven_trigger_pct
            if pnl_pct >= self.breakeven_trigger_pct and ticker not in self._breakeven_applied:
                pos.stop_loss_price = round(entry_price * 1.001, 2)  # Zajištění nákupní ceny + fee buffer
                self._breakeven_applied.add(ticker)
                self.tracker.save_portfolio()
                logger.info(f"🛡️ [BREAKEVEN AKTIVOVÁN] {ticker}: Zisk dosáhl +{pnl_pct:.2f} %. Stop Loss posunut na nákupní cenu ({pos.stop_loss_price:.2f} USD).")

            # 2. Dynamický Trailing Stop pro běžící vítěze
            if pnl_pct >= 2.0:
                trailing_level = round(peak - (snap.atr_5m * 1.2), 2)
                if trailing_level > pos.stop_loss_price:
                    pos.stop_loss_price = trailing_level
                    self.tracker.save_portfolio()
                    logger.info(f"📈 [TRAILING STOP POSUNUT] {ticker}: Nový Stop Loss {trailing_level:.2f} USD (High: {peak:.2f} USD).")

            # 3. Kontrola Take Profitu
            if current_price >= pos.take_profit_price:
                logger.info(f"🎯 [TAKE PROFIT ZASAŽEN] {ticker}: Cena {current_price:.2f} >= TP {pos.take_profit_price:.2f} USD (+{pnl_pct:.2f} % zisk).")
                exit_sig = PositionExitSignal(
                    ticker=ticker,
                    should_exit=True,
                    reason=ExitReason.TAKE_PROFIT,
                    exit_price=current_price,
                    message=f"Bleskový Day Trade Take Profit zasažen (+{pnl_pct:.2f} %)"
                )
                res = self.executor.execute_sell(ticker, exit_sig, pos.quantity)
                closed_reports.append({"ticker": ticker, "reason": "TAKE_PROFIT", "pnl_pct": pnl_pct, "res": res})
                self._peak_prices.pop(ticker, None)
                self._breakeven_applied.discard(ticker)
                continue

            # 4. Kontrola Stop Lossu
            if current_price <= pos.stop_loss_price:
                logger.info(f"🛑 [STOP LOSS ZASAŽEN] {ticker}: Cena {current_price:.2f} <= SL {pos.stop_loss_price:.2f} USD ({pnl_pct:.2f} % ztráta).")
                exit_sig = PositionExitSignal(
                    ticker=ticker,
                    should_exit=True,
                    reason=ExitReason.STOP_LOSS,
                    exit_price=current_price,
                    message=f"Day Trade Stop Loss aktivován k ochraně kapitálu ({pnl_pct:.2f} %)"
                )
                res = self.executor.execute_sell(ticker, exit_sig, pos.quantity)
                closed_reports.append({"ticker": ticker, "reason": "STOP_LOSS", "pnl_pct": pnl_pct, "res": res})
                self._peak_prices.pop(ticker, None)
                self._breakeven_applied.discard(ticker)
                continue

        return closed_reports

    def execute_daytrade_buys(self, signals: List[DayTradeSignal]) -> List[Dict[str, Any]]:
        """Vybere nejlepší BUY signály a bleskově otevře pozice až do limitu."""
        opened_reports: List[Dict[str, Any]] = []
        open_positions = self.tracker.get_open_positions()
        available_slots = max(0, self.max_active_positions - len(open_positions))

        if available_slots <= 0:
            return opened_reports

        buy_signals = [s for s in signals if s.action == SignalAction.BUY and s.ticker not in open_positions]
        buy_signals.sort(key=lambda s: s.score, reverse=True)

        for sig in buy_signals[:available_slots]:
            # Zformování TradingSignalu pro standardní OrderExecutor
            std_signal = TradingSignal(
                ticker=sig.ticker,
                action=SignalAction.BUY,
                confidence=0.85,
                suggested_allocation_eur=self.allocation_per_trade_eur,
                stop_loss_price=sig.stop_loss_price,
                take_profit_price=sig.take_profit_price,
                technical_score=sig.score / 2.0,
                sentiment_score=sig.score / 2.0,
                total_score=sig.score,
                time_horizon="INTRADAY",
                current_price=sig.current_price,
                atr_14=sig.atr_5m,
                rsi_14=sig.rsi_5m,
            )

            logger.info(
                f"🚀 [DAY TRADE VSTUP] {sig.ticker} | Skóre: {sig.score:.1f}/100 | "
                f"Cena: {sig.current_price:.2f} USD | TP: {sig.take_profit_price:.2f} | "
                f"SL: {sig.stop_loss_price:.2f} | Důvody: {', '.join(sig.reasons[:2])}"
            )

            buy_res = self.executor.execute_buy(std_signal)
            if buy_res:
                self._peak_prices[sig.ticker] = sig.current_price
                opened_reports.append({
                    "ticker": sig.ticker,
                    "price": sig.current_price,
                    "score": sig.score,
                    "tp": sig.take_profit_price,
                    "sl": sig.stop_loss_price,
                })
                time.sleep(1.0)

        return opened_reports

    def run_one_scan(self) -> Dict[str, Any]:
        """Provede jeden rychlý intradenní cyklus (audit pozic + sken grafů + nákupy)."""
        now = datetime.now()
        logger.info(f"⚡ [DAY TRADER SCAN] Zahajuji 5m intradenní sken ({now.strftime('%H:%M:%S')})...")

        # 1. Audit držených pozic
        closed = self.audit_active_positions()

        # 2. Skenování watchlistu
        evaluated_signals: List[DayTradeSignal] = []
        for ticker in self.watchlist:
            try:
                sig = self.evaluate_ticker_setup(ticker)
                if sig:
                    evaluated_signals.append(sig)
            except Exception as exc:
                logger.debug(f"Chyba při skenu {ticker}: {exc}")

        # 3. Exekuce nových nákupů
        opened = self.execute_daytrade_buys(evaluated_signals)

        return {
            "timestamp": now.isoformat(),
            "closed_trades": closed,
            "opened_trades": opened,
            "signals": evaluated_signals,
        }

    def start_loop(self, scan_interval_seconds: int = 60, close_at_eod: bool = False) -> None:
        """
        Spustí nepřetržitou Day Trading smyčku (výchozí interval: 60 sekund).
        Běží během otevřených trhů, sleduje grafy a provádí rychlé vstupy a výstupy.
        """
        print("\n" + "=" * 70)
        print("          ⚡ AI DAY TRADER & LEVERAGED MOMENTUM ENGINE ⚡       ")
        print("=" * 70)
        print(f"  Frekvence skenování:  Každých {scan_interval_seconds} sekund")
        print(f"  Pákový watchlist:     {', '.join(self.watchlist)}")
        print(f"  Alokace na obchod:    {self.allocation_per_trade_eur:,.0f} EUR")
        print(f"  Max otevřených pozic: {self.max_active_positions}")
        print(f"  Cílový Take Profit:   +{self.take_profit_pct:.1f} % (nebo 2x ATR)")
        print(f"  Těsný Stop Loss:      -{self.stop_loss_pct:.1f} % (nebo 1.2x ATR)")
        print(f"  Breakeven Stop:       Aktivní při zisku +{self.breakeven_trigger_pct:.1f} %")
        print(f"  Režim:                {'DRY-RUN (Simulace)' if self.dry_run else 'LIVE EXEKUCE (Demo účet)'}")
        print("=" * 70 + "\n")
        print("Stiskněte Ctrl+C pro bezpečné zastavení Day Tradera.\n")

        try:
            while True:
                market_stat = MarketHours.get_market_status()

                if market_stat["is_open"]:
                    cycle_res = self.run_one_scan()
                    signals = cycle_res.get("signals", [])

                    # Tisk živého terminálového přehledu
                    print(f"\n--- [DAY TRADER STATUS {datetime.now().strftime('%H:%M:%S')}] ---")
                    open_pos = self.tracker.get_open_positions()
                    if open_pos:
                        print(f"Otevřené pozice ({len(open_pos)} / {self.max_active_positions}):")
                        for t, p in open_pos.items():
                            peak = self._peak_prices.get(t, p.entry_price_usd)
                            print(
                                f"  • {t:<6} | Nákup: {p.entry_price_usd:.2f} | "
                                f"SL: {p.stop_loss_price:.2f} | TP: {p.take_profit_price:.2f} | "
                                f"Alokováno: {p.allocated_eur:.0f} EUR"
                            )
                    else:
                        print(f"Aktuálně nejsou otevřeny žádné pozice (Kapacita: {self.max_active_positions} volných slotů).")

                    print("\nTop intradenní momentum signály:")
                    signals.sort(key=lambda s: s.score, reverse=True)
                    for s in signals[:6]:
                        above_str = "NAD VWAP ▲" if (s.current_price > s.vwap) else "POD VWAP ▼"
                        action_icon = "🟢 BUY " if s.action == SignalAction.BUY else "⚪ HOLD"
                        print(
                            f"  {action_icon} {s.ticker:<6} | Skóre: {s.score:4.1f}/100 | "
                            f"Cena: {s.current_price:7.2f} USD ({above_str}) | "
                            f"5m RSI: {s.rsi_5m:4.1f} | Objem: {s.volume_surge:3.1f}x"
                        )
                    print(f"Příští sken za {scan_interval_seconds} sekund...\n")

                else:
                    print(f"🌙 [Trh je zavřen] {market_stat['status_text']}. Čekám na otevření...")

                time.sleep(scan_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Day Trader byl zastaven uživatelem (Ctrl+C).")
            print("\n🛑 Day Trader byl bezpečně ukončen.")
