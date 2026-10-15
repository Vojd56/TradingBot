from dataclasses import dataclass, field
from datetime import datetime, time as dtime
import logging
import time
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import yfinance as yf
from src.broker.t212_client import Trading212Client
from src.execution.order_executor import OrderExecutor
from src.execution.portfolio_tracker import PortfolioTracker, PositionRecord
from src.execution.ticker_mapper import TickerMapper
from src.llm.sentiment_engine import SentimentEngine
from src.strategy.autonomous_discovery import AutonomousDiscoveryEngine
from src.strategy.intraday_data import IntradayDataProvider, IntradaySnapshot
from src.strategy.macro_regime import MarketRegime, MarketRegimeDetector, RegimeSnapshot
from src.strategy.market_hours import MarketHours
from src.strategy.models import ExitReason, PositionExitSignal, SignalAction, TradingSignal

logger = logging.getLogger(__name__)

# Rozsáhlý agresivní intradenní watchlist (65+ vysoce likvidních a pákových titulů)
DEFAULT_DAYTRADE_WATCHLIST = [
    # 1. Páková ETF (denní páka 2x až 3x)
    "TQQQ",  # 3x Leveraged Nasdaq 100
    "SOXL",  # 3x Leveraged Semiconductor & AI
    "NVDL",  # 2x Leveraged Nvidia
    "TSLL",  # 2x Leveraged Tesla
    "UPRO",  # 3x Leveraged S&P 500
    "CONL",  # 2x Leveraged Coinbase & Crypto
    "FNGU",  # 3x Leveraged Big Tech FANG+
    "TECL",  # 3x Leveraged Technology
    "DPST",  # 3x Leveraged Regional Banks
    "LABU",  # 3x Leveraged Biotech
    # 2. AI, polovodiče & hardware titani
    "NVDA",
    "TSLA",
    "AMD",
    "ARM",
    "SMCI",
    "AVGO",
    "QCOM",
    "MU",
    "INTC",
    "MRVL",
    "ASML",
    # 3. Krypto & Fintech proxy s vysokou betou
    "MSTR",
    "COIN",
    "MARA",
    "RIOT",
    "CLSK",
    "HOOD",
    "SOFI",
    "AFRM",
    "UPST",
    "PYPL",
    # 4. Mega-Cap Momentum lídři
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    # 5. Rychle rostoucí Cloud, AI software & kyberbezpečnost
    "PLTR",
    "CRWD",
    "PANW",
    "DDOG",
    "NET",
    "SNOW",
    "APP",
    "DELL",
    "VRT",
    "ORCL",
    "NOW",
    # 6. Žhavé novinky, IPO a small/mid-cap AI příležitosti
    "BBAI",
    "SOUN",
    "RDDT",
    "TEM",
    "ALAB",
    "LUMN",
    "BRZE",
    "LASR",
    "VOYG",
    "SYM",
    # 7. Evropské momentum tituly (aktivní 09:00 - 17:30 CET)
    "RHM.DE",
    "RR.L",
    "ASML.AS",
    "SAP.DE",
    "AIR.PA",
    "SIE.DE",
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
    currency: str = "USD"
    timestamp: datetime = field(default_factory=datetime.now)


class DayTraderEngine:
    """
    Agresivní experimentální Day Trading & Leveraged Momentum Bot.
    
    Klíčové vlastnosti:
    1. Sleduje široké univerzum (65+ titulů) 5minutových svíček v rychlém paralelním stahování.
    2. Autonomně vyhledává nové akcie, čerstvá IPO a zakázky z médií a zařazuje je do sledování.
    3. Obchoduje páková ETF (TQQQ, SOXL, NVDL, TSLL) i volatilní krypto/tech tituly.
    4. Blesková exekuce každých 60 sekund.
    5. Rychlý Take Profit (+2.2 %), těsný Stop Loss (-1.2 %) a Breakeven Stop po dosažení +1.0 % zisku.
    """

    def __init__(
        self,
        client: Optional[Trading212Client] = None,
        tracker: Optional[PortfolioTracker] = None,
        executor: Optional[OrderExecutor] = None,
        data_provider: Optional[IntradayDataProvider] = None,
        discovery_engine: Optional[AutonomousDiscoveryEngine] = None,
        watchlist: Optional[List[str]] = None,
        max_active_positions: int = 5,
        allocation_per_trade_eur: float = 5000.0,
        take_profit_pct: float = 2.2,     # +2.2 % rychlý skalp zisku
        stop_loss_pct: float = 1.2,       # -1.2 % těsný stop loss
        breakeven_trigger_pct: float = 1.0, # Při +1.0 % zisku posun stop-lossu na nákup
        max_daily_loss_eur: float = 750.0,  # Maximální povolená denní ztráta (Circuit Breaker)
        max_consecutive_losses: int = 3,   # Stop po 3 ztrátách v řadě (ochrana před nepříznivým trhem)
        sentiment_engine: Optional[SentimentEngine] = None,
        regime_detector: Optional[MarketRegimeDetector] = None,
        dry_run: bool = False,
    ):
        self.client = client or Trading212Client()
        self.tracker = tracker or PortfolioTracker()
        self.dry_run = dry_run
        self.executor = executor or OrderExecutor(client=self.client, tracker=self.tracker, dry_run=self.dry_run)
        self.data_provider = data_provider or IntradayDataProvider(interval="5m")
        self.discovery_engine = discovery_engine or AutonomousDiscoveryEngine()
        self.sentiment_engine = sentiment_engine or SentimentEngine()
        self.regime_detector = regime_detector or MarketRegimeDetector(sentiment_engine=self.sentiment_engine)
        self.current_regime_snapshot: Optional[RegimeSnapshot] = None
        self.watchlist: List[str] = list(watchlist or DEFAULT_DAYTRADE_WATCHLIST)
        # Dynamické načtení ověřených příležitostí z Rady AI (ScheduledMarketScanner)
        try:
            from src.strategy.scheduled_scanner import ScheduledMarketScanner
            qualified = ScheduledMarketScanner.load_qualified_tickers()
            for q in qualified:
                if q not in self.watchlist:
                    self.watchlist.append(q)
            if qualified:
                logger.info(f"Day Trader zařadil {len(qualified)} schválených titulů z Rady AI do aktivního sledování.")
        except Exception:
            pass

        self.max_active_positions = max_active_positions
        self.allocation_per_trade_eur = allocation_per_trade_eur
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.breakeven_trigger_pct = breakeven_trigger_pct
        self.max_daily_loss_eur = max_daily_loss_eur
        self.max_consecutive_losses = max_consecutive_losses

        # Sledování stavu obchodu a statistik seance (inspirováno DavidTech / Kevin Davey pravidly)
        self._peak_prices: Dict[str, float] = {}
        self._breakeven_applied: Set[str] = set()
        self._dry_run_positions: Dict[str, PositionRecord] = {}
        self._last_discovery_ts: float = 0.0
        self._daily_realized_pnl_eur: float = 0.0
        self._consecutive_losses: int = 0
        self._circuit_breaker_triggered: bool = False
        self._session_trades_count: int = 0
        self._session_wins: int = 0

    @staticmethod
    def get_ticker_currency(ticker: str) -> str:
        """Určí měnu instrumentu podle burzovní přípony."""
        t = ticker.upper()
        if t.endswith((".DE", ".PA", ".AS", ".MC", ".MI")):
            return "EUR"
        if t.endswith(".L"):
            return "GBp"
        if t.endswith(".SW"):
            return "CHF"
        return "USD"

    def get_all_active_positions(self) -> Dict[str, PositionRecord]:
        """Vrátí všechny sledované pozice (reálné z trackeru + případné simulované v dry-run)."""
        res = dict(self.tracker.get_open_positions())
        if self.dry_run:
            res.update(self._dry_run_positions)
        return res

    def discover_new_daytrade_tickers(self, max_new: int = 4) -> List[str]:
        """Skenuje breaking zprávy na nová technologická IPO, kontrakty a zařazuje je do watchlistu."""
        try:
            logger.info("📡 [Day Trader Discovery] Prohledávám breaking news na nová IPO, AI kontrakty a momentum tituly...")
            new_opps = self.discovery_engine.discover_and_register_opportunities(max_new_per_run=max_new)
            added_tickers: List[str] = []
            for opp in new_opps:
                clean_t = opp.ticker.strip().upper()
                if clean_t not in self.watchlist:
                    self.watchlist.append(clean_t)
                    added_tickers.append(clean_t)
                    logger.info(
                        f"🚀 [DAY TRADER DISCOVERY] Nový titul zařazen do aktivního intradenního skenování: "
                        f"{clean_t} ({opp.company_name}) | Cena: {opp.price:.2f} USD | \"{opp.headline[:65]}...\""
                    )
            return added_tickers
        except Exception as exc:
            logger.debug(f"Chyba při autonomním vyhledávání titulů pro Day Trader: {exc}")
            return []

    def evaluate_snapshot(self, snap: IntradaySnapshot) -> DayTradeSignal:
        """Vyhodnotí intradenní profil a spočítá DayTradeScore (0 až 100)."""
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
            reasons.append(f"5m RSI přeprodané ({snap.rsi_5m:.1f}) - možný odraz")
        elif snap.rsi_5m > 72.0:
            score -= 12.0
            reasons.append(f"5m RSI překoupené ({snap.rsi_5m:.1f}) - riziko korekce")

        # 5. Proražení 30minutového maxima (+15 b)
        if snap.is_breaking_high:
            score += 15.0
            reasons.append("Proražení 30minutového lokálního maxima")

        # Zaokrouhlení skóre
        score = max(0.0, min(100.0, round(score, 1)))

        # Výpočet dynamického Stop Lossu a Take Profitu
        sl_dist = max(snap.current_price * (self.stop_loss_pct / 100.0), snap.atr_5m * 1.2)
        tp_dist = max(snap.current_price * (self.take_profit_pct / 100.0), snap.atr_5m * 2.0)

        sl_price = round(snap.current_price - sl_dist, 2)
        tp_price = round(snap.current_price + tp_dist, 2)

        action = SignalAction.BUY if score >= 62.0 else SignalAction.HOLD
        currency = self.get_ticker_currency(snap.ticker)

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
            currency=currency,
        )

    def audit_active_positions(self) -> List[Dict[str, Any]]:
        """
        Zkontroluje otevřené day trade pozice v reálném čase:
        - Take Profit (+2.2 %)
        - Stop Loss (-1.2 %)
        - Breakeven Stop (při +1.0 % zisku posun na nákup)
        - Trailing Stop pro běžící zisky
        """
        closed_reports: List[Dict[str, Any]] = []
        open_positions = self.get_all_active_positions()

        for ticker, pos in list(open_positions.items()):
            snap = self.data_provider.fetch_intraday_snapshot(ticker)
            if not snap:
                continue

            current_price = snap.current_price
            entry_price = pos.entry_price_usd
            if entry_price <= 0:
                continue

            curr = self.get_ticker_currency(ticker)
            pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

            # Evidence nejvyšší dosažené ceny
            peak = self._peak_prices.get(ticker, entry_price)
            if current_price > peak:
                self._peak_prices[ticker] = current_price
                peak = current_price

            # 1. Breakeven Stop při zisku >= breakeven_trigger_pct
            if pnl_pct >= self.breakeven_trigger_pct and ticker not in self._breakeven_applied:
                pos.stop_loss_price = round(entry_price * 1.001, 2)
                self._breakeven_applied.add(ticker)
                if ticker in self.tracker.positions:
                    self.tracker.save_portfolio()
                logger.info(f"🛡️ [BREAKEVEN AKTIVOVÁN] {ticker}: Zisk dosáhl +{pnl_pct:.2f} %. Stop Loss posunut na nákupní cenu ({pos.stop_loss_price:.2f} {curr}).")
                print(f"  🛡️ [BREAKEVEN POSUNUT] {ticker}: Zisk +{pnl_pct:.2f} % -> Nový SL: {pos.stop_loss_price:.2f} {curr}")

            # 2. Dynamický Trailing Stop pro běžící zisky
            if pnl_pct >= 2.0:
                trailing_level = round(peak - (snap.atr_5m * 1.2), 2)
                if trailing_level > pos.stop_loss_price:
                    pos.stop_loss_price = trailing_level
                    if ticker in self.tracker.positions:
                        self.tracker.save_portfolio()
                    logger.info(f"📈 [TRAILING STOP POSUNUT] {ticker}: Nový Stop Loss {trailing_level:.2f} {curr} (High: {peak:.2f} {curr}).")
                    print(f"  📈 [TRAILING STOP] {ticker}: Nový SL: {trailing_level:.2f} {curr}")

            # 3. Kontrola Take Profitu
            if current_price >= pos.take_profit_price:
                msg = f"🎯 [TAKE PROFIT ZASAŽEN] {ticker}: Cena {current_price:.2f} >= TP {pos.take_profit_price:.2f} {curr} (+{pnl_pct:.2f} % zisk)!"
                logger.info(msg)
                print(f"  {msg}")
                exit_sig = PositionExitSignal(
                    ticker=ticker,
                    should_exit=True,
                    reason=ExitReason.TAKE_PROFIT,
                    exit_price=current_price,
                    message=f"Bleskový Day Trade Take Profit zasažen (+{pnl_pct:.2f} %)"
                )
                res = self.executor.execute_sell(ticker, exit_sig, pos.quantity)
                trade_pnl_eur = pos.allocated_eur * (pnl_pct / 100.0)
                self._daily_realized_pnl_eur += trade_pnl_eur
                self._consecutive_losses = 0
                self._session_trades_count += 1
                self._session_wins += 1
                closed_reports.append({"ticker": ticker, "reason": "TAKE_PROFIT", "pnl_pct": pnl_pct, "res": res})
                self._peak_prices.pop(ticker, None)
                self._breakeven_applied.discard(ticker)
                if ticker in self._dry_run_positions:
                    del self._dry_run_positions[ticker]
                continue

            # 4. Kontrola Stop Lossu
            if current_price <= pos.stop_loss_price:
                msg = f"🛑 [STOP LOSS ZASAŽEN] {ticker}: Cena {current_price:.2f} <= SL {pos.stop_loss_price:.2f} {curr} ({pnl_pct:.2f} % ztráta)!"
                logger.info(msg)
                print(f"  {msg}")
                exit_sig = PositionExitSignal(
                    ticker=ticker,
                    should_exit=True,
                    reason=ExitReason.STOP_LOSS,
                    exit_price=current_price,
                    message=f"Day Trade Stop Loss aktivován k ochraně kapitálu ({pnl_pct:.2f} %)"
                )
                res = self.executor.execute_sell(ticker, exit_sig, pos.quantity)
                trade_pnl_eur = pos.allocated_eur * (pnl_pct / 100.0)
                self._daily_realized_pnl_eur += trade_pnl_eur
                self._consecutive_losses += 1
                self._session_trades_count += 1

                # Kontrola limitů Circuit Breakeru
                if self._consecutive_losses >= self.max_consecutive_losses or self._daily_realized_pnl_eur <= -self.max_daily_loss_eur:
                    self._circuit_breaker_triggered = True
                    cb_msg = (
                        f"🛑 [CIRCUIT BREAKER AKTIVOVÁN] Dosažen limit denní ztráty ({self._daily_realized_pnl_eur:.2f} EUR) "
                        f"nebo {self._consecutive_losses} ztrát v řadě! Nové nákupy pro dnešek pozastaveny pro ochranu kapitálu."
                    )
                    logger.warning(cb_msg)
                    print(f"\n{cb_msg}\n")

                closed_reports.append({"ticker": ticker, "reason": "STOP_LOSS", "pnl_pct": pnl_pct, "res": res})
                self._peak_prices.pop(ticker, None)
                self._breakeven_applied.discard(ticker)
                if ticker in self._dry_run_positions:
                    del self._dry_run_positions[ticker]
                continue

        return closed_reports

    def execute_daytrade_buys(self, signals: List[DayTradeSignal]) -> List[Dict[str, Any]]:
        """Vybere nejlepší BUY signály a bleskově otevře pozice až do limitu."""
        opened_reports: List[Dict[str, Any]] = []

        if self._circuit_breaker_triggered:
            logger.info("Circuit Breaker je aktivní: nové nákupy jsou pro dnešek zablokovány.")
            return opened_reports

        if self.current_regime_snapshot and not self.current_regime_snapshot.allow_new_buys:
            logger.warning(
                f"Makro režim ({self.current_regime_snapshot.regime.value}) blokuje nové long nákupy: {self.current_regime_snapshot.reasoning}"
            )
            return opened_reports

        open_positions = self.get_all_active_positions()
        available_slots = max(0, self.max_active_positions - len(open_positions))

        buy_signals = [s for s in signals if s.action == SignalAction.BUY and s.ticker not in open_positions]
        tradeable_buys = [s for s in buy_signals if MarketHours.is_ticker_market_open(s.ticker)]
        closed_market_buys = [s for s in buy_signals if not MarketHours.is_ticker_market_open(s.ticker)]

        if available_slots <= 0:
            if buy_signals:
                cand_list = ", ".join([f"{s.ticker} ({s.score:.0f})" for s in buy_signals[:3]])
                print(
                    f"  ⏸️ [KAPACITA PLNÁ] Nalezen BUY signál ({cand_list}), ale všechny sloty jsou obsazeny "
                    f"({len(open_positions)}/{self.max_active_positions}). Čekám na uvolnění slotu (TP nebo SL)."
                )
            return opened_reports

        if not tradeable_buys:
            if closed_market_buys:
                closed_list = ", ".join([s.ticker for s in closed_market_buys[:3]])
                print(f"  🌙 [BURZA ZAVŘENA] BUY signál pro {closed_list}, ale příslušná burza je uzavřena (např. Evropa po 17:30 CET).")
            return opened_reports

        tradeable_buys.sort(key=lambda s: s.score, reverse=True)

        for sig in tradeable_buys[:available_slots]:
            curr = getattr(sig, "currency", self.get_ticker_currency(sig.ticker))

            # --- 🤖 BLESKOVÝ AI RISK CHECK (AMD NPU / Lemonade Server) ---
            headlines: List[str] = []
            try:
                raw_news = yf.Ticker(sig.ticker).news or []
                for n in raw_news[:3]:
                    t_text = n.get("title") or n.get("content", {}).get("title", "")
                    if t_text:
                        headlines.append(t_text.strip())
            except Exception:
                pass

            ai_check = self.sentiment_engine.fast_daytrade_check(
                ticker=sig.ticker,
                current_price=sig.current_price,
                technical_score=sig.score,
                headlines=headlines,
                timeout_seconds=4.0,
            )

            # Pokud AI detekuje bezprostřední nebezpečí nebo krach -> VETO
            if not ai_check.get("approved", True):
                veto_msg = (
                    f"  🚫 [AI VETO - NÁKUP ZAMÍTNUT] {sig.ticker} | "
                    f"Riziko: {ai_check.get('reason', 'Detekováno vysoké riziko')} | Nákup zrušen!"
                )
                logger.warning(veto_msg)
                print(veto_msg)
                continue

            if ai_check.get("llm_used", False):
                ai_info_msg = (
                    f"  🤖 [AI NPU POTVRZENO] {sig.ticker} | "
                    f"Sentiment: {ai_check['sentiment']:+.2f} | {ai_check['reason']}"
                )
                logger.info(ai_info_msg)
                print(ai_info_msg)

            std_signal = TradingSignal(
                ticker=sig.ticker,
                action=SignalAction.BUY,
                confidence=0.85,
                suggested_allocation_eur=self.allocation_per_trade_eur,
                stop_loss_price=sig.stop_loss_price,
                take_profit_price=sig.take_profit_price,
                technical_score=round(min(50.0, sig.score / 2.0), 1),
                sentiment_score=round(max(0.0, min(50.0, (ai_check.get("sentiment", 0.0) * 25.0 + 25.0))), 1),
                total_score=sig.score,
                time_horizon="INTRADAY",
                current_price=sig.current_price,
                atr_14=sig.atr_5m,
                rsi_14=sig.rsi_5m,
                currency=curr,
            )

            if self.dry_run:
                print(
                    f"  ⚡ [SIMULACE NÁKUPU (--dry-run)] {sig.ticker} | Skóre: {sig.score:.1f}/100 | "
                    f"Cena: {sig.current_price:.2f} {curr} | SL: {sig.stop_loss_price:.2f} | "
                    f"TP: {sig.take_profit_price:.2f} | Důvody: {', '.join(sig.reasons[:2])}"
                )
            else:
                print(
                    f"  🚀 [ODESÍLÁM NÁKUP T212] {sig.ticker} | Skóre: {sig.score:.1f}/100 | "
                    f"Cena: {sig.current_price:.2f} {curr} | SL: {sig.stop_loss_price:.2f} | "
                    f"TP: {sig.take_profit_price:.2f} | Důvody: {', '.join(sig.reasons[:2])}"
                )

            logger.info(
                f"🚀 [DAY TRADE VSTUP] {sig.ticker} | Skóre: {sig.score:.1f}/100 | "
                f"Cena: {sig.current_price:.2f} {curr} | TP: {sig.take_profit_price:.2f} | "
                f"SL: {sig.stop_loss_price:.2f} | Důvody: {', '.join(sig.reasons[:2])}"
            )

            buy_res = self.executor.execute_buy(std_signal)
            if buy_res:
                self._peak_prices[sig.ticker] = sig.current_price
                if self.dry_run:
                    qty = buy_res.get("quantity", 1.0)
                    self._dry_run_positions[sig.ticker] = PositionRecord(
                        ticker=sig.ticker,
                        t212_ticker=TickerMapper.to_t212_ticker(sig.ticker),
                        quantity=qty,
                        entry_price_usd=sig.current_price,
                        stop_loss_price=sig.stop_loss_price,
                        take_profit_price=sig.take_profit_price,
                        allocated_eur=self.allocation_per_trade_eur,
                        status="OPEN",
                    )
                opened_reports.append({
                    "ticker": sig.ticker,
                    "price": sig.current_price,
                    "score": sig.score,
                    "tp": sig.take_profit_price,
                    "sl": sig.stop_loss_price,
                })
                time.sleep(0.5)

        return opened_reports

    def run_one_scan(self) -> Dict[str, Any]:
        """Provede jeden rychlý intradenní cyklus (audit pozic + rychlý batch sken grafů + nákupy)."""
        now = datetime.now()
        logger.info(f"⚡ [DAY TRADER SCAN] Skenuji {len(self.watchlist)} intradenních titulů ({now.strftime('%H:%M:%S')})...")

        # 0. Vyhodnocení globálního tržního režimu (Macro Regime Classifier na NPU)
        try:
            regime = self.regime_detector.evaluate_regime()
            self.current_regime_snapshot = regime
            self.take_profit_pct = regime.take_profit_pct
            self.stop_loss_pct = regime.stop_loss_pct
            self.breakeven_trigger_pct = regime.breakeven_pct
            self.max_active_positions = regime.max_active_positions
        except Exception as reg_err:
            logger.debug(f"Chyba při detekci makro režimu: {reg_err}")

        # 1. Audit držených pozic
        closed = self.audit_active_positions()

        # 2. Rychlé paralelní stažení a výpočet indikátorů pro celý široký watchlist
        snapshots = self.data_provider.fetch_batch_snapshots(self.watchlist)

        # 3. Vyhodnocení signálů
        evaluated_signals: List[DayTradeSignal] = []
        for ticker, snap in snapshots.items():
            try:
                sig = self.evaluate_snapshot(snap)
                evaluated_signals.append(sig)
            except Exception as exc:
                logger.debug(f"Chyba při vyhodnocení {ticker}: {exc}")

        # 4. Exekuce nových nákupů
        opened = self.execute_daytrade_buys(evaluated_signals)

        return {
            "timestamp": now.isoformat(),
            "closed_trades": closed,
            "opened_trades": opened,
            "signals": evaluated_signals,
            "total_monitored": len(self.watchlist),
        }

    def start_loop(self, scan_interval_seconds: int = 60, discovery_interval_minutes: int = 15) -> None:
        """
        Spustí nepřetržitou Day Trading smyčku (výchozí interval: 60 sekund).
        Průběžně vyhledává nové tituly z médií a provádí rychlé vstupy a výstupy.
        """
        print("\n" + "=" * 70)
        print("          ⚡ AI DAY TRADER & LEVERAGED MOMENTUM ENGINE ⚡       ")
        print("=" * 70)
        print(f"  Frekvence skenování:  Každých {scan_interval_seconds} sekund")
        try:
            from src.strategy.scheduled_scanner import ScheduledMarketScanner
            qualified = ScheduledMarketScanner.load_qualified_tickers()
            qual_note = f" (včetně {len(qualified)} s konsenzem Rady AI)" if qualified else ""
        except Exception:
            qual_note = ""
        print(f"  Sledováno titulů:     {len(self.watchlist)} aktivních akcií & pákových ETF{qual_note}")
        print(f"  Autonomní discovery:  Každých {discovery_interval_minutes} minut hledá nové akcie z médií")
        print(f"  Alokace na obchod:    {self.allocation_per_trade_eur:,.0f} EUR")
        print(f"  Max otevřených pozic: {self.max_active_positions}")
        print(f"  Cílový Take Profit:   +{self.take_profit_pct:.1f} % (nebo 2x ATR)")
        print(f"  Těsný Stop Loss:      -{self.stop_loss_pct:.1f} % (nebo 1.2x ATR)")
        print(f"  Breakeven Stop:       Aktivní při zisku +{self.breakeven_trigger_pct:.1f} %")
        print(f"  Circuit Breaker:      Stop po {self.max_consecutive_losses} ztrátách nebo -{self.max_daily_loss_eur:.0f} EUR ztráty")
        print(f"  AI Validátor (NPU):   AKTIVNÍ (Lemonade NPU provádí bleskový sanity check zpráv)")
        print(f"  Režim:                {'DRY-RUN (Simulace)' if self.dry_run else 'LIVE EXEKUCE (Demo účet)'}")
        print("=" * 70 + "\n")
        print("Stiskněte Ctrl+C pro bezpečné zastavení Day Tradera.\n")

        # Úvodní prohledání zpráv na čerstvé objevy
        self.discover_new_daytrade_tickers(max_new=4)
        self._last_discovery_ts = time.time()

        try:
            while True:
                market_stat = MarketHours.get_market_status()
                now_ts = time.time()

                # Periodické hledání nových titulů ze zpráv každých X minut
                if (now_ts - self._last_discovery_ts) >= (discovery_interval_minutes * 60):
                    self._last_discovery_ts = now_ts
                    self.discover_new_daytrade_tickers(max_new=4)

                if market_stat["is_open"]:
                    cycle_res = self.run_one_scan()
                    signals = cycle_res.get("signals", [])

                    # Tisk živého terminálového přehledu
                    print(f"\n--- [DAY TRADER STATUS {datetime.now().strftime('%H:%M:%S')}] ---")
                    mode_label = "🟡 SIMULACE (--dry-run: příkazy se na burzu neposílají)" if self.dry_run else "🟢 LIVE EXEKUCE (Reálný demo účet Trading 212)"
                    print(f"Režim bota:              {mode_label}")
                    if self.current_regime_snapshot:
                        snap = self.current_regime_snapshot
                        if snap.regime == MarketRegime.RISK_ON_BULL:
                            reg_icon = "🟢 RISK-ON BULL (Silný růst)"
                        elif snap.regime == MarketRegime.RISK_OFF_PANIC:
                            reg_icon = "🔴 RISK-OFF PANIC (Výprodeje - nákupy STOP)"
                        else:
                            reg_icon = "🟡 NEUTRAL CHOPPY (Rychlé skalpy)"
                        llm_tag = " [NPU AI]" if snap.llm_used else ""
                        print(f"Tržní AI režim:          {reg_icon}{llm_tag} | VIX: {snap.vix:.1f} | SPY: {snap.spy_change_pct:+.2f} % | TP: +{snap.take_profit_pct:.1f} % | BE: +{snap.breakeven_pct:.1f} %")
                    print(f"Sledováno v reálném čase: {len(self.watchlist)} titulů (Páková ETF, Tech lídři, Nové zprávy)")
                    win_pct = (self._session_wins / self._session_trades_count * 100.0) if self._session_trades_count > 0 else 0.0
                    pnl_sign = "+" if self._daily_realized_pnl_eur >= 0 else ""
                    cb_status = " 🛑 (CIRCUIT BREAKER AKTIVNÍ)" if self._circuit_breaker_triggered else " 🟢 (Aktivní)"
                    print(f"Denní PnL seance:        {pnl_sign}{self._daily_realized_pnl_eur:.2f} EUR | Obchody: {self._session_trades_count} (Win: {self._session_wins}, Win Rate: {win_pct:.0f} %){cb_status}")
                    open_pos = self.get_all_active_positions()
                    if open_pos:
                        print(f"Otevřené pozice ({len(open_pos)} / {self.max_active_positions}):")
                        for t, p in open_pos.items():
                            peak = self._peak_prices.get(t, p.entry_price_usd)
                            curr = self.get_ticker_currency(t)
                            sim_badge = " [SIMULACE]" if t in self._dry_run_positions else ""
                            print(
                                f"  • {t:<7}{sim_badge} | Nákup: {p.entry_price_usd:7.2f} {curr} | "
                                f"SL: {p.stop_loss_price:7.2f} | TP: {p.take_profit_price:7.2f} | "
                                f"Alokováno: {p.allocated_eur:.0f} EUR"
                            )
                    else:
                        print(f"Aktuálně nejsou otevřeny žádné pozice (Kapacita: {self.max_active_positions} volných slotů).")

                    print("\nTop intradenní momentum signály:")
                    signals.sort(key=lambda s: s.score, reverse=True)
                    for s in signals[:8]:
                        above_str = "NAD VWAP ▲" if (s.current_price > s.vwap) else "POD VWAP ▼"
                        market_open = MarketHours.is_ticker_market_open(s.ticker)
                        market_note = "" if market_open else " (🌙 Burza zavřena)"
                        action_icon = "🟢 BUY " if s.action == SignalAction.BUY else "⚪ HOLD"
                        curr = getattr(s, "currency", self.get_ticker_currency(s.ticker))
                        print(
                            f"  {action_icon} {s.ticker:<7}{market_note} | Skóre: {s.score:4.1f}/100 | "
                            f"Cena: {s.current_price:7.2f} {curr} ({above_str}) | "
                            f"5m RSI: {s.rsi_5m:4.1f} | Objem: {s.volume_surge:3.1f}x"
                        )
                    print(f"Příští sken za {scan_interval_seconds} sekund...\n")

                else:
                    print(f"🌙 [Trh je zavřen] {market_stat['status_text']}. Čekám na otevření...")

                time.sleep(scan_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Day Trader byl zastaven uživatelem (Ctrl+C).")
            print("\n🛑 Day Trader byl bezpečně ukončen.")
