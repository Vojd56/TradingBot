import logging
from typing import Any, Dict, Optional
import yfinance as yf
from config import settings
from src.broker.t212_client import Trading212APIError, Trading212Client
from src.execution.portfolio_tracker import PortfolioTracker
from src.execution.ticker_mapper import TickerMapper
from src.strategy.models import PositionExitSignal, SignalAction, TradingSignal
from src.strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Exekuční engine pro bezpečné odesílání tržních příkazů na Trading 212 API.
    Obsahuje bezpečnostní pojistky proti chybné exekuci na live účtu.
    """

    def __init__(
        self,
        client: Optional[Trading212Client] = None,
        tracker: Optional[PortfolioTracker] = None,
        risk_manager: Optional[RiskManager] = None,
        dry_run: bool = False,
    ):
        self.client = client or Trading212Client()
        self.tracker = tracker or PortfolioTracker()
        self.risk_manager = risk_manager or RiskManager()
        self.dry_run = dry_run

        # Pojistka pro kontrolu prostředí
        self._verify_environment_safety()

    def _verify_environment_safety(self) -> None:
        """Garantuje, že live exekuce neproběhne bez výslovného souhlasu."""
        if settings.trading_mode == "live":
            if settings.i_understand_live_risk.strip().lower() != "yes":
                raise RuntimeError(
                    "Bezpečnostní stop: TRADING_MODE je live, ale I_UNDERSTAND_LIVE_RISK není 'yes'!"
                )
            logger.warning("POZOR: OrderExecutor inicializován v LIVE režimu (reálné finance)!")
        else:
            logger.info("OrderExecutor inicializován v bezpečném DEMO režimu.")

    def get_exchange_rates(self) -> Dict[str, float]:
        """Získá aktuální tržní kurzy EUR/USD, EUR/GBP a EUR/CHF."""
        rates = {"EURUSD": 1.08, "EURGBP": 0.84, "EURCHF": 0.94}
        try:
            eur_usd = yf.Ticker("EURUSD=X").fast_info.get("last_price")
            if eur_usd and eur_usd > 0.5:
                rates["EURUSD"] = float(eur_usd)
        except Exception:
            pass

        try:
            eur_gbp = yf.Ticker("EURGBP=X").fast_info.get("last_price")
            if eur_gbp and eur_gbp > 0.4:
                rates["EURGBP"] = float(eur_gbp)
        except Exception:
            pass

        try:
            eur_chf = yf.Ticker("EURCHF=X").fast_info.get("last_price")
            if eur_chf and eur_chf > 0.5:
                rates["EURCHF"] = float(eur_chf)
        except Exception:
            pass

        return rates

    def get_eur_usd_rate(self) -> float:
        """Vrátí aktuální kurz EUR/USD."""
        return self.get_exchange_rates().get("EURUSD", 1.08)

    def execute_buy(self, signal: TradingSignal) -> Optional[Dict[str, Any]]:
        """
        Zadá nákupní příkaz na základě signálu a zaeviduje pozici.
        """
        if signal.action != SignalAction.BUY:
            logger.warning(f"Ignoruji pokus o nákup pro {signal.ticker}: akce není BUY ({signal.action}).")
            return None

        rates = self.get_exchange_rates()
        currency = getattr(signal, "currency", "USD")
        quantity = self.risk_manager.calculate_shares_quantity(
            allocated_eur=signal.suggested_allocation_eur,
            stock_price=signal.current_price,
            currency=currency,
            eur_usd_rate=rates["EURUSD"],
            eur_gbp_rate=rates["EURGBP"],
            eur_chf_rate=rates.get("EURCHF", 0.94),
        )

        if quantity <= 0.0:
            logger.warning(f"Vypočtené množství pro {signal.ticker} je {quantity} ks. Příkaz nebyl odeslán.")
            return None

        t212_ticker = TickerMapper.to_t212_ticker(signal.ticker)

        logger.info(
            f"EXEKUCE NÁKUPU: {signal.ticker} ({t212_ticker}) | "
            f"Množství: {quantity} ks | Alokace: {signal.suggested_allocation_eur:.2f} EUR "
            f"(@ {signal.current_price:.2f} {currency})"
        )

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Simulace nákupu: {signal.ticker} ({t212_ticker}) | "
                f"Množství: {quantity} ks (žádný příkaz nebyl odeslán)."
            )
            return {
                "id": "dry_run_buy",
                "ticker": t212_ticker,
                "quantity": quantity,
                "dry_run": True,
            }

        try:
            try:
                order_res = self.client.place_market_order(ticker=t212_ticker, quantity=quantity, extended_hours=True)
            except Trading212APIError:
                order_res = self.client.place_market_order(ticker=t212_ticker, quantity=quantity)

            # Výpočet stops, pokud nejsou přímo v signálu
            sl = signal.stop_loss_price or round(signal.current_price * 0.92, 2)
            tp = signal.take_profit_price or round(signal.current_price * 1.15, 2)

            # Záznam do perzistentního trackeru
            self.tracker.record_buy(
                ticker=signal.ticker,
                quantity=quantity,
                entry_price_usd=signal.current_price,
                stop_loss_price=sl,
                take_profit_price=tp,
                allocated_eur=signal.suggested_allocation_eur,
            )
            return order_res

        except Trading212APIError as exc:
            logger.error(f"Trading 212 API odmítlo nákupní příkaz pro {t212_ticker}: {exc}")
            return None

    def execute_sell(
        self,
        ticker: str,
        exit_signal: PositionExitSignal,
        quantity: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Zadá prodejní příkaz k uzavření pozice (záporné quantity na T212 API).
        """
        open_positions = self.tracker.get_open_positions()
        rec = open_positions.get(ticker)

        qty_to_sell = quantity or (rec.quantity if rec else None)
        if not qty_to_sell or qty_to_sell <= 0.0:
            logger.warning(f"Nelze provést prodej {ticker}: nulové nebo neznámé množství ({qty_to_sell}).")
            return None

        t212_ticker = TickerMapper.to_t212_ticker(ticker)

        logger.info(
            f"EXEKUCE PRODEJE: {ticker} ({t212_ticker}) | "
            f"Množství: {qty_to_sell} ks @ {exit_signal.exit_price:.2f} | "
            f"Důvod: {exit_signal.reason.value if exit_signal.reason else 'N/A'} - {exit_signal.message}"
        )

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Simulace prodeje: {ticker} ({t212_ticker}) | "
                f"Množství: {qty_to_sell} ks (žádný příkaz nebyl odeslán)."
            )
            return {
                "id": "dry_run_sell",
                "ticker": t212_ticker,
                "quantity": qty_to_sell,
                "dry_run": True,
            }

        try:
            # Na T212 API se prodej zadává se záporným množstvím
            sell_qty = -abs(round(qty_to_sell, 6))
            try:
                order_res = self.client.place_market_order(
                    ticker=t212_ticker,
                    quantity=sell_qty,
                    extended_hours=True
                )
            except Trading212APIError:
                order_res = self.client.place_market_order(
                    ticker=t212_ticker,
                    quantity=sell_qty
                )

            reason_str = f"{exit_signal.reason.value}: {exit_signal.message}" if exit_signal.reason else exit_signal.message
            self.tracker.record_sell(
                ticker=ticker,
                close_price_usd=exit_signal.exit_price,
                close_reason=reason_str,
            )
            return order_res

        except Trading212APIError as exc:
            logger.error(f"Trading 212 API odmítlo prodejní příkaz pro {t212_ticker}: {exc}")
            return None
