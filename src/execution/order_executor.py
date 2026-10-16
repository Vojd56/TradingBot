from datetime import datetime
import logging
from typing import Any, Dict, Optional
import yfinance as yf
from config import settings
from src.broker.t212_client import Trading212APIError, Trading212Client
from src.execution.broker_adapter import (
    BrokerAdapter,
    PaperBrokerAdapter,
    T212BrokerAdapter,
)
from src.execution.portfolio_tracker import PortfolioTracker
from src.execution.ticker_mapper import TickerMapper
from src.quant.risk_engine import (
    ApprovedOrder,
    OrderIntent,
    PortfolioState,
    RejectedOrder,
    RiskGatekeeper,
)
from src.strategy.models import PositionExitSignal, SignalAction, TradingSignal
from src.strategy.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Exekuční engine pro bezpečné odesílání tržních příkazů s povinnou verifikací
    přes RiskGatekeeper a volitelným broker adaptérem.
    """

    def __init__(
        self,
        client: Optional[Trading212Client] = None,
        tracker: Optional[PortfolioTracker] = None,
        risk_manager: Optional[RiskManager] = None,
        gatekeeper: Optional[RiskGatekeeper] = None,
        broker_adapter: Optional[BrokerAdapter] = None,
        dry_run: bool = False,
    ):
        self.client = client or Trading212Client()
        self.tracker = tracker or PortfolioTracker()
        self.risk_manager = risk_manager or RiskManager()
        self.gatekeeper = gatekeeper or RiskGatekeeper()
        self.dry_run = dry_run

        if broker_adapter is not None:
            self.broker_adapter = broker_adapter
        elif self.dry_run or settings.trading_mode == "backtest":
            self.broker_adapter = PaperBrokerAdapter()
        else:
            self.broker_adapter = T212BrokerAdapter(client=self.client)

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

    def get_portfolio_state(self) -> PortfolioState:
        """Sestaví aktuální stav portfolia pro vyhodnocení v RiskGatekeeperu."""
        open_positions = self.tracker.get_open_positions()
        open_dict = {ticker: rec.quantity for ticker, rec in open_positions.items()}

        # Missing account data must stop execution; zero is a valid balance.
        cash = float(self.broker_adapter.get_account_cash())

        positions_value = sum(rec.allocated_eur for rec in open_positions.values())
        equity = cash + positions_value

        return PortfolioState(
            equity=round(equity, 2),
            cash=round(cash, 2),
            daily_realized_pnl=0.0,
            daily_unrealized_pnl=0.0,
            open_positions_count=len(open_positions),
            open_positions=open_dict,
        )

    def execute_buy(self, signal: TradingSignal) -> Optional[Dict[str, Any]]:
        """
        Zadá nákupní příkaz na základě signálu po povinné prověrce v RiskGatekeeperu.
        ŽÁDNÝ PŘÍKAZ NEMŮŽE BÝT ODESLÁN BEZ RAZÍTKA ApprovedOrder.
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

        sl = signal.stop_loss_price or round(signal.current_price * 0.92, 2)
        tp = signal.take_profit_price or round(signal.current_price * 1.15, 2)

        # 1. POVINNÉ SCHVÁLENÍ V RISKGATEKEEPERU
        intent = OrderIntent(
            ticker=signal.ticker,
            side="BUY",
            entry_price=signal.current_price,
            stop_loss_price=sl,
            take_profit_price=tp,
            desired_shares=quantity,
            suggested_allocation_eur=signal.suggested_allocation_eur,
            strategy_id="ORDER_EXECUTOR",
            timestamp=datetime.now(),
        )

        portfolio_state = self.get_portfolio_state()
        eval_result = self.gatekeeper.evaluate_order_intent(intent, portfolio_state)

        if not eval_result.approved or not isinstance(eval_result, ApprovedOrder):
            logger.warning(f"Příkaz nákupu pro {signal.ticker} ZAMÍTNUT RiskGatekeeperem: {eval_result.reason}")
            return None

        approved_order: ApprovedOrder = eval_result
        t212_ticker = TickerMapper.to_t212_ticker(signal.ticker)

        logger.info(
            f"EXEKUCE NÁKUPU (SCHVÁLENO): {signal.ticker} ({t212_ticker}) | "
            f"Množství: {approved_order.shares} ks | Alokace: {signal.suggested_allocation_eur:.2f} EUR "
            f"(@ {signal.current_price:.2f} {currency}) [ID: {approved_order.client_order_id}]"
        )

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Simulace nákupu: {signal.ticker} ({t212_ticker}) | "
                f"Množství: {approved_order.shares} ks (žádný příkaz nebyl odeslán)."
            )
            return {
                "id": approved_order.client_order_id,
                "ticker": t212_ticker,
                "quantity": approved_order.shares,
                "dry_run": True,
            }

        try:
            order_res = self.broker_adapter.place_order(approved_order)
            if not order_res.get("is_paper"):
                logger.warning("Pokyn %s byl odeslán; tracker čeká na potvrzené fill události.", order_res.get("id"))
                return None

            # Záznam do perzistentního trackeru
            self.tracker.record_buy(
                ticker=signal.ticker,
                quantity=approved_order.shares,
                entry_price_usd=signal.current_price,
                stop_loss_price=approved_order.stop_loss,
                take_profit_price=approved_order.take_profit,
                allocated_eur=signal.suggested_allocation_eur,
            )
            return order_res

        except Exception as exc:
            logger.error(f"Broker adaptér selhal při nákupním příkazu pro {t212_ticker}: {exc}")
            return None

    def execute_sell(
        self,
        ticker: str,
        exit_signal: PositionExitSignal,
        quantity: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Zadá prodejní příkaz k uzavření pozice po prověrce v RiskGatekeeperu.
        """
        open_positions = self.tracker.get_open_positions()
        rec = open_positions.get(ticker)

        qty_to_sell = quantity or (rec.quantity if rec else None)
        if not qty_to_sell or qty_to_sell <= 0.0:
            logger.warning(f"Nelze provést prodej {ticker}: nulové nebo neznámé množství ({qty_to_sell}).")
            return None

        # 1. POVINNÉ SCHVÁLENÍ V RISKGATEKEEPERU
        intent = OrderIntent(
            ticker=ticker,
            side="SELL",
            entry_price=exit_signal.exit_price,
            stop_loss_price=round(exit_signal.exit_price * 0.95, 2),
            desired_shares=qty_to_sell,
            strategy_id="ORDER_EXECUTOR",
            timestamp=datetime.now(),
        )

        portfolio_state = self.get_portfolio_state()
        eval_result = self.gatekeeper.evaluate_order_intent(intent, portfolio_state)

        if not eval_result.approved or not isinstance(eval_result, ApprovedOrder):
            logger.warning(f"Příkaz prodeje pro {ticker} ZAMÍTNUT RiskGatekeeperem: {eval_result.reason}")
            return None

        approved_order: ApprovedOrder = eval_result
        t212_ticker = TickerMapper.to_t212_ticker(ticker)

        logger.info(
            f"EXEKUCE PRODEJE (SCHVÁLENO): {ticker} ({t212_ticker}) | "
            f"Množství: {approved_order.shares} ks @ {exit_signal.exit_price:.2f} | "
            f"Důvod: {exit_signal.reason.value if exit_signal.reason else 'N/A'} - {exit_signal.message} "
            f"[ID: {approved_order.client_order_id}]"
        )

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Simulace prodeje: {ticker} ({t212_ticker}) | "
                f"Množství: {approved_order.shares} ks (žádný příkaz nebyl odeslán)."
            )
            return {
                "id": approved_order.client_order_id,
                "ticker": t212_ticker,
                "quantity": approved_order.shares,
                "dry_run": True,
            }

        try:
            order_res = self.broker_adapter.place_order(approved_order)
            if not order_res.get("is_paper"):
                logger.warning("Prodej %s byl odeslán; realizovaný PnL se bez broker fill nezapisuje.", order_res.get("id"))
                return None

            reason_str = f"{exit_signal.reason.value}: {exit_signal.message}" if exit_signal.reason else exit_signal.message
            self.tracker.record_sell(
                ticker=ticker,
                close_price_usd=exit_signal.exit_price,
                close_reason=reason_str,
            )
            return order_res

        except Exception as exc:
            logger.error(f"Broker adaptér selhal při prodejním příkazu pro {t212_ticker}: {exc}")
            return None

