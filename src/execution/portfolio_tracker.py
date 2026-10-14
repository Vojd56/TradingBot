import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from src.execution.ticker_mapper import TickerMapper

logger = logging.getLogger(__name__)


class PositionRecord(BaseModel):
    """Záznam o sledované obchodní pozici."""
    ticker: str
    t212_ticker: str
    quantity: float
    entry_price_usd: float
    entry_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    stop_loss_price: float
    take_profit_price: float
    allocated_eur: float
    status: Literal["OPEN", "CLOSED"] = "OPEN"
    close_price_usd: Optional[float] = None
    close_date: Optional[str] = None
    close_reason: Optional[str] = None
    realized_pnl_usd: Optional[float] = None


class PortfolioTracker:
    """
    Perzistentní správa stavu portfolia a historie obchodů v souboru data/portfolio_state.json.
    """

    def __init__(self, storage_path: str = "data/portfolio_state.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.positions: Dict[str, PositionRecord] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self._save()
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_positions = data.get("positions", {})
                self.positions = {
                    ticker: PositionRecord(**rec)
                    for ticker, rec in raw_positions.items()
                }
                self.trade_history = data.get("trade_history", [])
        except Exception as exc:
            logger.error(f"Chyba při načítání stavu portfolia ze souboru {self.storage_path}: {exc}")
            self.positions = {}
            self.trade_history = []

    def _save(self) -> None:
        try:
            payload = {
                "last_updated": datetime.now().isoformat(),
                "positions": {t: rec.model_dump() for t, rec in self.positions.items()},
                "trade_history": self.trade_history,
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"Chyba při ukládání stavu portfolia do {self.storage_path}: {exc}")

    def get_open_positions(self) -> Dict[str, PositionRecord]:
        """Vrátí všechny aktuálně otevřené pozice."""
        return {
            ticker: rec
            for ticker, rec in self.positions.items()
            if rec.status == "OPEN"
        }

    def record_buy(
        self,
        ticker: str,
        quantity: float,
        entry_price_usd: float,
        stop_loss_price: float,
        take_profit_price: float,
        allocated_eur: float,
    ) -> PositionRecord:
        """Zaznamená nový nákup do perzistentního stavu."""
        t212_ticker = TickerMapper.to_t212_ticker(ticker)
        record = PositionRecord(
            ticker=ticker,
            t212_ticker=t212_ticker,
            quantity=quantity,
            entry_price_usd=entry_price_usd,
            entry_date=datetime.now().isoformat(),
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            allocated_eur=allocated_eur,
            status="OPEN",
        )
        self.positions[ticker] = record
        self.trade_history.append({
            "action": "BUY",
            "ticker": ticker,
            "quantity": quantity,
            "price_usd": entry_price_usd,
            "allocated_eur": allocated_eur,
            "timestamp": record.entry_date,
        })
        self._save()
        logger.info(f"Zaznamenán nákup: {ticker} | {quantity} ks @ {entry_price_usd} USD")
        return record

    def record_sell(
        self,
        ticker: str,
        close_price_usd: float,
        close_reason: str,
    ) -> Optional[PositionRecord]:
        """Zaznamená uzavření pozice a vypočte realizovaný zisk/ztrátu."""
        if ticker not in self.positions or self.positions[ticker].status != "OPEN":
            logger.warning(f"Nelze uzavřít neexistující nebo již uzavřenou pozici: {ticker}")
            return None

        record = self.positions[ticker]
        record.status = "CLOSED"
        record.close_price_usd = close_price_usd
        record.close_date = datetime.now().isoformat()
        record.close_reason = close_reason

        # PnL v USD
        pnl_usd = (close_price_usd - record.entry_price_usd) * record.quantity
        record.realized_pnl_usd = round(pnl_usd, 2)

        self.trade_history.append({
            "action": "SELL",
            "ticker": ticker,
            "quantity": record.quantity,
            "entry_price_usd": record.entry_price_usd,
            "close_price_usd": close_price_usd,
            "realized_pnl_usd": record.realized_pnl_usd,
            "reason": close_reason,
            "timestamp": record.close_date,
        })
        self._save()
        logger.info(
            f"Zaznamenán prodej: {ticker} @ {close_price_usd} USD | "
            f"Důvod: {close_reason} | PnL: {record.realized_pnl_usd:+.2f} USD"
        )
        return record

    def sync_with_broker(self, broker_positions: List[Dict[str, Any]]) -> None:
        """
        Synchronizuje lokální evidenci s reálným portfoliem na brokerovi.
        """
        broker_tickers = set()
        for p in broker_positions:
            raw_t212 = p.get("ticker", "")
            std_ticker = TickerMapper.to_standard_ticker(raw_t212)
            broker_tickers.add(std_ticker)

            # Pokud broker pozici má a my ji v otevřených nemáme, zaevidujeme ji
            if std_ticker not in self.positions or self.positions[std_ticker].status != "OPEN":
                qty = float(p.get("quantity", 0.0))
                avg_price = float(p.get("averagePrice", p.get("currentPrice", 0.0)))
                # Odhad defaultních stops (+/- 10%)
                sl = round(avg_price * 0.90, 2)
                tp = round(avg_price * 1.15, 2)
                self.positions[std_ticker] = PositionRecord(
                    ticker=std_ticker,
                    t212_ticker=raw_t212,
                    quantity=qty,
                    entry_price_usd=avg_price,
                    stop_loss_price=sl,
                    take_profit_price=tp,
                    allocated_eur=16.67,
                    status="OPEN",
                )
                logger.info(f"Synchronizována externí pozice brokera: {std_ticker} ({qty} ks)")

        # Pokud máme otevřenou pozici, kterou broker již nemá (byla např. manuálně prodána v aplikaci)
        for ticker, record in list(self.positions.items()):
            if record.status == "OPEN" and ticker not in broker_tickers:
                logger.info(f"Pozice {ticker} již na brokerovi neexistuje. Označuji jako CLOSED (externí výstup).")
                record.status = "CLOSED"
                record.close_date = datetime.now().isoformat()
                record.close_reason = "MANUAL_EXTERNAL_EXIT"

        self._save()
