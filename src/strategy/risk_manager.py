import logging
from typing import Dict, Optional, Tuple
from config import settings
from src.data.market_data import TechnicalMetrics
from src.llm.sentiment_engine import LLMStockVerdict
from src.strategy.models import ExitReason, PositionExitSignal

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Řízení rizik a alokace kapitálu přizpůsobené pro:
    - Testovací kapitál: 50 EUR
    - Maximální počet současných pozic: 3
    - Horizont držení: týdny až měsíce (střední riziko)
    - Dynamické výstupy na bázi ATR (Average True Range)
    """

    def __init__(
        self,
        total_capital_eur: Optional[float] = None,
        max_positions: Optional[int] = None,
        atr_sl_multiplier: float = 2.0,
        atr_tp_multiplier: float = 3.5,
    ):
        self.total_capital_eur = total_capital_eur or settings.initial_capital_eur
        self.max_positions = max_positions or settings.max_positions
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier

    @property
    def max_position_size_eur(self) -> float:
        """Maximální velikost jedné pozice v EUR."""
        return round(self.total_capital_eur / max(1, self.max_positions), 2)

    def calculate_allocation(
        self,
        current_open_positions_count: int,
        available_cash_eur: float,
        signal_score: float = 75.0,
        atr_14: float = 0.0,
        current_price: float = 0.0,
    ) -> float:
        """
        Dynamický Conviction & Volatility Risk Sizer:
        - Škáluje alokaci podle prioritního skóre (Tier 1: ~130%, Tier 2: ~100%, Tier 3: ~65%).
        - Upravuje velikost pozice podle volatility (Risk Parity dle ATR).
        - Zajišťuje nepřekročení limitu pozic a dostupné hotovosti.
        """
        if current_open_positions_count >= self.max_positions:
            logger.info(
                f"Dosažen limit pozic ({current_open_positions_count}/{self.max_positions}). "
                "Nová alokace zamítnuta."
            )
            return 0.0

        base_allocation = self.max_position_size_eur

        # 1. Conviction Multiplier (váha podle skóre a přesvědčení)
        if signal_score >= 83.0:
            # Tier 1: Klíčoví lídři s vysokou jistotou (+30 % alokace)
            conviction_mult = 1.30
        elif signal_score >= 76.0:
            # Tier 2: Solidní momentum (základní alokace 100 %)
            conviction_mult = 1.00
        else:
            # Tier 3: Spekulativní růst / vysoká beta (-35 % alokace)
            conviction_mult = 0.65

        # 2. Volatility Dampener (tlumení u divokých titulů)
        vol_mult = 1.0
        if current_price > 0 and atr_14 > 0:
            atr_pct = (atr_14 / current_price) * 100
            if atr_pct > 5.0:
                vol_mult = 0.75
            elif atr_pct > 3.5:
                vol_mult = 0.88

        target_allocation = round(base_allocation * conviction_mult * vol_mult, 2)

        # Bezpečnostní mantinely
        min_alloc = round(base_allocation * 0.40, 2)
        max_alloc = round(base_allocation * 1.50, 2)
        target_allocation = max(min_alloc, min(max_alloc, target_allocation))

        actual_allocation = min(target_allocation, max(0.0, available_cash_eur))

        if actual_allocation < 5.0:
            logger.warning(
                f"Dostupná hotovost ({actual_allocation:.2f} EUR) je nižší než minimální práh."
            )
            return 0.0

        return round(actual_allocation, 2)

    def calculate_shares_quantity(
        self,
        allocated_eur: float,
        stock_price: float,
        currency: str = "USD",
        eur_usd_rate: float = 1.08,
        eur_gbp_rate: float = 0.84,
        eur_chf_rate: float = 0.94,
    ) -> float:
        """
        Přepočte kapitál v EUR na počet frakčních akcií dle měny instrumentu:
        - EUR (např. Rheinmetall, SAP na Xetře): allocated_eur / cena_eur
        - GBp (pence na LSE, např. Rolls-Royce RR.L): (allocated_eur * EUR_GBP * 100) / cena_pence
        - GBP (libry): (allocated_eur * EUR_GBP) / cena_gbp
        - USD (americké akcie a ADR, např. NVDA, SMSN.L, SUNB): (allocated_eur * EUR_USD) / cena_usd
        """
        if allocated_eur <= 0.0 or stock_price <= 0.0:
            return 0.0

        curr_clean = currency.strip()
        if curr_clean.upper() == "EUR":
            raw_quantity = allocated_eur / stock_price
        elif curr_clean == "GBp":
            allocated_pence = allocated_eur * eur_gbp_rate * 100.0
            raw_quantity = allocated_pence / stock_price
        elif curr_clean.upper() in ("GBP", "GBX"):
            allocated_gbp = allocated_eur * eur_gbp_rate
            raw_quantity = allocated_gbp / stock_price
        elif curr_clean.upper() == "CHF":
            allocated_chf = allocated_eur * eur_chf_rate
            raw_quantity = allocated_chf / stock_price
        else:
            # Výchozí USD pro americké tituly a globální ADR
            allocated_usd = allocated_eur * eur_usd_rate
            raw_quantity = allocated_usd / stock_price

        # Zaokrouhlení na 4 desetinná místa pro frakční obchodování na Trading 212
        return round(raw_quantity, 4)

    def calculate_stops(
        self,
        current_price: float,
        atr_14: float
    ) -> Tuple[float, float]:
        """
        Vypočte dynamický Stop Loss a Take Profit na bázi násobků ATR.
        - Stop Loss = Cena - 2.0 * ATR
        - Take Profit = Cena + 3.5 * ATR (RRR 1 : 1.75)
        """
        if atr_14 <= 0.0:
            # Fallback na procentuální úrovně (SL -6 %, TP +12 %)
            stop_loss = round(current_price * 0.94, 2)
            take_profit = round(current_price * 1.12, 2)
            return stop_loss, take_profit

        stop_loss = round(max(0.01, current_price - (self.atr_sl_multiplier * atr_14)), 2)
        take_profit = round(current_price + (self.atr_tp_multiplier * atr_14), 2)
        return stop_loss, take_profit

    def evaluate_position_exit(
        self,
        ticker: str,
        current_price: float,
        entry_price: float,
        metrics: TechnicalMetrics,
        ai_verdict: Optional[LLMStockVerdict] = None,
        custom_stop_loss: Optional[float] = None,
        custom_take_profit: Optional[float] = None,
    ) -> PositionExitSignal:
        """
        Vyhodnotí, zda má být otevřená pozice uzavřena.
        Kritéria výstupu:
        1. Zásah Stop Lossu (ochrana kapitálu)
        2. Zásah Take Profitu (uzamčení zisku)
        3. Prolomení střednědobého trendu (cena klesla pod SMA 50 o více než 1.5 %)
        4. Zhoršení AI sentimentu na AVOID s vysokou důvěrou
        """
        # Pokud nejsou předány uložené stops, spočítáme je z entry ceny
        stop_loss, take_profit = self.calculate_stops(entry_price, metrics.atr_14)
        if custom_stop_loss is not None:
            stop_loss = custom_stop_loss
        if custom_take_profit is not None:
            take_profit = custom_take_profit

        # 1. Kontrola Stop Loss
        if current_price <= stop_loss:
            loss_pct = ((current_price - entry_price) / entry_price) * 100
            return PositionExitSignal(
                ticker=ticker,
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                exit_price=current_price,
                message=f"Zasažen Stop Loss na {stop_loss} USD (Ztráta: {loss_pct:.1f} %)."
            )

        # 2. Kontrola Take Profit
        if current_price >= take_profit:
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            return PositionExitSignal(
                ticker=ticker,
                should_exit=True,
                reason=ExitReason.TAKE_PROFIT,
                exit_price=current_price,
                message=f"Zasažen Take Profit na {take_profit} USD (Zisk: +{profit_pct:.1f} %)."
            )

        # 3. Kontrola prolomení klouzavého průměru SMA 50 (rozpad střednědobého trendu)
        sma_50_buffer = metrics.sma_50 * 0.985  # 1.5% tolerance proti falešným průrazům
        if current_price < sma_50_buffer:
            return PositionExitSignal(
                ticker=ticker,
                should_exit=True,
                reason=ExitReason.TREND_BREAK,
                exit_price=current_price,
                message=f"Cena ({current_price} USD) prolomila podporu 50denního SMA ({metrics.sma_50} USD)."
            )

        # 4. Zhoršení fundamentálního / zprávového sentimentu na AVOID
        if ai_verdict and ai_verdict.recommendation == "AVOID" and ai_verdict.confidence >= 0.70:
            return PositionExitSignal(
                ticker=ticker,
                should_exit=True,
                reason=ExitReason.AI_AVOID,
                exit_price=current_price,
                message=f"AI identifikovala závažné negativní katalyzátory (AVOID, důvěra {ai_verdict.confidence*100:.0f} %): {ai_verdict.reasoning}"
            )

        # Žádná výstupní podmínka nenastala
        return PositionExitSignal(
            ticker=ticker,
            should_exit=False,
            exit_price=current_price,
            message="Pozice odpovídá investičnímu plánu, pokračuje držení."
        )
