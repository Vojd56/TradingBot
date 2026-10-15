"""Modul 5: Carver Volatility Targeting Risk Engine.

Implementuje pokročilé dimenzování pozic a řízení rizik dle Sekce 7 a 11 v research.md:
1. Carverovo cílování volatility (Volatility Targeting):
   - Velikost pozice nepřímo úměrná realizované volatilitě instrumentu (ATR_20).
   - Každá pozice přispívá stejným peněžním rizikem do portfolia.
2. Pásmo nečinnosti 10 % (Inertia Buffer):
   - Zabraňuje zbytečnému přebalancování a poplatkům při drobných fluktuacích.
3. Half-Kelly a limity koncentrace:
   - Strop maximální alokace na 1 titul (max 10-15 % kapitálu).
   - Maximální riziko na obchod při Stop Lossu <= 1.5 % kapitálu.
   - Celková hrubá expozice <= 100 % (bezpůjčkový hotovostní/investiční účet).
4. Korelační filtr:
   - Krácení či zamítnutí pozice při vysoké korelaci (> 0.65 / > 0.80) s již drženými aktivy.
"""

import logging
import math
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from src.quant.features import StockFeatures
from src.quant.prediction_model import ExpectedValueResult
from src.quant.regime_detector import MarketRegime

logger = logging.getLogger(__name__)


class PositionSizingResult(BaseModel):
    """Kvantitativní výpočet velikosti alokace pro daný instrument."""
    ticker: str
    target_shares: float = Field(..., ge=0.0)
    target_value: float = Field(..., ge=0.0)
    portfolio_weight_pct: float = Field(..., ge=0.0, le=100.0)
    risk_per_trade_cash: float = Field(..., ge=0.0)
    risk_per_trade_pct: float = Field(..., ge=0.0, le=100.0)
    current_shares: float = 0.0
    shares_diff: float = 0.0
    is_rebalance_required: bool = True
    approved_for_execution: bool = True
    rationale: str = ""


class CarverRiskEngine:
    """Carverův alokační a rizikový modul pro cílování volatility."""

    def __init__(
        self,
        base_annual_target_vol: float = 0.15,       # Cílová volatilita portfolia 15 % p.a.
        max_position_weight_pct: float = 12.0,      # Max 12 % kapitálu na 1 titul (koncentrační limit)
        max_risk_per_trade_pct: float = 1.5,        # Max 1.5 % kapitálu riskováno při zasažení Stop Lossu
        target_positions_count: int = 10,           # Nominální diverzifikace na 10 pozic
        inertia_buffer_pct: float = 10.0,           # 10% Carverovo pásmo nečinnosti
        max_correlation_threshold: float = 0.65,    # Práh pro redukci na 50 %
        hard_correlation_threshold: float = 0.80,   # Práh pro zamítnutí
    ):
        self.base_annual_target_vol = base_annual_target_vol
        self.max_position_weight_pct = max_position_weight_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.target_positions_count = target_positions_count
        self.inertia_buffer_pct = inertia_buffer_pct
        self.max_correlation_threshold = max_correlation_threshold
        self.hard_correlation_threshold = hard_correlation_threshold

    def calculate_position_size(
        self,
        ev_result: ExpectedValueResult,
        features: StockFeatures,
        regime: MarketRegime,
        portfolio_equity: float,
        current_shares: float = 0.0,
        available_cash: Optional[float] = None,
        correlation_penalty_multiplier: float = 1.0,
    ) -> PositionSizingResult:
        """
        Vypočte optimální velikost pozice podle Carverova Volatility Targetingu s 10% pásmem nečinnosti.
        """
        ticker = ev_result.ticker
        curr_price = features.current_price

        if portfolio_equity <= 0 or curr_price <= 0:
            return PositionSizingResult(
                ticker=ticker,
                target_shares=0.0,
                target_value=0.0,
                portfolio_weight_pct=0.0,
                risk_per_trade_cash=0.0,
                risk_per_trade_pct=0.0,
                current_shares=current_shares,
                shares_diff=-current_shares,
                is_rebalance_required=False,
                approved_for_execution=False,
                rationale="Neplatný kapitál nebo nulová cena instrumentu.",
            )

        if not ev_result.passes_hurdle or not regime.allow_trading:
            return PositionSizingResult(
                ticker=ticker,
                target_shares=0.0,
                target_value=0.0,
                portfolio_weight_pct=0.0,
                risk_per_trade_cash=0.0,
                risk_per_trade_pct=0.0,
                current_shares=current_shares,
                shares_diff=-current_shares,
                is_rebalance_required=bool(current_shares > 0),
                approved_for_execution=False,
                rationale=f"Obchod neschválen: {ev_result.rationale}",
            )

        # 1. Cílová volatilita portfolia škálovaná tržním režimem
        target_annual_vol = self.base_annual_target_vol * regime.risk_multiplier
        daily_target_vol = target_annual_vol / math.sqrt(252)

        # 2. Celková denní peněžní volatilita portfolia (Daily Cash Volatility Target)
        total_daily_cash_vol = portfolio_equity * daily_target_vol

        # 3. Peněžní volatilita alokovaná na 1 pozici
        position_daily_cash_vol = total_daily_cash_vol / math.sqrt(self.target_positions_count)

        # 4. Denní volatilita 1 kusu instrumentu: ATR_20
        # Použijeme ATR_20 (v absolutní měně na akcii)
        instrument_daily_vol_per_share = max(0.01, features.atr_20)

        # 5. Nekonstrainovaný počet kusů (Carverův základní vzorec)
        unconstrained_shares = position_daily_cash_vol / instrument_daily_vol_per_share

        # Aplikace korelační penalizace (pokud má titul vysokou korelaci s existujícím portfoliem)
        unconstrained_shares *= correlation_penalty_multiplier

        # 6. Aplikace Half-Kelly a koncentračních stropů
        # Max alokace na 1 pozici v kapitálu
        max_allowed_value = portfolio_equity * (self.max_position_weight_pct / 100.0)
        target_value_initial = unconstrained_shares * curr_price

        capped_value = min(target_value_initial, max_allowed_value)

        # Omezení dostupnou hotovostí (pokud je specifikována)
        if available_cash is not None and available_cash > 0:
            capped_value = min(capped_value, available_cash * 0.98)  # 2% rezerva na skluz

        # 7. Riziko při Stop Lossu (nesmí překročit max_risk_per_trade_pct)
        sl_pct = ev_result.barriers.stop_loss_pct / 100.0
        max_allowed_loss_cash = portfolio_equity * (self.max_risk_per_trade_pct / 100.0)

        potential_loss_cash = capped_value * sl_pct
        if potential_loss_cash > max_allowed_loss_cash:
            # Přepočet velikosti pozice podle maximálního přípustného Stop Loss rizika
            capped_value = max_allowed_loss_cash / max(1e-4, sl_pct)

        target_shares = capped_value / curr_price
        target_value = round(target_shares * curr_price, 2)
        portfolio_weight = round((target_value / portfolio_equity) * 100.0, 2)
        risk_cash = round(target_value * sl_pct, 2)
        risk_pct = round((risk_cash / portfolio_equity) * 100.0, 2)

        # 8. Carverovo 10% pásmo nečinnosti (Inertia Buffer)
        shares_diff = target_shares - current_shares
        if current_shares > 0 and target_shares > 0:
            pct_change = abs(shares_diff) / target_shares * 100.0
            if pct_change < self.inertia_buffer_pct:
                logger.debug(
                    f"{ticker}: Změna pozice ({pct_change:.1f} %) je v 10% pásmu nečinnosti "
                    f"(držíme {current_shares:.2f}, cíl {target_shares:.2f}). Rebalancování vynecháno."
                )
                return PositionSizingResult(
                    ticker=ticker,
                    target_shares=round(current_shares, 4),
                    target_value=round(current_shares * curr_price, 2),
                    portfolio_weight_pct=round((current_shares * curr_price / portfolio_equity) * 100.0, 2),
                    risk_per_trade_cash=round(current_shares * curr_price * sl_pct, 2),
                    risk_per_trade_pct=round((current_shares * curr_price * sl_pct / portfolio_equity) * 100.0, 2),
                    current_shares=current_shares,
                    shares_diff=0.0,
                    is_rebalance_required=False,
                    approved_for_execution=True,
                    rationale=f"Pozice je v 10% pásmu nečinnosti (rozdíl {pct_change:.1f}% < 10%). Držíme stávající velikost.",
                )

        rationale = (
            f"Carver alokace: Cíl {target_shares:.2f} ks ({target_value:.2f} CZK / {portfolio_weight:.1f} % portfolia). "
            f"SL riziko: {risk_cash:.2f} CZK ({risk_pct:.2f} % ekvity). "
            f"Faktor režimu: {regime.risk_multiplier:.2f}."
        )

        return PositionSizingResult(
            ticker=ticker,
            target_shares=round(target_shares, 4),
            target_value=target_value,
            portfolio_weight_pct=portfolio_weight,
            risk_per_trade_cash=risk_cash,
            risk_per_trade_pct=risk_pct,
            current_shares=current_shares,
            shares_diff=round(shares_diff, 4),
            is_rebalance_required=True,
            approved_for_execution=True,
            rationale=rationale,
        )

    def evaluate_correlation(
        self,
        candidate_ticker: str,
        current_holdings: List[str],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> float:
        """
        Vyhodnotí korelaci kandidáta s existujícími drženými aktivy:
        - correlation > hard_threshold (0.80) -> 0.0 (tvrdé zamítnutí)
        - correlation > max_threshold (0.65) -> 0.5 (zkrácení na 50 %)
        - jinak -> 1.0 (plná alokace)
        """
        if not current_holdings or not correlation_matrix:
            return 1.0

        if candidate_ticker not in correlation_matrix:
            return 1.0

        max_corr = 0.0
        for holding in current_holdings:
            if holding != candidate_ticker and holding in correlation_matrix[candidate_ticker]:
                corr = abs(correlation_matrix[candidate_ticker][holding])
                if corr > max_corr:
                    max_corr = corr

        if max_corr >= self.hard_correlation_threshold:
            logger.warning(f"{candidate_ticker}: Extrémní korelace {max_corr:.2f} s portfoliem -> VETO.")
            return 0.0
        elif max_corr >= self.max_correlation_threshold:
            logger.info(f"{candidate_ticker}: Zvýšená korelace {max_corr:.2f} s portfoliem -> Krácení alokace na 50 %.")
            return 0.5

        return 1.0
