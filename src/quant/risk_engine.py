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

from datetime import datetime
import hashlib
import logging
import math
from typing import Any, Dict, List, Optional, Union
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

    @staticmethod
    def get_ticker_currency(ticker: str) -> str:
        """Určí nativní kótovací měnu instrumentu podle přípony tickeru."""
        clean = ticker.upper().strip()
        if clean.endswith(".L") or clean.endswith(".IL"):
            if clean in ("SMSN.IL", "SMSN.L"):
                return "USD"
            return "GBp"
        if clean.endswith(".DE") or clean.endswith(".PA") or clean.endswith(".AS"):
            return "EUR"
        if clean.endswith(".SW"):
            return "CHF"
        return "USD"

    @classmethod
    def get_share_price_usd(
        cls,
        ticker: str,
        raw_price: float,
        fx_rates: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Převede surovou cenu akcie z yfinance (která může být v GBp, EUR, CHF nebo USD) na USD.
        """
        if raw_price <= 0:
            return 0.0

        curr = cls.get_ticker_currency(ticker)
        rates = fx_rates or {"EUR": 1.08, "GBP": 1.30, "CHF": 1.12, "USD": 1.0}

        if curr == "GBp":
            # Pence na LSE: 100 pence = 1 GBP
            gbp_rate = rates.get("GBP", 1.30)
            return (raw_price / 100.0) * gbp_rate
        elif curr == "EUR":
            return raw_price * rates.get("EUR", 1.08)
        elif curr == "CHF":
            return raw_price * rates.get("CHF", 1.12)

        return raw_price

    def calculate_position_size(
        self,
        ev_result: ExpectedValueResult,
        features: StockFeatures,
        regime: MarketRegime,
        portfolio_equity: float,
        current_shares: float = 0.0,
        available_cash: Optional[float] = None,
        correlation_penalty_multiplier: float = 1.0,
        fx_rates: Optional[Dict[str, float]] = None,
    ) -> PositionSizingResult:
        """
        Vypočte optimální velikost pozice podle Carverova Volatility Targetingu s 10% pásmem nečinnosti.
        Veškeré finanční výpočty (ekvita, hotovost, risk) probíhají v USD; ceny neamerických akcií (EUR, GBp, CHF)
        jsou normalizovány přes live FX kurzy.
        """
        ticker = ev_result.ticker
        raw_price = features.current_price
        share_price_usd = self.get_share_price_usd(ticker, raw_price, fx_rates)
        if share_price_usd <= 0:
            share_price_usd = raw_price

        if portfolio_equity <= 0 or raw_price <= 0:
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

        # 2. Celková denní peněžní volatilita portfolia v USD (Daily Cash Volatility Target)
        total_daily_cash_vol = portfolio_equity * daily_target_vol

        # 3. Peněžní volatilita alokovaná na 1 pozici v USD
        position_daily_cash_vol = total_daily_cash_vol / math.sqrt(self.target_positions_count)

        # 4. Denní volatilita 1 kusu instrumentu v USD (ATR_20 normalizovaná do USD)
        raw_atr_20 = max(0.01, features.atr_20)
        atr_20_usd = (raw_atr_20 / raw_price) * share_price_usd if raw_price > 0 else raw_atr_20
        instrument_daily_vol_per_share_usd = max(0.01, atr_20_usd)

        # 5. Nekonstrainovaný počet kusů (Carverův základní vzorec)
        unconstrained_shares = position_daily_cash_vol / instrument_daily_vol_per_share_usd

        # Aplikace korelační penalizace (pokud má titul vysokou korelaci s existujícím portfoliem)
        unconstrained_shares *= correlation_penalty_multiplier

        # 6. Aplikace Half-Kelly a koncentračních stropů (v USD)
        max_allowed_value = portfolio_equity * (self.max_position_weight_pct / 100.0)
        target_value_initial = unconstrained_shares * share_price_usd

        capped_value = min(target_value_initial, max_allowed_value)

        # 7. Riziko při Stop Lossu (nesmí překročit max_risk_per_trade_pct)
        sl_pct = ev_result.barriers.stop_loss_pct / 100.0
        max_allowed_loss_cash = portfolio_equity * (self.max_risk_per_trade_pct / 100.0)

        potential_loss_cash = capped_value * sl_pct
        if potential_loss_cash > max_allowed_loss_cash:
            # Přepočet velikosti pozice podle maximálního přípustného Stop Loss rizika
            capped_value = max_allowed_loss_cash / max(1e-4, sl_pct)

        target_shares = capped_value / share_price_usd
        target_value = round(target_shares * share_price_usd, 2)
        portfolio_weight = round((target_value / portfolio_equity) * 100.0, 2)
        risk_cash = round(target_value * sl_pct, 2)
        risk_pct = round((risk_cash / portfolio_equity) * 100.0, 2)

        # 8. Rozdíl a Carverovo 10% pásmo nečinnosti (Inertia Buffer)
        shares_diff = target_shares - current_shares

        # Omezení nového nákupu dostupnou hotovostí (v USD)
        if shares_diff > 0 and available_cash is not None:
            max_buy_cash = max(0.0, available_cash * 0.90)
            max_add_shares = max_buy_cash / share_price_usd
            if shares_diff > max_add_shares:
                shares_diff = max_add_shares
                target_shares = current_shares + shares_diff
                target_value = round(target_shares * share_price_usd, 2)
                portfolio_weight = round((target_value / portfolio_equity) * 100.0, 2)
                risk_cash = round(target_value * sl_pct, 2)
                risk_pct = round((risk_cash / portfolio_equity) * 100.0, 2)

        # Pro Long-Only prodeje: nikdy neprodávat více, než je aktuálně vlastněno
        if shares_diff < 0:
            if current_shares <= 0:
                shares_diff = 0.0
                target_shares = 0.0
            elif abs(shares_diff) > current_shares:
                shares_diff = -current_shares
                target_shares = 0.0

        # Minimální smysluplná velikost obchodu (prachové obchody pod $10 vynechat)
        if abs(shares_diff) * share_price_usd < 10.0:
            return PositionSizingResult(
                ticker=ticker,
                target_shares=round(current_shares, 4),
                target_value=round(current_shares * share_price_usd, 2),
                portfolio_weight_pct=round((current_shares * share_price_usd / portfolio_equity) * 100.0, 2),
                risk_per_trade_cash=round(current_shares * share_price_usd * sl_pct, 2),
                risk_per_trade_pct=round((current_shares * share_price_usd * sl_pct / portfolio_equity) * 100.0, 2),
                current_shares=current_shares,
                shares_diff=0.0,
                is_rebalance_required=False,
                approved_for_execution=True,
                rationale="Změna pozice je pod minimálním finančním limitem brokera (< 10 USD). Rebalancování vynecháno.",
            )

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
                    target_value=round(current_shares * share_price_usd, 2),
                    portfolio_weight_pct=round((current_shares * share_price_usd / portfolio_equity) * 100.0, 2),
                    risk_per_trade_cash=round(current_shares * share_price_usd * sl_pct, 2),
                    risk_per_trade_pct=round((current_shares * share_price_usd * sl_pct / portfolio_equity) * 100.0, 2),
                    current_shares=current_shares,
                    shares_diff=0.0,
                    is_rebalance_required=False,
                    approved_for_execution=True,
                    rationale=f"Pozice je v 10% pásmu nečinnosti (rozdíl {pct_change:.1f}% < 10%). Držíme stávající velikost.",
                )

        rationale = (
            f"Carver alokace: Cíl {target_shares:.2f} ks ({target_value:.2f} USD / {portfolio_weight:.1f} % portfolia). "
            f"SL riziko: {risk_cash:.2f} USD ({risk_pct:.2f} % ekvity). "
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


class OrderIntent(BaseModel):
    """Záměr objednávky předložený ke schválení RiskGatekeeperu."""
    ticker: str
    side: str = "BUY"  # BUY nebo SELL
    entry_price: float = Field(..., gt=0.0)
    stop_loss_price: float = Field(..., gt=0.0)
    take_profit_price: Optional[float] = None
    desired_shares: Optional[float] = None
    suggested_allocation_eur: Optional[float] = None
    strategy_id: str = "QUANT_STRATEGY"
    timestamp: Optional[datetime] = None


class PortfolioState(BaseModel):
    """Aktuální reálný stav portfolia před exekucí."""
    equity: float = Field(..., ge=0.0)
    cash: float = Field(..., ge=0.0)
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0
    open_positions_count: int = 0
    open_positions: Dict[str, Any] = Field(default_factory=dict)  # ticker -> shares nebo detail


class MarketBar(BaseModel):
    """Tržní svíčka v čase exekučního rozhodnutí pro kontrolu zastaralosti (stale data)."""
    ticker: str
    close: float
    high: float
    low: float
    volume: float = 0.0
    timestamp: datetime


class ApprovedOrder(BaseModel):
    """Schválený obchodní příkaz opatřený razítkem RiskGatekeeperu."""
    approved: bool = True
    client_order_id: str
    ticker: str
    side: str
    shares: float = Field(..., gt=0.0)
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_per_trade_cash: float
    risk_per_trade_pct: float
    allocation_cash: float
    approved_at: datetime
    notes: str = ""


class RejectedOrder(BaseModel):
    """Zamítnutý obchodní záměr s uvedením konkrétního bezpečnostního jističe."""
    approved: bool = False
    ticker: str
    side: str
    reason: str
    rejected_at: datetime


class RiskGatekeeper:
    """
    Sjednocený strážce rizika a poslední instance před exekucí pokynů.
    
    Vymáhá striktní provozní bezpečnost dle Sekce 4 a 11 v research.md:
    1. Kontrola zastaralosti dat (Stale Data Guard): svíčka nesmí být starší než 120 sekund.
    2. Denní Kill-Switch (Max Daily Loss): při denním propadu > 2.0 % blokuje nové obchody.
    3. Limit současných pozic (Max Concurrent Positions): max 4 otevřené pozice.
    4. Carverovo dimenzování s Half-Kelly limity a kontrolou rizika SL:
       Q = floor( (Equity * max_risk_per_trade_pct) / |Entry - SL| )
    5. Pásmo nečinnosti 20 % (No-Trade / Inertia Buffer): zákaz rebalancování při malé odchylce.
    6. Idempotentní deterministický klíč (client_order_id): ochrana proti duplicitnímu odeslání.
    """

    def __init__(
        self,
        max_risk_per_trade_pct: float = 0.005,      # 0.5 % kapitálu na 1 obchod
        max_position_notional_pct: float = 0.20,    # Max 20 % kapitálu na 1 titul
        max_daily_loss_pct: float = 0.02,          # Max 2.0 % denní ztráta (Kill-Switch)
        max_concurrent_positions: int = 4,         # Max 4 současné otevřené pozice
        max_stale_seconds: int = 120,              # Max stáří svíčky 120 s
        inertia_buffer_pct: float = 20.0,          # 20 % No-Trade buffer proti churningu
        carver_engine: Optional[CarverRiskEngine] = None,
    ):
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_position_notional_pct = max_position_notional_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.max_stale_seconds = max_stale_seconds
        self.inertia_buffer_pct = inertia_buffer_pct
        self.carver_engine = carver_engine or CarverRiskEngine()
        self.seen_order_ids: Dict[str, datetime] = {}

    @staticmethod
    def generate_client_order_id(
        ticker: str,
        side: str,
        timestamp: datetime,
        strategy_id: str,
        shares: float,
    ) -> str:
        """
        Vygeneruje deterministický hash idempotence SHA-256 zkrácený na 24 hex znaků.
        Garantuje, že při síťovém timeoutu nebo duplicitním volání nevznikne dvojitá objednávka.
        """
        date_str = timestamp.strftime("%Y%m%d_%H%M")
        raw_key = f"{strategy_id}:{ticker}:{date_str}:{side}:{shares:.4f}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]

    def evaluate_order_intent(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
        market_bar: Optional[MarketBar] = None,
    ) -> Union[ApprovedOrder, RejectedOrder]:
        """
        Hlavní brána: prověří záměr a vydá ApprovedOrder nebo RejectedOrder.
        """
        now = datetime.now()
        side = intent.side.upper()
        values = [intent.entry_price, intent.stop_loss_price, portfolio.cash, portfolio.equity,
                  portfolio.daily_realized_pnl, portfolio.daily_unrealized_pnl]
        if intent.desired_shares is not None:
            values.append(intent.desired_shares)
        if intent.take_profit_price is not None:
            values.append(intent.take_profit_price)
        if side not in ("BUY", "SELL") or not all(math.isfinite(v) for v in values) or intent.entry_price <= 0:
            return RejectedOrder(ticker=intent.ticker, side=intent.side, reason="INVALID_INPUT", rejected_at=now)

        # 1. KONTROLA ZASTARALOSTI DAT (STALE DATA GUARD)
        if market_bar is not None and market_bar.timestamp is not None:
            bar_ts = market_bar.timestamp
            # Srovnání časových pásem
            ref_now = datetime.now(bar_ts.tzinfo) if bar_ts.tzinfo is not None else now
            age_seconds = abs((ref_now - bar_ts).total_seconds())
            if age_seconds > self.max_stale_seconds:
                reason = f"STALE_DATA: Tržní data pro {intent.ticker} jsou stará {age_seconds:.0f} s (limit {self.max_stale_seconds} s)."
                logger.warning(reason)
                return RejectedOrder(ticker=intent.ticker, side=intent.side, reason=reason, rejected_at=now)

        # 2. DENNÍ KILL-SWITCH (MAX DAILY LOSS GUARD)
        daily_total_pnl = portfolio.daily_realized_pnl + portfolio.daily_unrealized_pnl
        if side == "BUY" and portfolio.equity > 0:
            daily_loss_pct = daily_total_pnl / portfolio.equity
            if daily_loss_pct <= -self.max_daily_loss_pct:
                reason = (
                    f"DAILY_KILL_SWITCH: Celková denní ztráta {daily_loss_pct * 100:.2f} % "
                    f"překročila limit {-self.max_daily_loss_pct * 100:.1f} %. Obchodování zastaveno."
                )
                logger.critical(reason)
                return RejectedOrder(ticker=intent.ticker, side=intent.side, reason=reason, rejected_at=now)

        # 3. EXEKUCE PRODEJE (SELL INTENT)
        if intent.side.upper() == "SELL":
            if intent.ticker not in portfolio.open_positions:
                reason = f"SELL_NO_POSITION: Pokus o prodej neexistující pozice {intent.ticker}."
                logger.warning(reason)
                return RejectedOrder(ticker=intent.ticker, side=intent.side, reason=reason, rejected_at=now)

            pos_val = portfolio.open_positions[intent.ticker]
            cur_held = float(pos_val.get("shares", pos_val.get("quantity", 0.0))) if isinstance(pos_val, dict) else float(pos_val)
            shares_to_sell = intent.desired_shares if intent.desired_shares and intent.desired_shares > 0 else cur_held
            if not math.isfinite(cur_held) or cur_held <= 0 or shares_to_sell > cur_held or (intent.desired_shares is not None and intent.desired_shares <= 0):
                return RejectedOrder(ticker=intent.ticker, side=intent.side, reason="SELL_EXCEEDS_POSITION", rejected_at=now)
            shares_to_sell = math.floor(shares_to_sell * 10000) / 10000
            if shares_to_sell <= 0:
                return RejectedOrder(ticker=intent.ticker, side=intent.side, reason="SELL_BELOW_PRECISION", rejected_at=now)

            order_id = self.generate_client_order_id(
                ticker=intent.ticker,
                side="SELL",
                timestamp=intent.timestamp or now,
                strategy_id=intent.strategy_id,
                shares=shares_to_sell,
            )

            if order_id in self.seen_order_ids:
                reason = f"DUPLICATE_ORDER: Příkaz k prodeji s ID {order_id} již byl v minulosti schválen."
                logger.warning(reason)
                return RejectedOrder(ticker=intent.ticker, side="SELL", reason=reason, rejected_at=now)

            self.seen_order_ids[order_id] = now
            return ApprovedOrder(
                approved=True,
                client_order_id=order_id,
                ticker=intent.ticker,
                side="SELL",
                shares=round(shares_to_sell, 4),
                entry_price=intent.entry_price,
                stop_loss=intent.stop_loss_price,
                take_profit=intent.take_profit_price or 0.0,
                risk_per_trade_cash=0.0,
                risk_per_trade_pct=0.0,
                allocation_cash=round(shares_to_sell * intent.entry_price, 2),
                approved_at=now,
                notes="Schválený prodejní pokyn",
            )

        if portfolio.cash <= 0 or portfolio.equity <= 0:
            return RejectedOrder(ticker=intent.ticker, side=intent.side, reason="INSUFFICIENT_FUNDS_OR_RISK", rejected_at=now)
        if intent.desired_shares is not None and intent.desired_shares <= 0:
            return RejectedOrder(ticker=intent.ticker, side=intent.side, reason="INVALID_QUANTITY", rejected_at=now)

        # 4. LIMIT SOUČASNÝCH POZIC (MAX CONCURRENT POSITIONS GUARD)
        is_new_position = intent.ticker not in portfolio.open_positions
        if is_new_position and portfolio.open_positions_count >= self.max_concurrent_positions:
            reason = (
                f"MAX_CONCURRENT_POSITIONS: Dosažen limit {self.max_concurrent_positions} pozic "
                f"(aktuálně {portfolio.open_positions_count})."
            )
            logger.info(reason)
            return RejectedOrder(ticker=intent.ticker, side=intent.side, reason=reason, rejected_at=now)

        # 5. KONTROLA STOP LOSSU A VÝPOČET RIZIKA NA AKCII
        price_risk = intent.entry_price - intent.stop_loss_price
        if price_risk <= 0.0001:
            reason = f"INVALID_STOP_LOSS: Stop Loss ({intent.stop_loss_price:.2f}) musí být přísně pod nákupní cenou ({intent.entry_price:.2f})."
            logger.warning(reason)
            return RejectedOrder(ticker=intent.ticker, side=intent.side, reason=reason, rejected_at=now)

        # 6. VÝPOČET VELIKOSTI POZICE (VOLATILITY TARGETING & RISK BUDGET)
        if intent.take_profit_price is not None and intent.take_profit_price - intent.entry_price < 2 * price_risk - 1e-9:
            return RejectedOrder(ticker=intent.ticker, side=intent.side, reason="REWARD_RISK_BELOW_TWO", rejected_at=now)
        # Q = floor( (Equity * max_risk_per_trade_pct) / |Entry - SL| )
        risk_budget_cash = portfolio.equity * self.max_risk_per_trade_pct
        q_risk = math.floor(risk_budget_cash / price_risk)

        # Koncentrační strop (max 20 % kapitálu)
        max_notional = portfolio.equity * self.max_position_notional_pct
        existing = portfolio.open_positions.get(intent.ticker, 0.0)
        held = float(existing.get("shares", existing.get("quantity", 0.0))) if isinstance(existing, dict) else float(existing)
        if not math.isfinite(held) or held < 0:
            return RejectedOrder(ticker=intent.ticker, side=intent.side, reason="INVALID_POSITION", rejected_at=now)
        max_notional = max(0.0, max_notional - held * intent.entry_price)
        q_notional = math.floor(max_notional / intent.entry_price)
        target_shares = min(q_risk, q_notional)

        # Pokud uživatel zadal explicitní desired_shares (např. z CarverEngine), zohledníme ho
        if intent.desired_shares is not None and intent.desired_shares > 0:
            target_shares = min(target_shares, math.floor(intent.desired_shares))

        # Kontrola dostupné hotovosti
        max_cash_shares = math.floor(portfolio.cash / intent.entry_price)
        target_shares = min(target_shares, max_cash_shares)

        if target_shares <= 0:
            reason = f"INSUFFICIENT_FUNDS_OR_RISK: Vypočtené množství pro {intent.ticker} je 0 ks."
            logger.warning(reason)
            return RejectedOrder(ticker=intent.ticker, side=intent.side, reason=reason, rejected_at=now)

        # 7. CARVEROVO PÁSMO NEČINNOSTI (INERTIA / NO-TRADE BUFFER)
        pos_val = portfolio.open_positions.get(intent.ticker)
        if pos_val is not None:
            cur_held = float(pos_val.get("shares", pos_val.get("quantity", 0.0))) if isinstance(pos_val, dict) else float(pos_val)
            if cur_held > 0:
                pct_diff = abs(target_shares - cur_held) / cur_held * 100.0
                if pct_diff < self.inertia_buffer_pct:
                    reason = (
                        f"NO_TRADE_INERTIA_BUFFER: Rozdíl v pozici {intent.ticker} ({pct_diff:.1f} %) "
                        f"je uvnitř {self.inertia_buffer_pct:.0f}% pásma nečinnosti. Úprava vynechána."
                    )
                    logger.debug(reason)
                    return RejectedOrder(ticker=intent.ticker, side=intent.side, reason=reason, rejected_at=now)

        # 8. IDEMPOTENTNÍ KONTROLA DUPLICIT
        order_id = self.generate_client_order_id(
            ticker=intent.ticker,
            side="BUY",
            timestamp=intent.timestamp or now,
            strategy_id=intent.strategy_id,
            shares=float(target_shares),
        )

        if order_id in self.seen_order_ids:
            reason = f"DUPLICATE_ORDER: Příkaz k nákupu s ID {order_id} již byl zpracován."
            logger.warning(reason)
            return RejectedOrder(ticker=intent.ticker, side="BUY", reason=reason, rejected_at=now)

        self.seen_order_ids[order_id] = now

        risk_cash = target_shares * price_risk
        risk_pct = (risk_cash / portfolio.equity) * 100.0 if portfolio.equity > 0 else 0.0
        alloc_cash = target_shares * intent.entry_price
        tp_price = intent.take_profit_price or round(intent.entry_price + (2.0 * price_risk), 2)

        return ApprovedOrder(
            approved=True,
            client_order_id=order_id,
            ticker=intent.ticker,
            side="BUY",
            shares=float(target_shares),
            entry_price=round(intent.entry_price, 4),
            stop_loss=round(intent.stop_loss_price, 4),
            take_profit=round(tp_price, 4),
            risk_per_trade_cash=round(risk_cash, 2),
            risk_per_trade_pct=round(risk_pct, 2),
            allocation_cash=round(alloc_cash, 2),
            approved_at=now,
            notes="Schváleno RiskGatekeeperem",
        )

