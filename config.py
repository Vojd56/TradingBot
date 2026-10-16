"""Konfigurace a bezpečnostní správa prostředí pro TradingBot.

Implementuje Task AG-00:
1. Striktní enumerace TradingMode (BACKTEST, PAPER, LIVE) s výchozím bezpečným stavem PAPER.
2. Tvrdý bezpečnostní zámek validate_live_mode vyžadující CONFIRM_LIVE_TRADING=true.
3. Strukturovaný model transakčních nákladů TransactionCostConfig (spread, skluz, poplatky brokera, FX).
"""

from enum import Enum
import os
import sys
from typing import Any, List, Literal, Optional, Union
from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    """Provozní režimy bota."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class TransactionCostConfig(BaseModel):
    """
    Strukturovaný model transakčních nákladů.
    Všechny hodnoty jsou v bazických bodech (bps: 1 bps = 0.01 % = 0.0001).
    """
    spread_bps: float = Field(default=5.0, description="Poloviční bid-ask spread (5 bps = 0.05 %)")
    slippage_bps: float = Field(default=5.0, description="Modelovaný skluz exekuce (5 bps = 0.05 %)")
    commission_bps: float = Field(default=0.0, description="Poplatek brokera za pokyn (Trading 212 = 0 bps)")
    fx_conversion_bps: float = Field(default=15.0, description="Poplatek za měnovou konverzi (T212 FX = 15 bps = 0.15 %)")

    def total_cost_bps(self, is_cross_currency: bool = True) -> float:
        """Vrátí celkové jednosměrné náklady v bazických bodech (bps)."""
        fx = self.fx_conversion_bps if is_cross_currency else 0.0
        return self.spread_bps + self.slippage_bps + self.commission_bps + fx

    def total_cost_pct(self, is_cross_currency: bool = True) -> float:
        """Vrátí celkové jednosměrné náklady v procentech (např. 0.25 %)."""
        return self.total_cost_bps(is_cross_currency=is_cross_currency) / 100.0

    def roundtrip_cost_pct(self, is_cross_currency: bool = True) -> float:
        """Vrátí celkové roundtrip náklady (vstup + výstup) v procentech."""
        return self.total_cost_pct(is_cross_currency=is_cross_currency) * 2.0


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Režim obchodování (výchozí stav je bezpečný PAPER)
    trading_mode: TradingMode = Field(default=TradingMode.PAPER, alias="TRADING_MODE")
    confirm_live_trading: str = Field(default="false", alias="CONFIRM_LIVE_TRADING")
    i_understand_live_risk: str = Field(default="no", alias="I_UNDERSTAND_LIVE_RISK")

    @field_validator("trading_mode", mode="before")
    @classmethod
    def parse_trading_mode(cls, value: Any) -> TradingMode:
        if isinstance(value, TradingMode):
            return value
        if isinstance(value, str):
            val_clean = value.strip().lower()
            if val_clean in ("demo", "paper"):
                return TradingMode.PAPER
            elif val_clean == "backtest":
                return TradingMode.BACKTEST
            elif val_clean == "live":
                return TradingMode.LIVE
        return TradingMode.PAPER

    # Čtení ze systémových proměnných Windows i .env
    t212_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "TRADING_212_DEMO_API_KEY",
            "TRADING_212_API_KEY",
            "T212_API_KEY"
        )
    )
    t212_api_secret: str = Field(
        default="",
        validation_alias=AliasChoices(
            "TRADING_212_DEMO_SECRET_KEY",
            "TRADING_212_SECRET_KEY",
            "T212_API_SECRET"
        )
    )

    # Lokální LLM (Lemonade Server)
    lemonade_base_url: str = Field(default="http://localhost:8000/api/v1", alias="LEMONADE_BASE_URL")
    lemonade_api_key: str = Field(default="lemonade", alias="LEMONADE_API_KEY")
    lemonade_model: str = Field(default="Qwen2.5-7B-Instruct", alias="LEMONADE_MODEL")

    # Financial Modeling Prep (FMP) - volitelný institucionální zdroj fundamentů
    fmp_api_key: str = Field(default="", alias="FMP_API_KEY")

    # Kapitál a risk management
    initial_capital_eur: float = Field(default=50000.0, alias="INITIAL_CAPITAL_EUR")
    max_positions: int = Field(default=15, alias="MAX_POSITIONS")

    # Transakční náklady
    transaction_costs: TransactionCostConfig = Field(default_factory=TransactionCostConfig)

    @property
    def t212_base_url(self) -> str:
        if self.trading_mode == TradingMode.LIVE:
            return "https://live.trading212.com"
        return "https://demo.trading212.com"

    @property
    def watched_tickers(self) -> List[str]:
        """Automaticky načte globální univerzum sledovaných tickerů."""
        try:
            from src.strategy.opportunity_scanner import OpportunityScanner
            return OpportunityScanner().get_all_universe_tickers()
        except Exception:
            return ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]

    def validate_live_mode(self) -> None:
        """
        Kritická bezpečnostní pojistka:
        Režim LIVE smí naběhnout POUZE tehdy, pokud je v prostředí nebo .env
        explicitně nastaveno CONFIRM_LIVE_TRADING=true. Jinak vyhodí RuntimeError.
        """
        if self.trading_mode == TradingMode.LIVE:
            # Kontrola v os.environ i v načtené konfiguraci
            env_confirm = os.environ.get("CONFIRM_LIVE_TRADING", "").strip().lower()
            cfg_confirm = self.confirm_live_trading.strip().lower()
            is_confirmed = (env_confirm == "true" or cfg_confirm == "true")

            if not is_confirmed:
                raise RuntimeError(
                    "Bezpečnostní zámek aktivován: TRADING_MODE je nastaven na 'live', "
                    "ale chybí povinná proměnná CONFIRM_LIVE_TRADING=true v prostředí nebo .env. "
                    "Živé obchodování je tvrdě zablokováno pro ochranu reálného kapitálu."
                )

            if self.i_understand_live_risk.strip().lower() != "yes":
                raise RuntimeError(
                    "Bezpečnostní zámek aktivován: Spuštění v LIVE režimu vyžaduje "
                    "I_UNDERSTAND_LIVE_RISK=yes v .env souboru."
                )

    def enforce_safety_checks(self) -> None:
        """Provede bezpečnostní kontrolu při startu aplikace."""
        self.validate_live_mode()
        if self.trading_mode == TradingMode.LIVE:
            banner = "=" * 70
            sys.stderr.write(f"\n{banner}\n")
            sys.stderr.write("POZOR: APLIKACE JE NAKONFIGUROVÁNA V REŽIMU LIVE (REÁLNÉ PENÍZE)!\n")
            sys.stderr.write(f"{banner}\n\n")


settings = AppSettings()
settings.enforce_safety_checks()