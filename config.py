import os
import sys
from typing import Literal
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Režim obchodování
    trading_mode: Literal["demo", "live"] = Field(default="demo", alias="TRADING_MODE")
    i_understand_live_risk: str = Field(default="no", alias="I_UNDERSTAND_LIVE_RISK")

    # Čtení ze systémových proměnných Windows i .env
    t212_api_key: str = Field(
        ...,
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

    @property
    def t212_base_url(self) -> str:
        if self.trading_mode == "live":
            return "https://live.trading212.com"
        return "https://demo.trading212.com"

    def enforce_safety_checks(self) -> None:
        if self.trading_mode == "live":
            banner = "=" * 70
            sys.stderr.write(f"\n{banner}\n")
            sys.stderr.write("POZOR: APLIKACE JE NAKONFIGUROVÁNA V REŽIMU LIVE (REÁLNÉ PENÍZE)!\n")
            sys.stderr.write(f"{banner}\n\n")

            if self.i_understand_live_risk.strip().lower() != "yes":
                sys.stderr.write(
                    "CHYBA: Spuštění v LIVE režimu vyžaduje I_UNDERSTAND_LIVE_RISK=yes v .env souboru.\n"
                    "Proces je z bezpečnostních důvodů ukončen.\n"
                )
                sys.exit(1)


settings = AppSettings()
settings.enforce_safety_checks()