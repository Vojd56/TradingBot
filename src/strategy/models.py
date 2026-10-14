from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class SignalAction(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TREND_BREAK = "TREND_BREAK"
    AI_AVOID = "AI_AVOID"
    ALPHA_ROTATION = "ALPHA_ROTATION"
    MANUAL = "MANUAL"


class TradingSignal(BaseModel):
    """Investiční signál generovaný strategií po technickém i AI vyhodnocení."""
    ticker: str
    action: SignalAction
    total_score: float = Field(..., ge=0.0, le=100.0, description="Kombinované skóre 0-100")
    technical_score: float = Field(..., ge=0.0, le=50.0, description="Technická složka 0-50")
    sentiment_score: float = Field(..., ge=0.0, le=50.0, description="AI Sentimentová složka 0-50")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Spolehlivost signálu")
    
    current_price: float = Field(..., gt=0.0, description="Aktuální cena instrumentu")
    currency: str = Field(default="USD", description="Měna instrumentu (USD, EUR, GBp, GBP)")
    suggested_allocation_eur: float = Field(default=0.0, description="Doporučená alokace kapitálu v EUR")
    suggested_shares: float = Field(default=0.0, description="Doporučený počet frakčních akcií")
    
    stop_loss_price: Optional[float] = Field(default=None, description="Dynamická úroveň Stop Loss")
    take_profit_price: Optional[float] = Field(default=None, description="Dynamická úroveň Take Profit")
    atr_14: float = Field(default=0.0, description="Hodnota ATR 14 pro výpočet volatility")
    
    reasons: List[str] = Field(default_factory=list, description="Klíčové důvody rozhodnutí")
    timestamp: datetime = Field(default_factory=datetime.now)


class PositionExitSignal(BaseModel):
    """Signál pro uzavření existující otevřené pozice."""
    ticker: str
    should_exit: bool
    reason: Optional[ExitReason] = None
    exit_price: float
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
