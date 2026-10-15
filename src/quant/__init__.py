"""Kvantitativní analytické a rozhodovací moduly dle výzkumné studie research.md.

Obsahuje:
- features: Stacionární a bezrozměrné kauzální příznaky
- regime_detector: Globální detekce tržního režimu (SPY/QQQ)
- signal_engine: Ortogonální Carverův model (Momentum + Mean Reversion)
- prediction_model: Metoda trojité bariéry, výpočet EV_net a brána NO TRADE
- risk_engine: Carverovo Volatility Targeting, 10% pásmo nečinnosti, limity a korelace
- execution_engine: Idempotentní Long-only exekuce a pre-trade rekonsiliace
- monitoring: 6 bezpečnostních Kill-Switchů a online CUSUM telemetrie
"""

from src.quant.features import CausalFeatureEngine, StockFeatures

__all__ = [
    "CausalFeatureEngine",
    "StockFeatures",
]
