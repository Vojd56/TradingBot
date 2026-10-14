import logging
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# Reprezentativní globální světové univerzum (MSCI World & Global Leaders)
# Zahrnuje přední lídry z Evropy, USA i Asie pokrývající všechny klíčové sektory
DEFAULT_MARKET_UNIVERSE: Dict[str, List[str]] = {
    "EUROPE_DEFENSE_AEROSPACE_INDUSTRIALS": [
        "RHM.DE",   # Rheinmetall AG (Evropská obrana & zbrojní supercyklus - Xetra)
        "RR.L",     # Rolls-Royce Holdings (Letecké motory, SMR jaderná energetika - LSE)
        "AIR.PA",   # Airbus SE (Globální letectví & obrana - Euronext Paris)
        "BA.L",     # BAE Systems (Obranné systémy & námořnictvo - LSE)
        "HO.PA",    # Thales SA (Obranná avionika & radar - Euronext Paris)
        "SAF.PA",   # Safran SA (Letecké motory & komponenty - Euronext Paris)
        "SIE.DE",   # Siemens AG (Průmyslová automatizace & elektrifikace - Xetra)
        "SU.PA",    # Schneider Electric (Energetický management pro datacentra - Euronext Paris)
        "ABBN.SW",  # ABB Ltd (Robotika & elektrifikace - SIX Švýcarsko)
        "SUNB",     # Sunbelt Rentals / Ashtead (Infrastrukturní průmyslový pronájem)
    ],
    "EUROPE_TECH_SEMIS_SOFTWARE": [
        "ASML.AS",  # ASML Holding (Monopol na EUV litografii - Euronext Amsterdam)
        "SAP.DE",   # SAP SE (Evropský páteřní ERP & cloudový gigant - Xetra)
        "IFX.DE",   # Infineon Technologies (Automotive & výkonové čipy - Xetra)
        "ADYEN.AS", # Adyen NV (Globální digitální platební infrastruktura - Euronext Amsterdam)
        "PRX.AS",   # Prosus NV (Globální technologický investor & e-commerce - Euronext Amsterdam)
    ],
    "EUROPE_HEALTH_CONSUMER_LUXURY": [
        "MC.PA",    # LVMH (Světový lídr v luxusu & módě - Euronext Paris)
        "RMS.PA",   # Hermès International (Ultra-luxusní compounder - Euronext Paris)
        "NESN.SW",  # Nestlé SA (Globální potravinářský & spotřební gigant - SIX Švýcarsko)
        "OR.PA",    # L'Oréal SA (Světový lídr v kosmetice - Euronext Paris)
        "AZN.L",    # AstraZeneca (Globální biofarmacie & onkologie - LSE)
        "SAN.PA",   # Sanofi SA (Evropská farmacie & vakcíny - Euronext Paris)
        "NOVN.SW",  # Novartis AG (Inovativní biofarmacie - SIX Švýcarsko)
        "RO.SW",    # Roche Holding (Diagnostika & farmacie - SIX Švýcarsko)
    ],
    "EUROPE_ENERGY_COMMODITIES_FINANCE": [
        "SHEL.L",   # Shell plc (Globální LNG & energetika - LSE)
        "TTE.PA",   # TotalEnergies SE (Integrovaná energetika & LNG - Euronext Paris)
        "BP.L",     # BP plc (Energetika - LSE)
        "RIO.L",    # Rio Tinto (Těžba kovů, měď & železná ruda - LSE)
        "ALV.DE",   # Allianz SE (Pojišťovací & finanční gigant - Xetra)
        "HSBA.L",   # HSBC Holdings (Mezinárodní bankovnictví - LSE)
        "UBSG.SW",  # UBS Group (Globální wealth management - SIX Švýcarsko)
    ],
    "ASIA_EMERGING_GLOBAL_CHAMPIONS": [
        "SMSN.IL",  # Samsung Electronics GDR (Světový lídr v paměťových čipech & AI - LSE)
        "TSM",      # Taiwan Semiconductor (Slévárenský monopol pro výrobu čipů - US ADR)
        "SONY",     # Sony Group (Obrazové senzory, zábava & elektronika - US ADR)
        "TM",       # Toyota Motor (Globální automotive - US ADR)
        "BABA",     # Alibaba Group (E-commerce & cloud v Asii - US ADR)
        "SE",       # Sea Limited (E-commerce & fintech v JV Asii - US)
        "MELI",     # MercadoLibre (E-commerce & fintech v Latinské Americe - US)
        "NU",       # Nu Holdings (Digitální bankovní revoluce - US)
    ],
    "US_SEMIS_AI_HARDWARE": [
        "NVDA", "AMD", "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "ARM", "VRT", "MPWR"
    ],
    "US_SOFTWARE_CLOUD_AI": [
        "MSFT", "GOOGL", "AMZN", "META", "AAPL", "PLTR", "CRM", "NOW", "CRWD", "DDOG", "SNOW", "NET", "APP", "MSTR", "COIN"
    ],
    "US_AEROSPACE_DEFENSE_INDUSTRIALS": [
        "RTX", "LMT", "NOC", "GD", "GE", "CAT", "DE", "ETN", "TDG", "HEI"
    ],
    "US_ENERGY_NUCLEAR_RENAISSANCE": [
        "CEG", "VST", "XOM", "CVX"
    ],
    "US_QUALITY_COMPOUNDERS_FINANCE_HEALTH": [
        "BRK-B", "LLY", "ISRG", "UNH", "ABBV", "JPM", "V", "MA", "AXP", "COST", "WMT", "TSLA", "NVO"
    ],
}


class CandidateScreenerScore:
    def __init__(
        self,
        ticker: str,
        price: float,
        score: float,
        trend_bullish: bool,
        rsi_14: float,
        momentum_20d: float,
        reasons: List[str]
    ):
        self.ticker = ticker
        self.price = price
        self.score = score
        self.trend_bullish = trend_bullish
        self.rsi_14 = rsi_14
        self.momentum_20d = momentum_20d
        self.reasons = reasons

    def __repr__(self) -> str:
        return f"<Candidate {self.ticker}: Score={self.score:.1f}, Price={self.price:.2f}, RSI={self.rsi_14:.1f}, Mom20d={self.momentum_20d:+.1f}%>"


class OpportunityScanner:
    """
    Autonomní skener trhu a generátor investičních příležitostí.
    Pracuje ve dvou fázích:
    1. Rychlý kvantitativní screening širokého univerza (40+ akcií) v paralelním stahování
    2. Výběr top kandidátů s nejsilnějším technickým momentem a breakout potenciálem
       pro hloubkovou lokální AI analýzu na AMD NPU.
    Nevyžaduje žádné manuální zadávání tickerů ani externí Gemini API.
    """

    def __init__(self, universe: Optional[Dict[str, List[str]]] = None):
        self.universe = universe or DEFAULT_MARKET_UNIVERSE

    def get_all_universe_tickers(self) -> List[str]:
        """Vrátí deduplikovaný seznam všech sledovaných tickerů."""
        tickers: Set[str] = set()
        for cat, t_list in self.universe.items():
            for t in t_list:
                tickers.add(t.strip().upper())
        return sorted(list(tickers))

    def register_discovered_ticker(self, ticker: str, category: str = "AUTONOMOUS_DISCOVERIES") -> bool:
        """
        Dynamicky zaregistruje nově objevený ticker z médií do univerza.
        Vrátí True, pokud byl ticker nově přidán.
        """
        clean = ticker.strip().upper()
        if not clean:
            return False

        all_existing = self.get_all_universe_tickers()
        if clean in all_existing:
            return False

        self.universe.setdefault(category, [])
        if clean not in self.universe[category]:
            self.universe[category].append(clean)
            logger.info(f"✨ [OpportunityScanner] Nový dynamický titul zařazen do univerza: {clean} ({category})")
            return True
        return False

    def fast_quant_screen(
        self,
        candidate_tickers: Optional[List[str]] = None,
        top_n: int = 10
    ) -> List[CandidateScreenerScore]:
        """
        Stáhne paralelně roční cenová data pro všechny tickery v univerzu,
        spočítá trend, RSI, 20denní momentum a breakout signály
        a vrátí top_n nejlepších kandidátů.
        """
        tickers = candidate_tickers or self.get_all_universe_tickers()
        logger.info(f"Spouštím rychlý kvantitativní screening pro {len(tickers)} akcií...")

        try:
            # Paralelní stažení cenové historie za 1 rok
            raw_data = yf.download(
                tickers,
                period="1y",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False
            )
        except Exception as exc:
            logger.error(f"Chyba při paralelním stahování tržních dat: {exc}")
            return []

        scored_candidates: List[CandidateScreenerScore] = []

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = raw_data
                else:
                    if ticker not in raw_data.columns.levels[0]:
                        continue
                    df = raw_data[ticker]

                close = df["Close"].dropna()
                if len(close) < 120:
                    continue

                current_price = float(close.iloc[-1])
                sma_50 = float(close.rolling(window=50).mean().iloc[-1])
                sma_200 = float(close.rolling(window=200).mean().iloc[-1]) if len(close) >= 200 else sma_50

                # 20denní momentum
                price_20d_ago = float(close.iloc[-20]) if len(close) >= 20 else float(close.iloc[0])
                mom_20d = ((current_price - price_20d_ago) / price_20d_ago) * 100.0

                # 52týdenní maximum (blízkost k breakoutu)
                high_52w = float(df["High"].max())
                dist_to_high = ((current_price - high_52w) / high_52w) * 100.0

                # RSI 14
                delta = close.diff()
                gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, np.nan)
                rsi_series = 100 - (100 / (1 + rs))
                rsi_14 = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

                trend_bullish = current_price > sma_50 and sma_50 > sma_200
                score = 0.0
                reasons = []

                # Kvantitativní bodování
                # 1. Růstový trend
                if trend_bullish:
                    score += 35.0
                    reasons.append("Silný uptrend (Cena > SMA50 > SMA200)")
                elif current_price > sma_50:
                    score += 20.0
                    reasons.append("Cena nad SMA50")
                elif current_price > sma_200:
                    score += 10.0

                # 2. RSI momentum
                if 45.0 <= rsi_14 <= 65.0:
                    score += 25.0
                    reasons.append(f"Optimální momentum (RSI {rsi_14:.1f})")
                elif 65.0 < rsi_14 <= 75.0:
                    score += 20.0
                    reasons.append(f"Silný nákupní tlak (RSI {rsi_14:.1f})")
                elif 35.0 <= rsi_14 < 45.0:
                    score += 15.0
                    reasons.append("Zdravý pullback v trendu")
                elif rsi_14 > 78.0:
                    score += 5.0  # Překoupeno, penalizace

                # 3. 20denní momentum a breakout
                if mom_20d > 8.0:
                    score += 20.0
                    reasons.append(f"Vysoké krátkodobé momentum (+{mom_20d:.1f}% za 20 dní)")
                elif mom_20d > 2.0:
                    score += 12.0
                elif mom_20d < -8.0:
                    score -= 10.0

                # 4. Blízkost k 52týdennímu maximu (potenciál breakoutu)
                if dist_to_high > -5.0:
                    score += 15.0
                    reasons.append("Testování 52T maxima (breakout zóna)")
                elif dist_to_high > -15.0:
                    score += 8.0

                candidate = CandidateScreenerScore(
                    ticker=ticker,
                    price=round(current_price, 2),
                    score=round(max(0.0, score), 2),
                    trend_bullish=trend_bullish,
                    rsi_14=round(rsi_14, 1),
                    momentum_20d=round(mom_20d, 1),
                    reasons=reasons
                )
                scored_candidates.append(candidate)

            except Exception as e:
                logger.debug(f"Chyba při výpočtu skóre pro {ticker}: {e}")
                continue

        # Seřazení podle skóre sestupně
        scored_candidates.sort(key=lambda c: c.score, reverse=True)

        # Tržní inteligence: Pokud je otevřen evropský trh (09:00 - 17:30 CET),
        # garantujeme zařazení nejlepších evropských titulů do TOP výběru pro okamžitou exekuci
        from src.strategy.market_hours import MarketHours
        eu_tickers_set = set()
        for cat, t_list in self.universe.items():
            if "EUROPE" in cat.upper():
                for t in t_list:
                    eu_tickers_set.add(t.upper())

        if MarketHours.is_eu_market_open():
            eu_candidates = [
                c for c in scored_candidates
                if c.ticker.upper() in eu_tickers_set or any(c.ticker.upper().endswith(sfx) for sfx in [".DE", ".L", ".AS", ".SW", ".IL"])
            ]
            other_candidates = [c for c in scored_candidates if c not in eu_candidates]

            # Vybereme top evropské kandidáty (až polovinu limitu) a doplníme z globálního výběru
            top_eu_count = min(len(eu_candidates), max(3, top_n // 2))
            top_eu = eu_candidates[:top_eu_count]
            top_other = other_candidates[:max(0, top_n - len(top_eu))]
            top_candidates = top_eu + top_other
            top_candidates.sort(key=lambda c: c.score, reverse=True)
            logger.info(f"Evropský trh je otevřen -> Zařazuji {len(top_eu)} top evropských titulů k okamžitému auditu.")
        else:
            top_candidates = scored_candidates[:top_n]

        logger.info(
            f"Kvantitativní screening dokončen. Top {len(top_candidates)} vybraných příležitostí: "
            f"{', '.join([f'{c.ticker} ({c.score}b)' for c in top_candidates])}"
        )
        return top_candidates
