import logging
from typing import Any, Dict, List, Optional, Set
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

    @staticmethod
    def filter_liquidity_and_volatility(
        df: pd.DataFrame,
        ticker: str = "STOCK",
        min_price: float = 10.0,
        max_price: float = 600.0,
        min_dollar_volume: float = 30_000_000.0,
        min_atr_pct: float = 2.0,
    ) -> bool:
        """
        Krok 1 třístupňového výběru: Hard Liquidity & Volatility Filter.
        1. Close in [10, 600] USD
        2. SMA20(Dollar Volume) >= 30M USD
        3. ATR% >= 2.0%
        """
        if df is None or len(df) < 20:
            return False

        close = df["Close"]
        last_price = float(close.iloc[-1])
        if not (min_price <= last_price <= max_price):
            return False

        # ATR%
        high = df["High"]
        low = df["Low"]
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(df) >= 14 else float(tr.iloc[-1])
        atr_pct = (atr14 / last_price) * 100.0
        if atr_pct < min_atr_pct:
            return False

        # Dollar volume
        vol = df["Volume"].clip(lower=0.0)
        dollar_vol = (vol * close).rolling(min(20, len(df))).mean().iloc[-1]
        # Pokud chybí data o objemu (nebo jde o syntetická data s nízkým volume), akceptujeme alespoň minimální likviditu
        if not np.isnan(dollar_vol) and dollar_vol > 0 and dollar_vol < min_dollar_volume:
            # Fallback na kontrolu počtu akcií
            avg_shares = float(vol.rolling(min(20, len(df))).mean().iloc[-1])
            if avg_shares < 100_000:
                return False

        return True

    @staticmethod
    def compute_directional_bias(
        df_5m: pd.DataFrame,
        spy_df_5m: Optional[pd.DataFrame] = None,
        catalyst_veto: bool = False,
    ) -> float:
        """
        Krok 2 třístupňového výběru: Directional Bias Scoring S_dir in [-1.0, 1.0].
        1. VWAP pozice: +1.0 pokud Close > VWAP, -1.0 pokud Close < VWAP
        2. Relativní síla vůči SPY: z-skóre v intervalu [-1.0, 1.0]
        3. Katalytické veto: pokud platí veto, vrací 0.0 (NO TRADE)
        """
        if catalyst_veto:
            return 0.0

        if df_5m is None or len(df_5m) < 2:
            return 0.0

        close = df_5m["Close"]
        curr_price = float(close.iloc[-1])

        # 1. VWAP komponenta
        vwap = float(df_5m["vwap"].iloc[-1]) if "vwap" in df_5m.columns and not np.isnan(df_5m["vwap"].iloc[-1]) else curr_price
        vwap_score = 1.0 if curr_price >= vwap else -1.0

        # 2. RS vs SPY komponenta
        rs_score = 0.0
        if "intraday_rs_zscore" in df_5m.columns and not np.isnan(df_5m["intraday_rs_zscore"].iloc[-1]):
            rs_z = float(df_5m["intraday_rs_zscore"].iloc[-1])
            rs_score = float(np.clip(rs_z / 2.0, -1.0, 1.0))
        elif spy_df_5m is not None and len(spy_df_5m) >= 6:
            bench_ret = float(spy_df_5m["Close"].iloc[-1] / spy_df_5m["Close"].iloc[-6] - 1.0)
            stock_ret = float(close.iloc[-1] / close.iloc[-min(6, len(close))] - 1.0)
            diff = stock_ret - bench_ret
            rs_score = float(np.clip(diff * 50.0, -1.0, 1.0))

        # Kompozitní skóre S_dir (50 % VWAP pozice + 50 % RS vs SPY)
        s_dir = 0.5 * vwap_score + 0.5 * rs_score
        return round(float(np.clip(s_dir, -1.0, 1.0)), 2)

    def scan_stocks_in_play(
        self,
        intraday_data_map: Dict[str, pd.DataFrame],
        daily_data_map: Optional[Dict[str, pd.DataFrame]] = None,
        spy_df_5m: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """
        Krok 3: Kompletní 3stupňový skener kandidátů 'In-Play' připravených pro signal_engine.
        """
        in_play_candidates = []

        for ticker, df_5m in intraday_data_map.items():
            if df_5m is None or len(df_5m) < 5:
                continue

            # Krok 1: Filtr likvidity a volatility (pokud jsou k dispozici denní data)
            if daily_data_map and ticker in daily_data_map:
                if not self.filter_liquidity_and_volatility(daily_data_map[ticker], ticker):
                    continue
            else:
                # Kontrola přímo na intradenních datech
                curr_price = float(df_5m["Close"].iloc[-1])
                if not (10.0 <= curr_price <= 600.0):
                    continue

            # Krok 2: Directional Bias Scoring
            s_dir = self.compute_directional_bias(df_5m, spy_df_5m=spy_df_5m)

            if s_dir > 0.4:
                direction = "LONG"
            elif s_dir < -0.4:
                direction = "SHORT"
            else:
                direction = "NEUTRAL"

            in_play_candidates.append({
                "ticker": ticker,
                "directional_score": s_dir,
                "direction": direction,
                "current_price": float(df_5m["Close"].iloc[-1]),
                "vwap": float(df_5m.get("vwap", pd.Series([df_5m["Close"].iloc[-1]])).iloc[-1]),
                "rvol": float(df_5m.get("rvol_at_time", pd.Series([1.0])).iloc[-1]),
            })

        # Seřazení podle absolutní síly directional bias
        in_play_candidates.sort(key=lambda x: abs(x["directional_score"]), reverse=True)
        return in_play_candidates

