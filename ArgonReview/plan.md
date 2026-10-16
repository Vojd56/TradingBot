# Antigravity Implementation Plan — Konkrétní technické zadání

Tento dokument slouží jako přímý exekuční plán pro vývojového agenta **Google Antigravity**. Úkoly jsou seřazeny tak, aby agent **nejprve sjednotil architekturu, opravil datovou vrstvu, vybudoval validační backtester a risk management**, a teprve poté upravoval obchodní strategie, ML a LLM.

---

## Pravidla pro agenta Antigravity před zahájením práce
1. **Žádné slepé předpoklady:** Před úpravou jakéhokoliv souboru si přečti jeho aktuální obsah na disku, zkontroluj importy v `main.py`, `config.py` a existující testy v `tests/`.
2. **Zachování funkčnosti testů:** Po každém úkolu spusť `pytest` a ujisti se, že všechny stávající i nové testy procházejí.
3. **Výchozí bezpečnost:** Nikdy v kódu nenastavuj živé obchodování jako výchozí stav.

---

## Seznam implementačních úkolů (Task Breakdown)

### TASK AG-00: Baseline Audit, Konfigurační čistka a Bezpečnostní pojistka režimů
* **Priorita:** P0 (Kritická — První krok)
* **Cíl změny:** Zavést striktní enumeraci režimů (`BACKTEST`, `PAPER`, `LIVE`), ošetřit nákladové konstanty (vč. FX fee u T212) a zdokumentovat reálné propojení mezi `src/strategy/` a `src/quant/`.
* **Problém v současném projektu:** `config.py` a spouštěcí `.cmd` skripty nemají tvrdý dvoufázový zámek proti nechtěnému spuštění živých pokynů a neobsahují explicitní model transakčních a měnových nákladů podle brokera.
* **Dotčené soubory:** `config.py`, `main.py`, `.env.example`, `tests/test_execution.py`.
* **Požadovaný cílový stav:**
  * V `config.py` definovat `TradingMode(str, Enum): BACKTEST = "backtest", PAPER = "paper", LIVE = "live"`. Výchozí hodnota `PAPER` nebo `BACKTEST`.
  * Přidat metodu `validate_live_mode()`, která vyhodí `RuntimeError`, pokud je zvolen `LIVE`, ale v `.env` chybí `CONFIRM_LIVE_TRADING=true`.
  * Přidat datovou třídu `TransactionCostConfig(spread_bps, slippage_bps, commission_bps, fx_conversion_bps)`.
* **Testy:** Unit test ověřující, že pokus o inicializaci `LIVE` režimu bez `CONFIRM_LIVE_TRADING=true` okamžitě selže s `RuntimeError`.
* **Kritéria přijetí:** `pytest` prochází, `main.py --help` a bezpečný start v `BACKTEST`/`PAPER` režimu fungují bez chyby.
* **Závislosti:** Žádné.
* **Rizika a Rollback:** Nulové riziko (`git checkout config.py main.py`).

---

### TASK AG-01: Datová vrstva, Session Kalendář a Parquet Cache (`src/strategy/intraday_data.py` & `market_hours.py`)
* **Priorita:** P0 (Kritická)
* **Cíl změny:** Zajistit časově přesná, validovaná intradenní OHLCV data v časovém pásmu `America/New_York` s lokálním ukládáním do Parquetu pro rychlé a reprodukovatelné backtesty.
* **Problém v současném projektu:** `src/strategy/intraday_data.py` stahuje data za běhu bez trvalého sloupcového úložiště a hrozí míchání pre-market/after-hours svíček s hlavní obchodní seancí (09:30–16:00 EST).
* **Dotčené soubory:** `src/strategy/intraday_data.py`, `src/strategy/market_hours.py`, `requirements.txt` (doplnění `pyarrow` pokud chybí), nový test `tests/test_intraday_data_validation.py`.
* **Požadovaný cílový stav:**
  1. V `market_hours.py` implementovat přesnou funkci `tag_market_session(df: pd.DataFrame) -> pd.DataFrame`, která převede index do `America/New_York` a přidá sloupce `session` (`PRE`, `REGULAR`, `POST`), `trading_day` (datum seance) a `minutes_from_open` (0 až 390 pro regular session).
  2. V `intraday_data.py` přidat `ParquetBarStore` (`data/bars/{timeframe}/{symbol}.parquet`), který ukládá a inkrementálně aktualizuje stažené svíčky.
  3. Přidat funkci `validate_ohlcv_integrity(df)`, která odhalí duplicitní časová razítka, záporné objemy, porušení `Low <= Open, Close <= High` a nahlásí chybějící 5m svíčky uvnitř regular session.
  4. Přidat volitelnou podporu pro stahování hluboké 5m historie přes Alpaca Market Data REST API (pokud jsou v `.env` klíče `ALPACA_API_KEY`), s automatickým fallbackem na `yfinance`.
* **Testy:** Unit testy v `tests/test_intraday_data_validation.py` testující přechod zimního/letního času (US DST vs. EU DST v březnu), detekci poškozených svíček a uložení/načtení z Parquetu.
* **Kritéria přijetí:** Načtení 30 dní 5m dat z lokální Parquet cache trvá `< 50 ms`, sloupce `is_regular_session` a `minutes_from_open` jsou exaktně vypočteny.
* **Závislosti:** AG-00.

---

### TASK AG-02: Kauzální Intradenní Feature Engineering (`src/quant/features.py`)
* **Priorita:** P0 (Kritická)
* **Cíl změny:** Opravit a rozšířit `src/quant/features.py` o skutečné intradenní institucionální indikátory bez look-ahead biasu a bez noční kontaminace.
* **Problém v současném projektu:** Indikátory počítané přes kontinuální řadu bez session-resetu deformují VWAP a relativní objem při otevření trhu.
* **Dotčené soubory:** `src/quant/features.py`, `src/strategy/market_structure.py`, `tests/test_quant_features.py`.
* **Požadovaný cílový stav:**
  Do `src/quant/features.py` implementovat čisté, vektorizované funkce (vstupem je DataFrame z AG-01):
  1. `compute_session_vwap(df)`: Počítá VWAP a směrodatné odchylky ($\pm 1\sigma, \pm 2\sigma, \pm 2.5\sigma$) **resetované přesně v 09:30 EST každého obchodního dne** (pouze z `REGULAR` session svíček).
  2. `compute_opening_range(df, orb_minutes=15)`: Pro každý den vypočítá `orb_high`, `orb_low`, `orb_width` z prvních 15 minut (09:30–09:45 EST). **Kritické:** Před časem 09:45 EST musí být hodnota `orb_high` a `orb_low` rovna `NaN` (zákaz look-ahead biasu!).
  3. `compute_rvol_at_time(df, lookback_days=20)`: Poměr kumulativního objemu od 09:30 do aktuální minuty $m$ vůči klouzavému průměru kumulativního objemu v minutě $m$ za předchozích 20 obchodních dní (strictly lagged: `.shift(1)` po dnech).
  4. `compute_intraday_rs(stock_df, benchmark_df, window=6)`: Relativní síla akcie vůči benchmarku (SPY) očištěná o volatilitu.
* **Testy:** Rozšířit `tests/test_quant_features.py` o **Causal Leakage Test**: vzít prefix dat `df.iloc[:k]`, spočítat features a ověřit, že se na indexu `k-1` na 8 desetinných míst shodují s features spočítanými na celém `df` (`pd.testing.assert_series_equal`).
* **Kritéria přijetí:** Causal Leakage Test prochází na 100 %, Session VWAP se resetuje každý den v 09:30 EST.
* **Závislosti:** AG-01.

---

### TASK AG-03: Event-Driven Backtester & Walk-Forward Validační Engine (`src/quant/backtester.py`)
* **Priorita:** P0 (Kritická — Nutné před laděním strategií)
* **Cíl změny:** Vytvořit realistický backtesting engine, který měří čistý výnos po započtení spreadu, skluzu a komisí.
* **Problém v současném projektu:** Chybí dedikovaný modul pro realistickou bar-by-bar simulaci s konzervativním vyhodnocováním SL/TP kolizí a Walk-Forward validací.
* **Dotčené soubory:** Nový modul `src/quant/backtester.py`, úprava `src/quant/monitoring.py`, nový test `tests/test_backtester.py`.
* **Požadovaný cílový stav:**
  * Třída `IntradayBacktester`:
    * Přijímá OHLCV+Features DataFrame, instanci strategie/signal enginu, `RiskEngine` a `TransactionCostConfig`.
    * **Pravidlo exekuce:** Signál ze svíčky `t` se exekuuje na `Open` svíčky `t+1` + `half_spread` + `slippage`.
    * **Pravidlo SL/TP kolize:** Pokud na svíčce `t+k` platí `Low <= stop_loss` i `High >= take_profit`, obchod vždy končí na `stop_loss` (pesimistický předpoklad).
    * **Pravidlo EOD Flatten:** Všechny intradenní pozice otevřené v `15:55 EST` se automaticky uzavírají na `Close` svíčky 15:55.
  * Třída `WalkForwardEvaluator`: Rozdělí dataset na $K$ klouzavých oken (In-Sample $\to$ Embargo $\to$ Out-of-Sample) a agreguje OOS metriky: `Net Return`, `Sharpe`, `Sortino`, `Max Drawdown`, `Profit Factor`, `Win Rate`, `Avg Trade (bps)`, `Bootstrap 95% CI`.
* **Testy:** V `tests/test_backtester.py` otestovat:
  1. Že strategie nakupující na `Close` nedostane v backtesteru cenu `Close[t]`, ale `Open[t+1] + costs`.
  2. Že svíčka zasahující současně SL i TP vyhodnotí SL.
  3. Že na konci dne nezůstane žádná otevřená pozice přes noc.
* **Kritéria přijetí:** Deterministický výstup, 100% průchod testů v `tests/test_backtester.py`.
* **Závislosti:** AG-01, AG-02.

---

### TASK AG-04: Sjednocení Risk Managementu a Exekučního Adaptéru (`src/quant/risk_engine.py`, `src/strategy/risk_manager.py`, `src/execution/`)
* **Priorita:** P1 (Vysoká — Provozní bezpečnost)
* **Cíl změny:** Odstranit architektonickou duplicitu mezi `src/strategy/risk_manager.py` a `src/quant/risk_engine.py` a zavést jednotný `RiskGatekeeper` + čisté rozhraní `BrokerAdapter`.
* **Problém v současném projektu:** Dva různé moduly pro risk management a přímá vazba exekuce na `t212_client.py`.
* **Dotčené soubory:** `src/quant/risk_engine.py`, `src/strategy/risk_manager.py`, `src/execution/order_executor.py`, `src/execution/reconciler.py`, `tests/test_quant_risk_carver.py`, `tests/test_execution.py`.
* **Požadovaný cílový stav:**
  1. Rozšířit `src/quant/risk_engine.py` o třídu `RiskGatekeeper` (která zachovává Carverovy výpočty pro `test_quant_risk_carver.py`, aby nerozbila stávající testy, a přidává metodu `evaluate_order_intent(intent, portfolio_state, market_bar) -> ApprovedOrder | RejectedOrder`):
     * Kontrola `max_risk_per_trade_pct` (výpočet počtu akcií $Q = \lfloor \frac{\text{Equity} \cdot r_{max}}{\vert{}Entry - SL\vert{}} \rfloor$).
     * Kontrola `max_daily_loss_pct` (Daily Kill-Switch).
     * Kontrola `max_concurrent_positions` a `stale_data_seconds <= 120`.
     * Kontrola No-Trade Bufferu (zákaz úpravy pozice, pokud se cílová velikost liší o méně než 20 % od současné — ochrana proti přetáčivému churningu).
  2. V `src/strategy/risk_manager.py` ponechat tenký wrapper (fasádu), který deleguje všechna volání na `src/quant/risk_engine.py` (kvůli zpětné kompatibilitě importů).
  3. V `src/execution/order_executor.py` zavést idempotentní `client_order_id` deduplikaci a rozhraní `BrokerAdapter` (`PaperBrokerAdapter`, `T212BrokerAdapter`, a připravený slot pro `AlpacaBrokerAdapter`).
* **Testy:** Otestovat, že `RiskGatekeeper` zablokuje obchod při překročení denní ztráty (-2 %), při zastaralé svíčce (>120 s) a při duplicitním `client_order_id`. Všechny stávající testy v `test_quant_risk_carver.py` a `test_execution.py` musí projít.
* **Kritéria přijetí:** Žádný příkaz v `order_executor.py` nemůže být odeslán bez razítka `ApprovedOrder` z `RiskGatekeeper`.
* **Závislosti:** AG-00, AG-03.

---

### TASK AG-05: Sjednocení Detekce Režimu, Stocks-in-Play Skeneru a 3 Day-Trading Strategií
* **Priorita:** P1 (Vysoká — Obchodní logika a Edge)
* **Cíl změny:** Sloučit `macro_regime.py` a `regime_detector.py`, zpřísnit `opportunity_scanner.py` na třístupňový filtr a implementovat 3 čisté intradenní strategie do `signal_engine.py`.
* **Problém v současném projektu:** Dvojkolejnost mezi `src/strategy/day_trader.py` a `src/quant/signal_engine.py`, chybějící filtr na RVOL a časová okna.
* **Dotčené soubory:** `src/quant/regime_detector.py`, `src/strategy/macro_regime.py`, `src/strategy/opportunity_scanner.py`, `src/quant/signal_engine.py`, `src/strategy/day_trader.py`, `tests/test_quant_regime_and_signals.py`, `tests/test_strategy.py`.
* **Požadovaný cílový stav:**
  1. Sloučit logiku režimu trhu do `src/quant/regime_detector.py` (výstup: `MarketRegime`: `TRENDING_BULL`, `TRENDING_BEAR`, `CHOPPY_RANGE`, `HIGH_VOL_PANIC`).
  2. V `opportunity_scanner.py` implementovat třístupňový proces (1. Likvidita & $ATR\% \ge 2\%$ & $RVOL_{15m} \ge 1.5$; 2. Directional Bias vůči VWAP a SPY; 3. Předání do `signal_engine.py`).
  3. V `src/quant/signal_engine.py` implementovat modulární strategie:
     * `ORB15mStrategy` (vstupy 09:45–11:00 EST, RVOL potvrzení, SL/TP v násobcích ATR).
     * `VWAPPullbackStrategy` (vstupy 10:00–14:30 EST v trendovém režimu).
     * `VWAPMeanReversionStrategy` (vstupy v `CHOPPY_RANGE` režimu při deviaci $>2.5\sigma$).
  4. Upravit `src/strategy/day_trader.py` a `src/strategy/quant_trader.py`, aby oba používaly tento sjednocený `signal_engine.py` a `risk_engine.py`.
* **Testy:** Rozšířit `tests/test_quant_regime_and_signals.py` a `tests/test_strategy.py` o syntetické cenové scénáře pro ORB breakout a VWAP pullback a ověřit přesné vygenerování vstupu, SL, TP a Time-stopu.
* **Kritéria přijetí:** Obchodní strategie lze spustit v `IntradayBacktester` z AG-03 a vygenerovat kompletní porovnávací tabulku strategií.
* **Závislosti:** AG-02, AG-03, AG-04.

---

### TASK AG-06: Přepracování ML Predikčního Modelu na Triple-Barrier Meta-Labeling (`src/quant/prediction_model.py`)
* **Priorita:** P2 (Střední — Aplikovat až po změření čistě pravidlové baseline!)
* **Cíl změny:** Opravit `src/quant/prediction_model.py` tak, aby odhadoval kalibrovanou pravděpodobnost zasažení Take-Profitu před Stop-Lossem ($P(\text{TP before SL})$) bez data leakage.
* **Problém v současném projektu:** Predikce bez Triple-Barrier labelingu a Purged Cross-Validace vede k přeučení a chybně zkalibrovanému EV.
* **Dotčené soubory:** `src/quant/prediction_model.py`, `tests/test_quant_prediction_ev.py`.
* **Požadovaný cílový stav:**
  1. Implementovat funkci `create_triple_barrier_labels(df, signals, tp_atr_mult, sl_atr_mult, max_bars)` vracející binární target $Y \in \{0, 1\}$ a realizovaný čistý výnos v jednotkách $R$.
  2. Implementovat `PurgedTimeSeriesSplit(n_splits=5, embargo_bars=12)`, který odstraní překrývající se obchodní horizonty mezi trénovací a validační množinou.
  3. Jako výchozí meta-model použít **Regularizovanou Logistickou Regresi / Calibrated LightGBM** s Plattovým škálováním.
  4. Zachovat výpočet $EV = \hat{p} \cdot R_{TP} - (1 - \hat{p}) \cdot R_{SL} - R_{cost}$ a obchod povolit pouze pokud $EV > EV_{min\_threshold}$.
  5. Přidat přepínač `use_ml_meta_filter: bool` — pokud ML model na OOS validaci nezlepší Profit Factor oproti pravidlové baseline, automaticky zůstane vypnutý.
* **Testy:** Aktualizovat `tests/test_quant_prediction_ev.py` — otestovat, že `PurgedTimeSeriesSplit` má nulový překryv časových intervalů mezi train a test setem a že predikovaná pravděpodobnost $\hat{p} \in [0, 1]$.
* **Kritéria přijetí:** Žádný data leakage v cross-validaci, všechny testy v `test_quant_prediction_ev.py` procházejí.
* **Závislosti:** AG-03, AG-05.

---

### TASK AG-07: Izolace LLM vrstvy, Zrušení Live Hlasování a Zabezpečení Autonomních Skriptů (`src/llm/`, `src/strategy/trade_reflection.py`)
* **Priorita:** P2 (Střední — Čistota architektury a odstranění latence)
* **Cíl změny:** Vyjmout lokální LLM z kritické intradenní cesty, nahradit `multi_llm_committee.py` rychlým pre-market `CatalystClassifier` s cache a přepnout `trade_reflection.py` do pasivního analytického režimu.
* **Problém v současném projektu:** Volání více LLM rolí blokuje intradenní exekuci, nelze ho backtestovat a `trade_reflection.py` / `autonomous_discovery.py` zavádějí statisticky nevalidované změny za běhu.
* **Dotčené soubory:** `src/llm/multi_llm_committee.py`, `src/llm/sentiment_engine.py`, `src/strategy/intelligence_cache.py`, `src/strategy/trade_reflection.py`, `src/strategy/autonomous_discovery.py`, `tests/test_llm.py`, `tests/test_reflection.py`.
* **Požadovaný cílový stav:**
  1. V `sentiment_engine.py` definovat striktní datový kontrakt `CatalystAssessment` (vč. `trade_veto: bool` pro toxické události jako sekundární úpis/offering nebo fixní buyout). Výsledky ukládat do `intelligence_cache.py` s TTL.
  2. V `multi_llm_committee.py` přidat parametr `fast_mode: bool = True` (výchozí pro day trading), který místo vícenásobného debatování LLM provede jediný strukturovaný dotaz s pevným timeoutem (např. 3 sekundy) a při timeoutu bezpečně vrátí neutrální výsledek (nezablokuje hlavní vlákno).
  3. V `trade_reflection.py` zakázat přímý zápis do produkčních vah strategií; výstup ukládat jako strukturovaný Markdown/JSON report do `logs/reflections/` pro schválení člověkem.
* **Testy:** Upravit a rozšířit `tests/test_llm.py` a `tests/test_reflection.py` (test chování při výpadku/timeoutu lokálního LLM serveru a test nezměnitelnosti produkčních rizikových limitů ze strany `trade_reflection.py`).
* **Kritéria přijetí:** Běh hlavního obchodního cyklu při vypnutém/nedostupném LLM nezpomalí ani nespadne; testy procházejí.
* **Závislosti:** AG-04, AG-05.