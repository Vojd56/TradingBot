# Architecture Review & Target System Design — TradingBot (`fcfb210`)

## 1. Mapa současného repozitáře a odhalená "Dvojhlavá architektura"

Analýza stromu repozitáře `TradingBot` (commit `fcfb2106a62dbf99dd2ea1430781357f1b452a28`) odhaluje zásadní architektonický rys: projekt obsahuje **dvě paralelní a částečně duplicitní obchodní větve**, které vznikly postupným přidáváním výzkumných konceptů z `research.md`:

### 1.1 Přehled reálných modulů v projektu
* **Kořen a spouštěče:** `main.py`, `config.py`, `research.md`, `AGENTS.md` a 7 dávkových skriptů (`run_daemon.cmd`, `run_daytrader.cmd`, `run_quant_trader.cmd`, `run_market_scanner.cmd`, `run_scheduled_scanner.cmd`, `run_status.cmd`, `run_trade_memory.cmd`).
* **Broker & Exekuce (`src/broker/`, `src/execution/`):**
  * `src/broker/t212_client.py` (klient pro Trading 212 API)
  * `src/execution/order_executor.py`, `portfolio_tracker.py`, `reconciler.py`, `ticker_mapper.py`
* **LLM Vrstva (`src/llm/`):**
  * `src/llm/multi_llm_committee.py` (víceagentní hlasování / komise LLM)
  * `src/llm/sentiment_engine.py`
* **Větev A — Původní / Agentní & Pravidlová větev (`src/strategy/`):**
  * Orchestrátory: `day_trader.py`, `quant_trader.py`, `stock_picker.py`
  * Skenery: `opportunity_scanner.py`, `scheduled_scanner.py`, `autonomous_discovery.py`, `continuous_researcher.py`
  * Kontext a struktura: `intraday_data.py`, `market_hours.py`, `market_structure.py`, `macro_regime.py`, `models.py`
  * Riziko a paměť: `risk_manager.py`, `intelligence_cache.py`, `trade_reflection.py`
* **Větev B — Kvantitativní / Carverova větev (`src/quant/`):**
  * `features.py`, `regime_detector.py`, `signal_engine.py`, `prediction_model.py`, `risk_engine.py` (inspirováno Robertem Carverem — `test_quant_risk_carver.py`), `execution_engine.py`, `monitoring.py`.

---

## 2. Kritické architektonické problémy v současném návrhu

### Problém 1: Architektonická schizofrenie (`src/strategy/` vs. `src/quant/` vs. `src/execution/`)
V projektu existují **tři exekuční vrstvy** (`src/execution/order_executor.py`, `src/quant/execution_engine.py` a přímá volání v `day_trader.py`/`quant_trader.py`), **dva risk manažery** (`src/strategy/risk_manager.py` a `src/quant/risk_engine.py`) a **dva detektory režimu** (`src/strategy/macro_regime.py` a `src/quant/regime_detector.py`).
* **Důsledek:** Rozštěpení zodpovědnosti (Split-Brain). Pokud `day_trader.py` obchází `src/quant/risk_engine.py` nebo pokud `quant_trader.py` nekomunikuje správně s `reconciler.py`, hrozí nekontrolovaná expozice účtu a nekonzistence stavu portfolia.

### Problém 2: Zásadní nesoulad mezi brokerem (`t212_client.py`) a cílem (Intraday Day Trading)
Projekt používá `src/broker/t212_client.py` (Trading 212 API) jako jediného brokera v `src/broker/`.
* **Tvrdá realita mikrostruktury trhu:**
  1. Trading 212 Invest/ISA **není direct-access day trading broker**. Má přísné API rate limity, nepodporuje nativní server-side OCO (One-Cancels-Other) bracket příkazy (současné zadání Stop-Lossu a Take-Profitu k otevřené pozici bez blokace kapitálu) a neumožňuje shortování skutečných akcií (mimo CFD s obrovskými spready).
  2. Exekuce tržních příkazů na T212 probíhá přes interní/OTC routování a při vysoké volatilitě (9:30–9:45 EST) trpí skluzem a zpožděním potvrzení stavu v `reconciler.py`.
* **Řešení:** `t212_client.py` je skvělý pro **sekundární cíl (`stock_picker.py` — dlouhodobý stock picking a více-denní swing)**, ale pro **primární cíl (intradenní day trading)** musí systém podporovat plnohodnotný DMA/API broker adaptér (např. **Alpaca Trading API** nebo **IBKR**) a především lokální **BacktestSimAdapter**.

### Problém 3: Chybějící dedikovaný, realistický Backtesting Engine
V celém stromu `src/` **neexistuje samostatný balíček `src/backtest/`**. Testy v `tests/test_quant_e2e.py` a `test_quant_prediction_ev.py` testují průchod pipeline a výpočet očekávané hodnoty (EV), ale projekt postrádá plnohodnotný historický **Walk-Forward Event-Driven Backtester** s lokální databází historických 5m/1m svíček (Parquet), který by simuloval bid-ask spread, skluz (slippage) a kolize SL/TP uvnitř svíčky.
* Bez tohoto modulu jsou veškeré úpravy v `autonomous_discovery.py`, `continuous_researcher.py` a `multi_llm_committee.py` "letem naslepo".

### Problém 4: Překombinovaná LLM a "Autonomous" vrstva (`multi_llm_committee.py`, `trade_reflection.py`, `autonomous_discovery.py`)
Přítomnost modulů jako `multi_llm_committee.py` (kde více lokálních LLM promptů/rolí "debatuje" o obchodu) a `trade_reflection.py` (kde LLM zpětně hodnotí obchody a upravuje chování) přináší v intradenním obchodování kritická rizika:
1. **Latence a blokování:** Volání komise lokálních LLM (přes Ollama) trvá sekundy až desítky sekund na jeden ticker. V intradenním breakoutu (ORB) je po 30 sekundách pohyb dávno pryč nebo na lokálním vrcholu.
2. **Nedeterminismus a přeučení na velikost vzorku $N=1$:** Pokud `trade_reflection.py` mění parametry nebo váhy strategie na základě posledních pár ztrátových obchodů pomocí LLM úvah, zavádí do systému **recency bias** a ničí statistickou konzistenci strategie.

### Problém 5: Robert Carver Framework (`test_quant_risk_carver.py`, `src/quant/risk_engine.py`) aplikovaný na Intraday Equities
Robert Carver (*Systematic Trading*) navrhl svůj vynikající framework (Forecast Scaling $[-20, +20]$, IDM, Volatility Targeting, Inertia/Buffering) primárně pro **denní/vícedenní obchodování futures s nízkými transakčními náklady**.
* Pokud je v `src/quant/signal_engine.py` a `risk_engine.py` použito kontinuální přeškálování pozice podle změny forecastu každých 5 minut na akciích, bot bude neustále přikupovat a odprodávat malé části pozice (churning), což na spreadu a skluzu vykrvácí účet. Pro intradenní akcie je nutné kombinovat Carverovo **Volatility-Targeted Position Sizing** s **diskrétními událostními vstupy a pevnými bariérami (Bracket SL/TP + EOD Flatten)**.

---

## 3. Rozhodovací tabulka po auditu konkrétních souborů

| Současný soubor / modul | Verdikt | Konkrétní akce a zdůvodnění |
| :--- | :--- | :--- |
| `src/quant/features.py` | **OPRAVIT & ROZŠÍŘIT** | Stane se jediným zdrojem pravdy pro výpočet features. Doplnit Session-anchored VWAP (reset 09:30 EST), RVOL (relativní objem k dané minutě dne), Intraday RS vs. SPY a striktní ochranu proti look-ahead biasu. |
| `src/quant/regime_detector.py` & `src/strategy/macro_regime.py` | **SLOUČIT DO `src/quant/regime_detector.py`** | Odstranit duplicitu. Jeden deterministický modul detekující tržní režim (TRENDING_BULL, TRENDING_BEAR, CHOPPY_RANGE, HIGH_VOL_PANIC) na základě SPY/VIX/šířky trhu. |
| `src/quant/risk_engine.py` & `src/strategy/risk_manager.py` | **SLOUČIT & PŘEPRACOVAT** | Sloučit do jediného neprůstřelného `RiskGatekeeper` modulu. Zachovat Carverovo škálování podle cílové volatility, ale přidat tvrdá intradenní pravidla: Max Daily Drawdown (-2 %), Max Risk per Trade (0.5–1 %), Stale Data Veto a EOD Flatten (15:55 EST). |
| `src/execution/*` & `src/quant/execution_engine.py` | **SJEDNOTIT (`src/execution/`)** | Odstranit `src/quant/execution_engine.py` (přesunout jeho cost/EV kontroly do před-exekučního filtru). V `src/execution/` vytvořit čisté rozhraní `BrokerAdapter` (implementace: `BacktestAdapter`, `PaperAdapter`, `T212Adapter`, `AlpacaAdapter`). |
| `src/llm/multi_llm_committee.py` | **ZJEDNODUŠIT & VYJMOUT Z HOT PATH** | Nahradit pomalou "komisi" v reálném čase jediným rychlým, strukturovaným (Pydantic JSON) voláním `CatalystClassifier` v pre-marketu (nebo asynchronně při detekci breaking news), s ukládáním do `intelligence_cache.py`. |
| `src/strategy/autonomous_discovery.py` & `continuous_researcher.py` | **IZOLOVAT DO OFFLINE RESEARCHE** | Nesmí běžet jako démon měnící produkční parametry za běhu (`run_daemon.cmd`). Přesunout do offline výzkumného skriptu, který pouze generuje kandidátní reporty. |
| `src/strategy/trade_reflection.py` | **ODSTRANIT Z ROZHODOVACÍ LOGIKY** | Ponechat výhradně jako pasivní deníkový generátor (post-market markdown report pro tradera). Zakázat mu automaticky upravovat váhy nebo blokovat strategie bez statistické validace. |
| `src/quant/prediction_model.py` | **PŘEPRACOVAT** | Upravit na **Triple-Barrier Meta-Labeling** model (odhad pravděpodobnosti $P(\text{TP dříve než SL})$ pro signály z `signal_engine.py`), validovaný přes Purged Walk-Forward CV. |
| `src/backtest/` *(chybí)* | **NOVĚ IMPLEMENTOVAT** | Vytvořit deterministický `EventDrivenBacktester` + `ParquetDataStore` pro objektivní měření strategií před nasazením do paper/live tradingu. |

---

## 4. Cílová sjednocená architektura (Unified Target Architecture)

Cílem je **zrušit dvojkolejnost** mezi `src/strategy/` a `src/quant/` a zavést jedinou, čistou pipeline, kde `day_trader` a `stock_picker` jsou pouze dvě různé konfigurace horizontu a strategií nad stejným kvantitativním jádrem.