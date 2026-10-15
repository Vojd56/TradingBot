# Pravidla a instrukce pro AI Agenty (AGENTS.md)

Tento dokument slouží jako trvalá paměť a závazná pravidla pro AI agenty pracující v tomto repozitáři.

---

## 1. Pravidla pro GitHub commity a verzování (DŮLEŽITÉ)

* **Vždy pojmenovávat commity logicky, přesně a srozumitelně**, aby bylo na první pohled jasné, jaká funkcionalita byla přidána nebo změněna.
* Historie musí umožňovat se kdykoliv snadno vrátit ke konkrétní verzi (např. verze před day tradingem, verze se swingovým botem, atd.).
* **Příklady správných commit zpráv:**
  - `feat: implementace day tradingu (5m svíčky, VWAP, EMA Ribbon, páková ETF)`
  - `feat: rozšíření watchlistu na 65+ titulů a autonomní vyhledávání nových akcií ze zpráv`
  - `fix: oprava posunu na breakeven stop a výpočtu dynamického ATR`
  - `docs: aktualizace dokumentace a návodů k použití`
* **Bezpečnost repozitáře:** Soubor `.env` s reálnými API klíči NESMÍ být nikdy commitnut do Gitu (stále udržovat v `.gitignore`).

---

## 2. Přehled architektury projektu

Projekt obsahuje dva hlavní obchodní módy:

### A) Režim Day Trader (Agresivní / Experimentální)
* **Spuštění:** `python main.py --daytrade --interval 60` nebo skript `run_daytrader.cmd`
* **Časový rámec:** 5minutové intradenní svíčky, kontrola každých 60 sekund.
* **Watchlist:** 65+ vysoce likvidních titulů (páková ETF 2x/3x jako `TQQQ`, `SOXL`, `NVDL`, `TSLL`, technologičtí lídři, krypto proxy) + dynamicky objevované akcie ze světových zpráv.
* **Indikátory:** VWAP (Volume-Weighted Average Price), EMA Ribbon 9/21, Volume Surge Ratio, 5m RSI.
* **Exekuce & Risk:** Cílový Take Profit +2.2 %, těsný Stop Loss -1.2 %, automatický posun na Breakeven při zisku +1.0 %, dynamický Trailing Stop, kapacita až 15 souběžných pozic (alokace ~3 300 EUR / pozice).

### B) Režim Swing / Value Investor (Konzervativní)
* **Spuštění:** `python main.py --daemon` nebo skript `run_daemon.cmd`
* **Časový rámec:** Denní svíčky (1D), 200denní historie, audit každých 15 minut při otevřeném trhu.
* **Role AI (AMD NPU):** Lokální model `Qwen2.5-7B-Instruct-NPU` provádí hloubkovou analýzu fundamentů (P/E, ROIC, marže, zprávy) a ukládá výsledky do `IntelligenceCache`.
* **Kritéria:** Striktní vstup pouze při `BUY` signálu a celkovém skóre $\ge 70/100$.

### C) Rozdělený Hloubkový Skener Trhu & Rada AI (2x denně)
* **Spuštění:** `python main.py --scan` (okamžitý sken) nebo `python main.py --schedule-scans` (plánovač na 09:30 a 16:00 CET) či skript `run_market_scanner.cmd`
* **Účel:** Odděluje strategické hledání příležitostí od samotného tradingu. 2x denně (při otevření Evropy 09:30 CET a Wall Street 16:00 CET) hloubkově projde trh a ověří příležitosti.
* **Multi-LLM Rada modelů (Consensus Committee):**
  - Každou akcii nezávisle hodnotí 3 specializované AI persony:
    1. **Fundamentální analytik** (P/E, marže, ROIC, rozvaha)
    2. **Skeptický auditor / Ďáblův advokát** (hledá skrytá rizika a bubliny; má **právo VETA**)
    3. **Katalytický specialista** (zprávy, momentum, makro vítr)
  - **Ověření pravdivosti & Konsenzus:** Nákup je schválen pouze při jednohlasné (`UNANIMOUS`) nebo většinové (`MAJORITY`) shodě. Pokud skeptik zjistí kritické riziko (`VETO_TRIGGERED`) nebo panuje neshoda (`SPLIT_DECISION`), obchod je zablokován pro ochranu kapitálu.
  - Schválené příležitosti se ukládají do `data/qualified_opportunities.json`, odkud si je Day Trader bleskově načítá pro exekuci.

### D) Samoučící se paměť chyb (Trade Reflection) & Tržní struktura (FinMem & Minervini)
* **Spuštění přehledu paměti:** `python main.py --trade-memory` nebo skript `run_trade_memory.cmd`
* **Perzistentní paměť:** `data/trade_memory.json`
* **Princip fungování (FinMem architektura):**
  - Po uzavření každého obchodu (zejména na Stop Lossu) lokální model `Qwen2.5-7B` provede automatický **Post-Mortem rozbor**:
    - Kategorizuje chybu (`OPENING_VOLATILITY_CHOP`, `OVERBOUGHT_RSI`, `VWAP_OVEREXTENSION`, `LIQUIDITY_SWEEP_FAKEOUT`, `COUNTER_TREND_FIGHT`, atd.).
    - Zformuluje konkrétní ponaučení (*Lesson Learned*) a preventivní pravidlo.
  - **Aktivní ochrana před opakováním chyb:** Před každým dalším nákupem Day Trader prověří paměť. Pokud by se chystal opakovat stejnou chybu u daného instrumentu, vystaví **VETO**.
* **Kvantitativní tržní struktura (`MarketStructureAnalyzer`):**
  - **Liquidity Sweeps (Turtle Soup):** Penalizuje nákupy na samotném vrcholu falešného breakoutu při vyčerpaném RSI a objemu.
  - **VWAP Overextension:** Zabraňuje honění ceny příliš daleko nad VWAPem.
  - **Otevírací okno:** Chrání před ranními whip-saws v prvních 15 minutách po otevření burzy.
  - **Páková ETF:** Počítá volatilitní rozpad (*volatility drag*) pro 2x/3x ETF.

### E) Pokročilá 8modulová Kvantitativní Strategie dle `research.md` (Carver Volatility Targeting & EV_net)
* **Spuštění:** `python main.py --quant` (kontinuální smyčka) nebo `python main.py --quant --run-once` (jednorázový cyklus) či skript `run_quant_trader.cmd`
* **Architektura modulů (`src/quant/`):**
  1. `features.py` (**Causal Feature Engineering**): Bezrozměrné stacionární příznaky bez look-ahead biasu (volatilitně normalizované 3M a 12M-1M momentum, Z-skóre SMA20/50, komprese ATR_14/ATR_100, RVOL, sklon SMA200, průřezový percentilní rank).
  2. `regime_detector.py` (**Regime Detector**): Klasifikace tržního režimu benchmarku SPY (BULL_QUIET, BULL_VOLATILE, SIDEWAYS, BEAR_QUIET, CRISIS_PANIC) s tvrdým zastavením tradingu při volatilitě nad 95. percentilem.
  3. `signal_engine.py` (**Carver Orthogonal Signal Engine**): Nezávislá syntéza trendového momenta a mean-reversion (s ochranou proti padajícímu noži / falling knife) dynamicky vážená tržním režimem.
  4. `prediction_model.py` (**Triple-Barrier & $EV_{net}$ Gate**): Dynamické bariéry (TP, SL s garantovaným $R:R \ge 1:2$, časový stop $H$), realisticky kalibrovaná pravděpodobnost $\hat{p}_{win}$, výpočet čisté očekávané hodnoty $EV_{net} > \eta \cdot C_{total}$ a striktní výchozí stav **`NO TRADE`**.
  5. `risk_engine.py` (**Carver Volatility Targeting**): Velikost pozic nepřímo úměrná $ATR_{20}$, Carverovo 10% pásmo nečinnosti (Inertia Buffer) proti poplatkovému přebalancování, Half-Kelly limit koncentrace (max 12 % na titul), riziko SL $\le 1.5$ % ekvity a korelační filtr (> 0.65 / > 0.80).
  6. `execution_engine.py` (**Idempotent Execution**): Deterministické `client_order_id` (SHA-256 idempotence proti duplicitám při timeoutu), pre-trade verifikace stavu hotovosti a Long-only uzamčení pro Trading 212 Invest.
  7. `monitoring.py` (**6 Kill-Switches & CUSUM**): 6 bezpečnostních jističů (Denní ztráta > 3 %, Max DD > 10 %, Extrémní pohyb trhu > 4 %, Integrita dat, Desynchronizace portfolia, Skluz, API Loop Breaker) + statistický CUSUM detektor strukturálního rozpadu strategie.

---

## 3. Vývojové a provozní zásady
* **Komunikace s uživatelem:** Výhradně v češtině.
* **Bezpečnost kapitálu:** Demo režim na Trading 212 API jako výchozí stav.
* **Zachování funkčnosti obou módů:** Úpravy v Day Traderu nesmí rozbít swingového bota a naopak.

---

## 4. Striktní zásady auditu, správy portfolia a zákaz halucinací (ZERO-TOLERANCE POLICY)

### A) Zákaz halucinací a falšování dat
* **Žádné vymyšlené metriky:** Nikdy neodhadovat, neinterpolovat ani negenerovat fiktivní P&L, nákupní/prodejní ceny ani zůstatky.
* **Validace přes API:** Každý záznam v `trade memory` musí pocházet výhradně z oficiálního JSON výstupu brokera/burzy (`Order ID`, `Executed Price`, `Filled Qty`, `Fee`, `Timestamp`). Pokud chybí potvrzená exekuce z API, obchod NEEXISTUJE a nesmí být zapsán jako realizovaný.
* **Důsledné oddělené účtování:**
  - `Realized PnL` = (Prodejní cena − Nákupní vážená cena) × Množství − Poplatky (resp. oficiální `realisedProfitLoss` z broker API).
  - Nikdy nezaměňovat celkový objem prodeje (Gross Value) za zisk (PnL).
  - Vždy uvádět striktně odděleně: **Realized PnL**, **Unrealized PnL** a **Total Account Equity**.

### B) Audit a verifikace historie (Trade Memory Reconciliation)
* Před jakýmkoliv novým rozhodnutím porovnat stav lokální paměti se skutečným stavem na účtu přes API (`get_positions`, `get_account_cash`, `get_account_summary`, `get_all_order_history`).
* Zjistí-li se nesoulad (např. v paměti je evidován fiktivní zisk, ale celkové portfolio kleslo):
  1. Okamžitě označit chybný záznam jako `[INVALID/UNVERIFIED]`.
  2. Provést rekalkulaci na základě skutečně spárovaných nákupních a prodejních příkazů z API brokera.
  3. Upozornit uživatele na nalezenou odchylku s přesným rozpadem čísel.

### C) Analýza nákupů a tržní strategie
* **Fakta namísto spekulací:** Pracovat pouze s tvrdými tržními daty (volume, VWAP, support/resistance, funding rates) a doložitelnými fundamenty (oficiální zprávy, regulatorní hlášení).
* **Risk Management:**
  - Žádný nákup nesmí být proveden bez předem definovaného Stop-Lossu a kalkulace maximálního rizika na obchod s **R:R ratio minimálně 1:2**.
  - Pokud je celkové portfolio ve ztrátě, prioritou je ochrana kapitálu (*capital preservation*), nikoliv agresivní ředění ztrátových pozic (*averaging down*).

### D) Formát zápisu do paměti (Trade Log Schema)
Každý uzavřený trade se zapisuje striktně v tomto formátu:
```json
{
  "order_id": "STRING_FROM_BROKER",
  "asset": "TICKER",
  "direction": "BUY|SELL",
  "execution_price": 0.00,
  "quantity": 0.00,
  "gross_amount": 0.00,
  "fees": 0.00,
  "realized_pnl": 0.00,
  "realized_pnl_percent": 0.00,
  "api_verified": true,
  "notes": "Stručný faktický důvod vstupu/výstupu"
}
```
