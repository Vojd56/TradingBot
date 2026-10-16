# 🚀 AI Quantitative Trading Bot (Trading 212 & AMD Ryzen AI NPU)

Profesionální, deterministický systém pro algoritmické a intradenní obchodování na globálních akciových trzích (USA, Evropa, Asie). Systém kombinuje pokročilou **8modulovou kvantitativní strategii** (inspirovanou výzkumem Roberta Carvera, Marcos López de Prada a Triple-Barrier metodou) s **lokální Radou AI modelů** (Qwen 2.5-7B) běžící na **AMD Ryzen AI NPU**.

Exekuce příkazů probíhá plně automatizovaně a idempotentně přes **Trading 212 Invest API** s dynamickým přepočtem měn (CZK / EUR / USD) a robustním řízením rizik.

---

## 🏛️ Architektura systému

```text
TradingBot/
├── config.py                     # Centrální Pydantic konfigurace, bezpečnostní zámky režimů & model nákladů
├── main.py                       # Hlavní CLI vstupní bod, směrovač provozních módů a plánovač úloh
├── requirements.txt              # Závislosti projektu (Pydantic, yfinance, pyarrow, numpy, pandas, schedule...)
├── .env                          # Konfigurace API klíčů a provozních parametrů (nikdy se necommituje!)
│
├── src/
│   ├── quant/                    # 8modulová kvantitativní architektura (Carver & Triple-Barrier)
│   │   ├── features.py           # Causal Feature Engineering (z-skóre, normalizované momentum, RVOL, rank)
│   │   ├── regime_detector.py    # Klasifikace 5 tržních režimů benchmarku SPY (BULL_QUIET, CRISIS_PANIC...)
│   │   ├── signal_engine.py      # Carver ortogonální syntéza: trendové momentum + mean-reversion
│   │   ├── prediction_model.py   # Triple-Barrier metoda, odhad pravděpodobnosti p_win a brána EV_net > Hurdle
│   │   ├── risk_engine.py        # Carver Volatility Targeting, Half-Kelly limit, 10% Inertia buffer
│   │   ├── execution_engine.py   # Idempotentní exekuce (SHA-256 client_order_id), Long-only zámek, dust filtr
│   │   ├── monitoring.py         # 6 bezpečnostních Kill-Switches & statistický CUSUM detektor anomálií
│   │   └── backtester.py         # Realistický intradenní backtester bez look-ahead biasu (poplatky, skluz)
│   │
│   ├── broker/                   # Integrace s brokerem Trading 212 Invest API
│   │   ├── t212_client.py        # HTTP Basic Auth, rate-limiter, správa objednávek, auto-precision recovery
│   │   └── __init__.py
│   │
│   ├── data/                     # Datová infrastruktura a tržní feedy
│   │   ├── market_data.py        # yfinance stahování, technické indikátory (SMA, RSI, ATR, VWAP)
│   │   ├── parquet_store.py      # Sloupcové úložiště historických svíček s inkrementálním mergem
│   │   └── __init__.py
│   │
│   ├── llm/                      # Lokální LLM engine na AMD Ryzen AI NPU
│   │   ├── sentiment_engine.py   # Komunikace s lokálním OpenAI-kompatibilním endpointem (LM Studio / Lemonade)
│   │   └── __init__.py
│   │
│   ├── strategy/                 # Obchodní režimy, Rada AI a paměť reflexe chyb
│   │   ├── quant_trader.py       # Kontinuální řídicí smyčka kvantitativního enginu
│   │   ├── day_trader.py         # Intradenní 5m momentum trader (VWAP, EMA Ribbon, páková ETF)
│   │   ├── consensus_committee.py# Rada 3 AI person (Fundamentální, Skeptik s právem VETA, Katalyzátor)
│   │   ├── trade_reflection.py   # FinMem samoučící se paměť chyb (post-mortem analýza a aktivní VETO)
│   │   ├── market_hours.py       # Správa burzovních seancí NYSE/LSE/Euronext a ochrana před DST posuny
│   │   ├── opportunity_scanner.py# Globální MSCI World skener a správa tržního univerza (90+ aktiv)
│   │   ├── autonomous_discovery.py# Autonomní vytěžování čerstvých zpráv a objevování nových titulů
│   │   └── stock_picker.py       # Klasický swingový hodnotící engine
│   │
│   └── execution/                # Exekuční adaptéry a správa stavu
│       ├── ticker_mapper.py      # Mapování standardních tickerů na oficiální symboly T212 (např. TTE.PA -> FPp_EQ)
│       ├── broker_adapter.py     # Abstrakce brokera (Live Trading212 vs PaperBrokerAdapter)
│       ├── order_executor.py     # Validační a odesílací vrstva příkazů
│       └── portfolio_tracker.py  # Perzistentní tracker otevřených pozic a realizovaného P&L
│
├── data/
│   ├── bars/                     # Lokální sloupcová Parquet databáze intradenních a denních svíček
│   ├── trade_memory.json         # Perzistentní paměť chyb a post-mortem ponaučení z obchodů
│   ├── qualified_opportunities.json # Schválené nákupní příležitosti z Rady AI
│   └── intelligence_cache.json   # Cache fundamentálních a sentimentových analýz
│
└── logs/
    ├── trading_bot.log           # Systémový rotační log běhu
    └── reports/                  # Denní JSON auditní protokoly a statistiky exekuce
```

---

## 📊 Odkud bot bere data a jak je zpracovává

Bot nepracuje s fiktivními ani zpožděnými daty, ale integruje 4 navzájem se doplňující datové zdroje:

1. **Globální tržní data (yfinance & Parquet Bar Store):**
   - **Široké tržní univerzum (90+ lídrů):** Pokrývá klíčové americké tituly (Mag7, software, čipy, obranný sektor), přední evropské blue chips (Euronext Paris, LSE, Xetra, SIX Švýcarsko) a asijské giganty (ADR).
   - **Páková ETF (2x / 3x):** Pro intradenní momentum skenování (`TQQQ`, `SOXL`, `NVDL`, `TSLL`, `FNGU`).
   - **Sloupcové Parquet úložiště (`data/bars/`):** Všechny intradenní 5m a denní 1D svíčky jsou lokálně ukládány ve formátu Apache Parquet. Systém automaticky provádí inkrementální merge nově stažených svíček a validuje integritu (detekce děr v datech, neplatných objemů a prevence look-ahead biasu).

2. **Reálná data z účtu brokera (Trading 212 API):**
   - Před každým rozhodnutím bot dotazuje broker API na **skutečnou čistou ekvitu**, **volnou hotovost**, **otevřené pozice** a **nevyřízené příkazy**.
   - **Multi-měnová normalizace:** Zůstatek na účtu vedený v CZK (či EUR) je v reálném čase přes aktuální kurz brokera normalizován na USD, čímž odpadá jakékoliv zkreslení velikosti pozic.

3. **Autonomní vyhledávání nových akcií ze zpráv (`AutonomousDiscovery`):**
   - Kontinuálně monitoruje finanční RSS feedy (Yahoo Finance, Google News, breaking zprávy).
   - Detekuje zmínky o nových vládních a armádních AI zakázkách, fúzích, velkých partnerstvích nebo nových IPO.
   - Nově objevené tituly automaticky prověří a zařadí do skenovacího univerza.

---

## 🔬 Kvantitativní 8modulová strategie (`src/quant/`)

Deterministický rozhodovací aparát eliminuje emoce a náhodné tipy. Každý nákup musí projít přísným sítem:

### 1. Causal Feature Engineering (`features.py`)
Generuje bezrozměrné stacionární indikátory s nulovým look-ahead biasem:
- Normalizované momentum (3M a 12M-1M očištěné o volatilitu).
- Relativní odchylka od klouzavých průměrů ($Z\text{-skóre}$ vůči SMA 20 a SMA 50).
- Komprese volatility ($\text{ATR}_{14} / \text{ATR}_{100}$) identifikující přípravu na breakout.
- Relativní objem ($\text{RVOL}$) a sklon 200denního průměru.

### 2. Regime Detector (`regime_detector.py`)
Podle chování benchmarku SPY (cena vůči SMA 200 a 20denní realizovaná volatilita) klasifikuje trh do 5 režimů:
- `BULL_QUIET`: Stabilní býčí trh $\rightarrow$ 100% alokace na momentum.
- `BULL_VOLATILE`: Rostoucí trh s vyššími výkyvy $\rightarrow$ snížení vah, širší stop-lossy.
- `SIDEWAYS`: Trh bez trendu $\rightarrow$ váha přesunuta na Mean-Reversion.
- `BEAR_QUIET`: Počínající medvědí trh $\rightarrow$ konzervativní alokace.
- `CRISIS_PANIC`: Volatilita nad 95. percentilem $\rightarrow$ **tvrdé zastavení tradingu (0 % nákupů)**.

### 3. Carver Orthogonal Signal Engine (`signal_engine.py`)
Syntetizuje trendový komponent a návrat k průměru (mean-reversion) do kombinovaného signálu. Obsahuje zabudovaný **filtr padajícího nože** (*falling knife filter*), který zakazuje nákup titulů v silném volném pádu pod spodním Bollingerovým pásmem.

### 4. Triple-Barrier & $EV_{net}$ Gate (`prediction_model.py`)
Každý obchod je ohraničen 3 dynamickými bariérami:
- **Take Profit (horní bariéra):** Dynamicky dle ATR s garancí poměru rizika a zisku $R:R \ge 1:2$.
- **Stop Loss (dolní bariéra):** Těsná ochrana kapitálu dle lokální volatility.
- **Časový stop (vertikální bariéra $H$):** Pokud pozice nezačne generovat zisk do stanoveného počtu barů, je uzavřena.

> [!IMPORTANT]
> **Výchozí stav je `NO TRADE`:** Obchod je schválen k exekuci pouze tehdy, pokud realisticky odhadnutá pravděpodobnost úspěchu $\hat{p}_{win}$ a velikost bariér zajistí kladnou čistou očekávanou hodnotu po započtení nákladů:
> $$EV_{net} = \hat{p}_{win} \cdot \text{Gain} - (1 - \hat{p}_{win}) \cdot \text{Loss} - \text{Costs}_{total} > \text{Hurdle}$$

---

## 💰 Dynamická správa kapitálu (Carver Volatility Targeting)

Žádné zastaralé fixní limity typu 50 EUR. Systém dynamicky škáluje velikost pozic podle reálné velikosti portfolia:

1. **Carver Volatility Targeting:**
   Velikost pozice v hotovosti je nepřímo úměrná 20dennímu průměrnému rozpětí ($\text{ATR}_{20}$):
   $$\text{Pozice} \propto \frac{\text{Cílové denní riziko}}{\text{Cena} \times \text{ATR}_{20}}$$
   Volatilnější tituly (např. páková ETF nebo krypto proxy) dostávají menší počet kusů, zatímco stabilní compoundeři mají alokaci vyšší.
2. **Half-Kelly limit koncentrace:**
   Žádná jednotlivá akcie nesmí překročit **10 až 12 % celkové ekvity účtu**.
3. **Striktní risk na obchod:**
   Maximální možná ztráta při zasažení Stop Lossu nesmí překročit **1.5 % celkové ekvity portfolia**.
4. **Carverovo 10% pásmo nečinnosti (Inertia Buffer):**
   Pokud by úprava velikosti stávající pozice představovala změnu menší než 10 %, příkaz se neodešle. To brání zbytečnému poplatkovému přebalancování a drobným posunům.
5. **Deduplikace čekajících příkazů & Prachový filtr:**
   Pokud má titul již nevyřízený čekající příkaz z mimoburzovních hodin, další příkaz se neodesílá. Obchody s hodnotou menší než 2 USD jsou automaticky odfiltrovány.
6. **Automatická obnova a korekce přesnosti množství:**
   Systém bere v potaz rozdílnou přesnost burz (LSE v Londýně vyžaduje 2–3 desetinná místa, US burzy až 4–6). Pokud broker vrátí chybu přesnosti (`quantity-precision-mismatch`), bot ji automaticky zachytí, zaokrouhlí na brokerem povolený počet míst a příkaz okamžitě úspěšně zopakuje.

---

## 🧠 Zapojení lokální AI (AMD Ryzen AI NPU & Qwen 2.5-7B)

Umělá inteligence neběží v cloudu, ale přímo na vašem hardwaru přes NPU bez latence a poplatků za tokeny. Zastává dvě klíčové role:

### A) Rada AI – Konsenzuální komise (Consensus Committee)
Každého kandidáta nezávisle posoudí 3 specializované persony:
1. **Fundamentální analytik:** Zkoumá ocenění (P/E, forward P/E), marže, rozvahu, zadlužení a ROIC.
2. **Skeptický auditor / Ďáblův advokát:** Hledá skrytá rizika, hrozbu ředění akcií, regulatorní zásahy a tržní bubliny. **Disponuje absolutním právem VETA.**
3. **Katalytický specialista:** Hodnotí čerstvé zprávy, sílu makroekonomického větru a momentum zakázek.

Nákup je schválen pouze při jednohlasné (`UNANIMOUS`) nebo většinové (`MAJORITY`) shodě. Jakékoliv veto skeptika obchod zablokuje. Výsledky se ukládají do `data/qualified_opportunities.json`.

### B) Samoučící se paměť chyb (FinMem Trade Reflection)
* Soubor: `data/trade_memory.json`
* Po uzavření každého obchodu (především na Stop Lossu) lokální model provede **Post-Mortem analýzu**:
  - Kategorizuje příčinu chyby (např. `OVERBOUGHT_RSI`, `VWAP_OVEREXTENSION`, `OPENING_VOLATILITY_CHOP`, `COUNTER_TREND_FIGHT`).
  - Zformuluje konkrétní ponaučení a bezpečnostní pravidlo.
* **Aktivní ochrana před opakováním chyb:** Před každým dalším nákupem engine prověří historii. Pokud by se chystal zopakovat dřívější chybu u daného instrumentu, vystaví **okamžité VETO**.

---

## 💻 Spuštění a provozní režimy

Veškeré spouštění je připraveno ve Windows dávkových skriptech (`.cmd`) nebo přes standardní CLI rozhraní v aktivovaném virtuálním prostředí (`.\venv\Scripts\Activate.ps1`):

| Skript | CLI příkaz | Účel a chování |
|---|---|---|
| **`run_quant_trader.cmd`** | `python main.py --quant` | ⚡ **Kvantitativní 8modulová strategie (Hlavní)**<br>Kontinuální cyklus (kontrola každých 5 minut). Integruje Carver sizing, tržní režim SPY, Triple Barrier a paměť chyb. |
| **`run_daytrader.cmd`** | `python main.py --daytrade --interval 60` | 🔥 **Agresivní intradenní Day Trader**<br>5minutové svíčky, kontrola každých 60 s. Obchoduje VWAP odrazy, EMA ribbon 9/21, páková ETF, posun na breakeven stop při zisku +1.0 %. |
| **`run_market_scanner.cmd`** | `python main.py --scan` | 🔍 **Okamžitý hloubkový sken trhu**<br>Projde globální univerzum a svolá Radu AI pro schválení příležitostí. |
| **`run_scheduled_scanner.cmd`**| `python main.py --schedule-scans` | ⏰ **Plánovač skenů 2x denně**<br>Automaticky spustí Radu AI v 09:30 CET (otevření Evropy) a v 16:00 CET (otevření Wall Street). |
| **`run_trade_memory.cmd`** | `python main.py --trade-memory` | 🧠 **Audit paměti chyb & Zkušeností**<br>Zobrazí seznam všech post-mortem reflexí, úspěšnost a aktivní preventivní pravidla. |
| **`run_status.cmd`** | `python main.py --status` | 📊 **Stav účtu a otevřených pozic**<br>Bleskový výpis hotovosti na Trading 212, držených akcií a nerealizovaného P&L. |
| **`run_daemon.cmd`** | `python main.py --daemon` | 🐢 **Konzervativní Swing / Value Investor**<br>Pracuje s denními (1D) svíčkami, hloubkovým fundamentálním hodnocením a 200denním trendem. |

Pro jednorázový testovací cyklus kvantového enginu můžete použít:
```powershell
python main.py --quant --run-once
```

---

## 🔒 Bezpečnostní zámky režimů (Safety First)

* **Výchozí režim je vždy bezpečný:** Výchozí stav je `TRADING_MODE=paper` (nebo demo účet Trading 212).
* **Ochrana živého účtu (LIVE):** Přepnutí na reálné peníze vyžaduje v `.env` explicitní nastavení:
  ```env
  TRADING_MODE=live
  CONFIRM_LIVE_TRADING=true
  I_UNDERSTAND_LIVE_RISK=yes
  ```
  Bez těchto flagů se bot odmítne spustit, aby se vyloučil neúmyslný ostrý obchod.
* **6 bezpečnostních Kill-Switches:**
  1. *Daily Loss Kill-Switch:* Zastavení při denní ztrátě nad 3 %.
  2. *Max Drawdown Guard:* Zastavení při poklesu portfolia o více než 10 %.
  3. *Market Spike Guard:* Blokace při extrémním intradenním pohybu indexu o $\ge 4$ %.
  4. *Stale Data Guard:* Zamítnutí dat starších než 120 sekund.
  5. *Portfolio Desync Guard:* Kontrola souladu stavu bota s evidencí brokera.
  6. *CUSUM Detector:* Statistický detektor rozpadu výkonnosti strategie.

---

## 🧪 Testovací sada

Projekt obsahuje kompletní sadu automatizovaných testů (`pytest`):
```powershell
.\venv\Scripts\pytest.exe -v
```
Testy ověřují matematickou správnost příznaků bez look-ahead biasu, chování tržních režimů, Carverův inertia buffer, Triple-Barrier logiku, idempotenci exekuce i automatickou obnovu při chybách brokera.
