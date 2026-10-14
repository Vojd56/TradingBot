# 🚀 AI Quantitative Trading Bot (Trading 212 & AMD Ryzen AI NPU)

Kompletní produkční Python projekt kvantitativního autonomního trading bota navržený pro střednědobý a dlouhodobý **stock picking** amerických akcií. Systém kombinuje kvantitativní technickou analýzu (SMA 50/200, RSI, ATR) s lokální AI sentimentovou analýzou zpráv běžící přímo na **AMD Ryzen AI NPU** přes **Lemonade Server**. Exekuce příkazů probíhá automatizovaně přes **Trading 212 Invest API**.

---

## 🏛️ Architektura projektu

```text
TradingBot/
│
├── config.py                     # Centrální Pydantic konfigurace, čtení .env & safety checks
├── main.py                       # Vstupní bod, CLI rozhraní, plánovač úloh (Scheduler)
├── requirements.txt              # Závislosti projektu (Pydantic, yfinance, OpenAI, schedule...)
├── .env                          # Konfigurace prostředí a API klíčů
│
├── src/
│   ├── broker/                   # Komunikace s Trading 212 Invest API
│   │   ├── __init__.py
│   │   └── t212_client.py        # HTTP Basic Auth, rate-limity, správa portfolia a objednávek
│   │
│   ├── data/                     # Tržní data a technická analýza
│   │   ├── __init__.py
│   │   └── market_data.py        # yfinance pipeline: SMA 50/200, RSI 14, ATR 14, zprávy
│   │
│   ├── llm/                      # Lokální LLM sentiment engine na AMD NPU
│   │   ├── __init__.py
│   │   └── sentiment_engine.py   # OpenAI API specifikace, sanitizace JSONu, nouzový regex parser
│   │
│   ├── strategy/                 # Kvantitativní scoring & řízení rizik
│   │   ├── __init__.py
│   │   ├── models.py             # Datové modely signálů (TradingSignal, PositionExitSignal)
│   │   ├── risk_manager.py       # Alokace kapitálu (50 EUR, max 3 pozice), dynamický ATR Stop Loss/Take Profit
│   │   └── stock_picker.py       # Kompozitní scoring model (0-100 b), skenování watchlistu
│   │
│   └── execution/                # Exekuce a správa portfolia
│       ├── __init__.py
│       ├── ticker_mapper.py      # Převod mezi standardními tickery a T212 symboly (_US_EQ)
│       ├── portfolio_tracker.py  # Perzistentní evidence pozic a historie (data/portfolio_state.json)
│       ├── order_executor.py     # Bezpečné odesílání market příkazů, přepočet měn EUR/USD
│       └── reconciler.py         # Koordinátor: párování pozic, SL/TP výstupy, nákup kandidátů
│
├── data/
│   └── portfolio_state.json      # Perzistentní JSON databáze otevřených pozic a P&L
│
└── logs/
    ├── trading_bot.log           # Rotující systémové logy
    └── reports/                  # Denní auditní reporty ve formátu JSON
```

---

## ⚙️ Investiční profil & Strategie

| Parametr | Nastavení | Poznámka |
|---|---|---|
| **Investiční horizont** | Střednědobý / Dlouhodobý | Týdny až měsíce (swing / position trading) |
| **Rizikový profil** | Střední | Diverzifikace do prověřených velkých firem |
| **Testovací kapitál** | 50.0 EUR | Striktně alokováno max. 16.67 EUR na pozici |
| **Max otevřených pozic**| 3 | Zabránění přeexponování účtu |
| **Frekvence běhu** | Denně v 22:15 CET | Po uzavření řádných obchodních hodin v USA |
| **Stop Loss** | Cena $- 2.0 \times \text{ATR}_{14}$ | Dynamicky přizpůsoben volatilitě akcie |
| **Take Profit** | Cena $+ 3.5 \times \text{ATR}_{14}$ | Poměr rizika a zisku (RRR) $1 : 1.75$ |
| **Výstup na trend** | Proražení pod SMA 50 (-1.5 %) | Ochrana zisků při vyčerpání trendu |

---

## 🧠 Lokální LLM & Lemonade Server (AMD Ryzen AI NPU)

Projekt využívá lokální instanci **Lemonade Serveru** běžící na NPU. Komunikace probíhá přes standardní OpenAI specifikaci na `http://localhost:13305/api/v1`.

### 1. Stažení dedikovaného Instruct modelu
Pokud chcete stáhnout doporučený 7B model pro textový a finanční sentiment:
```powershell
lemonade pull Qwen2.5-7B-Instruct-NPU
```
V `.env` nastavte:
```env
LEMONADE_MODEL=Qwen2.5-7B-Instruct-NPU
```
*(Systém podporuje i model `Qwen2.5-Coder-7B-Instruct-NPU`, který je již otestován a plně funkční).*

---

## 🔒 Bezpečnostní pojistky (Safety Checks)

Aplikace striktně chrání vaše finance:
- Ve výchozím stavu je `TRADING_MODE=demo`.
- Pokud je v `.env` nastaveno `TRADING_MODE=live`, aplikace při startu **okamžitě zhavaruje s chybou**, pokud v `.env` není explicitně uvedeno:
  ```env
  I_UNDERSTAND_LIVE_RISK=yes
  ```

---

## 💻 Spuštění a používání

Aktivujte virtuální prostředí v PowerShellu:
```powershell
.\venv\Scripts\Activate.ps1
```

### 1. Zobrazení aktuálního stavu portfolia a účtu:
```powershell
python main.py --status
# Nebo poklepáním na run_status.cmd
```

### 2. Způsob 1: Dávkový / Plánovaný režim (Batch / Scheduled Mode)
Ideální pro spouštění v konkrétní časy (např. přes Windows Task Scheduler v 16:30 a 21:00 CET):
```powershell
# Okamžitý jednorázový cyklus:
python main.py --run-once --force
# Nebo poklepáním na run_trading_bot.cmd

# Simulovaný běh bez skutečných příkazů (Dry Run):
python main.py --run-once --dry-run --force

# Běh s lokálním plánovačem pro konkrétní časy:
python main.py --time "16:30,21:00"
```

### 3. Způsob 2: Nepřetržitý 24/7 Výzkumný a Obchodní Daemon (Continuous Intelligence & Live Trader)
**Doporučený pokročilý režim:**
1. **24/7 hloubkový výzkum na pozadí:** Vlákno výzkumu nepřetržitě prochází globální trhy, stahuje články, zprávy a fundamenty a nechává je analyzovat lokálním LLM na AMD Ryzen AI NPU. Žádný spěch ani časový stres.
2. **Ukládání do Intelligence Cache:** Všechny syntézy a skóre se ukládají do `data/intelligence_cache.json`.
3. **Blesková live exekuce:** V obchodních hodinách (Po–Pá 15:30–22:00 CET) bot okamžitě využívá předem zpracovaný výzkum z cache a exekuuje nákupy, prodeje a alfa rotace v řádu sekund za živé tržní ceny.
```powershell
# Spuštění 24/7 Daemona:
python main.py --daemon
# Nebo poklepáním na run_daemon.cmd

# Kdykoliv zobrazit stav výzkumné cache (analyzované akcie, sentiment, katalyzátory):
python main.py --research-status
# Nebo poklepáním na run_research_status.cmd
```

---

## 🧪 Testovací sada

Projekt obsahuje samostatné verifikační testy pro jednotlivé vrstvy:
- `python test.py` – Test datové vrstvy yfinance, RSI, ATR a zpráv
- `python test_llm.py` – Test sanitizace JSONu a propojení s NPU modelem
- `python test_strategy.py` – Test risk managementu, ATR stops a kvantitativního scoringu
- `python test_execution.py` – Test mapování tickerů a synchronizace s Trading 212 Demo API
