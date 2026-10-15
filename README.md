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

### 2. Spouštěcí skripty (Windows .cmd)

Všechny hlavní provozní úlohy lze spustit jednoduše poklepáním na připravené `.cmd` soubory v kořenovém adresáři:

| Skript | Funkce | Popis |
|---|---|---|
| **`run_daytrader.cmd`** | ⚡ **Day Trader (Hlavní)** | 5min svíčky, VWAP, EMA Ribbon, dynamické ATR, až 15 pozic na pákových ETF a akciích |
| **`run_market_scanner.cmd`** | 🔍 **Okamžitý Skener & Rada AI** | Okamžitý hloubkový průzkum trhu a konsenzuální schválení Radou 3 AI modelů |
| **`run_scheduled_scanner.cmd`** | ⏰ **Plánovač Skeneru** | Automatický sken 2x denně (09:30 CET Evropa a 16:00 CET USA) |
| **`run_trade_memory.cmd`** | 🧠 **Paměť chyb & Zkušenosti** | Přehled post-mortem analýz, ponaučení a bezpečnostních pravidel (FinMem) |
| **`run_status.cmd`** | 📊 **Stav účtu a pozic** | Bleskový výpis zůstatku účtu Trading 212 a otevřených pozic |
| **`run_daemon.cmd`** | 🐢 **Swing / Value Bot** | Konzervativní 24/7 výzkumný daemon (denní svíčky, NPU fundamenty) |

---

## 🧪 Testovací sada

Jednotkové a integrační testy jsou organizovány v adresáři `tests/`:
- `python tests/test_reflection.py` – Test tržní struktury (Turtle Soup, VWAP přepálení) a paměti chyb
- `python tests/test_strategy.py` – Test risk managementu, ATR stops a kvantitativního scoringu
- `python tests/test_execution.py` – Test mapování tickerů a správy portfolia
- `python tests/test_llm.py` – Test sanitizace JSON výstupů z lokálního LLM
