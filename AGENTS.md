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

---

## 3. Vývojové a provozní zásady
* **Komunikace s uživatelem:** Výhradně v češtině.
* **Bezpečnost kapitálu:** Demo režim na Trading 212 API jako výchozí stav.
* **Zachování funkčnosti obou módů:** Úpravy v Day Traderu nesmí rozbít swingového bota a naopak.
