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
* **Exekuce & Risk:** Cílový Take Profit +2.2 %, těsný Stop Loss -1.2 %, automatický posun na Breakeven při zisku +1.0 %, dynamický Trailing Stop.

### B) Režim Swing / Value Investor (Konzervativní)
* **Spuštění:** `python main.py --daemon` nebo skript `run_daemon.cmd`
* **Časový rámec:** Denní svíčky (1D), 200denní historie, audit každých 15 minut při otevřeném trhu.
* **Role AI (AMD NPU):** Lokální model `Qwen2.5-7B-Instruct-NPU` provádí hloubkovou analýzu fundamentů (P/E, ROIC, marže, zprávy) a ukládá výsledky do `IntelligenceCache`.
* **Kritéria:** Striktní vstup pouze při `BUY` signálu a celkovém skóre $\ge 70/100$.

---

## 3. Vývojové a provozní zásady
* **Komunikace s uživatelem:** Výhradně v češtině.
* **Bezpečnost kapitálu:** Demo režim na Trading 212 API jako výchozí stav.
* **Zachování funkčnosti obou módů:** Úpravy v Day Traderu nesmí rozbít swingového bota a naopak.
