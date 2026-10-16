# Antigravity Prompts — Samostatné exekuční prompty pro vývojového agenta

Níže jsou připravené, plně kontextové prompty pro **Google Antigravity**. Zadavej je agentovi **postupně jeden po druhém** (od Etapy 1 do Etapy 5). Po dokončení každé etapy zkontroluj výstup `pytest` a vytvoř git commit.

---

## ETAPA 1: Audit kódu, Bezpečnostní zámek režimů a Datová infrastruktura (AG-00 + AG-01)

```text
Jsi seniorní Python architekt a kvantitativní vývojář. Pracuješ v repozitáři TradingBot.
Tvým úkolem je provést Etapu 1 refaktoringu (Tasky AG-00 a AG-01): Zabezpečení provozních režimů, konfigurace transakčních nákladů a přepracování intradenní datové vrstvy.

NEJDŘÍVE PROVEĎ PRŮZKUM (Discovery):
1. Přečti si soubory `research.md`, `config.py`, `main.py`, `src/strategy/intraday_data.py`, `src/strategy/market_hours.py` a spusť `pytest`, abys ověřil výchozí stav testů.
2. Nikdy nepředpokládej obsah souborů naslepo — pracuj s reálným kódem v repozitáři.

POŽADOVANÉ ZMĚNY:
1. [config.py & main.py]:
   - Zaveď explicitní enum `TradingMode` (`BACKTEST`, `PAPER`, `LIVE`). Výchozí režim musí být bezpečný (`PAPER` nebo `BACKTEST`).
   - Implementuj tvrdou pojistku: Režim `LIVE` smí naběhnout POUZE tehdy, pokud je v prostředí explicitně nastaveno `CONFIRM_LIVE_TRADING=true`. Jinak vyhoď `RuntimeError`.
   - Přidej strukturovanou konfiguraci transakčních nákladů `TransactionCostConfig` (`spread_bps`, `slippage_bps`, `commission_bps`, `fx_conversion_bps` — důležité pro rozlišení USD účtu vs. Trading 212 účtu v cizí měně s 15 bps FX poplatkem).

2. [src/strategy/market_hours.py & src/strategy/intraday_data.py]:
   - V `market_hours.py` implementuj deterministické značkování obchodních seancí v časovém pásmu `America/New_York` (`PRE`, `REGULAR` 09:30-16:00 EST, `POST`), výpočet `trading_day` a `minutes_from_open` (0..390). Ošetři správně přechody mezi zimním a letním časem (DST).
   - V `intraday_data.py` implementuj lokální sloupcové úložiště `ParquetBarStore` (ukládání a načítání 1m/5m/1d OHLCV dat v `data/bars/`), aby backtesty a skenery nemusely při každém spuštění znovu stahovat historii ze sítě.
   - Implementuj validátor dat `validate_ohlcv_integrity(df)`, který odhalí a vyčistí duplicitní časová razítka, porušení `Low <= Open, Close <= High`, záporné objemy a označí chybějící svíčky uvnitř regular session.
   - Zachovej zpětnou kompatibilitu veřejných funkcí v `intraday_data.py` a `market_hours.py`, aby se nerozbily existující importy.

TESTY A VERIFIKACE:
- Vytvoř nový testovací soubor `tests/test_data_and_config_safety.py`, který otestuje:
  a) Zablokování `LIVE` režimu bez `CONFIRM_LIVE_TRADING=true`.
  b) Správné označení `REGULAR` session (09:30–16:00 America/New_York) včetně přechodu DST.
  c) Detekci nevalidních OHLCV svíček a uložení/načtení z Parquet cache.
- Spusť kompletní `pytest` a ujisti se, že procházejí všechny původní i nové testy.