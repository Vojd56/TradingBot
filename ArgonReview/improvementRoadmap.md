# Improvement Roadmap — Od prototypu k ziskovému systému

Tento dokument odpovídá na 8 klíčových architektonických otázek pro každou navrhovanou změnu a definuje exaktní pravidla pro výběr akcií, obchodní strategie, ML modely, validační framework a risk management.

---

## FÁZE 2 — Detailní zdůvodnění změn (8 otázek pro každou oblast)

### Oblast 2.1: Datová infrastruktura a Time-Series Storage (`src/strategy/intraday_data.py` → `src/data/`)
* **Typ změny:** Oprava technické chyby + Zvýšení spolehlivosti.
1. **Jaký problém řeší?** Eliminuje krátkou paměť intradenních dat, nekonzistenci časových pásem (CET vs. EST/EDT), kontaminaci indikátorů nočními daty a nemožnost provádět rychlé opakovatelné backtesty.
2. **Proč je současné řešení nedostatečné?** Modul `src/strategy/intraday_data.py` kombinovaný s `market_hours.py` nemá trvalé sloupcové úložiště (Parquet) s point-in-time validací a spoléhá na živé dotazování API, které pro intradenní intervaly vrací jen omezenou historii.
3. **Nejlepší navrhovaná alternativa:** Vytvořit balíček `src/data/` s `ParquetDataStore` (komprese ZSTD, partition by `symbol/timeframe`) a duálním providerem: **Alpaca Market Data API (Free)** pro hlubokou 1m/5m US historii + **yfinance** jako sekundární zdroj pro denní data a benchmarky.
4. **Další alternativy:** Placené Polygon.io ($29–$199/m) nebo ukládání OHLCV do SQLite (výrazně pomalejší při načítání matic do Pandas/Polars).
5. **Výhody, nevýhody a náklady:** Náklady $0, čtení 100 000 svíček pod 50 ms, 100% reprodukovatelnost. Nevýhoda: nutnost bezplatného API klíče u Alpacy pro stahování více než 60 dní 5m historie.
6. **Očekávaný přínos a měření:** Zkrácení doby přípravy dat pro backtest o >95 %, 0 NaN hodnot nebo neoznačených děr v regular session (09:30–16:00 EST).
7. **Bezpečná implementace a testování:** Zachovat zpětně kompatibilní fasádu v `intraday_data.py`, napsat unit testy na DST přechody (březen/listopad) a integritu svíček (`Low <= Open, Close <= High`).
8. **Co když se hypotéza nepotvrdí?** Pokud uživatel nechce registrovat bezplatný Alpaca klíč, systém automaticky spadne do režimu `YFinanceProvider` a začne lokálně akumulovat 5m svíčky do Parquetu každým dnem běhu (do té doby omezí intradenní backtest na posledních 60 dní).

---

### Oblast 2.2: Třístupňový výběr akcií (`src/strategy/opportunity_scanner.py` & `scheduled_scanner.py`)
* **Typ změny:** Skutečné zlepšení obchodní výkonnosti (Edge).
1. **Jaký problém řeší?** Odděluje výběr likvidních akcií v pohybu ("Stocks in Play") od samotného časování vstupu.
2. **Proč je současné řešení nedostatečné?** Současné skenery (`opportunity_scanner.py`, `autonomous_discovery.py`) míchají statické seznamy, indikátory a LLM sentiment dohromady bez přísného filtru relativního objemu v danou minutu dne (`RVOL_at_time`) a očekávaných nákladů (spread vs. ATR).
3. **Nejlepší navrhovaná alternativa:** Rozdělit skenování na **3 striktně sekvenční kroky**:

#### Krok 1: Výběr vhodných instrumentů (Hard Liquidity & Volatility Filter — 09:45 EST)
Akcie postupuje do denního watchlistu pouze pokud splňuje **všechny** podmínky (bez jakýchkoliv subjektivních vah):
* $Close_{prev} \in [\$10, \$600]$ (vyřazení penny stocks s obrovským procentuálním spreadem).
* $SMA_{20}(\text{Daily Dollar Volume}) \ge \$30\,000\,000$ (záruka likvidity).
* $ATR\%_{14d} = \frac{ATR_{14d}}{Close} \ge 2.0\%$ (akcie má dostatečný denní rozsah, aby pokryla náklady).
* **Podmínka "In-Play":** $RVOL_{09:30-09:45} \ge 1.5$ **NEBO** $\vert{}Gap\%_{open}\vert{} \ge 2.0\%$.

#### Krok 2: Určení obchodního směru a pravděpodobné výhody (Directional Bias Scoring)
Pro akcie, které prošly Krokem 1, se vypočítá testovatelný **Directional Edge Score** $S_{dir} \in [-1, +1]$ složený ze 3 ortogonálních komponent:
1. **VWAP Position ($\pm 1$):** $+1$ pokud $Close > Session\_VWAP$, jinak $-1$.
2. **Relative Strength vs. SPY:** $RS_{z} = \text{clip}\left(\frac{Ret_{stock, 30m} - \beta \cdot Ret_{SPY, 30m}}{\sigma_{diff}}, -1, 1\right)$.
3. **Catalyst Veto:** Pokud `intelligence_cache.py` hlásí `risk_veto == True` (např. sekundární úpis akcií / buyout), akcie je okamžitě vyřazena.
* **Pravidlo směru:** Pokud $S_{dir} > +0.4 \implies \text{LONG ONLY}$. Pokud $S_{dir} < -0.4 \implies \text{SHORT ONLY}$ (pouze na brokerech podporujících short, u T212 vyřadit). Mezi $-0.4$ a $+0.4 \implies \text{NO TRADE}$ (nebo pouze Mean Reversion).

#### Krok 3: Načasování vstupu a výstupu (Execution Timing)
* Předání filtrovaného seznamu do `src/quant/signal_engine.py`, který čeká na konkrétní mikrostrukturní spouštěč (ORB průraz nebo návrat k VWAPu).

---

### Oblast 2.3: Obchodní strategie (`src/quant/signal_engine.py` & `src/strategy/day_trader.py`)
* **Typ změny:** Skutečné zlepšení obchodní výkonnosti.
* Namísto nejasné kombinace všeho implementujeme v `src/quant/signal_engine.py` **3 čisté intradenní strategie** a **1 denní swing/stock-picking strategii**, které sdílejí jednotné rozhraní:

| Parametr | **Strategie 1: `ORB_15m_Momentum`** | **Strategie 2: `VWAP_Trend_Pullback`** | **Strategie 3: `VWAP_Band_Reversion`** |
| :--- | :--- | :--- | :--- |
| **Tržní režim (`regime_detector.py`)** | `TRENDING_BULL` nebo `NORMAL` (zakázáno v `HIGH_VOL_PANIC`) | `TRENDING_BULL` | `CHOPPY_RANGE` (nízký ADX na SPY) |
| **Obchodní okno (EST)** | Vstupy pouze `09:45 – 11:00 EST` | Vstupy `10:00 – 14:30 EST` | Vstupy `10:30 – 14:30 EST` |
| **Podmínka vstupu (LONG)** | 1) $RVOL_{15m} \ge 1.8$<br>2) Šířka ORB $(H_{15m}-L_{15m}) \le 0.75 \cdot ATR_{14d}$<br>3) 5m Close $> OR\_High$ a $> VWAP$<br>4) $RS_{SPY} > 0$ | 1) Akcie $> 80\%$ času od 09:30 nad VWAPem<br>2) $RS_{SPY} > 0.5\sigma$<br>3) Low svíčky otestuje pásmo $VWAP \pm 0.2 \cdot ATR_{5m}$<br>4) Býčí svíčka uzavře nad High předchozí 5m svíčky | 1) Akcie **nemá** silný zprávařský katalyzátor ($RVOL < 2.0$)<br>2) Cena klesne pod $VWAP - 2.5 \cdot \sigma_{VWAP}$<br>3) $RSI_{5m}(14) < 25$ a první svíčka uzavře zpět uvnitř pásma |
| **Stop-Loss (SL)** | $\max(OR\_Low, \; Entry - 1.2 \cdot ATR_{5m})$ | $\text{SwingLow}_{pullback} - 0.5 \cdot ATR_{5m}$ (max $1.2 \cdot ATR_{5m}$) | $Entry - 1.0 \cdot ATR_{5m}$ |
| **Take-Profit (TP)** | 50 % na $+1.5R$ (SL $\to$ BE), 50 % na $+2.5R$ | 100 % na testu `High of Day` (min. $+1.8R$) | 100 % návrat na `Session VWAP` |
| **Max doba držení (Time-Stop)** | 8 svíček (40 min) pokud zisk $< +0.5R$; pevný EOD výstup `15:55 EST` | 12 svíček (60 min) pokud nevznikne nové High; EOD `15:55 EST` | 6 svíček (30 min); EOD `15:55 EST` |
| **Kdy se NEOBCHODUJE** | Gap $> 8\%$ (riziko vyčerpání), 15 min před/po FOMC/CPI, spread $> 0.15\%$ | Polední útlum pokud $Volume_{5m} < 0.5 \cdot SMA_{20}(Vol)$ | Dny po Earnings nebo když SPY trenduje o $> 1.0\%$ |

---

### Oblast 2.4: Predikční modely (`src/quant/prediction_model.py`)
* **Typ změny:** Oprava metodologie ML + Zlepšení obchodní výkonnosti.
1. **Jaký problém řeší?** Zajišťuje, že výpočet $EV$ v `prediction_model.py` (testovaný v `test_quant_prediction_ev.py`) stojí na reálných, nezkreslených pravděpodobnostech.
2. **Proč je současné řešení nedostatečné?** Predikce bez Triple-Barrier labelingu a bez Purged Embargo Cross-Validace trpí překryvem labelů (auto-korelace sousedních svíček způsobuje masivní data leakage mezi train a test setem).
3. **Nejlepší navrhovaná alternativa:**
   * **Baseline (Model 0):** Čistě pravidlové strategie z Oblasti 2.3 ($P(\text{win})$ odhadnuto historickou frekvencí v daném tržním režimu pomocí Bayesovského Beta-Binomického updatu s konzervativním priorem $Beta(10, 10)$).
   * **Model 1 (Regularized Logistic Regression s L2 penalizací):** Nízkorozměrný model nad 8 stacionárními features (`rvol_at_time`, `vwap_dist_atr`, `orb_width_atr`, `spy_rs_zscore`, `regime_id`, `time_of_day_sin/cos`, `atr_pct`, `spread_to_atr_ratio`).
   * **Model 2 (LightGBM Meta-Labeling Classifier):** Mělké stromy (`max_depth=3`, `num_leaves=7`, `min_child_samples=50`) trénované na predikci toho, zda primární signál ze `signal_engine.py` zasáhne TP bariéru před SL bariérou.
4. **Pravidlo volby modelu:** Pokud LightGBM nepřekoná na Out-of-Sample datech jednoduchou Logistickou regresi nebo čistě pravidlovou Baseline v metrice **Net Profit Factor** a **Calmar Ratio**, systém automaticky použije jednodušší variantu.

---

### Oblast 2.5: Přepracování lokálního LLM (`src/llm/multi_llm_committee.py` & `sentiment_engine.py`)
* **Typ změny:** Zjednodušení + Odstranění z kritické cesty (Reliability & Speed).
1. **Jaký problém řeší?** Odstraňuje latenci, nedeterminismus a nemožnost backtestovat víceagentní LLM komisi při intradenním obchodování.
2. **Proč je současné řešení nedostatečné?** `multi_llm_committee.py` vytváří iluzi hluboké analýzy, ale při čtení tržních indikátorů generuje nekalibrovaný šum a zpomaluje běh o desítky sekund.
3. **Nejlepší navrhovaná alternativa:**
   * **Deaktivovat `multi_llm_committee.py` pro technické intradenní signály.**
   * Přepracovat `sentiment_engine.py` na **`CatalystClassifierService`**:
     * Volá se **pouze asynchronně před otevřením trhu** (v `run_scheduled_scanner.cmd`) nebo při příchodu nové zprávy pro kandidáta z `opportunity_scanner.py`.
     * Vstup: Titulky a perexy zpráv za posledních 24h pro daný ticker.
     * Výstup (vynucený JSON schema přes Ollama):
       ```json
       {
         "has_hard_catalyst": true,
         "catalyst_category": "EARNINGS_BEAT | GUIDANCE_RAISE | DILUTION_OFFERING | MNA_BUYOUT | FDA | MACRO_NOISE",
         "trade_veto": false,
         "veto_reason": null,
         "directional_conviction": 0.75
       }
       ```
     * Pokud `catalyst_category` je `DILUTION_OFFERING` (sekundární úpis akcií) nebo `MNA_BUYOUT` (odkup za pevnou cenu), nastaví `trade_veto = true` a ušetří bota jisté ztráty.
     * Pokud je Ollama nedostupná nebo trvá déle než `3.0 s` (timeout), systém pokračuje čistě na základě kvantitativních dat (`fallback: trade_veto=False`).

---

## FÁZE 3 — Experimentální a validační framework (`src/backtest/`)

Před nasazením jakékoliv změny do `run_daytrader.cmd` nebo `run_quant_trader.cmd` musí každá strategie projít novým modulem `src/backtest/engine.py`:

### 3.1 Pravidla realistické simulace (Anti-Delusion Rules)
1. **Exekuce na další svíčce (`Next-Bar Open Fill`):** Signál vygenerovaný na uzavření svíčky $t$ (např. 09:45:00) smí být vyplněn nejdříve na `Open` svíčky $t+1$ (09:45:01) zhoršeném o polovinu spreadu a skluz.
2. **Pesimistické řešení kolize uvnitř svíčky (`Intra-Bar Ambiguity Resolution`):** Pokud v jedné 5m svíčce `High >= TakeProfit` A ZÁROVEŇ `Low <= StopLoss`, backtester **vždy předpokládá, že jako první byl zasažen Stop-Loss** (pokud nejsou k dispozici 1m pod-svíčky pro rozlišení pořadí).
3. **Model transakčních nákladů (`CostModel`):**
   $$\text{Total Cost per Side (bps)} = \frac{1}{2}\text{Spread}_{bps} + \text{Slippage}_{bps}\left(\text{ATR}, \frac{\text{OrderSize}}{\text{BarVolume}}\right) + \text{BrokerFee}_{bps} + \text{FXFee}_{bps}$$
   * Výchozí stres-testovací nastavení pro US akcie: `half_spread = 3 bps`, `slippage = 5 bps` (u ORB v 9:30–9:45 `10 bps`), `fx_fee = 0 bps` (pro USD účet) nebo `15 bps` (pokud se testuje ne-USD T212 účet — aby bylo okamžitě vidět, jak FX poplatek ničí intradenní strategii).
4. **Walk-Forward Validace (Rolling WFO):**
   * Rozdělení historie na klouzavá okna: **6 měsíců In-Sample (trénink/kalibrace) $\to$ 2 dny Embargo $\to$ 2 měsíce Out-of-Sample (test)**.
   * Výsledná equity křivka se skládá **výhradně ze slepených Out-of-Sample oken**.

### 3.2 Kritéria úspěchu (Go / No-Go Gate pro přechod do Paper Tradingu)
Strategie je schválena pro Paper Trading pouze pokud na spojených **Out-of-Sample** oknech splní:
* **Počet obchodů:** $N \ge 100$ (statistická průkaznost).
* **Net Profit Factor (po všech nákladech):** $PF > 1.25$.
* **Out-of-Sample Sharpe Ratio (annualized):** $SR_{OOS} > 1.2$.
* **Maximální Drawdown:** $MaxDD < 8.0\%$.
* **Stabilita parametrů:** Změna klíčového parametru (např. ATR násobku pro SL nebo RVOL prahu) o $\pm 15\%$ nesmí propadnout čistý zisk do záporu (ochrana proti izolovanému špičkovému přeučení).

---

## FÁZE 4 — Neprůstřelný Risk Management a Provozní bezpečnost

Sloučený modul `src/quant/risk_engine.py` funguje jako **Hard Gatekeeper** (poslední instance před `src/execution/order_executor.py`):

1. **Tři striktně oddělené provozní režimy (`TRADING_MODE` v `config.py`):**
   * `BACKTEST` (výchozí režim — 0 síťových volání na brokera).
   * `PAPER` (běží na živých datech, posílá pokyny do simulátoru nebo Demo API).
   * `LIVE` (vyžaduje explicitní příznak `--mode=live` A ZÁROVEŇ proměnnou prostředí `CONFIRM_LIVE_TRADING=TRUE`).
2. **Tvrdé limity (Hard Limits):**
   * `MAX_RISK_PER_TRADE_PCT = 0.005` (0.5 % kapitálu účtu na jeden obchod podle vzdálenosti vstupu od Stop-Lossu).
   * `MAX_POSITION_NOTIONAL_PCT = 0.20` (jedna pozice nikdy nepřekročí 20 % účtu, ani když je Stop-Loss velmi blízko).
   * `MAX_DAILY_LOSS_PCT = 0.02` (při poklesu denní equity o 2.0 % — realizovaném + nerealizovaném — se aktivuje **Daily Kill-Switch**: uzavřou se všechny intradenní pozice a zablokují nové vstupy do dalšího obchodního dne).
   * `MAX_CONCURRENT_POSITIONS = 4` (a max 2 ze stejného sektoru).
3. **Provozní pojistky (Operational Safeguards):**
   * **Stale Data Guard:** Pokud je `now_utc - bar.timestamp_utc > 120 sekund`, pokyn je zamítnut.
   * **Idempotence & Duplicate Order Guard:** Každý signál generuje deterministický klíč `hash(symbol, strategy_id, bar_timestamp, side)`. `order_executor.py` odmítne odeslat pokyn se stejným hashem dvakrát.
   * **Broker Reconciliation Guard (`src/execution/reconciler.py`):** Před otevřením nové pozice porovná lokální `portfolio_tracker.py` se skutečným stavem u brokera. Při nesouladu (tzv. *Phantom/Orphan Position*) pozastaví nové vstupy a vypíše kritický alert.