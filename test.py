from src.data import MarketDataProvider


def test_market_data():
    print("\n--- Test Datové vrstvy (yfinance + Technická analýza + Zprávy) ---")

    provider = MarketDataProvider()
    test_watchlist = ["AAPL", "MSFT", "NVDA"]

    for ticker in test_watchlist:
        print(f"\n==================== Analýza: {ticker} ====================")
        data = provider.prepare_ticker_data(ticker)

        if not data:
            print(f"Nepodařilo se načíst data pro {ticker}.")
            continue

        m = data.metrics
        print(f"Společnost:    {data.company_name} ({data.sector})")
        print(f"Aktuální cena: {m.current_price} USD")
        print(f"SMA 50:        {m.sma_50} USD | SMA 200: {m.sma_200} USD")
        print(f"RSI (14):      {m.rsi_14} | ATR (14): {m.atr_14} USD")

        trend_status = "BÝČÍ (Vhodný pro nákup)" if m.trend_bullish else "NEUTRAL / MEDVĚDÍ (Nepotvrzen)"
        print(f"Trend filtr:   {trend_status}")
        print(f"RSI filtr:     {'OK (Není přehřáté)' if m.rsi_neutral_or_pullback else 'Mimo optimální pásmo'}")

        print(f"\nPoslední zprávy ({len(data.recent_news)}):")
        for i, news in enumerate(data.recent_news, 1):
            print(f"  {i}. [{news.published_at}] ({news.publisher}) {news.title}")

    print("\nFáze 3 úspěšně dokončena.\n")


if __name__ == "__main__":
    test_market_data()