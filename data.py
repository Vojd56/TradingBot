import base64
import json
import msvcrt
import os
import sys
import requests
from requests.auth import HTTPBasicAuth


class Trading212Client:
    def __init__(self, api_key: str, api_secret: str, mode: str = "live"):
        self.mode = mode.lower()
        self.auth = HTTPBasicAuth(api_key.strip(), api_secret.strip())

        if self.mode == "live":
            self.base_url = "https://live.trading212.com/api/v0"
        else:
            self.base_url = "https://demo.trading212.com/api/v0"

    def get_account_summary(self) -> dict:
        """Vrátí celkový přehled účtu (zůstatek, investovaná hodnota, volné finance)."""
        url = f"{self.base_url}/equity/account/summary"
        response = requests.get(url, auth=self.auth, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_positions(self) -> list:
        """Vrátí seznam všech otevřených pozic včetně aktuálních tržních cen."""
        url = f"{self.base_url}/equity/portfolio"
        response = requests.get(url, auth=self.auth, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_order_history(self, limit: int = 20, cursor: str | int = None) -> dict:
        """Vrátí historii objednávek s kurzorem pro stránkování."""
        url = f"{self.base_url}/equity/history/orders"
        params = {"limit": min(limit, 50)}
        if cursor:
            params["cursor"] = cursor

        response = requests.get(url, auth=self.auth, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_transaction_history(self, limit: int = 20, cursor: str | int = None) -> dict:
        """Vrátí historii finančních transakcí (vklady, výběry, dividendy)."""
        url = f"{self.base_url}/equity/history/transactions"
        params = {"limit": min(limit, 50)}
        if cursor:
            params["cursor"] = cursor

        response = requests.get(url, auth=self.auth, params=params, timeout=10)
        response.raise_for_status()
        return response.json()


def pause_and_exit(exit_code: int = 0):
    """Pozastaví provádění skriptu do stisku libovolné klávesy."""
    print("\nStiskněte libovolnou klávesu pro ukončení...")
    msvcrt.getwch()
    sys.exit(exit_code)


def main():
    print("=========================================")
    print("        TRADING 212 DATA CLIENT          ")
    print("=========================================")
    print("Vyberte režim účtu:")
    print("  [1] LIVE (Reálné finance)")
    print("  [2] DEMO (Cvičný účet)")
    print("  [Jiné] Konec")
    print("Volba: ", end="", flush=True)

    key_choice = msvcrt.getwch()
    print(key_choice)

    if key_choice == "1":
        mode = "live"
        api_key = os.environ.get("TRADING_212_API_KEY")
        api_secret = os.environ.get("TRADING_212_SECRET_KEY")
    elif key_choice == "2":
        mode = "demo"
        api_key = os.environ.get("TRADING_212_DEMO_API_KEY") or os.environ.get("TRADING_212_API_KEY")
        api_secret = os.environ.get("TRADING_212_DEMO_SECRET_KEY") or os.environ.get("TRADING_212_SECRET_KEY")
    else:
        print("Ukončeno uživatelem.")
        return

    if not api_key or not api_secret:
        print(f"\nCHYBA: Nalezeny neúplné proměnné prostředí pro {mode.upper()}!")
        print("Zkontrolujte nastavení proměnných ve Windows a restartujte terminál.")
        pause_and_exit(1)

    client = Trading212Client(api_key=api_key, api_secret=api_secret, mode=mode)

    try:
        print(f"\n--- Načítám data z Trading 212 ({mode.upper()}) ---")

        # 1. Informace o účtu
        summary = client.get_account_summary()
        currency = summary.get("currency", "")
        print(f"Měna účtu:           {currency}")
        print(f"Celková hodnota:     {summary.get('totalValue')} {currency}")
        print(f"Hodnota portfolia:   {summary.get('investments', {}).get('currentValue')} {currency}")
        print(f"Volné finance:       {summary.get('cash', {}).get('availableToTrade')} {currency}")

        # 2. Otevřené pozice
        positions = client.get_positions()
        print(f"\n--- Otevřené pozice ({len(positions)}) ---")
        if not positions:
            print("Žádné otevřené pozice.")
        else:
            for pos in positions:
                ticker = str(pos.get("ticker", "N/A"))
                qty = str(pos.get("quantity", "N/A"))
                price = str(pos.get("currentPrice", "N/A"))
                ppl = str(pos.get("ppl", "N/A"))
                print(f"- {ticker:<14} | Počet: {qty:<12} | Cena: {price:<8} | Zisk/Ztráta: {ppl} {currency}")

        # 3. Historie objednávek
        orders_data = client.get_order_history(limit=5)
        orders = orders_data.get("items", []) if isinstance(orders_data, dict) else orders_data
        print(f"\n--- Poslední objednávky ({len(orders)}) ---")
        if not orders:
            print("Žádné objednávky v historii.")
        else:
            for order in orders:
                data = order.get("order") if isinstance(order.get("order"), dict) else order

                oid = str(data.get("id", "N/A"))
                ticker = str(data.get("ticker", "N/A"))
                order_type = str(data.get("type", "N/A"))
                status = str(data.get("status", "N/A"))

                # Zpracování objemu podle typu zadání (kusy vs finanční hodnota)
                filled_qty = data.get("filledQuantity")
                ordered_qty = data.get("orderedQuantity")
                filled_val = data.get("filledValue")
                fill_price = data.get("fillPrice")

                if filled_qty is not None:
                    qty_str = f"{filled_qty} ks"
                elif ordered_qty is not None:
                    qty_str = f"{ordered_qty} ks"
                elif filled_val is not None:
                    qty_str = f"{filled_val} {currency}"
                else:
                    qty_str = "N/A"

                price_str = f" | Cena: {fill_price}" if fill_price is not None else ""

                print(
                    f"- ID: {oid:<14} | "
                    f"{ticker:<14} | "
                    f"Objem: {qty_str:<18} | "
                    f"Typ: {order_type:<8} | "
                    f"Stav: {status}{price_str}"
                )

    except requests.exceptions.HTTPError as err:
        print(f"\nHTTP Chyba {err.response.status_code}: {err.response.text}")
    except Exception as e:
        print(f"\nNeočekávaná chyba: {e}")

    pause_and_exit(0)


if __name__ == "__main__":
    main()