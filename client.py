from dotenv import load_dotenv
import os
from alpaca.trading.client import TradingClient

load_dotenv()

def get_trading_client() -> TradingClient:
    return TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )

if __name__ == "__main__":
    client = get_trading_client()
    account = client.get_account()
    print(f"Account status : {account.status}")
    print(f"Buying power   : ${float(account.buying_power):,.2f}")
    print(f"Portfolio value: ${float(account.portfolio_value):,.2f}")
    print(f"Cash           : ${float(account.cash):,.2f}")
