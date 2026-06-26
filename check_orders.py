import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

client = TradingClient(api_key=os.environ['ALPACA_API_KEY'], secret_key=os.environ['ALPACA_SECRET_KEY'], paper=True)

print('=== OPEN ORDERS ===')
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
if not orders:
    print('  None')
for o in orders:
    print(f'  {o.symbol} | {o.side} | qty={o.qty} | type={o.order_type} | status={o.status} | created={str(o.created_at)[:16]}')

print()
print('=== OPEN POSITIONS ===')
positions = client.get_all_positions()
if not positions:
    print('  None')
for p in positions:
    pnl_pct = float(p.unrealized_plpc) * 100
    entry = float(p.avg_entry_price)
    current = float(p.current_price)
    pnl_usd = float(p.unrealized_pl)
    print(f'  {p.symbol} | {p.side} | qty={p.qty} | entry=${entry:.4f} | current=${current:.4f} | PnL=${pnl_usd:.2f} ({pnl_pct:+.2f}%)')

print()
account = client.get_account()
portfolio = float(account.portfolio_value)
buying = float(account.buying_power)
print(f'Portfolio : ${portfolio:,.2f}')
print(f'Buying pw : ${buying:,.2f}')
