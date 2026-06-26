import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone
from pathlib import Path
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

client = TradingClient(api_key=os.environ['ALPACA_API_KEY'], secret_key=os.environ['ALPACA_SECRET_KEY'], paper=True)

# ── 1. Close AVAX ─────────────────────────────────────────────────────────────
print("=== CLOSING AVAX ===")
try:
    pos = client.get_open_position("AVAXUSD")
    qty = float(pos.qty)
    price = float(pos.current_price)
    entry = float(pos.avg_entry_price)
    pnl = float(pos.unrealized_pl)
    order = client.submit_order(MarketOrderRequest(
        symbol="AVAXUSD",
        qty=round(qty, 8),
        side=OrderSide.SELL,
        time_in_force=TimeInForce.IOC,
    ))
    print(f"  Order submitted: {order.id}")
    print(f"  Closed {qty:.6f} AVAX @ ~${price:.4f}  (entry ${entry:.4f})")
    print(f"  PnL: ${pnl:.2f}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 2. Fetch remaining positions & compute stops ───────────────────────────────
print()
print("=== REMAINING POSITIONS + STOP CHECK ===")

from strategy import fetch_bars, ema
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

SYMBOL_MAP = {
    "ETHUSD":  "ETH/USD",
    "LINKUSD": "LINK/USD",
    "SOLUSD":  "SOL/USD",
}

open_trades = {}
portfolio_value = float(client.get_account().portfolio_value)

for alpaca_sym, feed_sym in SYMBOL_MAP.items():
    try:
        pos = client.get_open_position(alpaca_sym)
    except Exception:
        print(f"  {alpaca_sym}: position not found, skipping")
        continue

    entry   = float(pos.avg_entry_price)
    current = float(pos.current_price)
    qty     = float(pos.qty)
    pnl_usd = float(pos.unrealized_pl)
    pnl_pct = float(pos.unrealized_plpc) * 100

    # Fetch 5m bars to get live EMA20 as stop reference
    df_5m = fetch_bars(feed_sym, TimeFrame(5, TimeFrameUnit.Minute), 80)
    e20_5m = float(ema(df_5m["close"], 20).iloc[-1])

    # Stop = 0.2% below EMA20 (strategy rule for long)
    stop = round(e20_5m * 0.998, 6)
    tp   = round(entry + 1.5 * (entry - stop), 6)  # 1.5R from original entry

    stop_hit = current <= stop
    risk_usd = qty * abs(entry - stop)
    pnl_r    = (current - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0

    status = "STOP HIT ⚠" if stop_hit else ("BE zone" if pnl_r >= 1.0 else "active")

    print(f"  {alpaca_sym}")
    print(f"    Entry    : ${entry:.4f}   Current: ${current:.4f}   PnL: ${pnl_usd:.2f} ({pnl_pct:+.2f}%)")
    print(f"    EMA20 5m : ${e20_5m:.4f}")
    print(f"    Stop     : ${stop:.4f}   TP: ${tp:.4f}   Status: {status}")
    print(f"    R position: {pnl_r:+.2f}R")
    print()

    open_trades[alpaca_sym] = {
        "side": "long",
        "entry": entry,
        "stop": stop,
        "tp": tp,
        "qty": qty,
        "breakeven_moved": pnl_r >= 1.0,
        "partial_taken": False,
        "trend_1h": "bullish",
    }

# ── 3. Rebuild state.json ──────────────────────────────────────────────────────
today = datetime.now(timezone.utc).date().isoformat()
state = {
    "date": today,
    "daily_loss_pct": 0.0,
    "consecutive_losses": 0,
    "trading_halted": False,
    "open_trades": open_trades,
}
STATE_FILE = Path(__file__).parent / "state.json"
with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=2)

print(f"=== state.json REBUILT — {len(open_trades)} positions tracked ===")
print(f"    Date: {today}")
for sym, t in open_trades.items():
    print(f"    {sym}: stop=${t['stop']:.4f}  tp=${t['tp']:.4f}")
