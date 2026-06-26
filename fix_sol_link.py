import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime, timezone
from pathlib import Path
from alpaca.trading.client import TradingClient

client = TradingClient(api_key=os.environ['ALPACA_API_KEY'], secret_key=os.environ['ALPACA_SECRET_KEY'], paper=True)

# ── 1. Close SOL ───────────────────────────────────────────────────────────────
print("=== CLOSING SOL ===")
try:
    pos = client.get_open_position("SOLUSD")
    current = float(pos.current_price)
    entry = float(pos.avg_entry_price)
    pnl = float(pos.unrealized_pl)
    client.close_position("SOLUSD")
    print(f"  Closed SOLUSD @ ~${current:.4f}  (entry ${entry:.4f})")
    print(f"  PnL locked in: ${pnl:.2f}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 2. Fix LINK in state.json ─────────────────────────────────────────────────
# LINK entry=$7.8742, current ~$8.16 (+3.7%)
# Old stop ($7.8758) was above entry — wrong. Move stop to breakeven.
# TP: set 1.5x the current move above entry (~$8.55)
print()
print("=== FIXING state.json ===")
STATE_FILE = Path(__file__).parent / "state.json"
with open(STATE_FILE) as f:
    state = json.load(f)

link_entry = 7.8742
link_stop = link_entry           # breakeven stop
link_tp = round(link_entry + 1.5 * (link_entry - 7.68), 6)  # ~0.20 risk unit → TP ~8.17

state["date"] = datetime.now(timezone.utc).date().isoformat()
state["open_trades"].pop("SOLUSD", None)
state["open_trades"]["LINKUSD"] = {
    "side": "long",
    "entry": link_entry,
    "stop": round(link_stop, 6),
    "tp": round(link_tp, 6),
    "qty": 12.430243387,
    "breakeven_moved": True,
    "partial_taken": False,
    "trend_1h": "bullish",
}

with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=2)

print(f"  Date updated to: {state['date']}")
print(f"  SOL removed from open_trades")
print(f"  LINK stop = ${link_stop:.4f} (breakeven)  TP = ${link_tp:.4f}")
print(f"  ETH unchanged: stop=${state['open_trades']['ETHUSD']['stop']:.4f}")
print()
print("Done. Remaining positions tracked:", list(state["open_trades"].keys()))
