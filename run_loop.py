import time
from datetime import datetime, timezone
from strategy import run

INTERVAL = 300  # 5 minutes

print(f"Bot started. Scanning every {INTERVAL // 60} minutes. Press Ctrl+C to stop.")

while True:
    try:
        run()
    except Exception as e:
        print(f"[ERROR] {datetime.now(timezone.utc).strftime('%H:%M UTC')} — {e}")

    next_run = datetime.now(timezone.utc)
    print(f"Next scan in {INTERVAL // 60} min...\n")
    time.sleep(INTERVAL)
