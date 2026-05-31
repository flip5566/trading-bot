"""
Long-Term Bot — Trade Monitor

Runs the bot on a configurable interval and stops once N buy trades
have been executed, then automatically launches the post-trade analysis.

Usage:
    # Wait for 5 buys (checks every 60 s — good for testing)
    python lt_monitor.py --paper-trade --dca-amount 1000 --target 5 --interval 60

    # Production: daily cadence
    python lt_monitor.py --paper-trade --dca-amount 1000 --target 5 --interval 86400
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def count_executed_buys(db) -> int:
    rows = db.get_recent_decisions(days=365)
    return sum(1 for r in rows if r.get("action") == "BUY_EXECUTED")


def run_cycle(db, trader, dca_amount: float) -> int:
    """Run one full collection + analysis + execution cycle. Returns buys executed this cycle."""
    from lt_bot.data_collector import collect_all
    from lt_bot.execution_engine import generate_daily_report

    collect_all(db)
    report = generate_daily_report(db, dca_amount, trader=trader)
    print(report)

    # Count new buys in this cycle (rough: just re-count total)
    return count_executed_buys(db)


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-Term Bot — Trade Monitor")
    parser.add_argument("--paper-trade", action="store_true",
                        help="Connect to Alpaca paper account and execute trades")
    parser.add_argument("--dca-amount", type=float, default=1_000.0,
                        help="Monthly base DCA amount USD")
    parser.add_argument("--target",   type=int,   default=5,
                        help="Stop after this many executed buy trades (default: 5)")
    parser.add_argument("--interval", type=int,   default=3600,
                        help="Seconds between cycles (default: 3600 = 1 hour)")
    args = parser.parse_args()

    from lt_bot.database import Database
    db = Database()

    trader = None
    if args.paper_trade:
        from lt_bot.trader import PaperTrader
        trader = PaperTrader()
        acct = trader.get_account_summary()
        print(f"\n  Alpaca paper account  |  Portfolio: ${acct['portfolio_value']:,.2f}"
              f"  |  Buying power: ${acct['buying_power']:,.2f}\n")

    initial_buys = count_executed_buys(db)
    print(f"  Starting with {initial_buys} previously executed buys in DB.")
    print(f"  Target: {args.target} buys total  |  Interval: {args.interval}s\n")

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n{'━'*66}")
        print(f"  CYCLE {cycle}  —  {now}")
        print(f"{'━'*66}\n")

        try:
            total_buys = run_cycle(db, trader, args.dca_amount)
        except Exception as exc:
            log.error("Cycle %d failed: %s", cycle, exc)
            total_buys = count_executed_buys(db)

        new_buys = total_buys - initial_buys
        print(f"\n  Progress: {new_buys}/{args.target} buys executed since monitor started")

        if new_buys >= args.target:
            print(f"\n  Target of {args.target} trades reached — launching analysis...\n")
            break

        remaining = args.target - new_buys
        print(f"  Waiting {args.interval}s before next cycle  "
              f"({remaining} more buy(s) needed)...\n")
        time.sleep(args.interval)

    # Run analysis
    import subprocess
    cmd = [sys.executable, "lt_analyze.py"]
    if args.paper_trade:
        cmd.append("--paper-trade")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
