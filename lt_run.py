"""
Long-Term AI Portfolio Manager — Daily Runner

Run once per day (e.g. via cron, Task Scheduler, or manually).
Generates a full portfolio analysis and recommendation report.

Usage:
    # Recommendations only (default)
    python lt_run.py --dca-amount 1000

    # With live paper trading (reads Alpaca account + executes orders)
    python lt_run.py --paper-trade --dca-amount 1000

    # Pass your allocation manually (recommendations only)
    python lt_run.py --allocation BTC=0.40 ETH=0.25 SOL=0.10 STABLECOIN=0.25

With --paper-trade the bot will:
  - Read your live Alpaca paper account balance and positions
  - Execute DCA buys on underweight assets when score >= 50
  - Execute partial sells on profit-taking signals
  - Skip all buying in CRASH regime or emergency mode
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Windows terminals often default to a non-UTF-8 code page; reconfigure stdout
# so the report's box-drawing characters print cleanly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path for module resolution
sys.path.insert(0, str(Path(__file__).parent))

from lt_bot.database import Database
from lt_bot.execution_engine import run, BASE_DCA_AMOUNT_USD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def parse_allocation(args: list[str]) -> dict[str, float] | None:
    """Parse KEY=VALUE pairs into a allocation dict, e.g. ['BTC=0.40', 'ETH=0.25']."""
    if not args:
        return None
    result: dict[str, float] = {}
    for item in args:
        parts = item.split("=")
        if len(parts) != 2:
            print(f"Skipping invalid allocation: {item}")
            continue
        try:
            result[parts[0].upper()] = float(parts[1])
        except ValueError:
            print(f"Skipping invalid value: {item}")
    if result:
        total = sum(result.values())
        if abs(total - 1.0) > 0.01:
            print(f"Warning: allocations sum to {total:.2f} (expected 1.0)")
    return result or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-Term AI Crypto Portfolio Manager (V1 — recommendations)"
    )
    parser.add_argument(
        "--allocation",
        nargs="*",
        metavar="TICKER=FRACTION",
        help="Current portfolio allocation, e.g. BTC=0.40 ETH=0.25 STABLECOIN=0.35",
    )
    parser.add_argument(
        "--dca-amount",
        type=float,
        default=float(os.environ.get("LT_DCA_AMOUNT", BASE_DCA_AMOUNT_USD)),
        help=f"Monthly base DCA amount in USD (default: {BASE_DCA_AMOUNT_USD:,.0f})",
    )
    parser.add_argument(
        "--paper-trade",
        action="store_true",
        help="Connect to Alpaca paper account and execute trades (default: recommendations only)",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip data collection and use cached data (faster, for testing)",
    )
    args = parser.parse_args()

    allocation = parse_allocation(args.allocation or [])

    db = Database()

    # Set up paper trader if requested
    trader = None
    if args.paper_trade:
        from lt_bot.trader import PaperTrader
        trader = PaperTrader()
        acct = trader.get_account_summary()
        print(f"\n  Alpaca paper account connected")
        print(f"  Portfolio value : ${acct['portfolio_value']:,.2f}")
        print(f"  Buying power    : ${acct['buying_power']:,.2f}\n")
        # Ignore manual --allocation when paper trading (we read live positions)
        allocation = None

    if args.skip_collect:
        from lt_bot.execution_engine import generate_daily_report
        report = generate_daily_report(db, args.dca_amount, allocation, trader)
        print(report)
    else:
        from lt_bot.data_collector import collect_all
        collect_all(db)
        from lt_bot.execution_engine import generate_daily_report
        report = generate_daily_report(db, args.dca_amount, allocation, trader)
        print(report)


if __name__ == "__main__":
    main()
