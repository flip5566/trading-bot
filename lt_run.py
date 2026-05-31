"""
Long-Term AI Portfolio Manager — Daily Runner

Run once per day (e.g. via cron, Task Scheduler, or manually).
Generates a full portfolio analysis and recommendation report.

Usage:
    python lt_run.py

Optional: pass --allocation to describe your current holdings.
    python lt_run.py --allocation BTC=0.40 ETH=0.25 SOL=0.10 STABLECOIN=0.25

Optional: set base DCA amount via environment variable:
    LT_DCA_AMOUNT=500 python lt_run.py

V1 generates recommendations only (no live trading).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

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
        "--skip-collect",
        action="store_true",
        help="Skip data collection and use cached data (faster, for testing)",
    )
    args = parser.parse_args()

    allocation = parse_allocation(args.allocation or [])

    db = Database()

    if args.skip_collect:
        from lt_bot.execution_engine import generate_daily_report
        report = generate_daily_report(db, args.dca_amount, allocation)
        print(report)
    else:
        from lt_bot.data_collector import collect_all
        collect_all(db)
        from lt_bot.execution_engine import generate_daily_report
        report = generate_daily_report(db, args.dca_amount, allocation)
        print(report)


if __name__ == "__main__":
    main()
