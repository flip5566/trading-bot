"""
Portfolio Management Engine

Controls target allocation per regime, detects rebalancing needs,
and generates buy/sell recommendations to bring portfolio back to targets.

Rebalancing trigger: monthly or when any allocation drifts > 5% from target.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from .config import (
    PORTFOLIO_TARGETS,
    LARGE_CAP_ASSETS,
    MAX_SINGLE_ASSET_PCT,
    MAX_ALTCOIN_PCT,
    MIN_STABLECOIN_PCT,
)
from .database import Database

log = logging.getLogger(__name__)

REBALANCE_DRIFT_THRESHOLD = 0.05   # 5% drift triggers rebalance recommendation
REBALANCE_MIN_DAYS        = 28     # Don't recommend rebalance within 28 days of last one


@dataclass
class AllocationPlan:
    regime:          str
    targets:         dict[str, float]          # ticker → target fraction
    current:         dict[str, float]          # ticker → current fraction
    actions:         list[dict]                 # list of {asset, action, drift, priority}
    needs_rebalance: bool


def expand_targets(regime: str) -> dict[str, float]:
    """
    Expand the high-level targets (BTC, ETH, LARGE_CAP, STABLECOIN) into
    per-ticker targets, splitting LARGE_CAP equally among LARGE_CAP_ASSETS.
    """
    base = PORTFOLIO_TARGETS.get(regime, PORTFOLIO_TARGETS["SIDEWAYS"])
    result: dict[str, float] = {}

    result["BTC"] = base["BTC"]
    result["ETH"] = base["ETH"]

    large_cap_total = base["LARGE_CAP"]
    per_alt = large_cap_total / len(LARGE_CAP_ASSETS) if LARGE_CAP_ASSETS else 0
    for asset in LARGE_CAP_ASSETS:
        result[asset] = per_alt

    result["STABLECOIN"] = base["STABLECOIN"]
    return result


def detect_rebalancing_need(
    targets: dict[str, float],
    current: dict[str, float],
) -> list[dict]:
    """
    Compare current vs target allocations.
    Returns a list of action dicts, sorted by absolute drift (most urgent first).
    """
    actions = []
    all_assets = set(targets) | set(current)

    for asset in all_assets:
        target  = targets.get(asset, 0.0)
        actual  = current.get(asset, 0.0)
        drift   = actual - target

        if abs(drift) < 0.01:  # < 1% drift — ignore
            continue

        action = "SELL" if drift > 0 else "BUY"
        if asset == "STABLECOIN":
            action = "REDUCE" if drift > 0 else "INCREASE"

        actions.append({
            "asset":    asset,
            "action":   action,
            "current":  round(actual * 100, 1),
            "target":   round(target * 100, 1),
            "drift_pct": round(drift * 100, 1),
            "urgent":   abs(drift) >= REBALANCE_DRIFT_THRESHOLD,
        })

    return sorted(actions, key=lambda x: abs(x["drift_pct"]), reverse=True)


def build_allocation_plan(
    db: Database,
    regime: str,
    current: dict[str, float] | None = None,
) -> AllocationPlan:
    """
    Build a full allocation plan comparing current to targets.

    `current` should be {ticker: fraction_of_portfolio}.
    If not provided, defaults to an empty portfolio (all in STABLECOIN).
    """
    targets = expand_targets(regime)

    if current is None:
        # No data — treat everything as stablecoin
        current = {"STABLECOIN": 1.0}

    actions = detect_rebalancing_need(targets, current)
    needs = any(a["urgent"] for a in actions)

    # Check time since last rebalance
    portfolio = db.get_latest_portfolio()
    if portfolio:
        last_ts = portfolio.get("ts", "")
        if last_ts:
            try:
                last_date = date.fromisoformat(last_ts)
                days_since = (date.today() - last_date).days
                if days_since < REBALANCE_MIN_DAYS and needs:
                    log.info(
                        "Rebalance needed but last was %d days ago (min %d) — suppressing",
                        days_since, REBALANCE_MIN_DAYS,
                    )
                    needs = False
            except ValueError:
                pass

    return AllocationPlan(
        regime          = regime,
        targets         = targets,
        current         = current,
        actions         = actions,
        needs_rebalance = needs,
    )


def format_allocation_table(plan: AllocationPlan) -> str:
    """Pretty-print the allocation plan as a text table."""
    lines = [
        f"  Regime: {plan.regime}",
        f"  {'Asset':<12} {'Current':>8} {'Target':>8} {'Action':<14}",
        f"  {'─'*12} {'─'*8} {'─'*8} {'─'*14}",
    ]
    for asset, target in plan.targets.items():
        current = plan.current.get(asset, 0.0)
        drift   = current - target
        if abs(drift) < 0.005:
            action_str = "OK"
        elif drift > 0:
            action_str = f"↓ REDUCE {drift*100:+.1f}%"
        else:
            action_str = f"↑ ADD {drift*100:+.1f}%"
        lines.append(
            f"  {asset:<12} {current*100:>7.1f}% {target*100:>7.1f}% {action_str:<14}"
        )
    if plan.needs_rebalance:
        lines.append("\n  *** REBALANCE RECOMMENDED ***")
    return "\n".join(lines)


def risk_guard(current: dict[str, float]) -> list[str]:
    """
    Check all risk limit rules.
    Returns list of warning strings (empty = all clear).
    """
    warnings = []

    for asset, frac in current.items():
        if asset == "STABLECOIN":
            continue
        if frac > MAX_SINGLE_ASSET_PCT:
            warnings.append(
                f"{asset} at {frac*100:.1f}% exceeds single-asset limit ({MAX_SINGLE_ASSET_PCT*100:.0f}%)"
            )

    alt_total = sum(v for k, v in current.items() if k in LARGE_CAP_ASSETS)
    if alt_total > MAX_ALTCOIN_PCT:
        warnings.append(
            f"Altcoin exposure {alt_total*100:.1f}% exceeds limit ({MAX_ALTCOIN_PCT*100:.0f}%)"
        )

    stable = current.get("STABLECOIN", 0.0)
    if stable < MIN_STABLECOIN_PCT:
        warnings.append(
            f"Stablecoin reserve {stable*100:.1f}% below minimum ({MIN_STABLECOIN_PCT*100:.0f}%)"
        )

    return warnings
