"""
Momentum Rotation Engine

Detects capital rotation between BTC, ETH, and altcoins.
Never rotates without trend confirmation.

Seasons:
  BTC_SEASON  — BTC dominance rising, ETH/BTC falling
  ETH_SEASON  — ETH/BTC rising, ETH outperforming
  ALT_SEASON  — Alts outperforming ETH and BTC
  NEUTRAL     — No clear rotation signal
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .database import Database

log = logging.getLogger(__name__)


@dataclass
class RotationSignal:
    season:           str
    btc_dominance:    float | None
    dom_trend:        str          # "rising" | "falling" | "flat"
    eth_btc_ratio:    float | None
    eth_btc_trend:    str
    recommendation:   str
    confidence:       str          # "HIGH" | "MEDIUM" | "LOW"


def _trend_direction(values: list[float], lookback: int = 7) -> str:
    """Simple linear-regression direction on last `lookback` values."""
    if len(values) < 3:
        return "flat"
    recent = values[-lookback:]
    if len(recent) < 2:
        return "flat"
    n = len(recent)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(recent) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, recent))
    den = sum((xi - x_mean) ** 2 for xi in x)
    if den == 0:
        return "flat"
    slope = num / den
    pct   = slope / (y_mean or 1)  # normalised slope
    if pct >  0.005:
        return "rising"
    if pct < -0.005:
        return "falling"
    return "flat"


def _alt_outperformance(db: Database, days: int = 14) -> bool:
    """
    Returns True if large-cap alts are collectively outperforming BTC over `days`.
    Uses relative change from price history.
    """
    from .config import LARGE_CAP_ASSETS

    btc_rows = db.get_price_history("BTC", "1D", days)
    if len(btc_rows) < 2:
        return False
    btc_change = btc_rows[-1]["close"] / btc_rows[0]["close"] - 1

    alt_changes = []
    for alt in LARGE_CAP_ASSETS:
        rows = db.get_price_history(alt, "1D", days)
        if len(rows) >= 2:
            alt_changes.append(rows[-1]["close"] / rows[0]["close"] - 1)

    if not alt_changes:
        return False

    avg_alt = sum(alt_changes) / len(alt_changes)
    return avg_alt > btc_change * 1.1   # alts need to beat BTC by 10%


def detect_rotation(db: Database) -> RotationSignal:
    """Analyse market data to determine the current season."""
    history = db.get_market_data_history(30)

    btc_dom    = None
    eth_btc    = None
    dom_trend  = "flat"
    eth_trend  = "flat"

    if history:
        dom_vals = [r["btc_dominance"] for r in history if r.get("btc_dominance")]
        eth_vals = [r["eth_btc_ratio"] for r in history if r.get("eth_btc_ratio")]

        if dom_vals:
            btc_dom  = dom_vals[-1]
            dom_trend = _trend_direction(dom_vals)
        if eth_vals:
            eth_btc   = eth_vals[-1]
            eth_trend = _trend_direction(eth_vals)

    alts_outperform = _alt_outperformance(db)

    # ── Classification logic ──────────────────────────────────────────────────

    if btc_dom and btc_dom > 55 and dom_trend == "rising":
        season       = "BTC_SEASON"
        recommendation = "Increase BTC allocation — capital flowing into BTC"
        confidence   = "HIGH" if dom_trend == "rising" else "MEDIUM"

    elif eth_btc and eth_trend == "rising" and (not dom_trend == "rising"):
        season         = "ETH_SEASON"
        recommendation = "Increase ETH allocation — ETH outperforming BTC"
        confidence     = "HIGH" if eth_trend == "rising" else "MEDIUM"

    elif alts_outperform and dom_trend == "falling":
        season         = "ALT_SEASON"
        recommendation = "Increase large-cap alt allocation — alt season underway"
        confidence     = "MEDIUM"

    else:
        season         = "NEUTRAL"
        recommendation = "No clear rotation — maintain current allocation"
        confidence     = "LOW"

    signal = RotationSignal(
        season         = season,
        btc_dominance  = btc_dom,
        dom_trend      = dom_trend,
        eth_btc_ratio  = eth_btc,
        eth_btc_trend  = eth_trend,
        recommendation = recommendation,
        confidence     = confidence,
    )

    log.info(
        "Rotation: %s (%s) — BTC dom %.1f%% (%s), ETH/BTC %s (%s)",
        season, confidence,
        (btc_dom or 0), dom_trend,
        f"{eth_btc:.4f}" if eth_btc else "n/a", eth_trend,
    )
    return signal


def format_rotation(signal: RotationSignal) -> str:
    lines = [
        f"  Season        : {signal.season} ({signal.confidence} confidence)",
        f"  BTC Dominance : {signal.btc_dominance:.1f}% ({signal.dom_trend})"
          if signal.btc_dominance else "  BTC Dominance : n/a",
        f"  ETH/BTC Ratio : {signal.eth_btc_ratio:.4f} ({signal.eth_btc_trend})"
          if signal.eth_btc_ratio else "  ETH/BTC Ratio : n/a",
        f"  Recommendation: {signal.recommendation}",
    ]
    return "\n".join(lines)
