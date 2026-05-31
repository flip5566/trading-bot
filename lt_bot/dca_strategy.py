"""
Smart DCA Strategy — dynamic allocation based on AI score and market conditions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import DCA_SCHEDULE, CORRECTION_BUY_THRESHOLD_PCT, ASSETS
from .database import Database

log = logging.getLogger(__name__)


@dataclass
class DCARecommendation:
    multiplier:      float
    base_amount_usd: float
    total_usd:       float
    reason:          str
    extra_triggers:  list[str] = field(default_factory=list)


def get_dca_multiplier(score: float) -> float:
    """Return the DCA multiplier for a given AI score."""
    for lo, hi, mult in DCA_SCHEDULE:
        if lo <= score <= hi:
            return mult
    return 0.0


def check_correction_trigger(db: Database, ticker: str = "BTC") -> tuple[bool, float]:
    """
    Return (triggered, drop_pct) if the asset has corrected ≥ 15% from its
    recent 30-day high while the longer-term trend is still intact.
    """
    history = db.get_price_history(ticker, "1D", 30)
    if len(history) < 10:
        return False, 0.0

    peak   = max(r["high"] for r in history)
    curr   = history[-1]["close"]
    drop   = (curr / peak - 1) if peak else 0.0

    # Check long-term trend is still intact (price > EMA200)
    ind = db.get_latest_indicators(ticker, "1D") or {}
    ema200 = ind.get("ema200") or 0
    trend_ok = curr > ema200 * 0.97  # allow slight breach

    triggered = drop <= CORRECTION_BUY_THRESHOLD_PCT and trend_ok
    return triggered, drop


def build_dca_recommendation(
    score: float,
    base_amount_usd: float,
    db: Database,
) -> DCARecommendation:
    """
    Build the full DCA recommendation including base multiplier and any
    additional correction/oversold triggers.
    """
    multiplier = get_dca_multiplier(score)
    total_usd  = base_amount_usd * multiplier
    extra: list[str] = []

    if multiplier == 0.0:
        reason = f"AI score {score:.1f} < 50 — DCA paused"
    else:
        reason = f"AI score {score:.1f} → {multiplier}× DCA multiplier"

    # Check if any asset has a major correction
    for ticker in ASSETS:
        triggered, drop_pct = check_correction_trigger(db, ticker)
        if triggered:
            ind = db.get_latest_indicators(ticker, "1D") or {}
            rsi = ind.get("rsi") or 50
            msg = (
                f"{ticker} correction {drop_pct*100:+.1f}% from 30d high"
                f" (RSI {rsi:.0f}) — trend intact"
            )
            extra.append(msg)
            # Add 0.5x extra allocation for correction opportunities
            total_usd += base_amount_usd * 0.5

    return DCARecommendation(
        multiplier      = multiplier,
        base_amount_usd = base_amount_usd,
        total_usd       = total_usd,
        reason          = reason,
        extra_triggers  = extra,
    )


def format_dca_recommendation(rec: DCARecommendation, regime: str) -> str:
    lines = [
        f"DCA Multiplier : {rec.multiplier}×",
        f"Base Amount    : ${rec.base_amount_usd:,.0f}",
        f"Total DCA      : ${rec.total_usd:,.0f}",
        f"Reason         : {rec.reason}",
    ]
    if rec.extra_triggers:
        lines.append("Extra triggers :")
        for t in rec.extra_triggers:
            lines.append(f"  + {t}")
    return "\n".join(lines)
