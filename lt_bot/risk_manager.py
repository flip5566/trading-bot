"""
Risk Management Engine

Enforces hard limits on portfolio composition and detects emergency conditions
that require defensive action.

Hard limits:
  - Max single asset:      40%
  - Max altcoin exposure:  40%
  - Min stablecoin reserve: 10%

Emergency mode triggers:
  - Market crash (BTC drops ≥ 10% in 1 day)
  - Extreme volatility (ATR ≥ 8% of price)
  - Liquidity collapse (volume < 30% of 20d MA)

Emergency actions:
  - Stop new buying
  - Protect capital (move to stablecoin)
  - Wait for recovery signal
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import (
    LARGE_CAP_ASSETS,
    MAX_SINGLE_ASSET_PCT,
    MAX_ALTCOIN_PCT,
    MIN_STABLECOIN_PCT,
    CRASH_DAILY_DROP_PCT,
    CRASH_ATR_PCT_THRESHOLD,
)
from .database import Database

log = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    status:          str           # "NORMAL" | "ELEVATED" | "EMERGENCY"
    emergency_mode:  bool
    violations:      list[str] = field(default_factory=list)
    warnings:        list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    crash_drop:      float | None = None
    atr_pct:         float | None = None


def _daily_change(db: Database) -> float:
    rows = db.get_price_history("BTC", "1D", 2)
    if len(rows) < 2:
        return 0.0
    prev = rows[-2]["close"]
    curr = rows[-1]["close"]
    return (curr / prev - 1) if prev else 0.0


def assess_risk(
    db: Database,
    current_allocation: dict[str, float] | None = None,
) -> RiskAssessment:
    """
    Run all risk checks.

    `current_allocation`: {ticker: fraction} — if None, only market-based checks run.
    """
    violations:  list[str] = []
    warnings:    list[str] = []
    recs:        list[str] = []

    # ── Portfolio limit checks ─────────────────────────────────────────────────
    if current_allocation:
        for asset, frac in current_allocation.items():
            if asset == "STABLECOIN":
                continue
            if frac > MAX_SINGLE_ASSET_PCT:
                violations.append(
                    f"{asset}: {frac*100:.1f}% > max {MAX_SINGLE_ASSET_PCT*100:.0f}%"
                )
                recs.append(f"REDUCE {asset} to ≤ {MAX_SINGLE_ASSET_PCT*100:.0f}%")

        alt_total = sum(v for k, v in current_allocation.items() if k in LARGE_CAP_ASSETS)
        if alt_total > MAX_ALTCOIN_PCT:
            violations.append(
                f"Altcoins: {alt_total*100:.1f}% > max {MAX_ALTCOIN_PCT*100:.0f}%"
            )
            recs.append("REDUCE altcoin exposure — sell largest alt positions first")

        stable = current_allocation.get("STABLECOIN", 0.0)
        if stable < MIN_STABLECOIN_PCT:
            violations.append(
                f"Stablecoin: {stable*100:.1f}% < min {MIN_STABLECOIN_PCT*100:.0f}%"
            )
            recs.append(f"ADD to stablecoin reserve to reach {MIN_STABLECOIN_PCT*100:.0f}%")

    # ── Market-based risk checks ───────────────────────────────────────────────
    ind = db.get_latest_indicators("BTC", "1D") or {}
    atr_pct    = ind.get("atr_pct") or 0.0
    daily_chg  = _daily_change(db)

    crash_drop   = None
    emergency    = False

    # Crash detection
    if daily_chg <= CRASH_DAILY_DROP_PCT:
        crash_drop = daily_chg
        violations.append(
            f"CRASH SIGNAL: BTC daily change {daily_chg*100:.1f}% "
            f"(threshold {CRASH_DAILY_DROP_PCT*100:.0f}%)"
        )
        emergency = True
        recs.append("EMERGENCY: Stop new buying immediately")
        recs.append("EMERGENCY: Move 25–40% of crypto to stablecoin")

    if atr_pct >= CRASH_ATR_PCT_THRESHOLD:
        violations.append(
            f"HIGH VOLATILITY: ATR {atr_pct:.1f}% ≥ threshold {CRASH_ATR_PCT_THRESHOLD*100:.0f}%"
        )
        if not emergency:
            emergency = True
            recs.append("High volatility — pause new entries, wait for stabilisation")

    # Volume collapse
    price_rows = db.get_price_history("BTC", "1D", 1)
    vol_ma = ind.get("volume_ma") or 0
    curr_vol = price_rows[-1]["volume"] if price_rows else 0
    if vol_ma > 0:
        vol_ratio = curr_vol / vol_ma
        if vol_ratio < 0.30:
            warnings.append(
                f"LOW VOLUME: current {vol_ratio*100:.0f}% of 20d MA — possible liquidity collapse"
            )

    # Funding rate extreme long positioning
    sent = db.get_latest_sentiment() or {}
    fund = sent.get("btc_funding_rate")
    if fund and fund > 0.05:
        warnings.append(
            f"Extreme positive funding ({fund*100:.3f}%) — longs overcrowded, crash risk elevated"
        )

    # RSI extreme
    rsi = ind.get("rsi") or 50
    if rsi > 80:
        warnings.append(f"BTC RSI {rsi:.0f} — extremely overbought territory")

    # ── Determine overall status ───────────────────────────────────────────────
    if emergency or violations:
        status = "EMERGENCY" if emergency else "ELEVATED"
    elif warnings:
        status = "ELEVATED"
    else:
        status = "NORMAL"

    return RiskAssessment(
        status          = status,
        emergency_mode  = emergency,
        violations      = violations,
        warnings        = warnings,
        recommendations = recs,
        crash_drop      = crash_drop,
        atr_pct         = atr_pct or None,
    )


def format_risk_assessment(ra: RiskAssessment) -> str:
    lines = [f"  Status: {ra.status}"]

    if ra.emergency_mode:
        lines.append("  *** EMERGENCY MODE ACTIVE ***")

    if ra.violations:
        lines.append("  VIOLATIONS:")
        for v in ra.violations:
            lines.append(f"    ✗ {v}")

    if ra.warnings:
        lines.append("  WARNINGS:")
        for w in ra.warnings:
            lines.append(f"    ⚠ {w}")

    if ra.recommendations:
        lines.append("  ACTIONS REQUIRED:")
        for r in ra.recommendations:
            lines.append(f"    → {r}")

    if ra.status == "NORMAL":
        lines.append("  All risk limits within bounds.")

    return "\n".join(lines)
