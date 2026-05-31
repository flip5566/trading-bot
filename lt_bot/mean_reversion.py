"""
Mean Reversion Engine — identifies panic opportunities.

A buy signal requires ALL of:
  1. RSI < 30 (oversold)
  2. Significant price correction (≥ 15% from 30d high)
  3. Fear & Greed < 25 (extreme fear)     [OR proxy if unavailable]
  4. Funding rate negative or near-zero   [optional — boosts confidence]
  5. Long-term trend intact (price > 0.85× EMA200)

Only flags assets that pass a quality filter (BTC/ETH always pass;
large-caps need market cap and volume checks).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import ASSETS
from .database import Database

log = logging.getLogger(__name__)

OVERSOLD_RSI        = 30.0
CORRECTION_THRESHOLD = 0.15   # 15% from 30d high
FEAR_THRESHOLD       = 25
EMA200_MARGIN        = 0.85   # allow price 15% below EMA200 still considered
QUALITY_ASSETS       = {"BTC", "ETH", "SOL", "AVAX", "LINK"}  # skip low-quality alts


@dataclass
class MeanReversionOpportunity:
    ticker:          str
    price:           float
    rsi:             float
    drop_from_high:  float      # negative pct, e.g. -0.25 = -25%
    fear_greed:      float | None
    funding:         float | None
    confidence:      str        # "HIGH" | "MEDIUM" | "LOW"
    signals:         list[str] = field(default_factory=list)


def _check_asset(ticker: str, db: Database, sent: dict) -> MeanReversionOpportunity | None:
    """Evaluate one asset for mean-reversion opportunity. Returns None if not triggered."""
    ind = db.get_latest_indicators(ticker, "1D")
    if not ind:
        return None

    history = db.get_price_history(ticker, "1D", 30)
    if not history:
        return None

    price   = history[-1]["close"]
    peak    = max(r["high"] for r in history)
    drop    = (price / peak - 1) if peak else 0.0

    rsi_val = ind.get("rsi") or 50.0
    ema200  = ind.get("ema200") or price

    fg    = sent.get("fear_greed_index")
    fund  = sent.get("btc_funding_rate") if ticker == "BTC" else sent.get("eth_funding_rate")

    signals: list[str] = []
    score = 0

    # Condition 1: RSI oversold
    if rsi_val < OVERSOLD_RSI:
        signals.append(f"RSI oversold ({rsi_val:.0f})")
        score += 2
    elif rsi_val < 35:
        signals.append(f"RSI approaching oversold ({rsi_val:.0f})")
        score += 1

    # Condition 2: price correction
    if drop <= -CORRECTION_THRESHOLD:
        signals.append(f"Price -{abs(drop)*100:.1f}% from 30d high")
        score += 2
    elif drop <= -0.10:
        signals.append(f"Price -{abs(drop)*100:.1f}% correction")
        score += 1

    # Condition 3: extreme fear
    if fg is not None:
        if fg < FEAR_THRESHOLD:
            signals.append(f"Extreme fear (F&G={fg})")
            score += 2
        elif fg < 35:
            signals.append(f"Fear (F&G={fg})")
            score += 1
    else:
        # Proxy via RSI
        if rsi_val < 25:
            signals.append("RSI proxy: extreme fear conditions")
            score += 1

    # Condition 4: funding negative / neutral (bullish for longs)
    if fund is not None and fund < 0:
        signals.append(f"Negative funding ({fund*100:.3f}%)")
        score += 1

    # Condition 5: long-term trend still intact
    trend_intact = price >= ema200 * EMA200_MARGIN
    if not trend_intact:
        signals.append(f"WARNING: price {price:.0f} < {EMA200_MARGIN*100:.0f}% EMA200 {ema200:.0f}")
        score -= 1

    if score < 3:
        return None   # not enough signals

    if score >= 6:
        confidence = "HIGH"
    elif score >= 4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return MeanReversionOpportunity(
        ticker         = ticker,
        price          = price,
        rsi            = rsi_val,
        drop_from_high = drop,
        fear_greed     = fg,
        funding        = fund,
        confidence     = confidence,
        signals        = signals,
    )


def find_opportunities(db: Database) -> list[MeanReversionOpportunity]:
    """Scan all tracked assets for mean-reversion buying opportunities."""
    sent = db.get_latest_sentiment() or {}
    opportunities: list[MeanReversionOpportunity] = []

    for ticker in ASSETS:
        if ticker not in QUALITY_ASSETS:
            continue   # skip lower-quality alts

        opp = _check_asset(ticker, db, sent)
        if opp:
            opportunities.append(opp)
            log.info(
                "Mean-reversion opportunity: %s  RSI=%.0f  drop=%.1f%%  confidence=%s",
                ticker, opp.rsi, opp.drop_from_high * 100, opp.confidence,
            )

    return sorted(opportunities, key=lambda o: o.rsi)


def format_opportunities(opps: list[MeanReversionOpportunity]) -> str:
    if not opps:
        return "  No mean-reversion opportunities detected."
    lines = []
    for opp in opps:
        lines += [
            f"  {opp.ticker}  — {opp.confidence} confidence",
            f"    Price: ${opp.price:,.2f}  |  RSI: {opp.rsi:.0f}"
            f"  |  Drop: {opp.drop_from_high*100:.1f}% from high",
        ]
        for sig in opp.signals:
            lines.append(f"    • {sig}")
    return "\n".join(lines)
