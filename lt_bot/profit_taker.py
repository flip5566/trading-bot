"""
Profit Taking Engine — partial exits, never sell everything.

Exit tiers:
  First signal  → recommend selling 10%
  Second signal → recommend selling another 15%
  Extreme       → move larger amount (25–40%) to stablecoins

Always maintain a core position.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import PT_RSI_OVERBOUGHT, PT_FUNDING_EXTREME, PT_FEAR_GREED_EUPHORIA, ASSETS
from .database import Database

log = logging.getLogger(__name__)


@dataclass
class ProfitSignal:
    ticker:     str
    tier:       str    # "TIER_1" | "TIER_2" | "TIER_3_EXTREME"
    sell_pct:   float  # fraction to sell, e.g. 0.10 = 10%
    reasons:    list[str] = field(default_factory=list)
    rsi:        float = 0.0
    price:      float = 0.0


def _profit_signals_for(ticker: str, db: Database, sent: dict) -> ProfitSignal | None:
    """Evaluate one asset for profit-taking signals."""
    ind = db.get_latest_indicators(ticker, "1D")
    if not ind:
        return None

    rows  = db.get_price_history(ticker, "1D", 1)
    price = rows[-1]["close"] if rows else 0.0
    rsi   = ind.get("rsi") or 50.0

    ema200 = ind.get("ema200") or price
    fg     = sent.get("fear_greed_index")
    fund   = sent.get("btc_funding_rate") if ticker == "BTC" else sent.get("eth_funding_rate")

    reasons: list[str] = []
    signal_count = 0

    # RSI overbought
    if rsi > PT_RSI_OVERBOUGHT:
        reasons.append(f"RSI overbought ({rsi:.0f} > {PT_RSI_OVERBOUGHT:.0f})")
        signal_count += 1

    # Fear & Greed extreme greed
    if fg is not None and fg >= PT_FEAR_GREED_EUPHORIA:
        reasons.append(f"Extreme greed (F&G={fg})")
        signal_count += 1

    # Funding rate extremely high (longs overcrowded)
    if fund is not None and fund >= PT_FUNDING_EXTREME:
        reasons.append(f"Extreme funding ({fund*100:.3f}%)")
        signal_count += 1

    # Price extended far above EMA200 (> 100% premium)
    if ema200 and price > ema200 * 2.0:
        prem = (price / ema200 - 1) * 100
        reasons.append(f"Price +{prem:.0f}% above EMA200 (parabolic)")
        signal_count += 1

    # BB: price above upper Bollinger Band
    bb_upper = ind.get("bb_upper")
    if bb_upper and price > bb_upper:
        reasons.append("Price above upper Bollinger Band")
        signal_count += 1

    if signal_count == 0:
        return None

    if signal_count >= 4:
        tier, sell_pct = "TIER_3_EXTREME", 0.35
    elif signal_count >= 3:
        tier, sell_pct = "TIER_2", 0.15
    else:
        tier, sell_pct = "TIER_1", 0.10

    log.info(
        "Profit signal %s for %s — %d signals, sell %.0f%%",
        tier, ticker, signal_count, sell_pct * 100,
    )
    return ProfitSignal(
        ticker   = ticker,
        tier     = tier,
        sell_pct = sell_pct,
        reasons  = reasons,
        rsi      = rsi,
        price    = price,
    )


def check_profit_signals(db: Database) -> list[ProfitSignal]:
    """Scan all assets for profit-taking signals."""
    sent = db.get_latest_sentiment() or {}
    signals: list[ProfitSignal] = []

    for ticker in ASSETS:
        sig = _profit_signals_for(ticker, db, sent)
        if sig:
            signals.append(sig)

    return sorted(signals, key=lambda s: s.sell_pct, reverse=True)


_TIER_LABELS = {
    "TIER_1":         "First signal — sell 10%",
    "TIER_2":         "Second signal — sell 15%",
    "TIER_3_EXTREME": "Extreme signal — sell 35% to stablecoins",
}


def format_profit_signals(signals: list[ProfitSignal]) -> str:
    if not signals:
        return "  No profit-taking signals. Hold current positions."
    lines = []
    for sig in signals:
        lines += [
            f"  {sig.ticker}  [{sig.tier}]  — {_TIER_LABELS.get(sig.tier, '')}",
            f"    Price: ${sig.price:,.2f}  |  RSI: {sig.rsi:.0f}",
        ]
        for r in sig.reasons:
            lines.append(f"    • {r}")
    return "\n".join(lines)
