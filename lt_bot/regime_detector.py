"""
Market Regime Detection Engine

Determines the current crypto market environment from BTC data + market signals.

States (in priority order):
  EUPHORIA    — overbought + extreme funding + extreme F&G
  CRASH       — large daily drop + high volatility + liquidation spike
  STRONG_BULL — price > EMA200, EMAs stacked, ADX > 25, volume rising
  BULL        — price > EMA200, positive momentum
  BEAR        — price < EMA200, bearish EMA stack
  SIDEWAYS    — mixed EMAs, low ADX, low volatility
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .database import Database
from .config import (
    CRASH_DAILY_DROP_PCT,
    CRASH_ATR_PCT_THRESHOLD,
    PT_FEAR_GREED_EUPHORIA,
    PT_RSI_OVERBOUGHT,
    PT_FUNDING_EXTREME,
)

log = logging.getLogger(__name__)

REGIMES = ["EUPHORIA", "CRASH", "STRONG_BULL", "BULL", "SIDEWAYS", "BEAR"]


@dataclass
class RegimeInputs:
    price:          float
    ema50:          float
    ema100:         float
    ema200:         float
    rsi:            float
    adx:            float
    atr_pct:        float
    volume:         float
    volume_ma:      float
    daily_change:   float       # pct, e.g. -0.12 = -12%
    fear_greed:     float | None
    funding_rate:   float | None  # decimal, e.g. 0.03 = 3%


def _inputs_from_db(db: Database) -> RegimeInputs | None:
    ind = db.get_latest_indicators("BTC", "1D")
    if not ind:
        log.warning("No BTC 1D indicators found in DB")
        return None

    history = db.get_price_history("BTC", "1D", limit=2)
    if len(history) < 2:
        daily_change = 0.0
    else:
        prev_close = history[-2]["close"]
        curr_close = history[-1]["close"]
        daily_change = (curr_close / prev_close - 1) if prev_close else 0.0

    sent = db.get_latest_sentiment() or {}
    price_rows = db.get_price_history("BTC", "1D", limit=1)
    price = price_rows[-1]["close"] if price_rows else 0.0

    return RegimeInputs(
        price        = price,
        ema50        = ind.get("ema50")  or price,
        ema100       = ind.get("ema100") or price,
        ema200       = ind.get("ema200") or price,
        rsi          = ind.get("rsi")    or 50.0,
        adx          = ind.get("adx")    or 20.0,
        atr_pct      = ind.get("atr_pct") or 2.0,
        volume       = ind.get("volume_ma") or 0.0,   # reuse; we compare ratio
        volume_ma    = ind.get("volume_ma") or 1.0,
        daily_change = daily_change,
        fear_greed   = sent.get("fear_greed_index"),
        funding_rate = sent.get("btc_funding_rate"),
    )


def _volume_expanding(db: Database, lookback: int = 5) -> bool:
    rows = db.get_price_history("BTC", "1D", limit=lookback + 1)
    if len(rows) < lookback + 1:
        return False
    recent_vol = sum(r["volume"] for r in rows[-lookback:]) / lookback
    older_vol  = sum(r["volume"] for r in rows[:lookback]) / lookback
    return recent_vol > older_vol * 1.1


def _higher_highs(db: Database, lookback: int = 14) -> bool:
    rows = db.get_price_history("BTC", "1D", limit=lookback)
    if len(rows) < lookback:
        return False
    highs = [r["high"] for r in rows]
    # At least 2 consecutive higher highs in recent period
    count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
    return count >= lookback // 2


def detect_regime(db: Database) -> str:
    inp = _inputs_from_db(db)
    if not inp:
        return "SIDEWAYS"

    p   = inp.price
    e50 = inp.ema50
    e100 = inp.ema100
    e200 = inp.ema200

    # ── 1. EUPHORIA ──────────────────────────────────────────────────────────
    euphoria_signals = 0
    if inp.rsi > PT_RSI_OVERBOUGHT:
        euphoria_signals += 1
    if inp.fear_greed is not None and inp.fear_greed >= PT_FEAR_GREED_EUPHORIA:
        euphoria_signals += 1
    if inp.funding_rate is not None and inp.funding_rate >= PT_FUNDING_EXTREME:
        euphoria_signals += 1
    if p > e50 * 1.20:  # price 20% above EMA50 — rapid expansion
        euphoria_signals += 1

    if euphoria_signals >= 3:
        log.info("Regime: EUPHORIA (%d/4 signals)", euphoria_signals)
        return "EUPHORIA"

    # ── 2. CRASH ─────────────────────────────────────────────────────────────
    crash_signals = 0
    if inp.daily_change <= CRASH_DAILY_DROP_PCT:
        crash_signals += 1
    if inp.atr_pct >= CRASH_ATR_PCT_THRESHOLD:
        crash_signals += 1
    if inp.fear_greed is not None and inp.fear_greed <= 10:
        crash_signals += 1

    if crash_signals >= 2:
        log.info("Regime: CRASH (%d/3 signals)", crash_signals)
        return "CRASH"

    # ── 3. STRONG BULL ───────────────────────────────────────────────────────
    if (
        p > e200
        and e50 > e100 > e200
        and inp.adx > 25
        and _volume_expanding(db)
        and _higher_highs(db)
    ):
        log.info("Regime: STRONG_BULL")
        return "STRONG_BULL"

    # ── 4. BULL ──────────────────────────────────────────────────────────────
    if p > e200 and inp.rsi > 45:
        log.info("Regime: BULL")
        return "BULL"

    # ── 5. BEAR ──────────────────────────────────────────────────────────────
    if p < e200 and e50 < e200:
        log.info("Regime: BEAR")
        return "BEAR"

    # ── 6. SIDEWAYS ──────────────────────────────────────────────────────────
    log.info("Regime: SIDEWAYS")
    return "SIDEWAYS"


def regime_summary(regime: str) -> str:
    """One-line action summary for the regime."""
    return {
        "STRONG_BULL": "Increase exposure — follow trends, buy pullbacks",
        "BULL":        "Normal accumulation — continue DCA",
        "SIDEWAYS":    "Reduce buying — wait for directional confirmation",
        "BEAR":        "Reduce exposure — increase stablecoin allocation",
        "CRASH":       "Protect capital — pause risky buying",
        "EUPHORIA":    "Take profit — distribute into stablecoins",
    }.get(regime, "Hold current positions")
