"""
AI Scoring Engine — produces a 0-100 composite market score.

Components and weights:
  Trend     25% — EMA alignment, MACD, market structure
  Momentum  15% — RSI, volume, relative strength
  Cycle     20% — BTC halving cycle phase, market cap expansion
  Liquidity 15% — volume vs MA, stablecoin supply
  Sentiment 10% — Fear & Greed, funding rate
  Risk      15% — volatility (ATR), drawdown, leverage

Score interpretation:
  85-100  Strong accumulation
  70-85   Buy / hold
  50-70   Normal DCA
  30-50   Defensive
  <30     Risk-off
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np

from .config import SCORE_WEIGHTS, BTC_HALVINGS
from .database import Database

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _score_label(score: float) -> str:
    if score >= 85:
        return "Strong accumulation"
    if score >= 70:
        return "Buy / hold"
    if score >= 50:
        return "Normal DCA"
    if score >= 30:
        return "Defensive"
    return "Risk-off"


def _days_since_last_halving() -> int:
    today = date.today()
    past = [h for h in BTC_HALVINGS if h <= today]
    if not past:
        return 0
    return (today - past[-1]).days


# ── Component scorers ──────────────────────────────────────────────────────────

def trend_score(db: Database) -> float:
    """25% weight — EMA alignment, MACD, market structure."""
    ind = db.get_latest_indicators("BTC", "1D")
    if not ind:
        return 50.0

    price = (db.get_price_history("BTC", "1D", 1) or [{}])[-1].get("close", 0)
    if not price:
        return 50.0

    e50  = ind.get("ema50")  or price
    e100 = ind.get("ema100") or price
    e200 = ind.get("ema200") or price

    # EMA alignment (0–50)
    if price > e50 > e100 > e200:
        ema_pts = 50
    elif price > e200 and e50 > e200:
        ema_pts = 35
    elif price > e200:
        ema_pts = 25
    elif price < e200 and e50 < e200:
        ema_pts = 10
    else:
        ema_pts = 18

    # MACD (0–30)
    macd_pts = 0
    macd_h = ind.get("macd_hist")
    macd_v = ind.get("macd")
    if macd_h is not None and macd_v is not None:
        if macd_h > 0 and macd_v > 0:
            macd_pts = 30
        elif macd_h > 0:
            macd_pts = 20
        elif macd_v > 0:
            macd_pts = 15
        else:
            macd_pts = 5

    # ADX trend strength (0–20)
    adx_val = ind.get("adx") or 15.0
    if adx_val >= 40:
        adx_pts = 20
    elif adx_val >= 25:
        adx_pts = 15
    elif adx_val >= 15:
        adx_pts = 8
    else:
        adx_pts = 3

    return _clamp(ema_pts + macd_pts + adx_pts)


def momentum_score(db: Database) -> float:
    """15% weight — RSI, volume, relative strength."""
    ind = db.get_latest_indicators("BTC", "1D")
    if not ind:
        return 50.0

    rsi_val = ind.get("rsi") or 50.0

    # RSI score (0–50): oversold is good for long-term buyers
    if rsi_val < 30:
        rsi_pts = 45   # deeply oversold — strong buying opportunity
    elif rsi_val < 40:
        rsi_pts = 50   # oversold
    elif rsi_val < 50:
        rsi_pts = 40   # mild weakness
    elif rsi_val < 60:
        rsi_pts = 50   # neutral-bullish
    elif rsi_val < 70:
        rsi_pts = 45   # strong momentum
    elif rsi_val < 80:
        rsi_pts = 35   # overbought — caution
    else:
        rsi_pts = 15   # extreme overbought

    # Volume vs MA (0–35)
    vol_ma = ind.get("volume_ma") or 0
    price_rows = db.get_price_history("BTC", "1D", 1)
    curr_vol = price_rows[-1]["volume"] if price_rows else 0
    if vol_ma > 0:
        ratio = curr_vol / vol_ma
        if ratio >= 2.0:
            vol_pts = 35
        elif ratio >= 1.5:
            vol_pts = 28
        elif ratio >= 1.0:
            vol_pts = 20
        elif ratio >= 0.7:
            vol_pts = 12
        else:
            vol_pts = 5
    else:
        vol_pts = 15

    # Relative strength of ETH vs BTC (0–15)
    btc_ind = db.get_indicator_history("BTC", "1D", 7)
    eth_ind = db.get_indicator_history("ETH", "1D", 7)
    rs_pts = 10  # neutral default
    if len(btc_ind) >= 2 and len(eth_ind) >= 2:
        btc_rows = db.get_price_history("BTC", "1D", 7)
        eth_rows = db.get_price_history("ETH", "1D", 7)
        if len(btc_rows) >= 2 and len(eth_rows) >= 2:
            btc_chg = btc_rows[-1]["close"] / btc_rows[0]["close"] - 1
            eth_chg = eth_rows[-1]["close"] / eth_rows[0]["close"] - 1
            if eth_chg > btc_chg:  # alts outperforming → alt season signal
                rs_pts = 15
            elif btc_chg > 0:
                rs_pts = 10
            else:
                rs_pts = 5

    return _clamp(rsi_pts + vol_pts + rs_pts)


def cycle_score(db: Database) -> float:
    """20% weight — BTC halving cycle phase + market cap expansion."""
    days = _days_since_last_halving()

    # Halving cycle phase scoring
    if days <= 90:
        cycle_pts = 65     # just after halving — early accumulation
    elif days <= 365:
        cycle_pts = 82     # first year post-halving — historically strong bull
    elif days <= 548:
        cycle_pts = 90     # ~18 months — peak bull window
    elif days <= 730:
        cycle_pts = 72     # second year — late bull, caution
    elif days <= 912:
        cycle_pts = 50     # distribution / top
    elif days <= 1280:
        cycle_pts = 30     # deep bear
    else:
        cycle_pts = 58     # pre-halving accumulation

    # Market cap expansion (0–20 bonus)
    md = db.get_market_data_history(14)
    cap_pts = 10  # neutral
    if len(md) >= 2:
        old_cap = md[0].get("total_market_cap") or 0
        new_cap = md[-1].get("total_market_cap") or 0
        if old_cap and new_cap:
            expansion = (new_cap / old_cap - 1)
            if expansion > 0.10:
                cap_pts = 20
            elif expansion > 0.05:
                cap_pts = 15
            elif expansion > 0:
                cap_pts = 10
            elif expansion > -0.05:
                cap_pts = 5
            else:
                cap_pts = 0

    return _clamp(cycle_pts * 0.8 + cap_pts)


def liquidity_score(db: Database) -> float:
    """15% weight — volume vs MA, stablecoin supply as dry powder."""
    # Volume ratio (0–60)
    ind = db.get_latest_indicators("BTC", "1D")
    if not ind:
        return 50.0

    vol_ma  = ind.get("volume_ma") or 0
    rows    = db.get_price_history("BTC", "1D", 1)
    curr_vol = rows[-1]["volume"] if rows else 0

    if vol_ma > 0:
        ratio = curr_vol / vol_ma
        if ratio >= 2.0:
            vol_pts = 60
        elif ratio >= 1.5:
            vol_pts = 50
        elif ratio >= 1.0:
            vol_pts = 40
        elif ratio >= 0.7:
            vol_pts = 25
        else:
            vol_pts = 10
    else:
        vol_pts = 30

    # Stablecoin supply (0–40): high stablecoin % = dry powder = bullish
    md = db.get_latest_market_data() or {}
    stable_cap = md.get("stablecoin_market_cap") or 0
    total_cap  = md.get("total_market_cap") or 1
    stable_pct = stable_cap / total_cap if total_cap > 0 else 0

    if stable_pct >= 0.15:
        stable_pts = 40   # lots of dry powder
    elif stable_pct >= 0.10:
        stable_pts = 30
    elif stable_pct >= 0.07:
        stable_pts = 20
    else:
        stable_pts = 10

    return _clamp(vol_pts + stable_pts)


def sentiment_score(db: Database) -> float:
    """10% weight — Fear & Greed, funding rate."""
    sent = db.get_latest_sentiment() or {}

    fg    = sent.get("fear_greed_index")
    fund  = sent.get("btc_funding_rate")   # decimal, e.g. 0.01 = 1%

    # Fear & Greed (0–60): extreme fear is buy signal
    if fg is not None:
        if fg <= 20:
            fg_pts = 60   # extreme fear — max opportunity
        elif fg <= 35:
            fg_pts = 50   # fear — good accumulation
        elif fg <= 55:
            fg_pts = 35   # neutral
        elif fg <= 75:
            fg_pts = 20   # greed — caution
        else:
            fg_pts = 5    # extreme greed — sell pressure
    else:
        fg_pts = 30       # no data — neutral

    # Funding rate (0–40): negative funding = shorts paying longs (bullish)
    if fund is not None:
        if fund < -0.01:
            fund_pts = 40   # negative funding — very bullish
        elif fund < 0:
            fund_pts = 35
        elif fund < 0.01:
            fund_pts = 25   # near-zero — neutral
        elif fund < 0.03:
            fund_pts = 15   # mild positive — slight caution
        else:
            fund_pts = 5    # high funding — longs crowded
    else:
        fund_pts = 20

    # Proxy: derive from RSI when external data unavailable
    ind = db.get_latest_indicators("BTC", "1D")
    if fg is None and ind:
        rsi_val = ind.get("rsi") or 50
        fg_proxy = 100 - rsi_val   # low RSI ≈ high fear
        if fg_proxy >= 70:
            fg_pts = 55
        elif fg_proxy >= 55:
            fg_pts = 40
        elif fg_proxy >= 45:
            fg_pts = 30
        else:
            fg_pts = 15

    return _clamp(fg_pts + fund_pts)


def risk_score(db: Database) -> float:
    """15% weight — volatility (ATR%), drawdown from recent high."""
    ind = db.get_latest_indicators("BTC", "1D")
    if not ind:
        return 50.0

    atr_pct = ind.get("atr_pct") or 2.0

    # Volatility (ATR as % of price) — lower is safer (0–50)
    if atr_pct <= 1.5:
        atr_pts = 50
    elif atr_pct <= 2.5:
        atr_pts = 40
    elif atr_pct <= 4.0:
        atr_pts = 30
    elif atr_pct <= 6.0:
        atr_pts = 18
    elif atr_pct <= 8.0:
        atr_pts = 8
    else:
        atr_pts = 2    # extreme — crash conditions

    # Drawdown from 90-day high (0–50)
    history = db.get_price_history("BTC", "1D", 90)
    dd_pts = 30  # neutral
    if history:
        highs = [r["high"] for r in history]
        curr  = history[-1]["close"]
        peak  = max(highs) if highs else curr
        dd_pct = (curr / peak - 1) * 100 if peak else 0

        if dd_pct >= -5:
            dd_pts = 35   # near ATH — watch for tops
        elif dd_pct >= -15:
            dd_pts = 45   # healthy pullback
        elif dd_pct >= -30:
            dd_pts = 50   # significant correction — opportunity
        elif dd_pct >= -50:
            dd_pts = 40   # deep drawdown — risk but value
        else:
            dd_pts = 20   # >50% drawdown — bear market

    return _clamp(atr_pts + dd_pts)


# ── Main scorer ────────────────────────────────────────────────────────────────

def calculate_score(db: Database, regime: str | None = None) -> dict:
    """
    Calculate the composite AI score and all components.

    Returns:
        {
          'total_score': float,
          'trend_score': float, 'momentum_score': float, 'cycle_score': float,
          'liquidity_score': float, 'sentiment_score': float, 'risk_score': float,
          'market_regime': str,
          'score_label': str,
        }
    """
    scores = {
        "trend_score":     trend_score(db),
        "momentum_score":  momentum_score(db),
        "cycle_score":     cycle_score(db),
        "liquidity_score": liquidity_score(db),
        "sentiment_score": sentiment_score(db),
        "risk_score":      risk_score(db),
    }

    w = SCORE_WEIGHTS
    total = (
        scores["trend_score"]     * w["trend"]     +
        scores["momentum_score"]  * w["momentum"]  +
        scores["cycle_score"]     * w["cycle"]      +
        scores["liquidity_score"] * w["liquidity"]  +
        scores["sentiment_score"] * w["sentiment"]  +
        scores["risk_score"]      * w["risk"]
    )
    total = _clamp(total)

    label   = _score_label(total)
    regime_ = regime or "UNKNOWN"

    result = {
        "total_score":     round(total, 1),
        "market_regime":   regime_,
        "score_label":     label,
        **{k: round(v, 1) for k, v in scores.items()},
    }

    db.upsert_score(result)
    log.info(
        "AI Score: %.1f (%s) | T=%.0f M=%.0f C=%.0f L=%.0f S=%.0f R=%.0f",
        total, label,
        scores["trend_score"], scores["momentum_score"], scores["cycle_score"],
        scores["liquidity_score"], scores["sentiment_score"], scores["risk_score"],
    )
    return result
