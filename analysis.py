"""
analysis.py — Market analysis + trade performance review

Two entry points:
  market_report()          Called at 10:30 AM and 10:30 PM (UTC+8)
  trade_review(trades)     Called by strategy.py every 5 closed trades
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Re-use indicator helpers from strategy
from strategy import fetch_bars, ema, rsi, vwap, get_trend, is_market_active, SYMBOLS

load_dotenv()
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

SGT = timezone(timedelta(hours=8))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    hl = df["high"] - df["low"]
    return float(hl.rolling(period).mean().iloc[-1])


def _pct_change_24h(df_1h: pd.DataFrame) -> float:
    if len(df_1h) < 24:
        return float("nan")
    return (df_1h["close"].iloc[-1] / df_1h["close"].iloc[-24] - 1) * 100


def _key_levels(df_1h: pd.DataFrame, lookback: int = 24) -> tuple[float, float]:
    recent = df_1h.tail(lookback)
    return float(recent["high"].max()), float(recent["low"].min())


def _vol_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    avg = df["volume"].rolling(lookback).mean().iloc[-1]
    cur = df["volume"].iloc[-1]
    return float(cur / avg) if avg > 0 else float("nan")


# ── Per-symbol snapshot ───────────────────────────────────────────────────────

def symbol_snapshot(symbol: str) -> dict:
    """Fetch all timeframes and return a structured analysis dict."""
    try:
        df_1h  = fetch_bars(symbol, TimeFrame.Hour, 60)
        df_15m = fetch_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), 60)
        df_5m  = fetch_bars(symbol, TimeFrame(5,  TimeFrameUnit.Minute), 80)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

    close_1h = df_1h["close"]
    close_15m = df_15m["close"]
    close_5m  = df_5m["close"]

    e20_1h = ema(close_1h, 20)
    e50_1h = ema(close_1h, 50)
    e20_5m = ema(close_5m, 20)

    price       = float(close_1h.iloc[-1])
    trend       = get_trend(df_1h)
    rsi_1h      = float(rsi(close_1h).iloc[-1])
    rsi_15m     = float(rsi(close_15m).iloc[-1])
    rsi_5m      = float(rsi(close_5m).iloc[-1])
    vwap_5m     = float(vwap(df_5m).iloc[-1])
    atr_1h      = _atr(df_1h)
    atr_pct     = atr_1h / price * 100
    chg_24h     = _pct_change_24h(df_1h)
    hi_24h, lo_24h = _key_levels(df_1h, 24)
    vol_r_1h    = _vol_ratio(df_1h)
    vol_r_5m    = _vol_ratio(df_5m)
    active      = is_market_active(df_5m)
    vwap_side   = "above" if price > vwap_5m else "below"

    # Distance from EMA20 / EMA50
    e20_dist_pct = (price / e20_1h.iloc[-1] - 1) * 100
    e50_dist_pct = (price / e50_1h.iloc[-1] - 1) * 100

    # Session outlook
    reasons: list[str] = []
    score = 0

    if trend == "bullish":
        score += 2
    elif trend == "bearish":
        score += 2
    else:
        score -= 2
        reasons.append("messy trend — EMAs not aligned")

    if active:
        score += 1
    else:
        score -= 1
        reasons.append("low volume / low ATR")

    if atr_pct < 0.3:
        score -= 1
        reasons.append(f"ATR only {atr_pct:.2f}% — low volatility")

    if trend == "bullish" and price < vwap_5m:
        score -= 1
        reasons.append("bullish trend but price below VWAP")
    elif trend == "bearish" and price > vwap_5m:
        score -= 1
        reasons.append("bearish trend but price above VWAP")
    else:
        score += 1

    if 40 < rsi_1h < 60:
        score -= 1
        reasons.append("RSI mid-range on 1H — no clear momentum")
    elif (trend == "bullish" and rsi_1h > 50) or (trend == "bearish" and rsi_1h < 50):
        score += 1

    if score >= 3:
        outlook = "FAVORABLE"
    elif score >= 1:
        outlook = "CAUTION"
    else:
        outlook = "AVOID"

    return {
        "symbol": symbol,
        "price": price,
        "change_24h_pct": chg_24h,
        "trend_1h": trend,
        "ema20_1h": float(e20_1h.iloc[-1]),
        "ema50_1h": float(e50_1h.iloc[-1]),
        "ema20_dist_pct": e20_dist_pct,
        "ema50_dist_pct": e50_dist_pct,
        "rsi_1h": rsi_1h,
        "rsi_15m": rsi_15m,
        "rsi_5m": rsi_5m,
        "vwap_5m": vwap_5m,
        "vwap_side": vwap_side,
        "atr_1h": atr_1h,
        "atr_pct": atr_pct,
        "high_24h": hi_24h,
        "low_24h": lo_24h,
        "vol_ratio_1h": vol_r_1h,
        "vol_ratio_5m": vol_r_5m,
        "active": active,
        "outlook": outlook,
        "caution_reasons": reasons,
        "score": score,
    }


# ── Market report ─────────────────────────────────────────────────────────────

def market_report() -> None:
    """Full market analysis — call at 10:30 AM and 10:30 PM SGT."""
    now_sgt = datetime.now(SGT)
    session = "AM" if now_sgt.hour < 12 else "PM"
    header = f"MARKET ANALYSIS  {now_sgt.strftime('%Y-%m-%d %H:%M')} SGT ({session})"
    divider = "=" * 64

    lines: list[str] = [divider, header, divider, ""]

    for symbol in SYMBOLS:
        snap = symbol_snapshot(symbol)

        if "error" in snap:
            lines.append(f"  {symbol}: data error — {snap['error']}")
            lines.append("")
            continue

        chg_str = f"{snap['change_24h_pct']:+.2f}%" if not np.isnan(snap["change_24h_pct"]) else "n/a"
        lines += [
            f"  {symbol}",
            f"  {'─'*40}",
            f"  Price      : ${snap['price']:,.2f}   (24h {chg_str})",
            f"  Trend 1H   : {snap['trend_1h'].upper()}",
            f"  EMA20 1H   : ${snap['ema20_1h']:,.2f}  ({snap['ema20_dist_pct']:+.2f}% from price)",
            f"  EMA50 1H   : ${snap['ema50_1h']:,.2f}  ({snap['ema50_dist_pct']:+.2f}% from price)",
            f"  RSI        : 1H={snap['rsi_1h']:.1f}  15m={snap['rsi_15m']:.1f}  5m={snap['rsi_5m']:.1f}",
            f"  VWAP 5m    : ${snap['vwap_5m']:,.2f}  (price {snap['vwap_side']} VWAP)",
            f"  ATR 1H     : ${snap['atr_1h']:.2f}  ({snap['atr_pct']:.2f}% of price)",
            f"  24h Range  : ${snap['low_24h']:,.2f} — ${snap['high_24h']:,.2f}",
            f"  Volume     : 1H×{snap['vol_ratio_1h']:.1f}  5m×{snap['vol_ratio_5m']:.1f}  "
            f"({'ACTIVE' if snap['active'] else 'QUIET'})",
            f"",
            f"  Outlook    : *** {snap['outlook']} ***",
        ]
        if snap["caution_reasons"]:
            lines.append(f"  Caution    : {'; '.join(snap['caution_reasons'])}")
        lines.append("")

    lines += [divider, ""]

    report_text = "\n".join(lines)
    print(report_text)

    # Save to file
    fname = REPORTS_DIR / f"market_{now_sgt.strftime('%Y%m%d_%H%M')}.txt"
    fname.write_text(report_text)


# ── Trade performance review ──────────────────────────────────────────────────

def trade_review(trades: list[dict], trigger_count: int) -> None:
    """
    Analyse the last block of trades.
    Prints and saves a report every 5 closed trades.
    `trigger_count` is the running total of closed trades (used for labelling).
    """
    if not trades:
        return

    n = len(trades)
    wins   = [t for t in trades if t.get("outcome") == "win"]
    losses = [t for t in trades if t.get("outcome") == "loss"]
    r_vals = [t["r_multiple"] for t in trades if "r_multiple" in t]
    pnl    = [t["pnl_pct"] * 100 for t in trades if "pnl_pct" in t]

    win_rate    = len(wins) / n * 100 if n else 0
    avg_r       = float(np.mean(r_vals)) if r_vals else 0
    total_pnl   = float(np.sum(pnl)) if pnl else 0
    avg_win_r   = float(np.mean([t["r_multiple"] for t in wins])) if wins else 0
    avg_loss_r  = float(np.mean([t["r_multiple"] for t in losses])) if losses else 0

    best  = max(trades, key=lambda t: t.get("r_multiple", -99))
    worst = min(trades, key=lambda t: t.get("r_multiple", 99))

    # Breakdown by symbol
    by_sym: dict[str, list] = {}
    for t in trades:
        by_sym.setdefault(t.get("symbol", "?"), []).append(t)

    # Breakdown by side
    longs  = [t for t in trades if t.get("side") == "long"]
    shorts = [t for t in trades if t.get("side") == "short"]

    def _wr(subset: list) -> str:
        if not subset:
            return "—"
        w = sum(1 for t in subset if t.get("outcome") == "win")
        return f"{w}/{len(subset)}  ({w/len(subset)*100:.0f}%)"

    now_sgt = datetime.now(SGT)
    title = f"TRADE REVIEW  (trades #{trigger_count - n + 1}–#{trigger_count})"
    divider = "─" * 56

    lines: list[str] = [
        "",
        "=" * 56,
        title,
        f"  Generated: {now_sgt.strftime('%Y-%m-%d %H:%M')} SGT",
        "=" * 56,
        "",
        f"  Trades analysed : {n}",
        f"  Win rate        : {win_rate:.0f}%  ({len(wins)}W / {len(losses)}L)",
        f"  Average R       : {avg_r:+.2f}R",
        f"  Total P&L       : {total_pnl:+.2f}%",
        f"  Avg win R       : {avg_win_r:+.2f}R",
        f"  Avg loss R      : {avg_loss_r:+.2f}R",
        "",
        divider,
        "  By side",
        divider,
        f"  Long  W/L : {_wr(longs)}",
        f"  Short W/L : {_wr(shorts)}",
        "",
        divider,
        "  By symbol",
        divider,
    ]

    for sym, ts in by_sym.items():
        lines.append(f"  {sym:<10} {_wr(ts)}  avg R {np.mean([t.get('r_multiple',0) for t in ts]):+.2f}")

    lines += [
        "",
        divider,
        "  Best / Worst",
        divider,
        f"  Best : {best.get('symbol')} {best.get('side')}  "
        f"entry={best.get('entry',0):.2f}  exit={best.get('exit',0):.2f}  "
        f"R={best.get('r_multiple',0):+.2f}",
        f"  Worst: {worst.get('symbol')} {worst.get('side')}  "
        f"entry={worst.get('entry',0):.2f}  exit={worst.get('exit',0):.2f}  "
        f"R={worst.get('r_multiple',0):+.2f}",
        "",
        divider,
        "  Trade log",
        divider,
    ]

    for i, t in enumerate(trades, 1):
        outcome_tag = "WIN " if t.get("outcome") == "win" else "LOSS"
        lines.append(
            f"  {i:>2}. [{outcome_tag}] {t.get('symbol','?'):>8} {t.get('side','?'):<5}"
            f"  entry={t.get('entry',0):>10.2f}"
            f"  exit={t.get('exit',0):>10.2f}"
            f"  R={t.get('r_multiple',0):>+6.2f}"
            f"  {t.get('closed_at','')[:16]}"
        )

    # Pattern notes
    lines += ["", divider, "  Patterns & notes", divider]
    if win_rate < 40:
        lines.append("  ⚠ Win rate below 40% — review entry conditions (RSI filter, VWAP alignment)")
    if avg_r > 0 and win_rate < 50:
        lines.append("  ✓ Positive avg R despite sub-50% win rate — good risk/reward discipline")
    if avg_r < 0:
        lines.append("  ⚠ Negative avg R — consider tighter stop placement or raising TP target")
    long_wr  = len([t for t in longs  if t.get("outcome")=="win"]) / len(longs)  if longs  else 0
    short_wr = len([t for t in shorts if t.get("outcome")=="win"]) / len(shorts) if shorts else 0
    if longs and shorts:
        better = "longs" if long_wr >= short_wr else "shorts"
        lines.append(f"  → {better.capitalize()} performing better this period — lean toward {better}")
    if not lines[-1].startswith("  "):
        lines.append("  No notable patterns detected.")
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    fname = REPORTS_DIR / f"trades_{now_sgt.strftime('%Y%m%d_%H%M')}_#{trigger_count}.txt"
    fname.write_text(report_text)


# ── Strategy deep review (every 50 trades) ───────────────────────────────────

def _bucket_stats(trades: list[dict], key: str, buckets: list[tuple]) -> list[dict]:
    """
    Split trades into buckets based on a numeric field and return win-rate stats.
    buckets = [(label, lo, hi), ...]  where lo <= value < hi
    """
    results = []
    for label, lo, hi in buckets:
        subset = [t for t in trades if lo <= (t.get(key) or 0) < hi]
        if not subset:
            continue
        wins = sum(1 for t in subset if t.get("outcome") == "win")
        avg_r = float(np.mean([t["r_multiple"] for t in subset if "r_multiple" in t])) if subset else 0
        results.append({
            "label": label,
            "n": len(subset),
            "wins": wins,
            "wr": wins / len(subset) * 100,
            "avg_r": avg_r,
        })
    return results


def _wr_line(label: str, subset: list[dict], flag: str = "") -> str:
    if not subset:
        return f"  {label:<28} —"
    wins = sum(1 for t in subset if t.get("outcome") == "win")
    wr = wins / len(subset) * 100
    avg_r = float(np.mean([t.get("r_multiple", 0) for t in subset]))
    bar_len = int(wr / 5)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    return f"  {label:<28} {bar}  {wr:5.1f}%  avg {avg_r:+.2f}R  (n={len(subset)})  {flag}"


def strategy_deep_review(all_trades: list[dict], trigger_count: int) -> None:
    """
    Comprehensive strategy breakdown every 50 trades.
    Shows what's working and what isn't across symbols, sides,
    RSI ranges, volume conditions, ATR conditions, and trend context.
    """
    if not all_trades:
        return

    now_sgt = datetime.now(SGT)
    n = len(all_trades)
    wins = [t for t in all_trades if t.get("outcome") == "win"]
    losses = [t for t in all_trades if t.get("outcome") == "loss"]
    overall_wr = len(wins) / n * 100
    overall_r = float(np.mean([t.get("r_multiple", 0) for t in all_trades]))

    divider_h = "═" * 64
    divider   = "─" * 64

    lines: list[str] = [
        "",
        divider_h,
        f" STRATEGY DEEP REVIEW  —  Trades #1–#{trigger_count}",
        f" Generated: {now_sgt.strftime('%Y-%m-%d %H:%M')} SGT",
        divider_h,
        "",
        f"  Total trades : {n}",
        f"  Win rate     : {overall_wr:.1f}%  ({len(wins)}W / {len(losses)}L)",
        f"  Avg R        : {overall_r:+.2f}R",
        f"  Total P&L    : {sum(t.get('pnl_pct',0) for t in all_trades)*100:+.2f}%",
        "",
    ]

    # ── By symbol ──
    lines += [divider, "  BY SYMBOL", divider]
    by_sym: dict[str, list] = {}
    for t in all_trades:
        by_sym.setdefault(t.get("symbol", "?"), []).append(t)
    sym_avg_r = {s: float(np.mean([t.get("r_multiple", 0) for t in ts])) for s, ts in by_sym.items()}
    best_sym = max(sym_avg_r, key=sym_avg_r.get) if sym_avg_r else None
    worst_sym = min(sym_avg_r, key=sym_avg_r.get) if sym_avg_r else None
    for sym, ts in sorted(by_sym.items()):
        flag = "← BEST" if sym == best_sym else ("← WEAKEST" if sym == worst_sym else "")
        lines.append(_wr_line(sym, ts, flag))
    lines.append("")

    # ── By side ──
    lines += [divider, "  BY DIRECTION", divider]
    longs  = [t for t in all_trades if t.get("side") == "long"]
    shorts = [t for t in all_trades if t.get("side") == "short"]
    lines.append(_wr_line("Longs", longs))
    lines.append(_wr_line("Shorts", shorts))
    lines.append("")

    # ── By trend ──
    lines += [divider, "  BY TREND AT ENTRY", divider]
    for trend_val in ("bullish", "bearish"):
        subset = [t for t in all_trades if t.get("trend_1h") == trend_val]
        lines.append(_wr_line(f"Trend {trend_val}", subset))
    lines.append("")

    # ── By RSI at entry ──
    lines += [divider, "  BY RSI AT ENTRY", divider]
    rsi_buckets = [
        ("RSI 30–40", 30, 40),
        ("RSI 40–50", 40, 50),
        ("RSI 50–55", 50, 55),
        ("RSI 55–60", 55, 60),
        ("RSI 60–65", 60, 65),
        ("RSI 65–70", 65, 70),
        ("RSI 70–80", 70, 80),
    ]
    rsi_stats = _bucket_stats(all_trades, "entry_rsi", rsi_buckets)
    best_rsi  = max(rsi_stats, key=lambda x: x["wr"]) if rsi_stats else None
    worst_rsi = min(rsi_stats, key=lambda x: x["wr"]) if rsi_stats else None
    for s in rsi_stats:
        flag = "← BEST" if s == best_rsi else ("← WORST — AVOID" if s == worst_rsi else "")
        subset = [t for t in all_trades if rsi_buckets[rsi_stats.index(s)][1] <= (t.get("entry_rsi") or 0) < rsi_buckets[rsi_stats.index(s)][2]]
        lines.append(_wr_line(s["label"], subset, flag))
    lines.append("")

    # ── By volume ratio at entry ──
    lines += [divider, "  BY VOLUME RATIO AT ENTRY  (breakout vol ÷ avg)", divider]
    vol_buckets = [
        ("Vol < 1.2×", 0, 1.2),
        ("Vol 1.2–1.5×", 1.2, 1.5),
        ("Vol 1.5–2.0×", 1.5, 2.0),
        ("Vol 2.0–3.0×", 2.0, 3.0),
        ("Vol > 3.0×",   3.0, 99),
    ]
    vol_stats = _bucket_stats(all_trades, "entry_vol_ratio", vol_buckets)
    best_vol  = max(vol_stats, key=lambda x: x["wr"]) if vol_stats else None
    for s in vol_stats:
        flag = "← BEST" if s == best_vol else ""
        lo, hi = next((b[1], b[2]) for b in vol_buckets if b[0] == s["label"])
        subset = [t for t in all_trades if lo <= (t.get("entry_vol_ratio") or 0) < hi]
        lines.append(_wr_line(s["label"], subset, flag))
    lines.append("")

    # ── By ATR% at entry ──
    lines += [divider, "  BY VOLATILITY (ATR%) AT ENTRY", divider]
    atr_buckets = [
        ("ATR < 0.2%  (very low)", 0,   0.2),
        ("ATR 0.2–0.4%",           0.2, 0.4),
        ("ATR 0.4–0.6%",           0.4, 0.6),
        ("ATR 0.6–1.0%",           0.6, 1.0),
        ("ATR > 1.0%  (high vol)", 1.0, 99),
    ]
    atr_stats = _bucket_stats(all_trades, "entry_atr_pct", atr_buckets)
    best_atr  = max(atr_stats, key=lambda x: x["wr"]) if atr_stats else None
    for s in atr_stats:
        flag = "← BEST" if s == best_atr else ""
        lo, hi = next((b[1], b[2]) for b in atr_buckets if b[0] == s["label"])
        subset = [t for t in all_trades if lo <= (t.get("entry_atr_pct") or 0) < hi]
        lines.append(_wr_line(s["label"], subset, flag))
    lines.append("")

    # ── Winning vs losing trade fingerprint ──
    lines += [divider, "  WINNER vs LOSER FINGERPRINT", divider]

    def _avg(lst: list[dict], key: str) -> str:
        vals = [t[key] for t in lst if t.get(key) is not None]
        return f"{float(np.mean(vals)):.2f}" if vals else "n/a"

    lines += [
        f"  {'Metric':<28} {'WINNERS':>12}  {'LOSERS':>12}",
        f"  {'─'*28} {'─'*12}  {'─'*12}",
        f"  {'Avg RSI at entry':<28} {_avg(wins,'entry_rsi'):>12}  {_avg(losses,'entry_rsi'):>12}",
        f"  {'Avg volume ratio':<28} {_avg(wins,'entry_vol_ratio'):>12}  {_avg(losses,'entry_vol_ratio'):>12}",
        f"  {'Avg ATR% at entry':<28} {_avg(wins,'entry_atr_pct'):>12}  {_avg(losses,'entry_atr_pct'):>12}",
        f"  {'Avg EMA20 dist%':<28} {_avg(wins,'entry_ema20_dist_pct'):>12}  {_avg(losses,'entry_ema20_dist_pct'):>12}",
        "",
    ]

    # ── Actionable recommendations ──
    lines += [divider, "  RECOMMENDATIONS", divider]
    recs: list[str] = []

    long_wr  = len([t for t in longs  if t.get("outcome")=="win"]) / len(longs)  if longs  else 0
    short_wr = len([t for t in shorts if t.get("outcome")=="win"]) / len(shorts) if shorts else 0
    if abs(long_wr - short_wr) > 0.15:
        better = "longs" if long_wr > short_wr else "shorts"
        worse  = "shorts" if long_wr > short_wr else "longs"
        recs.append(f"→ {better.capitalize()} outperforming {worse} — consider only trading {better} until {worse} win rate recovers")

    for sym, ts in by_sym.items():
        sym_wr = sum(1 for t in ts if t.get("outcome")=="win") / len(ts)
        if sym_wr < 0.4 and len(ts) >= 10:
            recs.append(f"→ {sym} win rate {sym_wr*100:.0f}% on {len(ts)} trades — consider pausing this symbol")

    if best_rsi and worst_rsi and best_rsi != worst_rsi:
        recs.append(f"→ Best RSI zone: {best_rsi['label']} ({best_rsi['wr']:.0f}% WR) — tighten entry filter to this range")
        if worst_rsi["wr"] < 45:
            recs.append(f"→ Worst RSI zone: {worst_rsi['label']} ({worst_rsi['wr']:.0f}% WR) — SKIP trades in this RSI range")

    if best_vol:
        recs.append(f"→ Best volume condition: {best_vol['label']} ({best_vol['wr']:.0f}% WR) — prioritise entries with this volume")

    if best_atr:
        recs.append(f"→ Best volatility condition: {best_atr['label']} ({best_atr['wr']:.0f}% WR) — focus on these ATR conditions")

    if overall_r < 0:
        recs.append("→ Negative avg R — stop loss may be too tight or TP too far; consider reducing TP_RATIO to 1.2R")
    elif overall_r > 1.2:
        recs.append("→ Strong avg R — strategy is executing well; maintain current parameters")

    if not recs:
        recs.append("→ No clear pattern edges yet — continue collecting data")

    for r in recs:
        lines.append(f"  {r}")

    lines += ["", divider_h, ""]

    report_text = "\n".join(lines)
    print(report_text)

    fname = REPORTS_DIR / f"deep_review_{now_sgt.strftime('%Y%m%d_%H%M')}_#{trigger_count}.txt"
    fname.write_text(report_text)


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    market_report()
