"""
Post-Trade Analysis — runs after N trades have been executed.

Reads the SQLite DB + live Alpaca positions and produces:
  1. Trade log with P&L per position
  2. Regime accuracy (did regime calls match subsequent price action?)
  3. Score calibration (did high scores lead to better entries?)
  4. Allocation drift (current vs intended)
  5. Specific improvement recommendations

Usage:
    python lt_analyze.py
    python lt_analyze.py --paper-trade   # also fetches live position P&L from Alpaca
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from lt_bot.database import Database
from lt_bot.config import REPORTS_DIR


# ── Helpers ────────────────────────────────────────────────────────────────────

def _divH(): return "═" * 66
def _divL(): return "─" * 66


def _pct(v: float) -> str:
    return f"{v:+.2f}%" if v == v else "n/a"   # nan guard


# ── Section builders ───────────────────────────────────────────────────────────

def section_trade_log(db: Database, live_positions: dict) -> list[str]:
    """All executed buys/sells with live P&L where available."""
    decisions = db.get_recent_decisions(days=365)
    executed  = [d for d in decisions if d.get("action") in ("BUY_EXECUTED", "SELL_EXECUTED")]

    if not executed:
        return [" No executed trades found in database."]

    lines = [
        f"  {'#':<3} {'Time':<12} {'Asset':<6} {'Action':<8} {'USD':>9}"
        f"  {'Entry note':<40} {'Live P&L':>9}",
        f"  {'─'*3} {'─'*12} {'─'*6} {'─'*8} {'─'*9}  {'─'*40} {'─'*9}",
    ]

    total_deployed = 0.0
    total_pnl_usd  = 0.0
    pnl_available  = False

    for i, d in enumerate(executed, 1):
        ticker  = d.get("symbol", "?")
        action  = d.get("action", "")
        usd     = d.get("recommended_usd") or 0.0
        ts      = (d.get("ts") or "")[:16].replace("T", " ")
        reason  = (d.get("reason") or "")[:40]

        pnl_str = "  —"
        if action == "BUY_EXECUTED" and ticker in live_positions:
            pos         = live_positions[ticker]
            entry_cost  = usd   # approximation: we used notional orders
            live_val    = pos.get("market_value", 0)
            unrealised  = pos.get("unrealized_pnl", 0)

            # Better: use Alpaca's unrealized_pnl across the whole position
            pnl_pct_pos = (pos.get("unrealized_pnl", 0) /
                           (pos.get("market_value", 1) - pos.get("unrealized_pnl", 0) + 1e-9)) * 100
            pnl_str     = f"{pnl_pct_pos:+.1f}%"
            total_pnl_usd += unrealised
            pnl_available  = True

        if action == "BUY_EXECUTED":
            total_deployed += usd

        action_short = "BUY" if "BUY" in action else "SELL"
        lines.append(
            f"  {i:<3} {ts:<12} {ticker:<6} {action_short:<8} ${usd:>8,.0f}"
            f"  {reason:<40} {pnl_str:>9}"
        )

    lines += [
        f"  {'─'*66}",
        f"  Total deployed : ${total_deployed:,.2f}",
    ]
    if pnl_available:
        pnl_pct_total = (total_pnl_usd / total_deployed * 100) if total_deployed else 0
        lines.append(f"  Unrealised P&L : ${total_pnl_usd:+,.2f}  ({pnl_pct_total:+.2f}%)")

    return lines


def section_regime_accuracy(db: Database) -> list[str]:
    """
    For each regime call in the score history, check whether BTC price
    moved in the expected direction over the next 7 days.
    """
    scores   = db.get_score_history(limit=60)
    btc_hist = db.get_price_history("BTC", "1D", limit=90)

    if not scores or len(btc_hist) < 8:
        return ["  Insufficient history for regime accuracy analysis (need 8+ days)."]

    # Build a date→close map for BTC
    price_map: dict[str, float] = {r["ts"]: r["close"] for r in btc_hist}
    dates_sorted = sorted(price_map.keys())

    results: dict[str, list[str]] = {}   # regime → list of "correct"/"wrong"

    for entry in scores:
        ts     = (entry.get("ts") or "")[:10]
        regime = entry.get("market_regime", "UNKNOWN")

        # Find BTC price at ts and 7 days later
        if ts not in price_map:
            continue

        idx = dates_sorted.index(ts) if ts in dates_sorted else -1
        if idx < 0 or idx + 7 >= len(dates_sorted):
            continue

        p_entry = price_map[ts]
        p_7d    = price_map[dates_sorted[idx + 7]]
        chg_7d  = (p_7d / p_entry - 1) * 100

        # Was the regime call directionally correct?
        bullish_regimes = {"STRONG_BULL", "BULL"}
        bearish_regimes = {"BEAR", "CRASH"}
        neutral_regimes = {"SIDEWAYS", "EUPHORIA"}

        if regime in bullish_regimes:
            outcome = "correct" if chg_7d > 2 else ("wrong" if chg_7d < -2 else "neutral")
        elif regime in bearish_regimes:
            outcome = "correct" if chg_7d < -2 else ("wrong" if chg_7d > 2 else "neutral")
        else:
            outcome = "neutral"

        results.setdefault(regime, []).append(outcome)

    if not results:
        return ["  No regime→outcome pairs found (need price data after each regime call)."]

    lines = [
        f"  {'Regime':<14} {'Calls':>6} {'Correct':>8} {'Wrong':>7} {'Accuracy':>9}",
        f"  {'─'*14} {'─'*6} {'─'*8} {'─'*7} {'─'*9}",
    ]
    for regime in sorted(results):
        outcomes = results[regime]
        n       = len(outcomes)
        correct = outcomes.count("correct")
        wrong   = outcomes.count("wrong")
        acc     = correct / n * 100 if n else 0
        lines.append(
            f"  {regime:<14} {n:>6} {correct:>8} {wrong:>7} {acc:>8.0f}%"
        )

    return lines


def section_score_calibration(db: Database) -> list[str]:
    """
    Bucket AI scores into bands and compare against subsequent 7-day BTC price change.
    Tells us whether high scores genuinely precede better outcomes.
    """
    scores   = db.get_score_history(limit=60)
    btc_hist = db.get_price_history("BTC", "1D", limit=90)

    if len(scores) < 5 or len(btc_hist) < 8:
        return ["  Need 5+ score records and 8+ days of price history for calibration."]

    price_map    = {r["ts"]: r["close"] for r in btc_hist}
    dates_sorted = sorted(price_map.keys())

    bands: dict[str, list[float]] = {
        "< 50":    [],
        "50 – 70": [],
        "70 – 85": [],
        "85+":     [],
    }

    for entry in scores:
        ts    = (entry.get("ts") or "")[:10]
        score = entry.get("total_score") or 0

        if ts not in price_map:
            continue
        idx = dates_sorted.index(ts) if ts in dates_sorted else -1
        if idx < 0 or idx + 7 >= len(dates_sorted):
            continue

        chg_7d = (price_map[dates_sorted[idx + 7]] / price_map[ts] - 1) * 100

        if score >= 85:      bands["85+"].append(chg_7d)
        elif score >= 70:    bands["70 – 85"].append(chg_7d)
        elif score >= 50:    bands["50 – 70"].append(chg_7d)
        else:                bands["< 50"].append(chg_7d)

    lines = [
        f"  {'Score band':<12} {'Samples':>8} {'Avg 7d':>8} {'Med 7d':>8} {'Win%':>7}",
        f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*7}",
    ]
    for band, changes in bands.items():
        if not changes:
            lines.append(f"  {band:<12} {'—':>8}")
            continue
        avg  = np.mean(changes)
        med  = np.median(changes)
        wins = sum(1 for c in changes if c > 0) / len(changes) * 100
        lines.append(
            f"  {band:<12} {len(changes):>8} {avg:>+7.1f}% {med:>+7.1f}% {wins:>6.0f}%"
        )

    return lines


def section_live_positions(live_positions: dict, db: Database) -> list[str]:
    """Current Alpaca paper positions with entry, current price and P&L."""
    if not live_positions:
        return ["  No open positions (portfolio 100% in cash/stablecoin)."]

    lines = [
        f"  {'Asset':<6} {'Qty':>12} {'Avg entry':>12} {'Current':>12}"
        f" {'Mkt value':>11} {'P&L $':>9} {'P&L%':>7}",
        f"  {'─'*6} {'─'*12} {'─'*12} {'─'*12} {'─'*11} {'─'*9} {'─'*7}",
    ]

    total_val = 0.0
    total_pnl = 0.0

    for ticker, pos in sorted(live_positions.items()):
        qty    = pos["qty"]
        entry  = pos["avg_entry"]
        curr   = pos["current_price"]
        mval   = pos["market_value"]
        upnl   = pos["unrealized_pnl"]
        pnl_pct = (upnl / (mval - upnl + 1e-9)) * 100

        total_val += mval
        total_pnl += upnl

        lines.append(
            f"  {ticker:<6} {qty:>12.6f} ${entry:>11,.2f} ${curr:>11,.2f}"
            f" ${mval:>10,.2f} ${upnl:>+8,.2f} {pnl_pct:>+6.1f}%"
        )

    lines += [
        f"  {'─'*66}",
        f"  {'TOTAL':<6} {'':>12} {'':>12} {'':>12}"
        f" ${total_val:>10,.2f} ${total_pnl:>+8,.2f}",
    ]
    return lines


def section_improvements(db: Database, live_positions: dict, scores: list[dict]) -> list[str]:
    """
    Rule-based improvement suggestions derived from the data collected so far.
    Grows richer with more trade history.
    """
    decisions = db.get_recent_decisions(days=365)
    executed  = [d for d in decisions if d.get("action") == "BUY_EXECUTED"]
    skipped   = [d for d in decisions if d.get("action") not in ("BUY_EXECUTED", "SELL_EXECUTED")]

    suggestions: list[str] = []

    # ── 1. Emergency mode false-positive rate ─────────────────────────────────
    emergency_skips = [d for d in skipped if "Emergency" in (d.get("reason") or "")]
    if len(emergency_skips) > len(executed) * 2:
        suggestions.append(
            "THRESHOLD: Emergency mode fired more than 2× as often as buys executed. "
            "Consider raising CRASH_ATR_PCT_THRESHOLD or requiring 2+ crash signals."
        )

    # ── 2. Bear-market entry evaluation ──────────────────────────────────────
    bear_buys = [d for d in executed if d.get("market_regime") in ("BEAR", "CRASH")]
    if bear_buys:
        btc_rows = db.get_price_history("BTC", "1D", 30)
        if len(btc_rows) >= 2:
            price_since_entry = btc_rows[-1]["close"] / btc_rows[0]["close"] - 1
            if price_since_entry < -0.05:
                suggestions.append(
                    f"ENTRY TIMING: Bought in BEAR regime and BTC is down {price_since_entry*100:.1f}% "
                    "since first DB record. Consider adding a '5-day momentum filter': "
                    "only DCA in BEAR if 5-day change > -3% (reduces buying into free-fall)."
                )
            elif price_since_entry > 0.05:
                suggestions.append(
                    f"ENTRY TIMING: Bear-regime entries up {price_since_entry*100:.1f}% — "
                    "accumulation strategy working as intended. Continue."
                )

    # ── 3. Score vs outcome ───────────────────────────────────────────────────
    score_hist = db.get_score_history(30)
    if score_hist:
        avg_score = np.mean([s["total_score"] for s in score_hist if s.get("total_score")])
        if avg_score < 45:
            suggestions.append(
                f"SCORE CALIBRATION: Average score {avg_score:.1f} has been consistently low. "
                "Check if Sentiment and Liquidity components have stale data "
                "(Fear & Greed or CoinGecko may have failed silently)."
            )

    # ── 4. Missing mean-reversion trades ─────────────────────────────────────
    mr_recs  = [d for d in decisions if d.get("action") == "MEAN_REVERSION_BUY"]
    mr_exec  = [d for d in executed if "mean_reversion" in (d.get("reason") or "").lower()]
    if mr_recs and not mr_exec:
        suggestions.append(
            "MEAN REVERSION: Opportunities were flagged but none were executed "
            "(likely blocked by emergency mode or regime filter). "
            "Consider allowing mean-reversion buys even in BEAR regime when "
            "RSI < 28 AND Fear & Greed < 20 — these are historically strong entries."
        )

    # ── 5. Concentration risk ─────────────────────────────────────────────────
    if live_positions:
        total_live = sum(p["market_value"] for p in live_positions.values())
        for ticker, pos in live_positions.items():
            share = pos["market_value"] / total_live if total_live else 0
            if share > 0.60:
                suggestions.append(
                    f"CONCENTRATION: {ticker} is {share*100:.0f}% of crypto holdings. "
                    "Diversify into ETH and large-caps on next DCA cycle."
                )

    # ── 6. Allocation drift ───────────────────────────────────────────────────
    latest_snap = db.get_latest_portfolio()
    if latest_snap and latest_snap.get("targets"):
        targets = latest_snap["targets"]
        allocs  = latest_snap.get("allocations") or {}
        max_drift = max(
            (abs(allocs.get(k, 0) - v) for k, v in targets.items()),
            default=0
        )
        if max_drift > 0.15:
            suggestions.append(
                f"DRIFT: Portfolio is {max_drift*100:.0f}% away from target on at least one asset. "
                "Run a rebalance cycle (bot will handle automatically once 28-day timer resets)."
            )

    # ── 7. DCA frequency ─────────────────────────────────────────────────────
    if len(executed) >= 3:
        timestamps = sorted(d["ts"] for d in executed)
        gaps = []
        for i in range(1, len(timestamps)):
            try:
                a = datetime.fromisoformat(timestamps[i-1].replace("Z", "+00:00"))
                b = datetime.fromisoformat(timestamps[i].replace("Z", "+00:00"))
                gaps.append((b - a).total_seconds() / 3600)
            except Exception:
                pass
        if gaps:
            avg_gap_h = np.mean(gaps)
            if avg_gap_h < 1:
                suggestions.append(
                    f"DCA FREQUENCY: Buys are averaging {avg_gap_h:.1f}h apart. "
                    "For a long-term strategy, daily cadence is healthier — "
                    "increase --interval to 86400 (24h) in lt_monitor.py."
                )

    if not suggestions:
        suggestions.append(
            "No issues detected from current data. Collect more trades for deeper analysis."
        )

    return [f"  {i+1}. {s}" for i, s in enumerate(suggestions)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Post-trade analysis for the LT bot")
    parser.add_argument("--paper-trade", action="store_true",
                        help="Fetch live position P&L from Alpaca paper account")
    args = parser.parse_args()

    db = Database()

    # Fetch live positions if paper trading is enabled
    live_positions: dict = {}
    account_summary: dict = {}
    if args.paper_trade:
        try:
            from lt_bot.trader import PaperTrader
            trader = PaperTrader()
            live_positions   = trader.get_positions()
            account_summary  = trader.get_account_summary()
        except Exception as exc:
            print(f"  Warning: could not fetch live positions — {exc}")

    scores = db.get_score_history(60)
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    divH = _divH()
    divL = _divL()

    lines = [
        divH,
        f" POST-TRADE ANALYSIS  —  {now}",
        divH,
        "",
    ]

    if account_summary:
        lines += [
            f"  Portfolio value : ${account_summary['portfolio_value']:,.2f}",
            f"  Buying power    : ${account_summary['buying_power']:,.2f}",
            f"  Cash            : ${account_summary['cash']:,.2f}",
            "",
        ]

    lines += [divL, " 1. LIVE POSITIONS", divL]
    lines += section_live_positions(live_positions, db)

    lines += ["", divL, " 2. EXECUTED TRADE LOG", divL]
    lines += section_trade_log(db, live_positions)

    lines += ["", divL, " 3. REGIME ACCURACY  (7-day forward price check)", divL]
    lines += section_regime_accuracy(db)

    lines += ["", divL, " 4. SCORE CALIBRATION  (AI score vs 7-day outcome)", divL]
    lines += section_score_calibration(db)

    lines += ["", divL, " 5. IMPROVEMENT RECOMMENDATIONS", divL]
    lines += section_improvements(db, live_positions, scores)

    lines += ["", divH, " Run `python lt_run.py --paper-trade` to continue accumulating.", divH]

    report = "\n".join(lines)
    print(report)

    fname = REPORTS_DIR / f"analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.txt"
    fname.write_text(report, encoding="utf-8")
    print(f"\n  Analysis saved to: {fname}")


if __name__ == "__main__":
    main()
