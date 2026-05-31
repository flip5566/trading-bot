"""
Execution Engine

Daily pipeline:
  1. Detect market regime
  2. Calculate AI score
  3. Determine DCA strategy
  4. Analyse portfolio allocation
  5. Check rotation signals
  6. Check mean-reversion opportunities
  7. Check profit-taking signals
  8. Run risk management checks
  9. Execute paper trades (if trader supplied)
 10. Generate and save daily report
 11. Record all decisions to DB
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import ASSETS, REPORTS_DIR
from .database import Database
from .regime_detector import detect_regime, regime_summary
from .scoring_engine import calculate_score
from .dca_strategy import build_dca_recommendation, format_dca_recommendation
from .portfolio_manager import build_allocation_plan, format_allocation_table, risk_guard
from .rotation_engine import detect_rotation, format_rotation
from .mean_reversion import find_opportunities, format_opportunities
from .profit_taker import check_profit_signals, format_profit_signals
from .risk_manager import assess_risk, format_risk_assessment
from .trader import PaperTrader, format_execution_summary

log = logging.getLogger(__name__)

BASE_DCA_AMOUNT_USD = 1_000.0   # adjust to your monthly investment budget


def _asset_summary_table(db: Database) -> str:
    """Per-asset snapshot: price, RSI, EMA alignment, ADX."""
    lines = [
        f"  {'Asset':<8} {'Price':>12} {'RSI':>5} {'ADX':>5} {'EMA Align':<18} {'1W RSI':>7}",
        f"  {'─'*8} {'─'*12} {'─'*5} {'─'*5} {'─'*18} {'─'*7}",
    ]
    for ticker in ASSETS:
        rows = db.get_price_history(ticker, "1D", 1)
        ind_1d = db.get_latest_indicators(ticker, "1D")
        ind_1w = db.get_latest_indicators(ticker, "1W")

        if not rows or not ind_1d:
            lines.append(f"  {ticker:<8}  (no data)")
            continue

        price  = rows[-1]["close"]
        rsi    = ind_1d.get("rsi") or 0
        adx    = ind_1d.get("adx") or 0
        e50    = ind_1d.get("ema50")  or price
        e100   = ind_1d.get("ema100") or price
        e200   = ind_1d.get("ema200") or price
        rsi_1w = (ind_1w.get("rsi") or 0) if ind_1w else 0

        if price > e50 > e100 > e200:
            align = "BULLISH ✓✓✓"
        elif price > e200:
            align = "ABOVE 200"
        elif price < e200 and e50 < e200:
            align = "BEARISH ✗✗"
        else:
            align = "MIXED"

        lines.append(
            f"  {ticker:<8} ${price:>11,.2f} {rsi:>5.0f} {adx:>5.0f} "
            f"{align:<18} {rsi_1w:>7.0f}"
        )
    return "\n".join(lines)


def generate_daily_report(
    db: Database,
    base_dca_usd: float = BASE_DCA_AMOUNT_USD,
    current_allocation: dict[str, float] | None = None,
    trader: PaperTrader | None = None,
) -> str:
    """
    Run the full analysis pipeline and return a formatted report string.

    Args:
        current_allocation: {ticker: fraction} of current portfolio.
                            If None, uses neutral defaults.
    """
    now    = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y-%m-%d %H:%M UTC")
    divH   = "═" * 66
    divL   = "─" * 66

    # ── Step 1: Regime ────────────────────────────────────────────────────────
    regime = detect_regime(db)

    # ── Step 2: AI Score ──────────────────────────────────────────────────────
    score_data = calculate_score(db, regime)
    total      = score_data["total_score"]

    # ── Step 3: DCA ───────────────────────────────────────────────────────────
    dca_rec = build_dca_recommendation(total, base_dca_usd, db)

    # ── Step 4: Portfolio allocation ─────────────────────────────────────────
    plan = build_allocation_plan(db, regime, current_allocation)

    # ── Step 5: Rotation ─────────────────────────────────────────────────────
    rotation = detect_rotation(db)

    # ── Step 6: Mean reversion ────────────────────────────────────────────────
    opps = find_opportunities(db)

    # ── Step 7: Profit taking ─────────────────────────────────────────────────
    profit_signals = check_profit_signals(db)

    # ── Step 8: Risk ──────────────────────────────────────────────────────────
    # If paper trader is active, read live allocation from Alpaca instead of manual input
    if trader is not None:
        live_fracs, _ = trader.get_allocation()
        current_allocation = live_fracs

    risk = assess_risk(db, current_allocation)
    guard_warnings = risk_guard(current_allocation or {})

    # ── Determine master action ───────────────────────────────────────────────
    if risk.emergency_mode:
        master_action = "PROTECT CAPITAL"
    elif profit_signals and profit_signals[0].tier == "TIER_3_EXTREME":
        master_action = "TAKE PROFIT"
    elif total >= 85:
        master_action = "STRONG BUY / DCA"
    elif total >= 70:
        master_action = "BUY / HOLD"
    elif total >= 50:
        master_action = "DCA (reduced)"
    elif total >= 30:
        master_action = "HOLD / DEFENSIVE"
    else:
        master_action = "RISK-OFF"

    # ── Record decisions to DB ────────────────────────────────────────────────
    db.record_decision({
        "symbol":          "PORTFOLIO",
        "action":          master_action,
        "reason":          f"Regime={regime} Score={total}",
        "ai_score":        total,
        "market_regime":   regime,
        "dca_multiplier":  dca_rec.multiplier,
        "recommended_usd": dca_rec.total_usd,
    })
    for opp in opps:
        db.record_decision({
            "symbol":          opp.ticker,
            "action":          "MEAN_REVERSION_BUY",
            "reason":          "; ".join(opp.signals),
            "ai_score":        total,
            "market_regime":   regime,
            "dca_multiplier":  1.5,
            "recommended_usd": base_dca_usd * 0.5,
        })
    for sig in profit_signals:
        db.record_decision({
            "symbol":          sig.ticker,
            "action":          f"SELL_{sig.tier}",
            "reason":          "; ".join(sig.reasons),
            "ai_score":        total,
            "market_regime":   regime,
            "dca_multiplier":  0,
            "recommended_usd": 0,
        })

    # ── Step 9: Execute paper trades ─────────────────────────────────────────
    exec_summary = None
    if trader is not None:
        # Rebuild plan with live allocation before executing
        plan = build_allocation_plan(db, regime, current_allocation)
        exec_summary = trader.execute_plan(
            db             = db,
            score_data     = score_data,
            regime         = regime,
            dca_rec        = dca_rec,
            profit_signals = profit_signals,
            risk_assessment = risk,
            target_allocation = plan.targets,
        )
        db.save_portfolio_snapshot({
            "total_value":  exec_summary.portfolio_value,
            "regime":       regime,
            "ai_score":     total,
            "allocations":  current_allocation or {},
            "targets":      plan.targets,
        })
    else:
        db.save_portfolio_snapshot({
            "total_value":  0,
            "regime":       regime,
            "ai_score":     total,
            "allocations":  current_allocation or {},
            "targets":      plan.targets,
        })

    # ── Build report string ───────────────────────────────────────────────────
    lines = [
        divH,
        f" CRYPTO AI PORTFOLIO MANAGER",
        f" {ts_str}",
        divH,
        "",
        f" MARKET REGIME : {regime}",
        f" ACTION        : {master_action}",
        f" AI SCORE      : {total:.1f} / 100  →  {score_data['score_label']}",
        f" REGIME GUIDE  : {regime_summary(regime)}",
        "",
        divL,
        " SCORE BREAKDOWN",
        divL,
        f"  {'Component':<18} {'Score':>6}  {'Weight':>7}  {'Contribution':>12}",
        f"  {'─'*18} {'─'*6}  {'─'*7}  {'─'*12}",
    ]
    from .config import SCORE_WEIGHTS
    for key, weight in SCORE_WEIGHTS.items():
        s  = score_data.get(f"{key}_score", 0)
        w  = weight
        c  = s * w
        lines.append(f"  {key.capitalize():<18} {s:>6.1f}  {w*100:>6.0f}%  {c:>12.1f}")

    lines += [
        "",
        divL,
        " DCA STRATEGY",
        divL,
        format_dca_recommendation(dca_rec, regime),
        "",
        divL,
        " PORTFOLIO ALLOCATION",
        divL,
        format_allocation_table(plan),
        "",
        divL,
        " MOMENTUM ROTATION",
        divL,
        format_rotation(rotation),
        "",
        divL,
        " MEAN REVERSION OPPORTUNITIES",
        divL,
        format_opportunities(opps),
        "",
        divL,
        " PROFIT TAKING SIGNALS",
        divL,
        format_profit_signals(profit_signals),
        "",
        divL,
        " RISK MANAGEMENT",
        divL,
        format_risk_assessment(risk),
    ]

    if guard_warnings:
        lines.append("  LIMIT GUARD:")
        for w in guard_warnings:
            lines.append(f"    ⚠ {w}")

    # Paper trading section
    if exec_summary is not None:
        lines += [
            "",
            divL,
            " PAPER TRADING — EXECUTED",
            divL,
            format_execution_summary(exec_summary),
        ]

    lines += [
        "",
        divL,
        " ASSET ANALYSIS  (1D indicators)",
        divL,
        _asset_summary_table(db),
        "",
        divH,
        f" Paper trading: {'ACTIVE' if trader else 'OFF (recommendations only)'}",
        divH,
    ]

    report = "\n".join(lines)

    # ── Save to file ──────────────────────────────────────────────────────────
    fname = REPORTS_DIR / f"lt_report_{now.strftime('%Y%m%d_%H%M')}.txt"
    fname.write_text(report, encoding="utf-8")
    log.info("Report saved: %s", fname)

    return report


def run(db: Database | None = None, current_allocation: dict[str, float] | None = None) -> None:
    """
    Entry point called by lt_run.py.
    Collects fresh data, then generates and prints the daily report.
    """
    from .data_collector import collect_all

    if db is None:
        db = Database()

    collect_all(db)

    report = generate_daily_report(db, current_allocation=current_allocation)
    print(report)
