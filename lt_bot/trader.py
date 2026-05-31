"""
Paper Trading Execution Layer (V2)

Reads the live Alpaca paper account, compares current allocation to targets,
and executes DCA buys / profit-taking sells based on the AI engine output.

Entry conditions:
  - AI score >= 50 (DCA multiplier > 0)
  - Regime is not CRASH
  - Asset is below target allocation
  - Sufficient buying power available

Exit conditions:
  - Profit-taking signals from profit_taker.py
  - Emergency mode (risk manager triggered)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from .config import ASSETS
from .database import Database

load_dotenv()
log = logging.getLogger(__name__)

MIN_ORDER_USD = 5.0   # Alpaca minimum for crypto notional orders

# Map our ticker → Alpaca symbol (no slash)
_ALPACA_SYM = {ticker: sym.replace("/", "") for ticker, sym in ASSETS.items()}
_TICKER_MAP  = {v: k for k, v in _ALPACA_SYM.items()}


@dataclass
class TradeResult:
    action:     str          # "BUY" | "SELL"
    ticker:     str
    amount_usd: float = 0.0
    qty:        float = 0.0
    reason:     str   = ""
    success:    bool  = False
    error:      str   = ""


@dataclass
class ExecutionSummary:
    portfolio_value:  float
    buying_power:     float
    trades:           list[TradeResult] = field(default_factory=list)
    skipped_reasons:  list[str]         = field(default_factory=list)


class PaperTrader:
    def __init__(self) -> None:
        self._client = TradingClient(
            api_key    = os.environ.get("ALPACA_API_KEY",    ""),
            secret_key = os.environ.get("ALPACA_SECRET_KEY", ""),
            paper      = True,
        )

    # ── Account helpers ───────────────────────────────────────────────────────

    def get_account_summary(self) -> dict:
        acct = self._client.get_account()
        return {
            "portfolio_value": float(acct.portfolio_value),
            "buying_power":    float(acct.buying_power),
            "cash":            float(acct.cash),
        }

    def get_positions(self) -> dict[str, dict]:
        """Returns {ticker: {qty, market_value, current_price, avg_entry, unrealized_pnl}}."""
        try:
            raw = self._client.get_all_positions()
        except Exception as exc:
            log.error("Error fetching positions: %s", exc)
            return {}

        result: dict[str, dict] = {}
        for pos in raw:
            ticker = _TICKER_MAP.get(pos.symbol)
            if not ticker:
                continue
            result[ticker] = {
                "qty":            float(pos.qty),
                "market_value":   float(pos.market_value),
                "current_price":  float(pos.current_price),
                "avg_entry":      float(pos.avg_entry_price),
                "unrealized_pnl": float(pos.unrealized_pl),
            }
        return result

    def get_allocation(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Returns (fractions, values) where:
          fractions : {ticker: fraction_of_portfolio}
          values    : {ticker: usd_value}
        """
        acct      = self.get_account_summary()
        positions = self.get_positions()
        total     = acct["portfolio_value"]

        fractions: dict[str, float] = {}
        values:    dict[str, float] = {}

        for ticker, pos in positions.items():
            v = pos["market_value"]
            values[ticker]    = v
            fractions[ticker] = v / total if total > 0 else 0.0

        cash = acct["cash"]
        values["STABLECOIN"]    = cash
        fractions["STABLECOIN"] = cash / total if total > 0 else 0.0

        return fractions, values

    # ── Order helpers ──────────────────────────────────────────────────────────

    def _buy_notional(self, ticker: str, amount_usd: float, reason: str, db: Database) -> TradeResult:
        """Buy $amount_usd of ticker at market (notional order)."""
        result = TradeResult(action="BUY", ticker=ticker, amount_usd=amount_usd, reason=reason)

        if amount_usd < MIN_ORDER_USD:
            result.error = f"below min order ${MIN_ORDER_USD}"
            return result

        alpaca_sym = _ALPACA_SYM.get(ticker)
        if not alpaca_sym:
            result.error = f"unknown ticker {ticker}"
            return result

        try:
            order = self._client.submit_order(MarketOrderRequest(
                symbol          = alpaca_sym,
                notional        = round(amount_usd, 2),
                side            = OrderSide.BUY,
                time_in_force   = TimeInForce.IOC,
            ))
            result.success = True
            log.info("BUY  %s  $%.2f  order=%s  [%s]", ticker, amount_usd, order.id, reason)
            db.record_decision({
                "symbol":          ticker,
                "action":          "BUY_EXECUTED",
                "reason":          reason,
                "recommended_usd": amount_usd,
                "executed":        1,
            })
        except Exception as exc:
            result.error = str(exc)
            log.error("BUY  %s  $%.2f  FAILED: %s", ticker, amount_usd, exc)

        return result

    def _sell_fraction(self, ticker: str, fraction: float, reason: str, db: Database) -> TradeResult:
        """Sell `fraction` of current position in ticker."""
        result = TradeResult(action="SELL", ticker=ticker, reason=reason)

        alpaca_sym = _ALPACA_SYM.get(ticker)
        if not alpaca_sym:
            result.error = f"unknown ticker {ticker}"
            return result

        try:
            pos      = self._client.get_open_position(alpaca_sym)
            qty      = float(pos.qty)
            sell_qty = round(qty * fraction, 8)
            price    = float(pos.current_price)

            if sell_qty <= 0:
                result.error = "zero qty"
                return result

            order = self._client.submit_order(MarketOrderRequest(
                symbol          = alpaca_sym,
                qty             = sell_qty,
                side            = OrderSide.SELL,
                time_in_force   = TimeInForce.IOC,
            ))
            result.qty        = sell_qty
            result.amount_usd = sell_qty * price
            result.success    = True
            log.info(
                "SELL %s  %.4f units (~$%.2f)  order=%s  [%s]",
                ticker, sell_qty, result.amount_usd, order.id, reason,
            )
            db.record_decision({
                "symbol":          ticker,
                "action":          "SELL_EXECUTED",
                "reason":          reason,
                "recommended_usd": result.amount_usd,
                "executed":        1,
            })
        except Exception as exc:
            result.error = str(exc)
            log.error("SELL %s  FAILED: %s", ticker, exc)

        return result

    # ── Main execution logic ───────────────────────────────────────────────────

    def execute_plan(
        self,
        db:              Database,
        score_data:      dict,
        regime:          str,
        dca_rec,                    # DCARecommendation
        profit_signals:  list,      # list[ProfitSignal]
        risk_assessment,            # RiskAssessment
        target_allocation: dict[str, float],
    ) -> ExecutionSummary:
        """
        Decide and execute trades based on the full AI engine output.

        Order of operations:
          1. Emergency mode  → abort all buying
          2. Profit taking   → partial sells on overbought signals
          3. DCA buying      → buy underweight assets with today's DCA budget
        """
        acct   = self.get_account_summary()
        summary = ExecutionSummary(
            portfolio_value = acct["portfolio_value"],
            buying_power    = acct["buying_power"],
        )
        score = score_data["total_score"]

        # ── 1. Emergency mode ─────────────────────────────────────────────────
        if risk_assessment.emergency_mode:
            msg = f"Emergency mode active ({regime}) — no new buys"
            log.warning(msg)
            summary.skipped_reasons.append(msg)
            return summary

        # ── 2. Profit taking (runs regardless of buy conditions) ──────────────
        for sig in profit_signals:
            if sig.ticker not in ASSETS:
                continue
            r = self._sell_fraction(
                sig.ticker,
                sig.sell_pct,
                f"{sig.tier}: {'; '.join(sig.reasons[:2])}",
                db,
            )
            summary.trades.append(r)

        # ── 3. DCA buying ─────────────────────────────────────────────────────
        if regime == "CRASH":
            summary.skipped_reasons.append(f"Regime={regime} — DCA paused")
            return summary

        total_dca = dca_rec.total_usd
        if total_dca <= 0:
            summary.skipped_reasons.append(
                f"Score {score:.1f} < 50 — DCA multiplier=0, no buying"
            )
            return summary

        # Cap at 95% of available buying power
        deploy_usd = min(total_dca, acct["buying_power"] * 0.95)
        if deploy_usd < MIN_ORDER_USD:
            summary.skipped_reasons.append(
                f"Buying power ${acct['buying_power']:.2f} too low for DCA"
            )
            return summary

        # Find underweight assets and rank by deficit
        current_fracs, current_values = self.get_allocation()
        pv = acct["portfolio_value"]

        candidates: list[tuple[str, float]] = []  # (ticker, deficit)
        for ticker, target_frac in target_allocation.items():
            if ticker == "STABLECOIN":
                continue
            deficit = target_frac - current_fracs.get(ticker, 0.0)
            if deficit > 0.005:    # only buy if >0.5% below target
                candidates.append((ticker, deficit))

        if not candidates:
            summary.skipped_reasons.append("All positions at or above target — no DCA buys")
            return summary

        # Distribute DCA proportionally to deficit size
        total_deficit = sum(d for _, d in candidates)
        for ticker, deficit in sorted(candidates, key=lambda x: -x[1]):
            share      = (deficit / total_deficit) * deploy_usd
            # Don't push asset above its target value
            target_val = target_allocation[ticker] * pv
            current_val = current_values.get(ticker, 0.0)
            buy_amount  = min(share, max(0.0, target_val - current_val))

            if buy_amount < MIN_ORDER_USD:
                summary.skipped_reasons.append(
                    f"{ticker}: calculated buy ${buy_amount:.2f} < min — skipped"
                )
                continue

            r = self._buy_notional(
                ticker,
                buy_amount,
                f"DCA {dca_rec.multiplier}x | {regime} | score={score:.1f}",
                db,
            )
            summary.trades.append(r)

        return summary


def format_execution_summary(s: ExecutionSummary) -> str:
    lines = [
        f"  Portfolio     : ${s.portfolio_value:,.2f}",
        f"  Buying power  : ${s.buying_power:,.2f}",
    ]

    buys  = [t for t in s.trades if t.action == "BUY"]
    sells = [t for t in s.trades if t.action == "SELL"]

    if buys:
        lines.append("  EXECUTED BUYS:")
        for t in buys:
            status = "OK" if t.success else f"FAILED ({t.error})"
            lines.append(f"    + {t.ticker:<6} ${t.amount_usd:>8,.2f}   [{status}]  {t.reason}")

    if sells:
        lines.append("  EXECUTED SELLS:")
        for t in sells:
            status = "OK" if t.success else f"FAILED ({t.error})"
            lines.append(
                f"    - {t.ticker:<6} {t.qty:.6f} units (~${t.amount_usd:>8,.2f})  [{status}]  {t.reason}"
            )

    if s.skipped_reasons:
        lines.append("  SKIPPED:")
        for r in s.skipped_reasons:
            lines.append(f"    · {r}")

    if not buys and not sells and not s.skipped_reasons:
        lines.append("  No trades executed.")

    return "\n".join(lines)
