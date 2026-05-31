"""
Backtesting Framework for the Long-Term AI Portfolio Manager

Tests the strategy against historical BTC price data from Alpaca.
Uses indicator-only scoring (no external APIs) so it works for any past date.

Covered periods:
  2017 — Bull market
  2018 — Bear market
  2020 — Accumulation
  2021 — Bull market
  2022 — Crash
  2024 — Current cycle

Benchmark: Buy-and-hold BTC from start of period.

Run:
    python lt_backtest.py
    python lt_backtest.py --period 2021
    python lt_backtest.py --start 2020-01-01 --end 2022-12-31 --capital 10000
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from lt_bot.indicators import calculate_all
from lt_bot.config import BTC_HALVINGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PERIODS = {
    "2017": ("2017-01-01", "2017-12-31"),
    "2018": ("2018-01-01", "2018-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2021": ("2021-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2024": ("2024-01-01", "2025-04-30"),
    "all":  ("2017-01-01", "2025-04-30"),
}


# ── Data fetch ─────────────────────────────────────────────────────────────────

def fetch_btc_history(start: str, end: str) -> pd.DataFrame:
    """
    Fetch BTC/USD daily OHLCV from Alpaca for the full indicator warm-up period.
    Returns bars from (start - 300 days) to end so EMA-200 is valid from `start`.
    """
    import os
    from dotenv import load_dotenv
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    load_dotenv()
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    warmup   = start_dt - timedelta(days=300)

    log.info("Fetching BTC/USD daily from %s to %s (with warm-up)", start, end)
    client = CryptoHistoricalDataClient(
        api_key    = os.environ.get("ALPACA_API_KEY", ""),
        secret_key = os.environ.get("ALPACA_SECRET_KEY", ""),
    )
    req = CryptoBarsRequest(
        symbol_or_symbols="BTC/USD",
        timeframe=TimeFrame.Day,
        start=warmup,
        end=end_dt + timedelta(days=1),
    )
    raw = client.get_crypto_bars(req)
    df  = raw.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs("BTC/USD", level=0)

    df = df.sort_index()[["open", "high", "low", "close", "volume"]]
    log.info("Fetched %d bars", len(df))
    return df


# ── Indicator-only scoring ─────────────────────────────────────────────────────

def _days_since_halving(dt: date) -> int:
    past = [h for h in BTC_HALVINGS if h <= dt]
    return (dt - past[-1]).days if past else 0


def score_row(row: pd.Series, df_full: pd.DataFrame, idx: int) -> dict:
    """
    Score a single daily bar using only technical indicators.
    Returns component scores and total.
    """
    close  = row["close"]
    e50    = row.get("ema50",  close)
    e100   = row.get("ema100", close)
    e200   = row.get("ema200", close)
    rsi    = row.get("rsi",    50.0)
    macd_h = row.get("macd_hist", 0.0)
    macd_v = row.get("macd",      0.0)
    adx    = row.get("adx",    20.0)
    atr_p  = row.get("atr_pct", 2.0)
    vol    = row.get("volume",   0.0)
    vol_ma = row.get("volume_ma", 1.0)
    bb_u   = row.get("bb_upper",  close * 1.05)
    bb_l   = row.get("bb_lower",  close * 0.95)

    # ── Trend score ──
    if close > e50 > e100 > e200:
        ema_pts = 50
    elif close > e200 and e50 > e200:
        ema_pts = 35
    elif close > e200:
        ema_pts = 25
    elif close < e200 and e50 < e200:
        ema_pts = 10
    else:
        ema_pts = 18

    macd_pts = 30 if (macd_h > 0 and macd_v > 0) else (20 if macd_h > 0 else (15 if macd_v > 0 else 5))
    adx_pts  = 20 if adx >= 40 else (15 if adx >= 25 else (8 if adx >= 15 else 3))
    t_score  = min(100, ema_pts + macd_pts + adx_pts)

    # ── Momentum score ──
    if   rsi < 30:   rsi_pts = 45
    elif rsi < 40:   rsi_pts = 50
    elif rsi < 50:   rsi_pts = 40
    elif rsi < 60:   rsi_pts = 50
    elif rsi < 70:   rsi_pts = 45
    elif rsi < 80:   rsi_pts = 35
    else:            rsi_pts = 15

    vol_ratio = (vol / vol_ma) if vol_ma > 0 else 1.0
    vol_pts   = 35 if vol_ratio >= 2.0 else (28 if vol_ratio >= 1.5 else (20 if vol_ratio >= 1.0 else (12 if vol_ratio >= 0.7 else 5)))
    m_score   = min(100, rsi_pts + vol_pts + 10)  # +10 neutral relative strength

    # ── Cycle score ──
    row_date = row.name.date() if hasattr(row.name, "date") else date.today()
    days_h   = _days_since_halving(row_date)
    if   days_h <= 90:   c_base = 65
    elif days_h <= 365:  c_base = 82
    elif days_h <= 548:  c_base = 90
    elif days_h <= 730:  c_base = 72
    elif days_h <= 912:  c_base = 50
    elif days_h <= 1280: c_base = 30
    else:                c_base = 58
    c_score = min(100, c_base)

    # ── Liquidity score ──
    lq_score = min(100, (
        (60 if vol_ratio >= 2.0 else 50 if vol_ratio >= 1.5 else 40 if vol_ratio >= 1.0 else 25 if vol_ratio >= 0.7 else 10)
        + 30   # assume neutral stablecoin supply (no external data)
    ))

    # ── Sentiment score (indicator proxy) ──
    bb_range = bb_u - bb_l if bb_u > bb_l else 1.0
    bb_pos   = (close - bb_l) / bb_range   # 0 = at lower band, 1 = at upper band
    if rsi < 25:     fg_proxy = 60
    elif rsi < 35:   fg_proxy = 50
    elif rsi < 50:   fg_proxy = 35
    elif rsi < 65:   fg_proxy = 30
    else:            fg_proxy = 10
    fund_proxy = 20   # neutral (no real funding data)
    s_score  = min(100, fg_proxy + fund_proxy)

    # ── Risk score ──
    if   atr_p <= 1.5: atr_pts = 50
    elif atr_p <= 2.5: atr_pts = 40
    elif atr_p <= 4.0: atr_pts = 30
    elif atr_p <= 6.0: atr_pts = 18
    elif atr_p <= 8.0: atr_pts = 8
    else:              atr_pts = 2

    # Drawdown from 90-day high
    start_i = max(0, idx - 90)
    high_90  = df_full.iloc[start_i : idx + 1]["high"].max()
    dd       = (close / high_90 - 1) * 100 if high_90 else 0
    if   dd >= -5:   dd_pts = 35
    elif dd >= -15:  dd_pts = 45
    elif dd >= -30:  dd_pts = 50
    elif dd >= -50:  dd_pts = 40
    else:            dd_pts = 20
    r_score = min(100, atr_pts + dd_pts)

    total = (
        t_score  * 0.25 +
        m_score  * 0.15 +
        c_score  * 0.20 +
        lq_score * 0.15 +
        s_score  * 0.10 +
        r_score  * 0.15
    )
    return {
        "trend": t_score, "momentum": m_score, "cycle": c_score,
        "liquidity": lq_score, "sentiment": s_score, "risk": r_score,
        "total": round(total, 1),
    }


# ── Regime detection (indicator-only) ─────────────────────────────────────────

def detect_regime_row(row: pd.Series, prev_close: float) -> str:
    close  = row["close"]
    e50    = row.get("ema50",  close)
    e100   = row.get("ema100", close)
    e200   = row.get("ema200", close)
    rsi    = row.get("rsi",    50.0)
    adx    = row.get("adx",    20.0)
    atr_p  = row.get("atr_pct", 2.0)
    vol    = row.get("volume",   1.0)
    vol_ma = row.get("volume_ma", 1.0)

    daily_chg = (close / prev_close - 1) if prev_close else 0

    # Euphoria: RSI > 80 AND price >20% above EMA50
    if rsi > 80 and close > e50 * 1.20:
        return "EUPHORIA"

    # Crash: >10% daily drop with high volatility
    if daily_chg <= -0.10 and atr_p >= 6.0:
        return "CRASH"

    vol_ratio = vol / vol_ma if vol_ma > 0 else 1.0

    # Strong bull
    if close > e200 and e50 > e100 > e200 and adx > 25 and vol_ratio > 1.2:
        return "STRONG_BULL"

    # Bull
    if close > e200 and rsi > 45:
        return "BULL"

    # Bear
    if close < e200 and e50 < e200:
        return "BEAR"

    return "SIDEWAYS"


def dca_multiplier(score: float) -> float:
    if score >= 85:  return 2.5
    if score >= 70:  return 1.0
    if score >= 50:  return 0.5
    return 0.0


# ── Simulation ────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    period:          str
    start_date:      str
    end_date:        str
    initial_capital: float
    final_value:     float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio:    float
    btc_hodl_return: float
    win_months_pct:  float
    n_buy_days:      int
    n_sell_days:     int
    n_hold_days:     int
    daily_series:    pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


def simulate(
    df_with_indicators: pd.DataFrame,
    start: str,
    end: str,
    initial_capital: float = 10_000.0,
    base_dca_daily: float  = 10.0,    # daily DCA amount in USD (≈ $300/month)
) -> SimulationResult:
    """
    Simulate the portfolio strategy over a historical period.

    Portfolio: BTC (crypto) + Cash (stablecoin proxy).
    - Buy: allocate cash based on DCA multiplier × base_dca_daily
    - Sell 10-35%: profit-taking signals (RSI + ADX overbought)
    - Correction buy: +50% DCA when price drops >15% from 30d high

    Returns SimulationResult with performance metrics.
    """
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt   = pd.Timestamp(end,   tz="UTC")

    df = df_with_indicators.copy()
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    if df.empty:
        raise ValueError(f"No data found between {start} and {end}")

    cash        = initial_capital
    btc_units   = 0.0
    portfolio_values: list[float] = []
    daily_log:   list[dict] = []

    buy_days  = 0
    sell_days = 0
    hold_days = 0

    prev_close = df.iloc[0]["open"]
    rolling_high_30 = []   # rolling 30-day high prices

    for idx_pos, (ts, row) in enumerate(df.iterrows()):
        close = row["close"]
        rolling_high_30.append(close)
        if len(rolling_high_30) > 30:
            rolling_high_30.pop(0)
        high_30 = max(rolling_high_30)

        # Score and regime
        scores = score_row(row, df, idx_pos)
        total_score = scores["total"]
        regime = detect_regime_row(row, prev_close)

        rsi    = row.get("rsi",  50.0)
        atr_p  = row.get("atr_pct", 2.0)

        action = "HOLD"
        trade_note = ""

        # ── Profit taking (partial exit) ──
        sell_pct = 0.0
        if regime == "EUPHORIA" or total_score < 30:
            sell_pct = 0.35
        elif rsi > 78 and atr_p > 4:
            sell_pct = 0.15
        elif rsi > 72:
            sell_pct = 0.10

        if sell_pct > 0 and btc_units > 0:
            units_to_sell = btc_units * sell_pct
            proceeds = units_to_sell * close
            btc_units -= units_to_sell
            cash      += proceeds
            action     = "SELL"
            sell_days += 1
            trade_note = f"profit-take {sell_pct*100:.0f}%"

        # ── Emergency: stop buying ──
        if regime == "CRASH":
            hold_days += 1
            action    = "HOLD"
            trade_note = "emergency — no buying"

        # ── DCA buying ──
        elif action != "SELL":
            mult        = dca_multiplier(total_score)
            dca_amount  = base_dca_daily * mult

            # Correction bonus: +50% if >15% pullback and trend intact
            drop_from_high = (close / high_30 - 1) if high_30 else 0
            if (
                drop_from_high <= -0.15
                and close > (row.get("ema200") or close) * 0.95
                and rsi < 40
            ):
                dca_amount += base_dca_daily * 0.5
                trade_note  = f"correction buy {drop_from_high*100:.1f}%"

            if dca_amount > 0 and cash >= dca_amount:
                units       = dca_amount / close
                btc_units  += units
                cash       -= dca_amount
                buy_days   += 1
                action      = "BUY"
            else:
                hold_days  += 1

        # Portfolio value
        portfolio_val = cash + btc_units * close
        portfolio_values.append(portfolio_val)

        daily_log.append({
            "date":      ts.strftime("%Y-%m-%d"),
            "price":     close,
            "portfolio": portfolio_val,
            "btc_units": btc_units,
            "cash":      cash,
            "score":     total_score,
            "regime":    regime,
            "action":    action,
            "note":      trade_note,
            "rsi":       rsi,
        })

        prev_close = close

    if not portfolio_values:
        raise ValueError("Simulation produced no data")

    # ── Metrics ──────────────────────────────────────────────────────────────
    final_val    = portfolio_values[-1]
    total_return = (final_val / initial_capital - 1) * 100

    # Max drawdown
    running_max = initial_capital
    max_dd      = 0.0
    for v in portfolio_values:
        running_max = max(running_max, v)
        dd = (v / running_max - 1) * 100
        max_dd = min(max_dd, dd)

    # Sharpe (daily risk-free rate ≈ 0)
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe  = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0.0

    # Monthly win rate
    daily_df = pd.DataFrame(daily_log)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.set_index("date")
    monthly  = daily_df["portfolio"].resample("ME").last()
    monthly_ret = monthly.pct_change().dropna()
    win_months  = (monthly_ret > 0).mean() * 100

    # BTC buy-and-hold benchmark
    start_price = df.iloc[0]["open"]
    end_price   = df.iloc[-1]["close"]
    btc_hodl    = (end_price / start_price - 1) * 100

    return SimulationResult(
        period           = f"{start} to {end}",
        start_date       = start,
        end_date         = end,
        initial_capital  = initial_capital,
        final_value      = final_val,
        total_return_pct = total_return,
        max_drawdown_pct = max_dd,
        sharpe_ratio     = round(sharpe, 2),
        btc_hodl_return  = btc_hodl,
        win_months_pct   = win_months,
        n_buy_days       = buy_days,
        n_sell_days      = sell_days,
        n_hold_days      = hold_days,
        daily_series     = pd.DataFrame(daily_log),
    )


# ── Report formatter ───────────────────────────────────────────────────────────

def format_result(r: SimulationResult, period_name: str = "") -> str:
    divH = "═" * 62
    divL = "─" * 62

    alpha = r.total_return_pct - r.btc_hodl_return

    lines = [
        divH,
        f" BACKTEST: {period_name or r.period}",
        divH,
        f"  Period            : {r.start_date}  →  {r.end_date}",
        f"  Initial capital   : ${r.initial_capital:>12,.2f}",
        f"  Final value       : ${r.final_value:>12,.2f}",
        divL,
        f"  Strategy return   : {r.total_return_pct:>+8.1f}%",
        f"  BTC buy-and-hold  : {r.btc_hodl_return:>+8.1f}%",
        f"  Alpha vs HODL     : {alpha:>+8.1f}%",
        divL,
        f"  Max drawdown      : {r.max_drawdown_pct:>+8.1f}%",
        f"  Sharpe ratio      : {r.sharpe_ratio:>8.2f}",
        f"  Win months        : {r.win_months_pct:>8.1f}%",
        divL,
        f"  Buy days          : {r.n_buy_days:>8}",
        f"  Sell days         : {r.n_sell_days:>8}",
        f"  Hold days         : {r.n_hold_days:>8}",
        divH,
    ]

    # Quick assessment
    notes = []
    if r.max_drawdown_pct > -50:
        notes.append("Max drawdown better than BTC HODL (typically -80%+ in bears)")
    if r.sharpe_ratio > 1.0:
        notes.append("Sharpe > 1.0 — good risk-adjusted return")
    if alpha > 0:
        notes.append(f"Strategy outperformed buy-and-hold by {alpha:.1f}%")
    elif alpha < -10:
        notes.append("Strategy underperformed — DCA timing cost vs full exposure")

    if notes:
        lines.append(" NOTES:")
        for n in notes:
            lines.append(f"  • {n}")
        lines.append(divH)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backtest(
    start: str,
    end: str,
    initial_capital: float = 10_000.0,
    base_dca_daily: float  = 10.0,
    period_name: str = "",
) -> SimulationResult:
    df_raw  = fetch_btc_history(start, end)
    df_ind  = calculate_all(df_raw)

    log.info("Running simulation %s → %s", start, end)
    result = simulate(df_ind, start, end, initial_capital, base_dca_daily)

    report = format_result(result, period_name or f"{start} — {end}")
    print(report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the Long-Term AI Portfolio Manager"
    )
    parser.add_argument(
        "--period",
        choices=list(PERIODS.keys()),
        help="Named test period (2017, 2018, 2020, 2021, 2022, 2024, all)",
    )
    parser.add_argument("--start",   default=None, help="Custom start date YYYY-MM-DD")
    parser.add_argument("--end",     default=None, help="Custom end date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital USD")
    parser.add_argument("--dca",     type=float, default=10.0,
                        help="Daily base DCA amount USD (default 10 ~= $300/month)")
    parser.add_argument("--all-periods", action="store_true",
                        help="Run all named test periods sequentially")
    args = parser.parse_args()

    if args.all_periods:
        print("\n" + "=" * 62)
        print(" COMPREHENSIVE BACKTEST — ALL PERIODS")
        print("=" * 62 + "\n")
        for name, (s, e) in PERIODS.items():
            if name == "all":
                continue
            try:
                run_backtest(s, e, args.capital, args.dca, name)
                print()
            except Exception as exc:
                print(f"Period {name} failed: {exc}\n")
        return

    if args.period:
        start, end = PERIODS[args.period]
        run_backtest(start, end, args.capital, args.dca, args.period)
    elif args.start and args.end:
        run_backtest(args.start, args.end, args.capital, args.dca)
    else:
        # Default: run all named periods
        for name, (s, e) in PERIODS.items():
            if name == "all":
                continue
            try:
                run_backtest(s, e, args.capital, args.dca, name)
                print()
            except Exception as exc:
                print(f"Period {name} failed: {exc}\n")


if __name__ == "__main__":
    main()
