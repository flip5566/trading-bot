"""
Short-Term Momentum Pullback Strategy
======================================
Timeframes : 1H (trend) → 15m (setup) → 5m (entry)
Entry long : trend bullish + 15m breakout candle + 5m pullback to EMA20 +
             RSI > 50 + above VWAP + continuation candle
Entry short: trend bearish + 15m breakdown candle + 5m pullback to EMA20 +
             RSI < 50 + below VWAP + continuation candle
Risk       : RISK_PCT per trade | SL below pullback low | TP at 1.5R
Daily limit: 2 consecutive losses OR 2% account drawdown → halt
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
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS: list[str] = [
    "BTC/USD",   # Bitcoin
    "ETH/USD",   # Ethereum
    "SOL/USD",   # Solana
    "XRP/USD",   # XRP
    "ADA/USD",   # Cardano
    "DOGE/USD",  # Dogecoin
    "AVAX/USD",  # Avalanche
    "LINK/USD",  # Chainlink
    "DOT/USD",   # Polkadot
    "LTC/USD",   # Litecoin
    "BCH/USD",   # Bitcoin Cash
    "UNI/USD",   # Uniswap
]
RISK_PCT: float = 0.004           # 0.4% of portfolio per trade
MAX_NOTIONAL_PCT: float = 0.15    # cap position at 15% of portfolio (protects against tiny-stop alts)
MAX_DAILY_LOSS_PCT: float = 0.02  # halt at −2%
MAX_CONSECUTIVE_LOSSES: int = 2   # halt after 2 losses in a row
TP_RATIO: float = 1.5             # take profit at 1.5R

EMA_FAST: int = 20
EMA_SLOW: int = 50
RSI_PERIOD: int = 14
VOL_LOOKBACK: int = 20
VOL_THRESHOLD: float = 1.0        # breakout volume must be ≥ 1.0× average

STATE_FILE = Path(__file__).parent / "state.json"


# ── Indicators ────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP reset at UTC midnight."""
    df = df.copy()
    idx = df.index
    if hasattr(idx, "normalize"):
        df["_date"] = idx.normalize()
    else:
        df["_date"] = pd.to_datetime(idx).normalize()
    df["_tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpv"] = df["_tp"] * df["volume"]
    df["_cum_tpv"] = df.groupby("_date")["_tpv"].cumsum()
    df["_cum_v"] = df.groupby("_date")["volume"].cumsum()
    return df["_cum_tpv"] / df["_cum_v"]


# ── Data helpers ──────────────────────────────────────────────────────────────

def fetch_bars(symbol: str, tf: TimeFrame, bars: int) -> pd.DataFrame:
    per_bar_minutes = {
        TimeFrame.Hour: 60,
        TimeFrame(15, TimeFrameUnit.Minute): 15,
        TimeFrame(5, TimeFrameUnit.Minute): 5,
    }.get(tf, 60)

    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=per_bar_minutes * bars * 3)  # 3× buffer

    req = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end,
    )
    raw = CryptoHistoricalDataClient().get_crypto_bars(req)
    df = raw.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    df = df.sort_index()
    return df.tail(bars)


# ── Trend (1H) ────────────────────────────────────────────────────────────────

def get_trend(df_1h: pd.DataFrame) -> str:
    """'bullish' | 'bearish' | 'neutral'"""
    close = df_1h["close"]
    e20 = ema(close, EMA_FAST)
    e50 = ema(close, EMA_SLOW)
    p, f, s = close.iloc[-1], e20.iloc[-1], e50.iloc[-1]
    if p > f > s:
        return "bullish"
    if p < f < s:
        return "bearish"
    return "neutral"


# ── Market activity filter ────────────────────────────────────────────────────

def is_market_active(df: pd.DataFrame) -> bool:
    """Reject low-volume / choppy conditions."""
    vol = df["volume"]
    avg_vol = vol.rolling(VOL_LOOKBACK).mean().iloc[-1]
    recent_vol = vol.iloc[-5:].mean()

    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    atr_pct = atr / df["close"].iloc[-1]

    return (recent_vol >= avg_vol * 0.7) and (atr_pct >= 0.001)


# ── 15m setup confirmation ────────────────────────────────────────────────────

def find_15m_breakout(df_15m: pd.DataFrame, direction: str) -> dict | None:
    """
    Scan the last 8 15m bars (~2h) for a strong breakout/breakdown candle.
    Returns info dict or None.
    """
    close = df_15m["close"]
    high = df_15m["high"]
    low = df_15m["low"]
    open_ = df_15m["open"]
    vol = df_15m["volume"]
    avg_vol = vol.rolling(VOL_LOOKBACK).mean()

    for back in range(1, 9):
        idx = -back
        rng = high.iloc[idx] - low.iloc[idx]
        if rng < 1e-9 or avg_vol.iloc[idx] == 0:
            continue
        body = abs(close.iloc[idx] - open_.iloc[idx])
        body_pct = body / rng
        vol_ratio = vol.iloc[idx] / avg_vol.iloc[idx]
        bullish = close.iloc[idx] > open_.iloc[idx]
        bearish = close.iloc[idx] < open_.iloc[idx]

        if direction == "bullish" and bullish and body_pct > 0.5 and vol_ratio >= VOL_THRESHOLD:
            return {"level": high.iloc[idx], "vol_ratio": vol_ratio}
        if direction == "bearish" and bearish and body_pct > 0.5 and vol_ratio >= VOL_THRESHOLD:
            return {"level": low.iloc[idx], "vol_ratio": vol_ratio}

    return None


# ── 5m entry detection ────────────────────────────────────────────────────────

def _strong_candle_idx(df_5m: pd.DataFrame, direction: str) -> int | None:
    """Return iloc index of the most recent strong breakout candle (3–20 bars ago)."""
    close = df_5m["close"]
    high = df_5m["high"]
    low = df_5m["low"]
    open_ = df_5m["open"]
    vol = df_5m["volume"]
    avg_vol = vol.rolling(VOL_LOOKBACK).mean()

    for back in range(3, 21):
        idx = -back
        rng = high.iloc[idx] - low.iloc[idx]
        if rng < 1e-9 or avg_vol.iloc[idx] == 0:
            continue
        body = abs(close.iloc[idx] - open_.iloc[idx])
        vol_ratio = vol.iloc[idx] / avg_vol.iloc[idx]
        bullish = close.iloc[idx] > open_.iloc[idx]
        bearish = close.iloc[idx] < open_.iloc[idx]

        if direction == "bullish" and bullish and body / rng > 0.5 and vol_ratio >= VOL_THRESHOLD:
            return idx
        if direction == "bearish" and bearish and body / rng > 0.5 and vol_ratio >= VOL_THRESHOLD:
            return idx
    return None


def find_5m_long_entry(df_5m: pd.DataFrame) -> dict | None:
    """
    Long entry conditions on 5m:
    • Recent bullish breakout candle (high vol, large body)
    • Price pulled back to EMA20 zone without closing below
    • RSI > 50 (not above 70)
    • Above VWAP
    • Continuation candle: current close > previous bar high
    • Current volume above average
    """
    if len(df_5m) < 30:
        return None

    close = df_5m["close"]
    high  = df_5m["high"]
    low   = df_5m["low"]
    vol   = df_5m["volume"]
    e20   = ema(close, EMA_FAST)
    r     = rsi(close, RSI_PERIOD)
    avg_v = vol.rolling(VOL_LOOKBACK).mean()
    vwap_ = vwap(df_5m)

    curr_rsi  = r.iloc[-1]
    curr_vwap = vwap_.iloc[-1]
    curr_close = close.iloc[-1]
    curr_vol   = vol.iloc[-1]
    curr_avg_v = avg_v.iloc[-1]

    if not (50 < curr_rsi < 70):
        return None
    if curr_close <= curr_vwap:
        return None
    if curr_close <= high.iloc[-2]:       # not a continuation break
        return None
    if curr_avg_v > 0 and curr_vol < curr_avg_v * 0.8:
        return None

    # Find breakout candle
    bo_idx = _strong_candle_idx(df_5m, "bullish")
    if bo_idx is None:
        return None

    # Bars between breakout and current; check pullback to EMA20
    pullback_low = None
    for i in range(bo_idx + 1, -1):     # from after breakout to bar before current
        bar_low   = low.iloc[i]
        bar_close = close.iloc[i]
        bar_e20   = e20.iloc[i]
        # Touched EMA20 zone but didn't close below it
        if bar_low <= bar_e20 * 1.005 and bar_close >= bar_e20 * 0.997:
            pullback_low = bar_low
            break

    if pullback_low is None:
        return None

    entry = curr_close
    stop  = pullback_low * 0.998
    risk  = entry - stop
    if risk <= 0:
        return None

    atr_pct = float((df_5m["high"] - df_5m["low"]).rolling(14).mean().iloc[-1] / entry * 100)
    vol_ratio = float(curr_vol / curr_avg_v) if curr_avg_v > 0 else 1.0
    ema20_dist_pct = float((entry / e20.iloc[-1] - 1) * 100)

    return {
        "side": "long",
        "entry": entry,
        "stop": stop,
        "tp": entry + TP_RATIO * risk,
        "rsi": curr_rsi,
        "vol_ratio": round(vol_ratio, 2),
        "atr_pct": round(atr_pct, 3),
        "ema20_dist_pct": round(ema20_dist_pct, 3),
    }


def find_5m_short_entry(df_5m: pd.DataFrame) -> dict | None:
    """
    Short entry conditions on 5m:
    • Recent bearish breakdown candle (high vol, large body)
    • Pullback toward EMA20 from below without closing above
    • RSI < 50 (not below 30)
    • Below VWAP
    • Continuation candle: current close < previous bar low
    • Current volume above average
    """
    if len(df_5m) < 30:
        return None

    close = df_5m["close"]
    high  = df_5m["high"]
    low   = df_5m["low"]
    vol   = df_5m["volume"]
    e20   = ema(close, EMA_FAST)
    r     = rsi(close, RSI_PERIOD)
    avg_v = vol.rolling(VOL_LOOKBACK).mean()
    vwap_ = vwap(df_5m)

    curr_rsi   = r.iloc[-1]
    curr_vwap  = vwap_.iloc[-1]
    curr_close = close.iloc[-1]
    curr_vol   = vol.iloc[-1]
    curr_avg_v = avg_v.iloc[-1]

    if not (30 < curr_rsi < 50):
        return None
    if curr_close >= curr_vwap:
        return None
    if curr_close >= low.iloc[-2]:        # not a breakdown continuation
        return None
    if curr_avg_v > 0 and curr_vol < curr_avg_v * 0.8:
        return None

    bo_idx = _strong_candle_idx(df_5m, "bearish")
    if bo_idx is None:
        return None

    pullback_high = None
    for i in range(bo_idx + 1, -1):
        bar_high  = high.iloc[i]
        bar_close = close.iloc[i]
        bar_e20   = e20.iloc[i]
        if bar_high >= bar_e20 * 0.995 and bar_close <= bar_e20 * 1.003:
            pullback_high = bar_high
            break

    if pullback_high is None:
        return None

    entry = curr_close
    stop  = pullback_high * 1.002
    risk  = stop - entry
    if risk <= 0:
        return None

    atr_pct = float((df_5m["high"] - df_5m["low"]).rolling(14).mean().iloc[-1] / entry * 100)
    vol_ratio = float(curr_vol / curr_avg_v) if curr_avg_v > 0 else 1.0
    ema20_dist_pct = float((entry / e20.iloc[-1] - 1) * 100)

    return {
        "side": "short",
        "entry": entry,
        "stop": stop,
        "tp": entry - TP_RATIO * risk,
        "rsi": curr_rsi,
        "vol_ratio": round(vol_ratio, 2),
        "atr_pct": round(atr_pct, 3),
        "ema20_dist_pct": round(ema20_dist_pct, 3),
    }


# ── Position sizing ───────────────────────────────────────────────────────────

def compute_qty(portfolio_value: float, entry: float, stop: float) -> float:
    """
    Fixed-risk position sizing with notional cap.
    Risk-based qty is capped at MAX_NOTIONAL_PCT of portfolio to prevent
    oversized positions on low-priced coins with tight stop distances.
    """
    risk_dollars = portfolio_value * RISK_PCT
    stop_distance = abs(entry - stop)
    if stop_distance == 0 or entry == 0:
        return 0.0
    qty = risk_dollars / stop_distance
    max_qty = (portfolio_value * MAX_NOTIONAL_PCT) / entry
    return min(qty, max_qty)


# ── State / daily protection ──────────────────────────────────────────────────

def load_state() -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    default: dict = {
        "date": today,
        "daily_loss_pct": 0.0,
        "consecutive_losses": 0,
        "trading_halted": False,
        "open_trades": {},
    }
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        if state.get("date") != today:
            state.update(default)
    else:
        state = default
    return state


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def check_daily_limits(state: dict) -> bool:
    """Returns True if trading should be halted."""
    if state["trading_halted"]:
        return True
    if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
        return True
    if abs(state["daily_loss_pct"]) >= MAX_DAILY_LOSS_PCT:
        return True
    return False


def record_closed_trade(
    state: dict,
    symbol: str,
    side: str,
    entry: float,
    exit_price: float,
    stop: float,
    pnl_pct: float,
    trade_context: dict | None = None,
) -> dict:
    risk = abs(entry - stop)
    r_multiple = ((exit_price - entry) / risk) if side == "long" else ((entry - exit_price) / risk)
    r_multiple = r_multiple if risk > 0 else 0.0
    outcome = "win" if pnl_pct >= 0 else "loss"

    trade_record: dict = {
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "exit": exit_price,
        "stop": stop,
        "r_multiple": round(r_multiple, 3),
        "pnl_pct": round(pnl_pct, 6),
        "outcome": outcome,
        "closed_at": datetime.now(timezone.utc).isoformat(),
    }
    # Attach entry context if available
    if trade_context:
        for key in ("entry_rsi", "entry_vol_ratio", "entry_atr_pct",
                    "entry_ema20_dist_pct", "trend_1h"):
            if key in trade_context:
                trade_record[key] = trade_context[key]

    state.setdefault("trade_history", []).append(trade_record)
    state["total_closed"] = len(state["trade_history"])

    state["daily_loss_pct"] += min(0.0, pnl_pct)
    state["consecutive_losses"] = state["consecutive_losses"] + 1 if pnl_pct < 0 else 0

    if check_daily_limits(state):
        state["trading_halted"] = True
        print("  *** DAILY LIMIT HIT — trading halted for the rest of the day ***")

    total = state["total_closed"]

    # Quick recap every 5 trades
    if total % 5 == 0:
        from analysis import trade_review
        trade_review(state["trade_history"][-5:], trigger_count=total)

    # Deep strategy review every 50 trades
    if total % 50 == 0:
        from analysis import strategy_deep_review
        strategy_deep_review(state["trade_history"], trigger_count=total)

    return state


# ── Position management ───────────────────────────────────────────────────────

def manage_open_positions(trading_client: TradingClient, state: dict) -> dict:
    """
    For each tracked open trade:
    • If stop hit  → close position, record loss
    • If TP hit    → close position, record win
    • If +1R       → move stop to breakeven (state only)
    • If +1.5R     → take partial (50 % qty) and trail
    """
    open_trades: dict = state.get("open_trades", {})
    portfolio_value = float(trading_client.get_account().portfolio_value)

    for sym, trade in list(open_trades.items()):
        try:
            pos = trading_client.get_open_position(sym)
        except Exception:
            del open_trades[sym]
            continue

        price  = float(pos.current_price)
        entry  = trade["entry"]
        stop   = trade["stop"]
        tp     = trade["tp"]
        side   = trade["side"]
        qty    = float(pos.qty)
        risk   = abs(entry - stop)
        pnl_r  = ((price - entry) / risk) if side == "long" else ((entry - price) / risk)

        print(f"  {sym} [{side}] price={price:.2f}  PnL={pnl_r:+.2f}R  stop={stop:.2f}  tp={tp:.2f}")

        # Stop hit
        stop_hit = (side == "long" and price <= stop) or (side == "short" and price >= stop)
        # TP hit
        tp_hit = (side == "long" and price >= tp) or (side == "short" and price <= tp)

        if stop_hit or tp_hit:
            reason = "TP" if tp_hit else "STOP"
            print(f"  → {reason} hit — closing {sym}")
            try:
                close_side = OrderSide.SELL if side == "long" else OrderSide.BUY
                trading_client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=round(qty, 8),
                    side=close_side, time_in_force=TimeInForce.IOC,
                ))
                pnl_pct = ((price - entry) / entry) * (1 if side == "long" else -1) * (qty * entry / portfolio_value)
                orig_stop = trade.get("stop", entry)
                state = record_closed_trade(
                    state, sym, side, entry, price, orig_stop, pnl_pct,
                    trade_context=trade,
                )
            except Exception as e:
                print(f"  Close order error: {e}")
            del open_trades[sym]
            continue

        # Move stop to breakeven at +1R
        if pnl_r >= 1.0 and not trade.get("breakeven_moved"):
            print(f"  +1R reached — moving stop to breakeven ({entry:.2f})")
            trade["stop"] = entry
            trade["breakeven_moved"] = True

        # Partial exit at +1.5R (50 % of position)
        if pnl_r >= 1.5 and not trade.get("partial_taken") and qty > 0:
            partial_qty = round(qty * 0.5, 8)
            print(f"  +1.5R — taking 50 % partial ({partial_qty:.8f} units)")
            try:
                close_side = OrderSide.SELL if side == "long" else OrderSide.BUY
                trading_client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=partial_qty,
                    side=close_side, time_in_force=TimeInForce.IOC,
                ))
                trade["partial_taken"] = True
                trade["qty"] = round(qty - partial_qty, 8)
            except Exception as e:
                print(f"  Partial order error: {e}")

        open_trades[sym] = trade

    state["open_trades"] = open_trades
    return state


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    state = load_state()

    trading_client = TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )

    account = trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    buying_power    = float(account.buying_power)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"\n{'='*60}")
    print(f" {now_str}")
    print(f" Portfolio: ${portfolio_value:,.2f}   Buying power: ${buying_power:,.2f}")
    print(f" Daily P&L: {state['daily_loss_pct']*100:+.2f}%   Consec. losses: {state['consecutive_losses']}")
    print(f"{'='*60}")

    # ── Silent stop/TP monitor (Alpaca crypto has no native stop orders) ──
    if state.get("open_trades"):
        open_trades: dict = state["open_trades"]
        for sym, trade in list(open_trades.items()):
            try:
                pos = trading_client.get_open_position(sym)
            except Exception:
                del open_trades[sym]
                continue
            price  = float(pos.current_price)
            entry  = trade["entry"]
            stop   = trade["stop"]
            tp     = trade["tp"]
            side   = trade["side"]
            qty    = float(pos.qty)
            risk   = abs(entry - stop)
            pnl_r  = ((price - entry) / risk) if side == "long" and risk else ((entry - price) / risk) if risk else 0

            stop_hit = (side == "long" and price <= stop) or (side == "short" and price >= stop)
            tp_hit   = (side == "long" and price >= tp)  or (side == "short" and price <= tp)

            if stop_hit or tp_hit:
                reason = "TP" if tp_hit else "STOP"
                print(f"\n  *** {reason} HIT: {sym} @ ${price:.4f} (entry=${entry:.4f}) ***")
                try:
                    close_side = OrderSide.SELL if side == "long" else OrderSide.BUY
                    trading_client.submit_order(MarketOrderRequest(
                        symbol=sym, qty=round(qty, 8),
                        side=close_side, time_in_force=TimeInForce.IOC,
                    ))
                    pnl_pct = ((price - entry) / entry) * (1 if side == "long" else -1) * (qty * entry / portfolio_value)
                    state = record_closed_trade(state, sym, side, entry, price, stop, pnl_pct, trade)
                    print(f"  Closed {sym}  PnL={pnl_r:+.2f}R")
                except Exception as e:
                    print(f"  Close error: {e}")
                del open_trades[sym]
                continue

            # Move stop to breakeven at +1R (silent)
            if pnl_r >= 1.0 and not trade.get("breakeven_moved"):
                print(f"  {sym}: +1R — stop moved to breakeven (${entry:.4f})")
                trade["stop"] = entry
                trade["breakeven_moved"] = True

            # Partial exit at +1.5R (silent)
            if pnl_r >= 1.5 and not trade.get("partial_taken") and qty > 0:
                partial_qty = round(qty * 0.5, 8)
                print(f"  {sym}: +1.5R — taking 50% partial")
                try:
                    close_side = OrderSide.SELL if side == "long" else OrderSide.BUY
                    trading_client.submit_order(MarketOrderRequest(
                        symbol=sym, qty=partial_qty,
                        side=close_side, time_in_force=TimeInForce.IOC,
                    ))
                    trade["partial_taken"] = True
                    trade["qty"] = round(qty - partial_qty, 8)
                except Exception as e:
                    print(f"  Partial error: {e}")

            open_trades[sym] = trade
        state["open_trades"] = open_trades
        save_state(state)

    # ── Daily protection check ──
    if check_daily_limits(state):
        print(" Trading halted for today (daily limit reached).")
        return

    # ── Scan for new entries ──
    for symbol in SYMBOLS:
        alpaca_sym = symbol.replace("/", "")
        print(f"[{symbol}]")

        # Skip if already in position
        held_qty = 0.0
        try:
            pos = trading_client.get_open_position(alpaca_sym)
            held_qty = float(pos.qty)
        except Exception:
            pass
        if held_qty != 0:
            print(f"  Position open ({held_qty:.6f} units) — skipping entry scan")
            print()
            continue

        # Fetch bars
        try:
            df_1h  = fetch_bars(symbol, TimeFrame.Hour, 60)
            df_15m = fetch_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), 60)
            df_5m  = fetch_bars(symbol, TimeFrame(5,  TimeFrameUnit.Minute), 80)
        except Exception as e:
            print(f"  Data fetch error: {e}")
            print()
            continue

        # 1. Trend filter (1H)
        trend = get_trend(df_1h)
        e20_1h = ema(df_1h["close"], EMA_FAST).iloc[-1]
        e50_1h = ema(df_1h["close"], EMA_SLOW).iloc[-1]
        print(f"  1H trend : {trend.upper()}  (EMA{EMA_FAST}={e20_1h:.0f}  EMA{EMA_SLOW}={e50_1h:.0f})")

        if trend == "neutral":
            print("  → Skip: messy EMAs")
            print()
            continue

        # 2. Market activity filter (5m)
        if not is_market_active(df_5m):
            print("  → Skip: low volume / choppy")
            print()
            continue

        # 3. 15m setup confirmation
        bo_15m = find_15m_breakout(df_15m, trend)
        if bo_15m is None:
            print(f"  → Skip: no 15m {'breakout' if trend=='bullish' else 'breakdown'} candle")
            print()
            continue
        print(f"  15m {'breakout' if trend=='bullish' else 'breakdown'}: level={bo_15m['level']:.2f}  vol×{bo_15m['vol_ratio']:.1f}")

        # 4. 5m entry
        if trend == "bullish":
            setup = find_5m_long_entry(df_5m)
        else:
            setup = find_5m_short_entry(df_5m)

        if setup is None:
            print("  → Skip: 5m entry conditions not met (pullback / continuation / RSI / VWAP)")
            print()
            continue

        # 5. Position sizing
        entry = setup["entry"]
        stop  = setup["stop"]
        tp    = setup["tp"]
        qty   = compute_qty(portfolio_value, entry, stop)
        notional = qty * entry
        rr = TP_RATIO

        print(f"  ✓ SETUP: {setup['side'].upper()}")
        print(f"    Entry={entry:.2f}  Stop={stop:.2f}  TP={tp:.2f}  RR={rr}")
        print(f"    Qty={qty:.6f}  Notional=${notional:.2f}  RSI={setup['rsi']:.1f}")

        if notional < 1.0 or qty <= 0:
            print("  → Skip: position too small")
            print()
            continue

        # 6. Execute
        side = OrderSide.BUY if setup["side"] == "long" else OrderSide.SELL
        try:
            order = trading_client.submit_order(MarketOrderRequest(
                symbol=alpaca_sym,
                qty=round(qty, 8),
                side=side,
                time_in_force=TimeInForce.GTC,
            ))
            print(f"  → Order submitted: {order.id}")
            state["open_trades"][alpaca_sym] = {
                "side": setup["side"],
                "entry": entry,
                "stop": stop,
                "tp": tp,
                "qty": qty,
                "order_id": str(order.id),
                "breakeven_moved": False,
                "partial_taken": False,
                # entry context for post-trade analysis
                "entry_rsi": setup.get("rsi"),
                "entry_vol_ratio": setup.get("vol_ratio"),
                "entry_atr_pct": setup.get("atr_pct"),
                "entry_ema20_dist_pct": setup.get("ema20_dist_pct"),
                "trend_1h": trend,
            }
        except Exception as e:
            print(f"  → Order failed: {e}")

        print()

    save_state(state)


if __name__ == "__main__":
    run()
