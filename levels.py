from dotenv import load_dotenv
load_dotenv()

from strategy import (
    fetch_bars, ema, rsi, vwap, get_trend,
    find_15m_breakout, find_5m_long_entry, find_5m_short_entry,
    compute_qty, RISK_PCT, MAX_NOTIONAL_PCT, TP_RATIO
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

client = TradingClient(
    api_key=os.environ["ALPACA_API_KEY"],
    secret_key=os.environ["ALPACA_SECRET_KEY"],
    paper=True,
)
portfolio_value = float(client.get_account().portfolio_value)

TARGETS = ["XRP/USD", "ADA/USD"]

for symbol in TARGETS:
    print(f"\n{'='*52}")
    print(f"  {symbol}")
    print(f"{'='*52}")

    df_1h  = fetch_bars(symbol, TimeFrame.Hour, 60)
    df_15m = fetch_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), 60)
    df_5m  = fetch_bars(symbol, TimeFrame(5,  TimeFrameUnit.Minute), 80)

    trend  = get_trend(df_1h)
    price  = float(df_1h["close"].iloc[-1])
    rsi_1h = float(rsi(df_1h["close"]).iloc[-1])
    e20    = float(ema(df_1h["close"], 20).iloc[-1])
    e50    = float(ema(df_1h["close"], 50).iloc[-1])
    vwap_  = float(vwap(df_5m).iloc[-1])

    print(f"  Price      : ${price:.4f}")
    print(f"  Trend 1H   : {trend.upper()}")
    print(f"  RSI 1H     : {rsi_1h:.1f}")
    print(f"  EMA20/50   : ${e20:.4f} / ${e50:.4f}")
    print(f"  VWAP 5m    : ${vwap_:.4f}  ({'above' if price > vwap_ else 'below'})")
    print()

    # 15m setup
    bo = find_15m_breakout(df_15m, trend)
    if bo:
        print(f"  15m setup  : ✓ {'breakout' if trend=='bullish' else 'breakdown'} level=${bo['level']:.4f}  vol×{bo['vol_ratio']:.1f}")
    else:
        print(f"  15m setup  : ✗ No {'breakout' if trend=='bullish' else 'breakdown'} candle found")

    # 5m entry
    if trend == "bullish":
        setup = find_5m_long_entry(df_5m)
    else:
        setup = find_5m_short_entry(df_5m)

    if setup:
        entry    = setup["entry"]
        stop     = setup["stop"]
        tp       = setup["tp"]
        risk_per = abs(entry - stop)
        qty      = compute_qty(portfolio_value, entry, stop)
        notional = qty * entry
        risk_usd = qty * risk_per
        reward   = qty * abs(tp - entry)

        print(f"  5m entry   : ✓ {setup['side'].upper()} setup confirmed")
        print()
        print(f"  ┌─────────────────────────────────┐")
        print(f"  │  ENTRY       ${entry:.4f}             │")
        print(f"  │  STOP LOSS   ${stop:.4f}  ({(stop/entry-1)*100:+.2f}%)   │")
        print(f"  │  TAKE PROFIT ${tp:.4f}  ({(tp/entry-1)*100:+.2f}%)   │")
        print(f"  │  R:R         1 : {TP_RATIO}                  │")
        print(f"  ├─────────────────────────────────┤")
        print(f"  │  Qty         {qty:.2f} units           │")
        print(f"  │  Notional    ${notional:.2f}            │")
        print(f"  │  Risk $      ${risk_usd:.2f}             │")
        print(f"  │  Reward $    ${reward:.2f}             │")
        print(f"  └─────────────────────────────────┘")
        print(f"  RSI 5m: {setup['rsi']:.1f}  |  Vol ratio: {setup['vol_ratio']}x  |  ATR: {setup['atr_pct']}%")
    else:
        # Show indicative levels even without confirmed entry
        print(f"  5m entry   : ✗ Not triggered yet")
        print()
        # Indicative levels based on EMA20 and ATR
        atr = float((df_5m["high"] - df_5m["low"]).rolling(14).mean().iloc[-1])
        if trend == "bullish":
            ind_entry = price
            ind_stop  = float(ema(df_5m["close"], 20).iloc[-1]) * 0.998
            ind_tp    = ind_entry + TP_RATIO * (ind_entry - ind_stop)
        else:
            ind_entry = price
            ind_stop  = float(ema(df_5m["close"], 20).iloc[-1]) * 1.002
            ind_tp    = ind_entry - TP_RATIO * (ind_stop - ind_entry)

        ind_risk = abs(ind_entry - ind_stop)
        ind_qty  = compute_qty(portfolio_value, ind_entry, ind_stop)

        print(f"  ── INDICATIVE levels (entry not confirmed) ──")
        print(f"  Entry        ~${ind_entry:.4f}  (current price)")
        print(f"  Stop Loss    ~${ind_stop:.4f}  ({(ind_stop/ind_entry-1)*100:+.2f}%)")
        print(f"  Take Profit  ~${ind_tp:.4f}  ({(ind_tp/ind_entry-1)*100:+.2f}%)")
        print(f"  R:R          1 : {TP_RATIO}")
        print(f"  Indicative qty: {ind_qty:.2f} units  (${ind_qty*ind_entry:.2f} notional)")
        # What's missing
        issues = []
        if not bo:
            issues.append("15m breakout candle")
        rsi_5m = float(rsi(df_5m["close"]).iloc[-1])
        if trend == "bullish":
            if not (50 < rsi_5m < 70): issues.append(f"5m RSI={rsi_5m:.0f} (needs 50-70)")
            if df_5m["close"].iloc[-1] <= vwap_: issues.append("price below VWAP")
        else:
            if not (30 < rsi_5m < 50): issues.append(f"5m RSI={rsi_5m:.0f} (needs 30-50)")
            if df_5m["close"].iloc[-1] >= vwap_: issues.append("price above VWAP")
        if not issues:
            issues.append("pullback + continuation candle")
        print(f"  Waiting for  : {', '.join(issues)}")

def _missing(trend, bo, setup, df_5m, vwap_, rsi_1h):
    issues = []
    if not bo:
        issues.append("15m breakout candle")
    rsi_5m = float(rsi(df_5m["close"]).iloc[-1])
    if trend == "bullish":
        if rsi_5m <= 50 or rsi_5m >= 70: issues.append(f"5m RSI ({rsi_5m:.0f}) must be 50–70")
        if df_5m["close"].iloc[-1] <= vwap_: issues.append("price below VWAP")
    else:
        if rsi_5m <= 30 or rsi_5m >= 50: issues.append(f"5m RSI ({rsi_5m:.0f}) must be 30–50")
        if df_5m["close"].iloc[-1] >= vwap_: issues.append("price above VWAP")
    if not issues:
        issues.append("pullback/continuation candle")
    return ", ".join(issues)
