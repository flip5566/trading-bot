import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
from strategy import (
    fetch_bars, ema, rsi, vwap, get_trend, is_market_active,
    find_15m_breakout, find_5m_long_entry, find_5m_short_entry,
    SYMBOLS, VOL_THRESHOLD, VOL_LOOKBACK, compute_qty
)
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

client = TradingClient(api_key=os.environ['ALPACA_API_KEY'], secret_key=os.environ['ALPACA_SECRET_KEY'], paper=True)
portfolio_value = float(client.get_account().portfolio_value)

print(f"=== SIGNAL SCAN  (VOL_THRESHOLD={VOL_THRESHOLD}x) ===")
print()

signals = []

for symbol in SYMBOLS:
    try:
        df_1h  = fetch_bars(symbol, TimeFrame.Hour, 60)
        df_15m = fetch_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), 60)
        df_5m  = fetch_bars(symbol, TimeFrame(5, TimeFrameUnit.Minute), 80)
        trend  = get_trend(df_1h)
        active = is_market_active(df_5m)

        if trend == "neutral":
            print(f"   {symbol:<10} [NEUTRAL ]  skip")
            continue
        if not active:
            print(f"   {symbol:<10} [{trend.upper():<8}]  low volume/ATR")
            continue

        bo = find_15m_breakout(df_15m, trend)
        if bo is None:
            vol = df_15m["volume"]
            avg = vol.rolling(VOL_LOOKBACK).mean()
            best = max((vol.iloc[-i] / avg.iloc[-i] for i in range(1, 9) if avg.iloc[-i] > 0), default=0)
            print(f"   {symbol:<10} [{trend.upper():<8}]  no 15m breakout (best {best:.2f}x, need {VOL_THRESHOLD}x)")
            continue

        setup = find_5m_long_entry(df_5m) if trend == "bullish" else find_5m_short_entry(df_5m)
        if setup is None:
            r5 = float(rsi(df_5m["close"]).iloc[-1])
            p  = float(df_5m["close"].iloc[-1])
            v5 = float(vwap(df_5m).iloc[-1])
            cont = (p > float(df_5m["high"].iloc[-2])) if trend == "bullish" else (p < float(df_5m["low"].iloc[-2]))
            issues = []
            if trend == "bullish":
                if not (50 < r5 < 70): issues.append(f"RSI={r5:.0f} (need 50-70)")
                if p <= v5: issues.append("below VWAP")
            else:
                if not (30 < r5 < 50): issues.append(f"RSI={r5:.0f} (need 30-50)")
                if p >= v5: issues.append("above VWAP")
            if not cont: issues.append("no continuation candle")
            block = ", ".join(issues) if issues else "no EMA20 pullback found"
            print(f"   {symbol:<10} [{trend.upper():<8}]  15m OK (vol x{bo['vol_ratio']:.1f}) | 5m blocked: {block}")
            continue

        entry    = setup["entry"]
        stop     = setup["stop"]
        tp       = setup["tp"]
        qty      = compute_qty(portfolio_value, entry, stop)
        notional = qty * entry
        print(f">>> {symbol:<10} [{trend.upper():<8}]  ** SIGNAL: {setup['side'].upper()} **")
        print(f"    Entry={entry:.4f}  Stop={stop:.4f}  TP={tp:.4f}")
        print(f"    Qty={qty:.4f}  Notional=${notional:.2f}  RSI={setup['rsi']:.0f}  Vol={setup['vol_ratio']}x")
        signals.append(symbol)

    except Exception as e:
        print(f"   {symbol:<10} ERROR: {e}")

print()
if signals:
    print(f"LIVE SIGNALS: {signals}")
else:
    print("No live signals right now.")
