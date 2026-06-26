from dotenv import load_dotenv
load_dotenv()

from strategy import fetch_bars, ema, rsi, SYMBOLS
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import numpy as np

WATCHLIST = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "AVAX/USD", "LINK/USD", "DOT/USD", "UNI/USD"]

print("=== MARKET DROP ANALYSIS — TODAY ===")
print()

for symbol in WATCHLIST:
    try:
        df_1h  = fetch_bars(symbol, TimeFrame.Hour, 72)
        df_15m = fetch_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), 96)

        close = df_1h["close"]
        vol   = df_1h["volume"]

        price_now = close.iloc[-1]
        price_24h = close.iloc[-24] if len(close) >= 24 else close.iloc[0]
        price_48h = close.iloc[-48] if len(close) >= 48 else close.iloc[0]
        price_72h = close.iloc[-72] if len(close) >= 72 else close.iloc[0]

        chg_24h = (price_now / price_24h - 1) * 100
        chg_48h = (price_now / price_48h - 1) * 100
        chg_72h = (price_now / price_72h - 1) * 100

        avg_vol = vol.rolling(20).mean()
        recent_vol_ratio = float(vol.iloc[-3:].mean() / avg_vol.iloc[-1]) if avg_vol.iloc[-1] > 0 else 0

        # Worst 1H candles in last 24h
        df_24h = df_1h.tail(24).copy()
        df_24h["chg"] = (df_24h["close"] - df_24h["open"]) / df_24h["open"] * 100
        worst = df_24h.nsmallest(2, "chg")

        # Volume on down vs up hours
        df_24h["is_down"] = df_24h["chg"] < 0
        down_vol = df_24h[df_24h["is_down"]]["volume"].mean()
        up_vol   = df_24h[~df_24h["is_down"]]["volume"].mean()
        vol_bias = "SELL PRESSURE" if down_vol > up_vol * 1.2 else ("BUY PRESSURE" if up_vol > down_vol * 1.2 else "NEUTRAL")

        rsi_now = float(rsi(close).iloc[-1])
        e20 = float(ema(close, 20).iloc[-1])
        e50 = float(ema(close, 50).iloc[-1])
        side = "above" if price_now > e20 else "below"

        print(f"{symbol}")
        print(f"  Price       : ${price_now:,.4f}")
        print(f"  24h change  : {chg_24h:+.2f}%")
        print(f"  48h change  : {chg_48h:+.2f}%")
        print(f"  72h change  : {chg_72h:+.2f}%")
        print(f"  RSI (1H)    : {rsi_now:.1f}")
        print(f"  EMA20/50    : ${e20:.4f} / ${e50:.4f}  (price {side} EMA20)")
        print(f"  Volume bias : {vol_bias}  (down-vol avg {down_vol:.1f} vs up-vol avg {up_vol:.1f})")
        print(f"  Recent vol  : {recent_vol_ratio:.2f}x avg (last 3 bars)")
        print(f"  Biggest down candles (24h):")
        for _, row in worst.iterrows():
            ts = str(row.name)[:16]
            print(f"    {ts}  {row['chg']:+.2f}%  vol={row['volume']:.2f}")
        print()

    except Exception as e:
        print(f"{symbol}: ERROR — {e}")
        print()

print("=== SUMMARY ===")
