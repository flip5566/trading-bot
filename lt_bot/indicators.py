"""
Technical indicator calculations — pure pandas/numpy, no external TA library.
All functions accept pd.Series or pd.DataFrame and return the same.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Primitives ─────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    fast_ema  = ema(series, fast)
    slow_ema  = ema(series, slow)
    macd_line = fast_ema - slow_ema
    sig_line  = ema(macd_line, signal)
    histogram = macd_line - sig_line
    return macd_line, sig_line, histogram


def true_range(df: pd.DataFrame) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low  - close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(com=period - 1, adjust=False).mean()


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""
    mid   = sma(series, period)
    sigma = series.rolling(window=period).std(ddof=0)
    upper = mid + std_dev * sigma
    lower = mid - std_dev * sigma
    return upper, mid, lower


def adx(
    df: pd.DataFrame,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, +DI, -DI)."""
    high  = df["high"]
    low   = df["low"]

    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Keep only the dominant direction
    mask_plus  = (plus_dm > minus_dm)
    mask_minus = (minus_dm >= plus_dm)
    plus_dm  = plus_dm.where(mask_plus,  0.0)
    minus_dm = minus_dm.where(mask_minus, 0.0)

    atr_val  = atr(df, period)
    plus_di  = 100 * ema(plus_dm,  period) / atr_val.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / atr_val.replace(0, np.nan)

    dx     = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_   = ema(dx, period)
    return adx_, plus_di, minus_di


def volume_ma(series: pd.Series, period: int = 20) -> pd.Series:
    return sma(series, period)


# ── All-in-one ─────────────────────────────────────────────────────────────────

def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicators to a DataFrame with columns: open, high, low, close, volume.
    Returns a new DataFrame; original is not modified.
    Requires at least 210 rows for EMA-200 to be meaningful.
    """
    df = df.copy()
    close = df["close"]

    df["ema50"]  = ema(close, 50)
    df["ema100"] = ema(close, 100)
    df["ema200"] = ema(close, 200)

    df["rsi"] = rsi(close, 14)

    macd_line, sig_line, hist = macd(close, 12, 26, 9)
    df["macd"]        = macd_line
    df["macd_signal"] = sig_line
    df["macd_hist"]   = hist

    atr_val = atr(df, 14)
    df["atr"]     = atr_val
    df["atr_pct"] = atr_val / close.replace(0, np.nan) * 100

    adx_val, plus_di, minus_di = adx(df, 14)
    df["adx"]      = adx_val
    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di

    bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)
    df["bb_upper"]  = bb_upper
    df["bb_middle"] = bb_mid
    df["bb_lower"]  = bb_lower

    df["volume_ma"] = volume_ma(df["volume"], 20)

    return df


def df_to_indicator_rows(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame with indicator columns to a list of row dicts for DB upsert."""
    cols = [
        "ts", "ema50", "ema100", "ema200", "rsi", "macd", "macd_signal", "macd_hist",
        "adx", "plus_di", "minus_di", "atr", "atr_pct",
        "bb_upper", "bb_middle", "bb_lower", "volume_ma",
    ]
    # Add ts column from index if not present
    if "ts" not in df.columns:
        df = df.copy()
        df["ts"] = df.index.astype(str).str[:10]

    rows = []
    for _, row in df.iterrows():
        r = {}
        for col in cols:
            val = row.get(col, None)
            r[col] = None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
        rows.append(r)
    return rows
