"""
Data Collection Engine — fetches price, market, and sentiment data.

Sources:
  - Alpaca Crypto API  : OHLCV daily bars (authenticated)
  - CoinGecko API      : market caps, BTC dominance (free, no key needed)
  - Alternative.me     : Fear & Greed Index (free)
  - Binance Public API : funding rates, open interest (free)
"""
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone, timedelta

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from .config import ASSETS, COINGECKO_IDS, DAILY_BARS
from .database import Database
from .indicators import calculate_all, df_to_indicator_rows

load_dotenv()
log = logging.getLogger(__name__)

COINGECKO_GLOBAL  = "https://api.coingecko.com/api/v3/global"
COINGECKO_PRICES  = "https://api.coingecko.com/api/v3/simple/price"
FEAR_GREED_URL    = "https://api.alternative.me/fng/?limit=1"
BINANCE_FUND_URL  = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_OI_URL    = "https://fapi.binance.com/fapi/v1/openInterest"

_REQUEST_TIMEOUT  = 15
_RATE_LIMIT_PAUSE = 1.2   # seconds between CoinGecko calls (free tier ≈ 30/min)


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning("GET %s failed (attempt %d): %s", url, attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ── Alpaca price data ──────────────────────────────────────────────────────────

def _alpaca_client() -> CryptoHistoricalDataClient:
    return CryptoHistoricalDataClient(
        api_key=os.environ.get("ALPACA_API_KEY", ""),
        secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
    )


def fetch_daily_bars(symbol_slash: str, limit: int = DAILY_BARS) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars for a symbol like 'BTC/USD'.
    Returns DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
    """
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=int(limit * 1.6))  # buffer for weekends/gaps

    req = CryptoBarsRequest(
        symbol_or_symbols=symbol_slash,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    client = _alpaca_client()
    raw = client.get_crypto_bars(req)
    df  = raw.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol_slash, level=0)

    df = df.sort_index()[["open", "high", "low", "close", "volume"]]
    return df.tail(limit)


def collect_price_and_indicators(ticker: str, db: Database) -> bool:
    """Fetch daily bars for `ticker`, calculate indicators, store both to DB."""
    symbol = ASSETS.get(ticker)
    if not symbol:
        log.error("Unknown ticker: %s", ticker)
        return False

    try:
        df = fetch_daily_bars(symbol)
        if df.empty:
            log.warning("%s: no data returned", symbol)
            return False
    except Exception as exc:
        log.error("%s: fetch error — %s", symbol, exc)
        return False

    # Store raw OHLCV
    price_rows = [
        {
            "ts":     str(idx)[:10],
            "open":   float(r["open"]),
            "high":   float(r["high"]),
            "low":    float(r["low"]),
            "close":  float(r["close"]),
            "volume": float(r["volume"]),
        }
        for idx, r in df.iterrows()
    ]
    db.upsert_price_bars(ticker, "1D", price_rows)

    # Calculate and store indicators
    df_ind = calculate_all(df)
    ind_rows = df_to_indicator_rows(df_ind)
    db.upsert_indicators(ticker, "1D", ind_rows)

    # Build weekly bars by resampling and store with indicators
    weekly = df.resample("W").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()

    if len(weekly) >= 5:
        weekly_rows = [
            {
                "ts":     str(idx)[:10],
                "open":   float(r["open"]),
                "high":   float(r["high"]),
                "low":    float(r["low"]),
                "close":  float(r["close"]),
                "volume": float(r["volume"]),
            }
            for idx, r in weekly.iterrows()
        ]
        db.upsert_price_bars(ticker, "1W", weekly_rows)

        if len(weekly) >= 30:
            weekly_ind = calculate_all(weekly)
            weekly_ind_rows = df_to_indicator_rows(weekly_ind)
            db.upsert_indicators(ticker, "1W", weekly_ind_rows)

    log.info("%s: stored %d daily bars + indicators", ticker, len(df))
    return True


# ── CoinGecko market data ──────────────────────────────────────────────────────

def collect_market_data(db: Database) -> bool:
    """Fetch global crypto market data from CoinGecko."""
    data = _get(COINGECKO_GLOBAL)
    if not data:
        log.warning("CoinGecko global data unavailable")
        return False

    g = data.get("data", {})

    btc_dom   = g.get("market_cap_percentage", {}).get("btc")
    total_cap = g.get("total_market_cap", {}).get("usd")

    # Fetch BTC and ETH individual market caps
    time.sleep(_RATE_LIMIT_PAUSE)
    prices = _get(COINGECKO_PRICES, params={
        "ids":               "bitcoin,ethereum,tether,usdc,binance-usd",
        "vs_currencies":     "usd",
        "include_market_cap": "true",
    })

    btc_cap   = None
    eth_cap   = None
    stable_cap = None

    if prices:
        btc_cap  = prices.get("bitcoin",  {}).get("usd_market_cap")
        eth_cap  = prices.get("ethereum", {}).get("usd_market_cap")
        usdt_cap = prices.get("tether",   {}).get("usd_market_cap") or 0
        usdc_cap = prices.get("usdc",     {}).get("usd_market_cap") or 0
        busd_cap = prices.get("binance-usd", {}).get("usd_market_cap") or 0
        stable_cap = usdt_cap + usdc_cap + busd_cap

    alt_cap = None
    if total_cap and btc_cap and eth_cap:
        alt_cap = total_cap - btc_cap - eth_cap - (stable_cap or 0)

    eth_btc = None
    if prices:
        btc_p = prices.get("bitcoin",  {}).get("usd")
        eth_p = prices.get("ethereum", {}).get("usd")
        if btc_p and eth_p:
            eth_btc = eth_p / btc_p

    db.upsert_market_data({
        "btc_dominance":        btc_dom,
        "eth_btc_ratio":        eth_btc,
        "total_market_cap":     total_cap,
        "altcoin_market_cap":   alt_cap,
        "stablecoin_market_cap": stable_cap,
        "btc_market_cap":       btc_cap,
        "eth_market_cap":       eth_cap,
    })
    log.info("Market data collected — BTC dom=%.1f%%  total cap=$%.2fT",
             btc_dom or 0, (total_cap or 0) / 1e12)
    return True


# ── Fear & Greed Index ─────────────────────────────────────────────────────────

def collect_fear_greed(db: Database) -> dict:
    """Fetch current Fear & Greed index from alternative.me."""
    raw = _get(FEAR_GREED_URL)
    if not raw:
        return {}

    item  = raw.get("data", [{}])[0]
    value = item.get("value")
    label = item.get("value_classification", "")
    return {"fear_greed_index": int(value) if value else None, "fear_greed_label": label}


# ── Binance funding rates ──────────────────────────────────────────────────────

def _binance_funding(pair: str) -> float | None:
    """Latest funding rate for a Binance perpetual futures pair (e.g. BTCUSDT)."""
    raw = _get(BINANCE_FUND_URL, params={"symbol": pair, "limit": 1})
    if raw and isinstance(raw, list) and raw:
        try:
            return float(raw[0]["fundingRate"])
        except (KeyError, ValueError):
            pass
    return None


def _binance_oi(pair: str) -> float | None:
    raw = _get(BINANCE_OI_URL, params={"symbol": pair})
    if raw:
        try:
            return float(raw["openInterest"])
        except (KeyError, ValueError):
            pass
    return None


def collect_sentiment_data(db: Database) -> bool:
    """Gather Fear & Greed + Binance funding rates and open interest."""
    fg = collect_fear_greed(db)

    btc_fund = _binance_funding("BTCUSDT")
    eth_fund = _binance_funding("ETHUSDT")
    btc_oi   = _binance_oi("BTCUSDT")

    db.upsert_sentiment({
        "fear_greed_index":  fg.get("fear_greed_index"),
        "fear_greed_label":  fg.get("fear_greed_label"),
        "btc_funding_rate":  btc_fund,
        "eth_funding_rate":  eth_fund,
        "btc_open_interest": btc_oi,
        "btc_liq_long":      None,
        "btc_liq_short":     None,
    })
    log.info(
        "Sentiment: F&G=%s  BTC fund=%.4f%%  OI=%s",
        fg.get("fear_greed_index", "n/a"),
        (btc_fund or 0) * 100,
        f"{btc_oi/1e9:.2f}B" if btc_oi else "n/a",
    )
    return True


# ── Orchestrator ───────────────────────────────────────────────────────────────

def collect_all(db: Database) -> None:
    """Run the full data collection pipeline."""
    log.info("=== Data collection started ===")

    for ticker in ASSETS:
        collect_price_and_indicators(ticker, db)
        time.sleep(0.3)   # gentle rate-limit

    collect_market_data(db)
    time.sleep(_RATE_LIMIT_PAUSE)

    collect_sentiment_data(db)
    log.info("=== Data collection complete ===")
