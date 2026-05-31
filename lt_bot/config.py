"""
Long-term bot configuration — all constants in one place.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

# ── Asset universe ─────────────────────────────────────────────────────────────

# Alpaca symbol format → used for OHLCV data
ASSETS: dict[str, str] = {
    "BTC":  "BTC/USD",
    "ETH":  "ETH/USD",
    "SOL":  "SOL/USD",
    "ADA":  "ADA/USD",
    "AVAX": "AVAX/USD",
    "LINK": "LINK/USD",
    "DOT":  "DOT/USD",
}

# Which assets are "large cap alts" (not BTC or ETH)
LARGE_CAP_ASSETS: list[str] = ["SOL", "ADA", "AVAX", "LINK", "DOT"]

# CoinGecko IDs for market/price queries
COINGECKO_IDS: dict[str, str] = {
    "BTC":  "bitcoin",
    "ETH":  "ethereum",
    "SOL":  "solana",
    "ADA":  "cardano",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "DOT":  "polkadot",
}

# ── Portfolio targets per regime ───────────────────────────────────────────────

# Fractions must sum to 1.0 per regime.
# LARGE_CAP allocation is split equally among LARGE_CAP_ASSETS when constructing
# specific buy/sell orders.
PORTFOLIO_TARGETS: dict[str, dict[str, float]] = {
    "STRONG_BULL": {"BTC": 0.35, "ETH": 0.30, "LARGE_CAP": 0.25, "STABLECOIN": 0.10},
    "BULL":        {"BTC": 0.35, "ETH": 0.25, "LARGE_CAP": 0.30, "STABLECOIN": 0.10},
    "SIDEWAYS":    {"BTC": 0.40, "ETH": 0.20, "LARGE_CAP": 0.15, "STABLECOIN": 0.25},
    "BEAR":        {"BTC": 0.30, "ETH": 0.10, "LARGE_CAP": 0.00, "STABLECOIN": 0.60},
    "CRASH":       {"BTC": 0.20, "ETH": 0.05, "LARGE_CAP": 0.00, "STABLECOIN": 0.75},
    "EUPHORIA":    {"BTC": 0.25, "ETH": 0.15, "LARGE_CAP": 0.10, "STABLECOIN": 0.50},
}

# ── Scoring weights (must sum to 1.0) ─────────────────────────────────────────

SCORE_WEIGHTS: dict[str, float] = {
    "trend":     0.25,
    "momentum":  0.15,
    "cycle":     0.20,
    "liquidity": 0.15,
    "sentiment": 0.10,
    "risk":      0.15,
}

# ── DCA schedule: (score_min, score_max, multiplier) ──────────────────────────

DCA_SCHEDULE: list[tuple[float, float, float]] = [
    (85, 100, 2.5),
    (70,  85, 1.0),
    (50,  70, 0.5),
    ( 0,  50, 0.0),
]

# Extra DCA trigger: price correction while trend intact
CORRECTION_BUY_THRESHOLD_PCT = -0.15   # >15% price drop triggers extra DCA

# ── Risk limits ────────────────────────────────────────────────────────────────

MAX_SINGLE_ASSET_PCT    = 0.40   # No single asset > 40% of portfolio
MAX_ALTCOIN_PCT         = 0.40   # Total large-cap alts ≤ 40%
MIN_STABLECOIN_PCT      = 0.10   # Always keep ≥ 10% stablecoin reserve

# Emergency mode triggers
CRASH_DAILY_DROP_PCT    = -0.10  # BTC drops ≥ 10% in 1 day
CRASH_ATR_PCT_THRESHOLD = 0.08   # ATR ≥ 8% of price

# Profit-taking triggers
PT_RSI_OVERBOUGHT       = 75.0
PT_FUNDING_EXTREME      = 0.05   # Funding rate ≥ 0.05% per 8h (annualised ~54%)
PT_FEAR_GREED_EUPHORIA  = 80     # F&G ≥ 80

# ── Bitcoin halving dates (for cycle scoring) ──────────────────────────────────

BTC_HALVINGS: list[date] = [
    date(2012, 11, 28),
    date(2016,  7,  9),
    date(2020,  5, 11),
    date(2024,  4, 19),
]

# ── Technical indicator periods ────────────────────────────────────────────────

EMA_PERIODS      = [50, 100, 200]
RSI_PERIOD       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
ADX_PERIOD       = 14
ATR_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
VOLUME_MA_PERIOD = 20

# ── File / DB paths ────────────────────────────────────────────────────────────

ROOT_DIR    = Path(__file__).parent.parent
DB_PATH     = ROOT_DIR / "lt_bot.db"
REPORTS_DIR = ROOT_DIR / "lt_reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Number of daily bars to fetch (enough for EMA-200 + buffer)
DAILY_BARS = 400
