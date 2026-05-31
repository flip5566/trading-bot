"""
SQLite database wrapper — stores all bot state and history.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DB_PATH


# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS price_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    UNIQUE(symbol, timeframe, ts)
);

CREATE TABLE IF NOT EXISTS indicators (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    ema50       REAL,
    ema100      REAL,
    ema200      REAL,
    rsi         REAL,
    macd        REAL,
    macd_signal REAL,
    macd_hist   REAL,
    adx         REAL,
    plus_di     REAL,
    minus_di    REAL,
    atr         REAL,
    atr_pct     REAL,
    bb_upper    REAL,
    bb_middle   REAL,
    bb_lower    REAL,
    volume_ma   REAL,
    UNIQUE(symbol, timeframe, ts)
);

CREATE TABLE IF NOT EXISTS market_data (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                   TEXT    NOT NULL UNIQUE,
    btc_dominance        REAL,
    eth_btc_ratio        REAL,
    total_market_cap     REAL,
    altcoin_market_cap   REAL,
    stablecoin_market_cap REAL,
    btc_market_cap       REAL,
    eth_market_cap       REAL
);

CREATE TABLE IF NOT EXISTS sentiment_data (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                     TEXT    NOT NULL UNIQUE,
    fear_greed_index       REAL,
    fear_greed_label       TEXT,
    btc_funding_rate       REAL,
    eth_funding_rate       REAL,
    btc_open_interest      REAL,
    btc_liq_long           REAL,
    btc_liq_short          REAL
);

CREATE TABLE IF NOT EXISTS ai_scores (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT    NOT NULL UNIQUE,
    total_score    REAL,
    trend_score    REAL,
    momentum_score REAL,
    cycle_score    REAL,
    liquidity_score REAL,
    sentiment_score REAL,
    risk_score     REAL,
    market_regime  TEXT,
    score_label    TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT    NOT NULL UNIQUE,
    total_value         REAL,
    regime              TEXT,
    ai_score            REAL,
    allocations_json    TEXT,
    targets_json        TEXT
);

CREATE TABLE IF NOT EXISTS trading_decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT    NOT NULL,
    symbol           TEXT,
    action           TEXT,
    reason           TEXT,
    ai_score         REAL,
    market_regime    TEXT,
    dca_multiplier   REAL,
    recommended_usd  REAL,
    executed         INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS performance_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT    NOT NULL UNIQUE,
    portfolio_value REAL,
    daily_return    REAL,
    running_max     REAL,
    drawdown_pct    REAL
);
"""


# ── Database class ─────────────────────────────────────────────────────────────

class Database:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._init()

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── Price / indicators ────────────────────────────────────────────────────

    def upsert_price_bars(self, symbol: str, timeframe: str, rows: list[dict]) -> None:
        sql = """
        INSERT INTO price_history (symbol, timeframe, ts, open, high, low, close, volume)
        VALUES (:symbol, :timeframe, :ts, :open, :high, :low, :close, :volume)
        ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET
            open=excluded.open, high=excluded.high, low=excluded.low,
            close=excluded.close, volume=excluded.volume
        """
        with self._conn() as conn:
            for r in rows:
                conn.execute(sql, {**r, "symbol": symbol, "timeframe": timeframe})

    def upsert_indicators(self, symbol: str, timeframe: str, rows: list[dict]) -> None:
        sql = """
        INSERT INTO indicators
            (symbol, timeframe, ts, ema50, ema100, ema200, rsi, macd, macd_signal,
             macd_hist, adx, plus_di, minus_di, atr, atr_pct, bb_upper, bb_middle,
             bb_lower, volume_ma)
        VALUES
            (:symbol, :timeframe, :ts, :ema50, :ema100, :ema200, :rsi, :macd,
             :macd_signal, :macd_hist, :adx, :plus_di, :minus_di, :atr, :atr_pct,
             :bb_upper, :bb_middle, :bb_lower, :volume_ma)
        ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET
            ema50=excluded.ema50, ema100=excluded.ema100, ema200=excluded.ema200,
            rsi=excluded.rsi, macd=excluded.macd, macd_signal=excluded.macd_signal,
            macd_hist=excluded.macd_hist, adx=excluded.adx, plus_di=excluded.plus_di,
            minus_di=excluded.minus_di, atr=excluded.atr, atr_pct=excluded.atr_pct,
            bb_upper=excluded.bb_upper, bb_middle=excluded.bb_middle,
            bb_lower=excluded.bb_lower, volume_ma=excluded.volume_ma
        """
        with self._conn() as conn:
            for r in rows:
                conn.execute(sql, {**r, "symbol": symbol, "timeframe": timeframe})

    def get_price_history(self, symbol: str, timeframe: str, limit: int = 300) -> list[dict]:
        sql = """
        SELECT ts, open, high, low, close, volume
        FROM price_history
        WHERE symbol=? AND timeframe=?
        ORDER BY ts DESC LIMIT ?
        """
        with self._conn() as conn:
            rows = conn.execute(sql, (symbol, timeframe, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_latest_indicators(self, symbol: str, timeframe: str = "1D") -> dict | None:
        sql = """
        SELECT * FROM indicators
        WHERE symbol=? AND timeframe=?
        ORDER BY ts DESC LIMIT 1
        """
        with self._conn() as conn:
            row = conn.execute(sql, (symbol, timeframe)).fetchone()
        return dict(row) if row else None

    def get_indicator_history(self, symbol: str, timeframe: str, limit: int = 60) -> list[dict]:
        sql = """
        SELECT * FROM indicators
        WHERE symbol=? AND timeframe=?
        ORDER BY ts DESC LIMIT ?
        """
        with self._conn() as conn:
            rows = conn.execute(sql, (symbol, timeframe, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Market data ───────────────────────────────────────────────────────────

    def upsert_market_data(self, data: dict) -> None:
        data.setdefault("ts", self._now())
        sql = """
        INSERT INTO market_data
            (ts, btc_dominance, eth_btc_ratio, total_market_cap,
             altcoin_market_cap, stablecoin_market_cap, btc_market_cap, eth_market_cap)
        VALUES
            (:ts, :btc_dominance, :eth_btc_ratio, :total_market_cap,
             :altcoin_market_cap, :stablecoin_market_cap, :btc_market_cap, :eth_market_cap)
        ON CONFLICT(ts) DO UPDATE SET
            btc_dominance=excluded.btc_dominance,
            eth_btc_ratio=excluded.eth_btc_ratio,
            total_market_cap=excluded.total_market_cap,
            altcoin_market_cap=excluded.altcoin_market_cap,
            stablecoin_market_cap=excluded.stablecoin_market_cap,
            btc_market_cap=excluded.btc_market_cap,
            eth_market_cap=excluded.eth_market_cap
        """
        with self._conn() as conn:
            conn.execute(sql, data)

    def get_latest_market_data(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM market_data ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def get_market_data_history(self, limit: int = 30) -> list[dict]:
        sql = "SELECT * FROM market_data ORDER BY ts DESC LIMIT ?"
        with self._conn() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Sentiment data ────────────────────────────────────────────────────────

    def upsert_sentiment(self, data: dict) -> None:
        data.setdefault("ts", self._now())
        sql = """
        INSERT INTO sentiment_data
            (ts, fear_greed_index, fear_greed_label, btc_funding_rate,
             eth_funding_rate, btc_open_interest, btc_liq_long, btc_liq_short)
        VALUES
            (:ts, :fear_greed_index, :fear_greed_label, :btc_funding_rate,
             :eth_funding_rate, :btc_open_interest, :btc_liq_long, :btc_liq_short)
        ON CONFLICT(ts) DO UPDATE SET
            fear_greed_index=excluded.fear_greed_index,
            fear_greed_label=excluded.fear_greed_label,
            btc_funding_rate=excluded.btc_funding_rate,
            eth_funding_rate=excluded.eth_funding_rate,
            btc_open_interest=excluded.btc_open_interest,
            btc_liq_long=excluded.btc_liq_long,
            btc_liq_short=excluded.btc_liq_short
        """
        with self._conn() as conn:
            conn.execute(sql, data)

    def get_latest_sentiment(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sentiment_data ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # ── AI scores ─────────────────────────────────────────────────────────────

    def upsert_score(self, data: dict) -> None:
        data.setdefault("ts", self._now())
        sql = """
        INSERT INTO ai_scores
            (ts, total_score, trend_score, momentum_score, cycle_score,
             liquidity_score, sentiment_score, risk_score, market_regime, score_label)
        VALUES
            (:ts, :total_score, :trend_score, :momentum_score, :cycle_score,
             :liquidity_score, :sentiment_score, :risk_score, :market_regime, :score_label)
        ON CONFLICT(ts) DO UPDATE SET
            total_score=excluded.total_score,
            trend_score=excluded.trend_score,
            momentum_score=excluded.momentum_score,
            cycle_score=excluded.cycle_score,
            liquidity_score=excluded.liquidity_score,
            sentiment_score=excluded.sentiment_score,
            risk_score=excluded.risk_score,
            market_regime=excluded.market_regime,
            score_label=excluded.score_label
        """
        with self._conn() as conn:
            conn.execute(sql, data)

    def get_latest_score(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM ai_scores ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def get_score_history(self, limit: int = 30) -> list[dict]:
        sql = "SELECT * FROM ai_scores ORDER BY ts DESC LIMIT ?"
        with self._conn() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Portfolio snapshots ───────────────────────────────────────────────────

    def save_portfolio_snapshot(self, data: dict) -> None:
        data.setdefault("ts", self._now())
        if "allocations" in data:
            data["allocations_json"] = json.dumps(data.pop("allocations"))
        if "targets" in data:
            data["targets_json"] = json.dumps(data.pop("targets"))
        sql = """
        INSERT INTO portfolio_snapshots
            (ts, total_value, regime, ai_score, allocations_json, targets_json)
        VALUES
            (:ts, :total_value, :regime, :ai_score, :allocations_json, :targets_json)
        ON CONFLICT(ts) DO UPDATE SET
            total_value=excluded.total_value,
            regime=excluded.regime,
            ai_score=excluded.ai_score,
            allocations_json=excluded.allocations_json,
            targets_json=excluded.targets_json
        """
        with self._conn() as conn:
            conn.execute(sql, data)

    def get_latest_portfolio(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("allocations_json"):
            d["allocations"] = json.loads(d["allocations_json"])
        if d.get("targets_json"):
            d["targets"] = json.loads(d["targets_json"])
        return d

    # ── Trading decisions ─────────────────────────────────────────────────────

    def record_decision(self, data: dict) -> None:
        data.setdefault("ts", datetime.now(timezone.utc).isoformat())
        sql = """
        INSERT INTO trading_decisions
            (ts, symbol, action, reason, ai_score, market_regime,
             dca_multiplier, recommended_usd, executed)
        VALUES
            (:ts, :symbol, :action, :reason, :ai_score, :market_regime,
             :dca_multiplier, :recommended_usd, :executed)
        """
        data.setdefault("executed", 0)
        with self._conn() as conn:
            conn.execute(sql, data)

    def get_recent_decisions(self, days: int = 7) -> list[dict]:
        sql = """
        SELECT * FROM trading_decisions
        WHERE ts >= date('now', ?)
        ORDER BY ts DESC
        """
        with self._conn() as conn:
            rows = conn.execute(sql, (f"-{days} days",)).fetchall()
        return [dict(r) for r in rows]

    # ── Performance tracking ──────────────────────────────────────────────────

    def record_performance(self, data: dict) -> None:
        data.setdefault("ts", self._now())
        sql = """
        INSERT INTO performance_history
            (ts, portfolio_value, daily_return, running_max, drawdown_pct)
        VALUES
            (:ts, :portfolio_value, :daily_return, :running_max, :drawdown_pct)
        ON CONFLICT(ts) DO UPDATE SET
            portfolio_value=excluded.portfolio_value,
            daily_return=excluded.daily_return,
            running_max=excluded.running_max,
            drawdown_pct=excluded.drawdown_pct
        """
        with self._conn() as conn:
            conn.execute(sql, data)
