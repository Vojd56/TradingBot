"""Durable, conservative guard for broker submissions without native idempotency.

One unresolved submission per instrument. UNKNOWN submissions require manual
broker reconciliation; deleting this database is not a safe retry mechanism.
"""
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone


class SubmissionJournal:
    def __init__(self, path="data/order_submissions.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS submissions (
                ticker TEXT PRIMARY KEY, payload TEXT NOT NULL, state TEXT NOT NULL,
                broker_id TEXT, updated_at TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS completed_submissions (
                broker_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, payload TEXT NOT NULL,
                state TEXT NOT NULL, updated_at TEXT NOT NULL)""")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def reserve(self, ticker, payload, context=None):
        with self._connect() as db:
            try:
                db.execute("INSERT INTO submissions VALUES (?, ?, 'UNKNOWN', NULL, ?)",
                           (ticker, json.dumps({"request": payload, "context": context or {}}, sort_keys=True, allow_nan=False), datetime.now(timezone.utc).isoformat()))
            except sqlite3.IntegrityError as exc:
                raise RuntimeError(f"Unresolved broker submission for {ticker}; reconcile before retry.") from exc

    def acknowledge(self, ticker, broker_id):
        with self._connect() as db:
            db.execute("UPDATE submissions SET state='SUBMITTED', broker_id=? WHERE ticker=?",
                       (str(broker_id), ticker))

    def reject(self, ticker):
        """Only use for an explicit HTTP rejection, never a transport timeout."""
        with self._connect() as db:
            db.execute("DELETE FROM submissions WHERE ticker=?", (ticker,))

    def unresolved(self, ticker):
        with self._connect() as db:
            return db.execute("SELECT state, broker_id FROM submissions WHERE ticker=?", (ticker,)).fetchone()

    def reconcile_terminal(self, ticker, broker_order):
        """Release only a matching broker-confirmed terminal order."""
        if broker_order.get("status") not in ("FILLED", "CANCELLED", "REJECTED", "EXPIRED"):
            return False
        broker_id = broker_order.get("id")
        if broker_id is None:
            return False
        with self._connect() as db:
            row = db.execute("SELECT payload FROM submissions WHERE ticker=? AND broker_id=?",
                             (ticker, str(broker_id))).fetchone()
            if row is None:
                return False
            db.execute("INSERT OR REPLACE INTO completed_submissions VALUES (?, ?, ?, ?, ?)",
                       (str(broker_id), ticker, row[0], broker_order["status"], datetime.now(timezone.utc).isoformat()))
            result = db.execute("DELETE FROM submissions WHERE ticker=? AND broker_id=?",
                                (ticker, str(broker_id)))
            return result.rowcount == 1

    def contexts(self):
        """Local protection levels/ownership, never claimed execution prices."""
        with self._connect() as db:
            rows = db.execute("SELECT ticker, payload FROM completed_submissions WHERE state='FILLED' ORDER BY updated_at").fetchall()
            rows += db.execute("SELECT ticker, payload FROM submissions ORDER BY updated_at").fetchall()
        contexts = {}
        for ticker, payload in rows:
            data = json.loads(payload)
            context = data.get("context", {})
            if data.get("request", {}).get("quantity", 0) > 0 and context:
                contexts[ticker] = context
        return contexts
