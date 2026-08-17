"""
Persistent activity-event storage (Requirement 24) using SQLite.

Schema:
  activity_events(id, person_id, activity, start_time, end_time, duration,
                   confidence, source_id, evidence_path, alert_status)
  alerts(id, event_id, person_id, alert_type, timestamp, message, status)
  persons(person_id, first_seen, last_seen)
"""
from __future__ import annotations
import sqlite3
import os
import csv
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS activity_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    activity TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    duration REAL,
    confidence REAL,
    source_id TEXT,
    evidence_path TEXT,
    alert_status TEXT DEFAULT 'none'
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    person_id INTEGER NOT NULL,
    alert_type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    message TEXT,
    status TEXT DEFAULT 'active',
    FOREIGN KEY(event_id) REFERENCES activity_events(id)
);

CREATE TABLE IF NOT EXISTS persons (
    person_id INTEGER PRIMARY KEY,
    first_seen REAL,
    last_seen REAL
);
"""


class ActivityDatabase:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- persons ----
    def upsert_person(self, person_id: int, timestamp: float):
        cur = self.conn.execute("SELECT person_id FROM persons WHERE person_id=?", (person_id,))
        if cur.fetchone() is None:
            self.conn.execute(
                "INSERT INTO persons(person_id, first_seen, last_seen) VALUES (?,?,?)",
                (person_id, timestamp, timestamp),
            )
        else:
            self.conn.execute(
                "UPDATE persons SET last_seen=? WHERE person_id=?", (timestamp, person_id)
            )
        self.conn.commit()

    # ---- activity events ----
    def insert_activity_event(self, person_id: int, activity: str, start_time: float,
                               end_time: Optional[float] = None, confidence: float = 0.0,
                               source_id: str = "default", evidence_path: Optional[str] = None,
                               alert_status: str = "none") -> int:
        duration = (end_time - start_time) if end_time is not None else None
        cur = self.conn.execute(
            """INSERT INTO activity_events
               (person_id, activity, start_time, end_time, duration, confidence,
                source_id, evidence_path, alert_status)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (person_id, activity, start_time, end_time, duration, confidence,
             source_id, evidence_path, alert_status),
        )
        self.conn.commit()
        return cur.lastrowid

    def close_activity_event(self, event_id: int, end_time: float):
        cur = self.conn.execute("SELECT start_time FROM activity_events WHERE id=?", (event_id,))
        row = cur.fetchone()
        if row is None:
            return
        duration = end_time - row["start_time"]
        self.conn.execute(
            "UPDATE activity_events SET end_time=?, duration=? WHERE id=?",
            (end_time, duration, event_id),
        )
        self.conn.commit()

    def query_events(self, person_id: Optional[int] = None, activity: Optional[str] = None,
                      start: Optional[float] = None, end: Optional[float] = None) -> List[sqlite3.Row]:
        q = "SELECT * FROM activity_events WHERE 1=1"
        params: List[Any] = []
        if person_id is not None:
            q += " AND person_id=?"
            params.append(person_id)
        if activity is not None:
            q += " AND activity=?"
            params.append(activity)
        if start is not None:
            q += " AND start_time>=?"
            params.append(start)
        if end is not None:
            q += " AND start_time<=?"
            params.append(end)
        q += " ORDER BY start_time DESC"
        cur = self.conn.execute(q, params)
        return cur.fetchall()

    def count_events_by_activity(self) -> Dict[str, int]:
        cur = self.conn.execute("SELECT activity, COUNT(*) c FROM activity_events GROUP BY activity")
        return {row["activity"]: row["c"] for row in cur.fetchall()}

    # ---- alerts ----
    def insert_alert(self, person_id: int, alert_type: str, timestamp: float,
                      message: str = "", event_id: Optional[int] = None, status: str = "active") -> int:
        cur = self.conn.execute(
            """INSERT INTO alerts(event_id, person_id, alert_type, timestamp, message, status)
               VALUES (?,?,?,?,?,?)""",
            (event_id, person_id, alert_type, timestamp, message, status),
        )
        self.conn.commit()
        return cur.lastrowid

    def acknowledge_alert(self, alert_id: int):
        self.conn.execute("UPDATE alerts SET status='acknowledged' WHERE id=?", (alert_id,))
        self.conn.commit()

    def query_alerts(self, status: Optional[str] = None) -> List[sqlite3.Row]:
        if status:
            cur = self.conn.execute("SELECT * FROM alerts WHERE status=? ORDER BY timestamp DESC", (status,))
        else:
            cur = self.conn.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
        return cur.fetchall()

    # ---- export ----
    def export_events_csv(self, path: str, person_id: Optional[int] = None,
                           activity: Optional[str] = None) -> str:
        rows = self.query_events(person_id=person_id, activity=activity)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "person_id", "activity", "start_time", "end_time",
                              "duration", "confidence", "source_id", "evidence_path", "alert_status"])
            for r in rows:
                writer.writerow([r["id"], r["person_id"], r["activity"], r["start_time"], r["end_time"],
                                  r["duration"], r["confidence"], r["source_id"], r["evidence_path"],
                                  r["alert_status"]])
        return path

    def close(self):
        self.conn.close()
