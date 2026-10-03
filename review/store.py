from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from detect.issue import Issue

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "review.sqlite"
STATUSES = ("Open", "Investigating", "Explained", "Resolved")
TRANSITIONS = {
    "Open": {"Investigating"},
    "Investigating": {"Explained"},
    "Explained": {"Resolved"},
    "Resolved": set(),
}


def _connect(path: Path | str | None = None) -> sqlite3.Connection:
    db_path = Path(path) if path else DEFAULT_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_store(path: Path | str | None = None) -> None:
    with _connect(path) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS cases (
                issue_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('Open','Investigating','Explained','Resolved')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_id TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(issue_id) REFERENCES cases(issue_id)
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_id TEXT NOT NULL,
                decision TEXT NOT NULL CHECK(decision IN ('accept','reject')),
                comment TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(issue_id) REFERENCES cases(issue_id)
            );
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_id TEXT NOT NULL,
                author TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(issue_id) REFERENCES cases(issue_id)
            );
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sync_issues(issues: list[Issue], path: Path | str | None = None) -> None:
    initialize_store(path)
    timestamp = _now()
    with _connect(path) as connection:
        for item in issues:
            payload = json.dumps(item.to_dict(), ensure_ascii=False, default=str)
            inserted = connection.execute(
                "INSERT OR IGNORE INTO cases(issue_id,payload,status,created_at,updated_at) VALUES(?,?,?,?,?)",
                (item.issue_id, payload, item.status, timestamp, timestamp),
            ).rowcount
            if inserted:
                connection.execute(
                    "INSERT INTO audit_log(issue_id,action,details,created_at) VALUES(?,?,?,?)",
                    (item.issue_id, "Detected", "Issue produced by the deterministic reconciliation pipeline.", timestamp),
                )
            else:
                connection.execute(
                    "UPDATE cases SET payload=? WHERE issue_id=?",
                    (payload, item.issue_id),
                )


def list_cases(path: Path | str | None = None) -> list[dict[str, Any]]:
    initialize_store(path)
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT issue_id,payload,status,created_at,updated_at FROM cases ORDER BY updated_at DESC"
        ).fetchall()
    return [
        {**json.loads(row["payload"]), "status": row["status"], "created_at": row["created_at"], "updated_at": row["updated_at"]}
        for row in rows
    ]


def get_case(issue_id: str, path: Path | str | None = None) -> dict[str, Any] | None:
    initialize_store(path)
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT issue_id,payload,status,created_at,updated_at FROM cases WHERE issue_id=?",
            (issue_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        **json.loads(row["payload"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def transition_case(
    issue_id: str,
    new_status: str,
    action: str,
    details: str = "",
    path: Path | str | None = None,
) -> None:
    if new_status not in STATUSES:
        raise ValueError(f"Unknown case status: {new_status}")
    initialize_store(path)
    timestamp = _now()
    with _connect(path) as connection:
        row = connection.execute("SELECT status FROM cases WHERE issue_id=?", (issue_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown issue: {issue_id}")
        current = str(row["status"])
        if new_status != current and new_status not in TRANSITIONS[current]:
            raise ValueError(f"Invalid case transition: {current} -> {new_status}")
        connection.execute(
            "UPDATE cases SET status=?,updated_at=? WHERE issue_id=?",
            (new_status, timestamp, issue_id),
        )
        connection.execute(
            "INSERT INTO audit_log(issue_id,action,details,created_at) VALUES(?,?,?,?)",
            (issue_id, action, details, timestamp),
        )


def add_comment(
    issue_id: str,
    body: str,
    author: str = "Reviewer",
    path: Path | str | None = None,
) -> None:
    clean_body = body.strip()
    if not clean_body:
        raise ValueError("Comment cannot be empty.")
    initialize_store(path)
    timestamp = _now()
    with _connect(path) as connection:
        if connection.execute("SELECT 1 FROM cases WHERE issue_id=?", (issue_id,)).fetchone() is None:
            raise KeyError(f"Unknown issue: {issue_id}")
        connection.execute(
            "INSERT INTO comments(issue_id,author,body,created_at) VALUES(?,?,?,?)",
            (issue_id, author.strip() or "Reviewer", clean_body, timestamp),
        )
        connection.execute(
            "INSERT INTO audit_log(issue_id,action,details,created_at) VALUES(?,?,?,?)",
            (issue_id, "Comment added", clean_body, timestamp),
        )


def save_feedback(
    issue_id: str,
    decision: str,
    comment: str = "",
    path: Path | str | None = None,
) -> None:
    if decision not in {"accept", "reject"}:
        raise ValueError("Feedback decision must be 'accept' or 'reject'.")
    initialize_store(path)
    timestamp = _now()
    with _connect(path) as connection:
        if connection.execute("SELECT 1 FROM cases WHERE issue_id=?", (issue_id,)).fetchone() is None:
            raise KeyError(f"Unknown issue: {issue_id}")
        connection.execute(
            "INSERT INTO feedback(issue_id,decision,comment,created_at) VALUES(?,?,?,?)",
            (issue_id, decision, comment.strip(), timestamp),
        )
        connection.execute(
            "INSERT INTO audit_log(issue_id,action,details,created_at) VALUES(?,?,?,?)",
            (issue_id, f"Finding {decision}ed", comment.strip(), timestamp),
        )


def list_audit_log(path: Path | str | None = None) -> list[dict[str, Any]]:
    initialize_store(path)
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT id,issue_id,action,details,created_at FROM audit_log ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def list_comments(issue_id: str, path: Path | str | None = None) -> list[dict[str, Any]]:
    initialize_store(path)
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT author,body,created_at FROM comments WHERE issue_id=? ORDER BY id",
            (issue_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_feedback(path: Path | str | None = None) -> list[dict[str, Any]]:
    initialize_store(path)
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT issue_id,decision,comment,created_at FROM feedback ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]
