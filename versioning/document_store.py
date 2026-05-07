"""SQLite-backed document version registry.

Tables
------
documents      — one row per (doc_id, kb_version); tracks checksum + status
kb_versions    — one row per snapshot; docs added/changed/unchanged counts
query_log      — append-only audit trail of every versioned query
latest_version — single-row table: which version is "current"
"""
from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path

from config import PATHS

_DB_PATH: Path = PATHS.get("versions_db",  # type: ignore[arg-type]
                            Path(__file__).parent.parent / "storage" / "versions.db")


class DocumentStore:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or _DB_PATH)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id      TEXT    NOT NULL,
                    version     INTEGER NOT NULL,
                    checksum    TEXT    NOT NULL,
                    timestamp   TEXT    NOT NULL,
                    status      TEXT    NOT NULL DEFAULT 'active',
                    source_path TEXT,
                    title       TEXT,
                    PRIMARY KEY (doc_id, version)
                );

                CREATE TABLE IF NOT EXISTS kb_versions (
                    version         INTEGER PRIMARY KEY,
                    timestamp       TEXT    NOT NULL,
                    batch_name      TEXT,
                    docs_added      INTEGER DEFAULT 0,
                    docs_changed    INTEGER DEFAULT 0,
                    docs_unchanged  INTEGER DEFAULT 0,
                    reason          TEXT,
                    collection_name TEXT
                );

                CREATE TABLE IF NOT EXISTS query_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT    NOT NULL,
                    query        TEXT    NOT NULL,
                    version_used INTEGER,
                    answer_hash  TEXT
                );

                CREATE TABLE IF NOT EXISTS latest_version (
                    id      INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL
                );
                """
            )

    # ── document operations ──────────────────────────────────────────

    def add_doc(
        self,
        doc_id: str,
        version: int,
        checksum: str,
        status: str,
        source_path: str,
        title: str,
    ) -> None:
        ts = datetime.datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(doc_id, version, checksum, timestamp, status, source_path, title) "
                "VALUES (?,?,?,?,?,?,?)",
                (doc_id, version, checksum, ts, status, source_path, title),
            )

    def get_checksum(self, doc_id: str) -> str | None:
        """Return the checksum of the highest-version record for this doc_id."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT checksum FROM documents WHERE doc_id=? "
                "ORDER BY version DESC LIMIT 1",
                (doc_id,),
            ).fetchone()
        return row[0] if row else None

    def get_doc_history(self, doc_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT doc_id, version, checksum, timestamp, status, source_path, title "
                "FROM documents WHERE doc_id=? ORDER BY version DESC",
                (doc_id,),
            ).fetchall()
        cols = ["doc_id", "version", "checksum", "timestamp",
                "status", "source_path", "title"]
        return [dict(zip(cols, r)) for r in rows]

    def get_all_doc_ids(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT doc_id FROM documents"
            ).fetchall()
        return [r[0] for r in rows]

    # ── version operations ───────────────────────────────────────────

    def bump_version(self) -> int:
        """Return next version number (max existing + 1, or 1 for first run)."""
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(version) FROM kb_versions").fetchone()
        return (row[0] or 0) + 1

    def log_version(
        self,
        version: int,
        batch_name: str,
        docs_added: int,
        docs_changed: int,
        docs_unchanged: int,
        reason: str,
        collection_name: str,
    ) -> None:
        ts = datetime.datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kb_versions "
                "(version, timestamp, batch_name, docs_added, docs_changed, "
                "docs_unchanged, reason, collection_name) VALUES (?,?,?,?,?,?,?,?)",
                (version, ts, batch_name, docs_added, docs_changed,
                 docs_unchanged, reason, collection_name),
            )

    def set_latest(self, version: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO latest_version (id, version) VALUES (1,?)",
                (version,),
            )

    def get_latest(self) -> int | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT version FROM latest_version WHERE id=1"
            ).fetchone()
        return row[0] if row else None

    def get_history(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT version, timestamp, batch_name, docs_added, docs_changed, "
                "docs_unchanged, reason, collection_name "
                "FROM kb_versions ORDER BY version DESC"
            ).fetchall()
        cols = ["version", "timestamp", "batch_name", "docs_added",
                "docs_changed", "docs_unchanged", "reason", "collection_name"]
        return [dict(zip(cols, r)) for r in rows]

    def get_version_info(self, version: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT version, timestamp, batch_name, docs_added, docs_changed, "
                "docs_unchanged, reason, collection_name "
                "FROM kb_versions WHERE version=?",
                (version,),
            ).fetchone()
        if not row:
            return None
        cols = ["version", "timestamp", "batch_name", "docs_added",
                "docs_changed", "docs_unchanged", "reason", "collection_name"]
        return dict(zip(cols, row))

    def docs_at_version(self, version: int) -> list[dict]:
        """All docs that were active at or before this version (latest entry ≤ version)."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT d.doc_id, d.version, d.checksum, d.timestamp,
                       d.status, d.source_path, d.title
                FROM documents d
                INNER JOIN (
                    SELECT doc_id, MAX(version) AS mv
                    FROM documents WHERE version <= ?
                    GROUP BY doc_id
                ) latest ON d.doc_id = latest.doc_id AND d.version = latest.mv
                WHERE d.status = 'active'
                ORDER BY d.doc_id
                """,
                (version,),
            ).fetchall()
        cols = ["doc_id", "version", "checksum", "timestamp",
                "status", "source_path", "title"]
        return [dict(zip(cols, r)) for r in rows]

    # ── query audit log ──────────────────────────────────────────────

    def log_query(self, query: str, version_used: int, answer_hash: str) -> None:
        ts = datetime.datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO query_log (timestamp, query, version_used, answer_hash) "
                "VALUES (?,?,?,?)",
                (ts, query, version_used, answer_hash),
            )

    def get_query_log(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT timestamp, query, version_used, answer_hash "
                "FROM query_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        cols = ["timestamp", "query", "version_used", "answer_hash"]
        return [dict(zip(cols, r)) for r in rows]
