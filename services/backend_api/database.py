"""
EduVision — SQLite database layer.

Tables
------
students          : student metadata (id, name, email, avatar_path, enrolled_at)
face_embeddings   : one row per embedding template; student may have many
sessions          : one row per class session
behavior_events   : per-frame final_behavior records pushed by the pipeline

The enrollment JSON consumed by FaceRecognizer is kept in sync automatically:
every mutating operation on students / face_embeddings calls export_enrollment_json().
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _REPO_ROOT / "data" / "eduvision.db"
_ENROLLMENT_JSON = _REPO_ROOT / "data" / "enrollments.json"
_AVATARS_DIR = _REPO_ROOT / "data" / "avatars"

# Behaviour states understood by the pipeline (from temporal_aggregator.py)
CANONICAL_STATES = frozenset(
    {
        "focused",
        "drowsy",
        "sleeping",
        "using_phone",
        "off_task",
        "side_talking",
        "raising_hand",
        "away_from_seat",
    }
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embedding_to_blob(embedding: list[float] | np.ndarray) -> bytes:
    arr = np.asarray(embedding, dtype=np.float32)
    return arr.tobytes()


def _blob_to_embedding(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class Database:
    """Thread-safe SQLite wrapper for EduVision."""

    def __init__(self, db_path: str | Path = _DB_PATH) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _AVATARS_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self.create_tables()

    # ------------------------------------------------------------------
    # Connection management (one connection per thread)
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._get_conn()
        with self._lock, conn:
            yield conn

    def delete_session(self, session_id: int) -> bool:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return bool(cur.rowcount)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        with self._tx() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS students (
                    student_id   TEXT PRIMARY KEY,
                    name         TEXT NOT NULL,
                    email        TEXT,
                    avatar_path  TEXT,
                    enrolled_at  TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id   TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                    embedding    BLOB NOT NULL,
                    model_name   TEXT NOT NULL DEFAULT 'buffalo_s',
                    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_embeddings_student
                    ON face_embeddings(student_id);

                CREATE TABLE IF NOT EXISTS sessions (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_name     TEXT NOT NULL,
                    start_time     TEXT NOT NULL DEFAULT (datetime('now')),
                    end_time       TEXT,
                    status         TEXT NOT NULL DEFAULT 'active'
                                   CHECK(status IN ('active', 'ended'))
                );

                CREATE TABLE IF NOT EXISTS behavior_events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    student_id   TEXT,
                    track_id     INTEGER,
                    state        TEXT NOT NULL,
                    confidence   REAL NOT NULL DEFAULT 0.0,
                    source       TEXT,
                    ts           REAL NOT NULL,
                    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_events_session
                    ON behavior_events(session_id);

                CREATE INDEX IF NOT EXISTS idx_events_student
                    ON behavior_events(student_id);
                """
            )

    # ------------------------------------------------------------------
    # Students
    # ------------------------------------------------------------------

    def upsert_student(
        self,
        student_id: str,
        name: str,
        email: Optional[str] = None,
        avatar_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Insert or update a student record."""
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO students (student_id, name, email, avatar_path)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(student_id) DO UPDATE SET
                    name        = excluded.name,
                    email       = excluded.email,
                    avatar_path = COALESCE(excluded.avatar_path, avatar_path)
                """,
                (student_id, name, email, avatar_path),
            )
        return self.get_student(student_id)  # type: ignore[return-value]

    def get_student(self, student_id: str) -> Optional[dict[str, Any]]:
        row = self._get_conn().execute(
            """
            SELECT s.*,
                   COUNT(e.id) AS embedding_count
            FROM students s
            LEFT JOIN face_embeddings e ON e.student_id = s.student_id
            WHERE s.student_id = ?
            GROUP BY s.student_id
            """,
            (student_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_students(self) -> list[dict[str, Any]]:
        rows = self._get_conn().execute(
            """
            SELECT s.*,
                   COUNT(e.id) AS embedding_count
            FROM students s
            LEFT JOIN face_embeddings e ON e.student_id = s.student_id
            GROUP BY s.student_id
            ORDER BY s.name
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_student(self, student_id: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute(
                "DELETE FROM students WHERE student_id = ?", (student_id,)
            )
        if cur.rowcount:
            self.export_enrollment_json()
        return bool(cur.rowcount)

    # ------------------------------------------------------------------
    # Face embeddings
    # ------------------------------------------------------------------

    def add_embedding(
        self,
        student_id: str,
        embedding: list[float] | np.ndarray,
        model_name: str = "buffalo_s",
    ) -> int:
        """Add one face embedding template for a student."""
        blob = _embedding_to_blob(embedding)
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO face_embeddings (student_id, embedding, model_name) VALUES (?, ?, ?)",
                (student_id, blob, model_name),
            )
            row_id: int = cur.lastrowid  # type: ignore[assignment]
        self.export_enrollment_json()
        return row_id

    def get_embeddings(self, student_id: str) -> list[list[float]]:
        rows = self._get_conn().execute(
            "SELECT embedding FROM face_embeddings WHERE student_id = ? ORDER BY id",
            (student_id,),
        ).fetchall()
        return [_blob_to_embedding(r["embedding"]) for r in rows]

    def clear_embeddings(self, student_id: str) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                "DELETE FROM face_embeddings WHERE student_id = ?", (student_id,)
            )
        if cur.rowcount:
            self.export_enrollment_json()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Enrollment JSON export (keeps pipeline in sync with DB)
    # ------------------------------------------------------------------

    def export_enrollment_json(
        self,
        path: str | Path = _ENROLLMENT_JSON,
        model_name: str = "buffalo_s",
    ) -> Path:
        """
        Write the enrollment JSON file consumed by FaceRecognizer.
        Only students that have at least one embedding are included.
        """
        rows = self._get_conn().execute(
            """
            SELECT s.student_id, s.name,
                   GROUP_CONCAT(e.id) AS emb_ids
            FROM students s
            JOIN face_embeddings e ON e.student_id = s.student_id
            WHERE e.model_name = ?
            GROUP BY s.student_id
            ORDER BY s.student_id
            """,
            (model_name,),
        ).fetchall()

        students_payload = []
        for row in rows:
            embeddings = self.get_embeddings(row["student_id"])
            students_payload.append(
                {
                    "student_id": row["student_id"],
                    "name": row["name"],
                    "embeddings": embeddings,
                }
            )

        dimension: Optional[int] = None
        if students_payload and students_payload[0]["embeddings"]:
            dimension = len(students_payload[0]["embeddings"][0])

        payload = {
            "metadata": {
                "version": 1,
                "model": model_name,
                "embedding_dimension": dimension,
                "student_count": len(students_payload),
            },
            "students": students_payload,
        }

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)
        return out

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, class_name: str) -> dict[str, Any]:
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (class_name) VALUES (?)", (class_name,)
            )
            session_id: int = cur.lastrowid  # type: ignore[assignment]
        return self.get_session(session_id)  # type: ignore[return-value]

    def end_session(self, session_id: int) -> Optional[dict[str, Any]]:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET end_time = datetime('now'), status = 'ended'
                WHERE id = ? AND status = 'active'
                """,
                (session_id,),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: int) -> Optional[dict[str, Any]]:
        row = self._get_conn().execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        return self._enrich_session(dict(row))

    def get_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._get_conn().execute(
            "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._enrich_session(dict(r)) for r in rows]

    def _enrich_session(self, session: dict[str, Any]) -> dict[str, Any]:
        """Attach aggregated stats to a session dict."""
        sid = session["id"]
        row = self._get_conn().execute(
            """
            SELECT
                COUNT(DISTINCT student_id) AS student_count,
                COUNT(*) AS event_count,
                ROUND(
                    100.0 * SUM(CASE WHEN state = 'focused' THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(*), 0)
                , 1) AS attention_pct
            FROM behavior_events
            WHERE session_id = ?
            """,
            (sid,),
        ).fetchone()
        if row:
            session["student_count"] = row["student_count"] or 0
            session["event_count"] = row["event_count"] or 0
            session["attention_pct"] = row["attention_pct"] or 0.0
        else:
            session.update({"student_count": 0, "event_count": 0, "attention_pct": 0.0})

        # Report exists?
        report_path = _REPO_ROOT / "data" / "reports" / f"session_{sid}.json"
        session["has_report"] = report_path.exists()
        return session

    # ------------------------------------------------------------------
    # Behavior events
    # ------------------------------------------------------------------

    def log_event(
        self,
        session_id: int,
        state: str,
        ts: float,
        student_id: Optional[str] = None,
        track_id: Optional[int] = None,
        confidence: float = 0.0,
        source: Optional[str] = None,
    ) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                """
                INSERT INTO behavior_events
                    (session_id, student_id, track_id, state, confidence, source, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, student_id, track_id, state, confidence, source, ts),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def log_events_bulk(
        self, session_id: int, events: list[dict[str, Any]]
    ) -> int:
        """
        Bulk-insert a list of final_behavior dicts (as produced by the pipeline).
        Returns the number of rows inserted.
        """
        rows = [
            (
                session_id,
                e.get("student_id"),
                e.get("track_id"),
                e.get("state", "focused"),
                float(e.get("confidence", 0.0)),
                e.get("source"),
                float(e.get("ts", 0.0)),
            )
            for e in events
            if e.get("state") in CANONICAL_STATES
        ]
        if not rows:
            return 0
        with self._tx() as conn:
            conn.executemany(
                """
                INSERT INTO behavior_events
                    (session_id, student_id, track_id, state, confidence, source, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def get_events(
        self,
        session_id: int,
        limit: int = 200,
        student_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM behavior_events WHERE session_id = ?"
        params: list[Any] = [session_id]
        if student_id:
            query += " AND student_id = ?"
            params.append(student_id)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self._get_conn().execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_attendance(self, session_id: int) -> list[dict[str, Any]]:
        """Return first/last seen timestamps per student for a session."""
        rows = self._get_conn().execute(
            """
            SELECT
                e.student_id,
                s.name,
                MIN(e.ts) AS entry_time,
                MAX(e.ts) AS last_seen,
                ROUND((MAX(e.ts) - MIN(e.ts)) / 60.0, 1) AS duration_min,
                COUNT(*) AS event_count
            FROM behavior_events e
            LEFT JOIN students s ON s.student_id = e.student_id
            WHERE e.session_id = ?
              AND e.student_id IS NOT NULL
            GROUP BY e.student_id
            ORDER BY entry_time
            """,
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_summary(self, session_id: int) -> dict[str, Any]:
        """Aggregate behavior distribution and attention % for one session."""
        rows = self._get_conn().execute(
            """
            SELECT state, COUNT(*) AS cnt
            FROM behavior_events
            WHERE session_id = ?
            GROUP BY state
            """,
            (session_id,),
        ).fetchall()
        distribution = {r["state"]: r["cnt"] for r in rows}
        total = sum(distribution.values()) or 1
        attention_pct = round(100.0 * distribution.get("focused", 0) / total, 1)

        student_count = self._get_conn().execute(
            "SELECT COUNT(DISTINCT student_id) FROM behavior_events WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]

        return {
            "behavior_distribution": distribution,
            "avg_attention_pct": attention_pct,
            "total_events": total,
            "student_count": student_count,
        }

    # ------------------------------------------------------------------
    # Reports (simple JSON file storage)
    # ------------------------------------------------------------------

    def save_report(self, session_id: int, data: dict[str, Any]) -> Path:
        reports_dir = _REPO_ROOT / "data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"session_{session_id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_report(self, session_id: int) -> Optional[dict[str, Any]]:
        path = _REPO_ROOT / "data" / "reports" / f"session_{session_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------

_db: Optional[Database] = None
_db_lock = threading.Lock()


def get_db() -> Database:
    """Return the module-level Database singleton (created on first call)."""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = Database()
    return _db
