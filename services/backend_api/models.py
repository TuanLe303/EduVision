"""
EduVision — Pydantic response models for the FastAPI backend.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------


class StudentOut(BaseModel):
    student_id: str
    name: str
    email: Optional[str] = None
    avatar_url: Optional[str] = None   # served via /api/avatars/<filename>
    enrolled: bool = False             # True when embedding_count > 0
    embedding_count: int = 0
    enrolled_at: Optional[str] = None

    @classmethod
    def from_db(cls, row: dict[str, Any], base_url: str = "") -> "StudentOut":
        avatar_path = row.get("avatar_path")
        avatar_url: Optional[str] = None
        if avatar_path:
            from pathlib import Path
            filename = Path(avatar_path).name
            avatar_url = f"{base_url}/api/avatars/{filename}"
        count = row.get("embedding_count", 0) or 0
        return cls(
            student_id=row["student_id"],
            name=row["name"],
            email=row.get("email"),
            avatar_url=avatar_url,
            enrolled=count > 0,
            embedding_count=count,
            enrolled_at=row.get("enrolled_at"),
        )


class EnrollRequest(BaseModel):
    student_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    email: Optional[str] = None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class SessionOut(BaseModel):
    id: int
    class_name: str
    start_time: str
    end_time: Optional[str] = None
    status: str
    student_count: int = 0
    event_count: int = 0
    attention_pct: float = 0.0
    has_report: bool = False

    @classmethod
    def from_db(cls, row: dict[str, Any]) -> "SessionOut":
        return cls(
            id=row["id"],
            class_name=row["class_name"],
            start_time=row["start_time"],
            end_time=row.get("end_time"),
            status=row.get("status", "active"),
            student_count=row.get("student_count", 0) or 0,
            event_count=row.get("event_count", 0) or 0,
            attention_pct=row.get("attention_pct", 0.0) or 0.0,
            has_report=bool(row.get("has_report", False)),
        )


class StartSessionRequest(BaseModel):
    class_name: str = Field(..., min_length=1)

# ---------------------------------------------------------------------------
# Pipeline Stream Management
# ---------------------------------------------------------------------------

class StartPipelineRequest(BaseModel):
    source: str = Field(default="0", description="RTSP URL or Camera Index")
    target_fps: float = Field(default=8.0, description="Target processing FPS")

class PipelineStatusOut(BaseModel):
    is_running: bool
    source: Optional[str] = None
    pid: Optional[int] = None


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------


class AttendanceOut(BaseModel):
    student_id: Optional[str]
    name: Optional[str]
    entry_time: Optional[float]
    last_seen: Optional[float] = None
    duration_min: Optional[float] = None
    event_count: int = 0

    @classmethod
    def from_db(cls, row: dict[str, Any]) -> "AttendanceOut":
        return cls(
            student_id=row.get("student_id"),
            name=row.get("name"),
            entry_time=row.get("entry_time"),
            last_seen=row.get("last_seen"),
            duration_min=row.get("duration_min"),
            event_count=row.get("event_count", 0) or 0,
        )


# ---------------------------------------------------------------------------
# Behavior events
# ---------------------------------------------------------------------------


class EventOut(BaseModel):
    id: int
    session_id: int
    student_id: Optional[str]
    track_id: Optional[int]
    state: str
    confidence: float
    source: Optional[str]
    ts: float

    @classmethod
    def from_db(cls, row: dict[str, Any]) -> "EventOut":
        return cls(**{k: row[k] for k in cls.model_fields if k in row})


class EventsBulkRequest(BaseModel):
    """Payload pushed by the vision pipeline for a batch of final_behavior frames."""
    events: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class SummaryOut(BaseModel):
    behavior_distribution: dict[str, int] = Field(default_factory=dict)
    avg_attention_pct: float = 0.0
    total_events: int = 0
    student_count: int = 0

    @classmethod
    def from_db(cls, data: dict[str, Any]) -> "SummaryOut":
        return cls(**data)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class GenerateReportRequest(BaseModel):
    session_id: int
    provider: str = "google"
    language: str = "vi"


class ReportOut(BaseModel):
    session_id: int
    provider: str
    generated_at: str
    content: str
