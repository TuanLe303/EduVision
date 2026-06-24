"""Aggregate frame-level vision AI JSONL output into a session summary."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


_BEHAVIOR_STATES = (
    "focused",
    "drowsy",
    "sleeping",
    "using_phone",
    "off_task",
    "away_from_seat",
    "side_talking",
    "raising_hand",
)


@dataclass
class StudentSummary:
    track_id: int
    student_id: Optional[str]
    name: Optional[str]
    present_frames: int
    behavior_counts: dict[str, int] = field(default_factory=dict)
    behavior_fractions: dict[str, float] = field(default_factory=dict)
    dominant_behavior: str = "unknown"
    attention_score: float = 0.0
    events: list[str] = field(default_factory=list)


@dataclass
class ClassStats:
    total_students: int
    avg_attention_score: float
    behavior_distribution: dict[str, float]
    total_frames: int
    duration_seconds: float
    start_timestamp: Optional[float]
    end_timestamp: Optional[float]


@dataclass
class SessionSummary:
    students: list[StudentSummary]
    class_stats: ClassStats


def aggregate_jsonl(path: str | Path, min_present_frames: int = 5) -> SessionSummary:
    """Read a JSONL file produced by the vision_ai pipeline and compute session stats.

    Each line must be a JSON object with at least a ``behavior`` list as output
    by ``services.vision_ai.src.main``.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return aggregate_frames(
            (json.loads(line) for line in f if line.strip()),
            min_present_frames=min_present_frames,
        )


def aggregate_frames(
    frames: Iterable[dict], min_present_frames: int = 5
) -> SessionSummary:
    """Aggregate in-memory frame records at the pre-report pipeline boundary."""

    # Per-track accumulators
    behavior_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    student_ids: dict[int, Optional[str]] = {}
    student_names: dict[int, Optional[str]] = {}
    events_by_track: dict[int, list[str]] = defaultdict(list)

    total_frames = 0
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None

    for frame in frames:
        total_frames += 1

        ts = frame.get("timestamp")
        if ts is not None:
            if start_ts is None or ts < start_ts:
                start_ts = ts
            if end_ts is None or ts > end_ts:
                end_ts = ts

        # Collect student names from recognition results
        for str_track_id, rec in frame.get("recognition", {}).items():
            if rec is None:
                continue
            tid = int(str_track_id)
            sid = rec.get("student_id")
            name = rec.get("name")
            if sid and (student_ids.get(tid) is None):
                student_ids[tid] = sid
            if name and (student_names.get(tid) is None):
                student_names[tid] = name

        # final_behavior is the public result after temporal gating and seat
        # priority. Fall back for old JSONL files produced before it existed.
        behaviors = frame.get("final_behavior")
        if behaviors is None:
            behaviors = frame.get("behavior", [])
        for behavior in behaviors:
            track_id = behavior.get("track_id")
            if track_id is None:
                continue

            state = behavior.get("state")
            if state not in _BEHAVIOR_STATES:
                continue
            behavior_counts[track_id][state] += 1

            # Fall back to student_id embedded in behavior record
            if track_id not in student_ids:
                student_ids[track_id] = behavior.get("student_id")
            elif student_ids[track_id] is None:
                sid = behavior.get("student_id")
                if sid:
                    student_ids[track_id] = sid

            for event in behavior.get("events", []):
                events_by_track[track_id].append(event)

    students: list[StudentSummary] = []
    all_attention: list[float] = []
    class_state_totals: dict[str, int] = defaultdict(int)
    class_total_count = 0

    for track_id, counts in behavior_counts.items():
        total = sum(counts.values())
        if total < min_present_frames:
            continue

        fractions = {
            state: counts.get(state, 0) / total for state in _BEHAVIOR_STATES
        }
        dominant = max(counts, key=lambda k: counts[k]) if counts else "unknown"
        attention = fractions.get("focused", 0.0) + fractions.get("raising_hand", 0.0)

        all_attention.append(attention)
        for state, count in counts.items():
            class_state_totals[state] += count
            class_total_count += count

        sid = student_ids.get(track_id)
        students.append(
            StudentSummary(
                track_id=track_id,
                student_id=sid,
                name=student_names.get(track_id),
                present_frames=total,
                behavior_counts=dict(counts),
                behavior_fractions=fractions,
                dominant_behavior=dominant,
                attention_score=attention,
                events=events_by_track[track_id],
            )
        )

    students.sort(key=lambda s: s.track_id)

    avg_attention = sum(all_attention) / len(all_attention) if all_attention else 0.0
    # fix: max(1, …) prevents ZeroDivisionError when the session has no behaviour events
    behavior_dist = {
        state: class_state_totals.get(state, 0) / max(1, class_total_count)
        for state in _BEHAVIOR_STATES
    }
    duration = (
        end_ts - start_ts
        if start_ts is not None and end_ts is not None
        else 0.0
    )

    class_stats = ClassStats(
        total_students=len(students),
        avg_attention_score=avg_attention,
        behavior_distribution=behavior_dist,
        total_frames=total_frames,
        duration_seconds=duration,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
    )

    return SessionSummary(students=students, class_stats=class_stats)
