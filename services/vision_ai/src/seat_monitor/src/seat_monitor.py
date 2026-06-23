from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Optional, Sequence

import numpy as np


_CONFIG_PATH = (
    Path(__file__).resolve().parents[5]
    / "configs"
    / "services"
    / "seat_monitor"
    / "seat_monitor.yaml"
)


@dataclass
class SeatAssignment:
    student_id: str
    seat_id: str
    anchor: tuple[float, float]
    roi: list[float]
    baseline_bbox: list[float]
    attendance_status: str
    assignment_status: str
    first_seen_frame: int
    first_seen_timestamp: float


@dataclass(frozen=True)
class SeatResult:
    student_id: str
    track_id: Optional[int]
    seat_id: str
    state: str
    attendance_status: str
    assignment_status: str
    confidence: float
    reason: str
    spatial_score: float = 0.0
    temporal_score: float = 0.0


@dataclass
class _RuntimeState:
    state: str = "seated"
    outside_frames: int = 0
    inside_frames: int = 0
    missing_frames: int = 0
    current_track_id: Optional[int] = None
    empty_seat_frames: int = 0
    away_identity_confirmed: bool = False


@dataclass(frozen=True)
class _Sample:
    bbox: list[float]
    frame_index: int
    timestamp: float


def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    import yaml

    with _CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class SeatMonitor:
    """Monitor fixed-camera student-to-seat assignments from tracks and identity."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        cfg = {**_load_config(), **(config or {})}
        self.calibration_frames = int(cfg.get("calibration_frames", 60))
        self.min_calibration_samples = int(cfg.get("min_calibration_samples", 12))
        self.late_calibration_frames = int(cfg.get("late_calibration_frames", 20))
        self.min_recognition_confidence = float(cfg.get("min_recognition_confidence", 0.65))
        self.roi_padding_x = float(cfg.get("roi_padding_x", 0.25))
        self.roi_padding_y = float(cfg.get("roi_padding_y", 0.20))
        self.leave_distance = float(cfg.get("leave_distance", 0.80))
        self.return_distance = float(cfg.get("return_distance", 0.50))
        self.leave_confirm_frames = int(cfg.get("leave_confirm_frames", 20))
        self.return_confirm_frames = int(cfg.get("return_confirm_frames", 10))
        self.occlusion_frames = int(cfg.get("occlusion_frames", 15))

        self.session_state = "idle"
        self.assignments: dict[str, SeatAssignment] = {}
        self._runtime: dict[str, _RuntimeState] = {}
        self._samples: dict[str, list[_Sample]] = defaultdict(list)
        self._track_to_student: dict[int, str] = {}
        self._session_start_frame = 0
        self._session_start_timestamp = 0.0
        self._seat_counter = 0

    def start_session(self, frame_index: int = 0, timestamp: float = 0.0) -> None:
        self.reset()
        self.session_state = "calibrating"
        self._session_start_frame = int(frame_index)
        self._session_start_timestamp = float(timestamp)

    def end_session(self) -> None:
        self.session_state = "ended"

    def update(
        self,
        tracks: Sequence[Any],
        recognition_by_track: Mapping[int, Any],
        frame_index: int,
        timestamp: float,
    ) -> list[SeatResult]:
        if self.session_state in {"idle", "ended"}:
            return []

        tracks_by_id = {int(_field(track, "track_id")): track for track in tracks}
        self._update_identity_bindings(recognition_by_track)
        confirmed_tracks = {
            int(track_id): student_id
            for track_id, recognition in recognition_by_track.items()
            if (student_id := _recognized_student_id(recognition)) is not None
            and _recognition_confidence(recognition) >= self.min_recognition_confidence
        }

        visible_students: dict[str, tuple[int, list[float]]] = {}
        for track_id, track in tracks_by_id.items():
            student_id = self._track_to_student.get(track_id)
            if student_id is None:
                continue
            bbox = [float(value) for value in _field(track, "bbox")]
            visible_students[student_id] = (track_id, bbox)
            if student_id not in self.assignments:
                self._samples[student_id].append(
                    _Sample(bbox=bbox, frame_index=int(frame_index), timestamp=float(timestamp))
                )

        if self.session_state == "calibrating":
            if frame_index - self._session_start_frame >= self.calibration_frames:
                self._finalize_initial_assignments()
                self.session_state = "active"
        else:
            self._finalize_late_assignments()

        results = self._update_assignments(
            visible_students,
            tracks_by_id,
            confirmed_tracks,
        )
        return self._mark_seats_occupied_by_other(results, visible_students)

    def confirm_assignment(self, student_id: str, seat_id: Optional[str] = None) -> None:
        assignment = self.assignments[str(student_id)]
        if seat_id is not None:
            assignment.seat_id = str(seat_id)
        assignment.assignment_status = "confirmed"

    def reassign(
        self,
        student_id: str,
        bbox: Sequence[float],
        seat_id: Optional[str] = None,
    ) -> None:
        assignment = self.assignments[str(student_id)]
        baseline = [float(value) for value in bbox]
        assignment.baseline_bbox = baseline
        assignment.anchor = _bottom_center(baseline)
        assignment.roi = _expanded_roi(baseline, self.roi_padding_x, self.roi_padding_y)
        assignment.assignment_status = "reassigned"
        if seat_id is not None:
            assignment.seat_id = str(seat_id)
        self._runtime[str(student_id)] = _RuntimeState()

    def reset(self) -> None:
        self.session_state = "idle"
        self.assignments.clear()
        self._runtime.clear()
        self._samples.clear()
        self._track_to_student.clear()
        self._seat_counter = 0

    def _update_identity_bindings(self, recognition_by_track: Mapping[int, Any]) -> None:
        for raw_track_id, recognition in recognition_by_track.items():
            student_id = _recognized_student_id(recognition)
            confidence = _recognition_confidence(recognition)
            if student_id is None or confidence < self.min_recognition_confidence:
                continue
            track_id = int(raw_track_id)

            # A seat assignment belongs to the student for the whole session,
            # while a track binding only identifies the student's current track.
            # Keep an unobserved binding so short occlusions retain identity, but
            # replace an older binding once recognition confirms a new track for
            # the same student. This prevents two active tracks from silently
            # representing one identity.
            previous_track_ids = [
                bound_track_id
                for bound_track_id, bound_student_id in self._track_to_student.items()
                if bound_track_id != track_id and bound_student_id == student_id
            ]
            for previous_track_id in previous_track_ids:
                del self._track_to_student[previous_track_id]
            self._track_to_student[track_id] = student_id

    def _finalize_initial_assignments(self) -> None:
        candidates: list[tuple[str, list[_Sample], tuple[float, float]]] = []
        for student_id, samples in self._samples.items():
            if len(samples) < self.min_calibration_samples:
                continue
            baseline = _median_bbox(samples)
            candidates.append((student_id, samples, _bottom_center(baseline)))
        candidates.sort(key=lambda item: (item[2][1], item[2][0]))
        for student_id, samples, _ in candidates:
            self._create_assignment(student_id, samples, "present", "confirmed")

    def _finalize_late_assignments(self) -> None:
        for student_id, samples in list(self._samples.items()):
            if student_id in self.assignments or len(samples) < self.late_calibration_frames:
                continue
            self._create_assignment(student_id, samples, "late", "provisional")

    def _create_assignment(
        self,
        student_id: str,
        samples: Sequence[_Sample],
        attendance_status: str,
        assignment_status: str,
    ) -> None:
        baseline = _median_bbox(samples)
        self._seat_counter += 1
        self.assignments[student_id] = SeatAssignment(
            student_id=student_id,
            seat_id=f"seat_{self._seat_counter:03d}",
            anchor=_bottom_center(baseline),
            roi=_expanded_roi(baseline, self.roi_padding_x, self.roi_padding_y),
            baseline_bbox=baseline,
            attendance_status=attendance_status,
            assignment_status=assignment_status,
            first_seen_frame=samples[0].frame_index,
            first_seen_timestamp=samples[0].timestamp,
        )
        self._runtime[student_id] = _RuntimeState()

    def _update_assignments(
        self,
        visible_students: Mapping[str, tuple[int, list[float]]],
        tracks_by_id: Mapping[int, Any],
        confirmed_tracks: Mapping[int, str],
    ) -> list[SeatResult]:
        results: list[SeatResult] = []
        for student_id, assignment in self.assignments.items():
            runtime = self._runtime.setdefault(student_id, _RuntimeState())
            visible = visible_students.get(student_id)
            if visible is None:
                runtime.current_track_id = None
                runtime.missing_frames += 1
                runtime.inside_frames = 0
                runtime.outside_frames = 0
                runtime.empty_seat_frames = 0
                runtime.away_identity_confirmed = False
                runtime.state = (
                    "temporarily_occluded"
                    if runtime.missing_frames <= self.occlusion_frames
                    else "missing"
                )
                results.append(
                    self._result(
                        assignment,
                        runtime,
                        confidence=0.0,
                        reason="student is not currently observed",
                        spatial_score=0.0,
                        temporal_score=min(
                            1.0,
                            runtime.missing_frames / max(1, self.occlusion_frames),
                        ),
                    )
                )
                continue

            track_id, bbox = visible
            runtime.current_track_id = track_id
            runtime.missing_frames = 0
            point = _bottom_center(bbox)
            distance = _normalized_distance(point, assignment.anchor, assignment.baseline_bbox)
            inside_roi = _point_inside(point, assignment.roi)
            clearly_outside = not inside_roi and distance > self.leave_distance
            clearly_returned = inside_roi or distance < self.return_distance
            seat_is_empty = not any(
                _point_inside(
                    _bottom_center([float(value) for value in _field(other, "bbox")]),
                    assignment.roi,
                )
                for other in tracks_by_id.values()
            )
            identity_confirmed_elsewhere = confirmed_tracks.get(track_id) == student_id

            # Spatial and temporal evidence are intentionally state-neutral here.
            # The final confidence below selects the evidence that corresponds to
            # the state actually returned, instead of reusing seated confidence
            # for away_from_seat.
            spatial_away_score = 0.0 if inside_roi else _normalized_away_score(
                distance,
                self.return_distance,
                self.leave_distance,
            )

            if clearly_outside:
                runtime.outside_frames += 1
                runtime.inside_frames = 0
                runtime.empty_seat_frames = (
                    runtime.empty_seat_frames + 1 if seat_is_empty else 0
                )
                runtime.away_identity_confirmed = (
                    runtime.away_identity_confirmed or identity_confirmed_elsewhere
                )
                enough_spatial_evidence = runtime.outside_frames >= self.leave_confirm_frames
                enough_empty_seat_evidence = (
                    runtime.empty_seat_frames >= self.leave_confirm_frames
                )
                confirmed_away = (
                    enough_spatial_evidence
                    and enough_empty_seat_evidence
                    and runtime.away_identity_confirmed
                )
                runtime.state = "away_from_seat" if confirmed_away else "candidate_away"
                reason = (
                    f"outside seat for {runtime.outside_frames} frame(s); "
                    f"seat empty for {runtime.empty_seat_frames} frame(s); "
                    f"identity confirmed elsewhere={runtime.away_identity_confirmed}"
                )
            elif clearly_returned:
                runtime.inside_frames += 1
                runtime.outside_frames = 0
                runtime.empty_seat_frames = 0
                runtime.away_identity_confirmed = False
                if runtime.state in {"away_from_seat", "candidate_away", "missing", "temporarily_occluded"}:
                    runtime.state = (
                        "returned"
                        if runtime.inside_frames >= self.return_confirm_frames
                        else runtime.state
                    )
                elif runtime.state == "returned":
                    runtime.state = "seated"
                else:
                    runtime.state = "seated"
                reason = "student is inside assigned seat region"
            else:
                reason = "seat position is within hysteresis band"

            temporal_away_score = min(
                1.0,
                min(runtime.outside_frames, runtime.empty_seat_frames)
                / max(1, self.leave_confirm_frames),
            )
            spatial_seated_score = 1.0 - spatial_away_score
            temporal_seated_score = min(
                1.0,
                runtime.inside_frames / max(1, self.return_confirm_frames),
            )

            if runtime.state in {"candidate_away", "away_from_seat"}:
                spatial_score = spatial_away_score
                temporal_score = (
                    temporal_away_score if runtime.away_identity_confirmed else 0.0
                )
            elif runtime.state in {"seated", "returned"}:
                spatial_score = spatial_seated_score
                temporal_score = temporal_seated_score
            else:
                # missing/occluded while a track has just reappeared has not yet
                # accumulated enough return evidence for a confident seat state.
                spatial_score = 0.0
                temporal_score = 0.0

            confidence = spatial_score * temporal_score
            results.append(
                self._result(
                    assignment,
                    runtime,
                    confidence,
                    reason,
                    spatial_score=spatial_score,
                    temporal_score=temporal_score,
                )
            )
        return results

    def _mark_seats_occupied_by_other(
        self,
        results: Sequence[SeatResult],
        visible_students: Mapping[str, tuple[int, list[float]]],
    ) -> list[SeatResult]:
        output: list[SeatResult] = []
        points = {
            student_id: _bottom_center(bbox)
            for student_id, (_, bbox) in visible_students.items()
        }
        for result in results:
            if result.state not in {"missing", "temporarily_occluded"}:
                output.append(result)
                continue
            assignment = self.assignments[result.student_id]
            occupant = next(
                (
                    other_id
                    for other_id, point in points.items()
                    if other_id != result.student_id and _point_inside(point, assignment.roi)
                ),
                None,
            )
            if occupant is None:
                output.append(result)
                continue
            output.append(
                SeatResult(
                    student_id=result.student_id,
                    track_id=None,
                    seat_id=result.seat_id,
                    state="occupied_by_other",
                    attendance_status=result.attendance_status,
                    assignment_status=result.assignment_status,
                    confidence=1.0,
                    reason=f"assigned seat is occupied by {occupant}",
                    spatial_score=1.0,
                    temporal_score=1.0,
                )
            )
        return output

    @staticmethod
    def _result(
        assignment: SeatAssignment,
        runtime: _RuntimeState,
        confidence: float,
        reason: str,
        spatial_score: float = 0.0,
        temporal_score: float = 0.0,
    ) -> SeatResult:
        return SeatResult(
            student_id=assignment.student_id,
            track_id=runtime.current_track_id,
            seat_id=assignment.seat_id,
            state=runtime.state,
            attendance_status=assignment.attendance_status,
            assignment_status=assignment.assignment_status,
            confidence=float(confidence),
            reason=reason,
            spatial_score=float(spatial_score),
            temporal_score=float(temporal_score),
        )


def _recognized_student_id(result: Any) -> Optional[str]:
    if result is None:
        return None
    if isinstance(result, dict):
        if result.get("recognized") is False or result.get("matched") is False:
            return None
        value = result.get("student_id") or result.get("identity") or result.get("label")
    else:
        if getattr(result, "recognized", True) is False or getattr(result, "matched", True) is False:
            return None
        value = (
            getattr(result, "student_id", None)
            or getattr(result, "identity", None)
            or getattr(result, "label", None)
        )
    return str(value) if value not in {None, "", "unknown"} else None


def _recognition_confidence(result: Any) -> float:
    if isinstance(result, dict):
        value = result.get("confidence", result.get("similarity", result.get("score", 0.0)))
    else:
        value = getattr(result, "confidence", getattr(result, "similarity", getattr(result, "score", 0.0)))
    return float(value or 0.0)


def _field(item: Any, name: str) -> Any:
    return item[name] if isinstance(item, dict) else getattr(item, name)


def _median_bbox(samples: Sequence[_Sample]) -> list[float]:
    return [float(median(sample.bbox[index] for sample in samples)) for index in range(4)]


def _bottom_center(bbox: Sequence[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3]))


def _expanded_roi(bbox: Sequence[float], padding_x: float, padding_y: float) -> list[float]:
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    return [
        float(bbox[0]) - width * padding_x,
        float(bbox[1]) - height * padding_y,
        float(bbox[2]) + width * padding_x,
        float(bbox[3]) + height * padding_y,
    ]


def _point_inside(point: tuple[float, float], roi: Sequence[float]) -> bool:
    return roi[0] <= point[0] <= roi[2] and roi[1] <= point[1] <= roi[3]


def _normalized_distance(
    point: tuple[float, float],
    anchor: tuple[float, float],
    baseline_bbox: Sequence[float],
) -> float:
    scale = max(1.0, float(baseline_bbox[3]) - float(baseline_bbox[1]))
    return float(np.hypot(point[0] - anchor[0], point[1] - anchor[1]) / scale)


def _normalized_away_score(
    distance: float,
    return_distance: float,
    leave_distance: float,
) -> float:
    """Map the hysteresis interval to spatial evidence for leaving the seat."""

    width = max(1e-6, leave_distance - return_distance)
    return max(0.0, min(1.0, (distance - return_distance) / width))
