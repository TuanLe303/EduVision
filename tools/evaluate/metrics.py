"""Metrics for the public, frame-level boundary of the EduVision pipeline."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Optional, Sequence


BEHAVIOR_STATES = (
    "focused",
    "drowsy",
    "sleeping",
    "using_phone",
    "off_task",
    "away_from_seat",
    "side_talking",
    "raising_hand",
)


@dataclass(frozen=True)
class _Frame:
    frame_index: int
    timestamp: float
    labels: dict[str, str]
    attendees: set[str]
    person_count: int


@dataclass(frozen=True)
class _Event:
    student_id: str
    state: str
    start: float
    end: float


def evaluate_records(
    prediction_records: Iterable[dict[str, Any]],
    ground_truth_records: Iterable[dict[str, Any]],
    *,
    fps: Optional[float] = None,
    event_iou_threshold: float = 0.5,
    runtime_seconds: Optional[float] = None,
    video_duration_seconds: Optional[float] = None,
    peak_ram_mb: Optional[float] = None,
    peak_vram_mb: Optional[float] = None,
) -> dict[str, Any]:
    """Evaluate final EduVision outputs against frame-aligned ground truth.

    Records are joined by ``frame_index``. Predictions use ``final_behavior``
    and ``recognition`` from :class:`VisionPipeline`. Ground truth accepts a
    simpler ``students`` or ``final_behavior`` list; see ``tools/evaluate/README.md``.
    """
    predictions = list(prediction_records)
    ground_truth = list(ground_truth_records)
    pred_by_index = _normalize_predictions(predictions, fps)
    gt_by_index = _normalize_ground_truth(ground_truth, fps)
    frame_indices = sorted(set(pred_by_index) | set(gt_by_index))
    if not frame_indices:
        raise ValueError("prediction and ground-truth inputs are both empty")

    pred_frames: list[_Frame] = []
    gt_frames: list[_Frame] = []
    for frame_index in frame_indices:
        timestamp = _joined_timestamp(frame_index, pred_by_index, gt_by_index, fps)
        pred_frames.append(
            pred_by_index.get(frame_index, _Frame(frame_index, timestamp, {}, set(), 0))
        )
        gt_frames.append(
            gt_by_index.get(frame_index, _Frame(frame_index, timestamp, {}, set(), 0))
        )

    attendance = _attendance_metrics(pred_frames, gt_frames)
    behavior = _behavior_metrics(pred_frames, gt_frames)
    frame_seconds = _frame_seconds(gt_frames, pred_frames, fps)
    events = _event_metrics(pred_frames, gt_frames, frame_seconds, event_iou_threshold)
    count_errors = [
        abs(pred.person_count - truth.person_count)
        for pred, truth in zip(pred_frames, gt_frames)
    ]
    processing_ms = [
        float(record["processing_ms"])
        for record in predictions
        if record.get("processing_ms") is not None
    ]
    performance = _performance_metrics(
        frame_count=len(predictions),
        processing_ms=processing_ms,
        runtime_seconds=runtime_seconds,
        video_duration_seconds=video_duration_seconds,
        peak_ram_mb=peak_ram_mb,
        peak_vram_mb=peak_vram_mb,
    )

    return {
        "evaluated_frames": len(frame_indices),
        "attendance": attendance,
        "student_behavior": behavior,
        "events": events,
        "student_count_mae": _mean(count_errors),
        "performance": performance,
        "settings": {"event_iou_threshold": event_iou_threshold, "fps": fps},
    }


def _normalize_predictions(records: Sequence[dict[str, Any]], fps: Optional[float]) -> dict[int, _Frame]:
    track_identity: dict[int, str] = {}
    output: dict[int, _Frame] = {}
    for position, record in enumerate(records):
        frame_index = int(record.get("frame_index", position + 1))
        timestamp = _timestamp(record, frame_index, fps)
        attendees: set[str] = set()
        for raw_track_id, recognition in record.get("recognition", {}).items():
            student_id = _recognized_student(recognition)
            if student_id is not None:
                track_identity[int(raw_track_id)] = student_id
                attendees.add(student_id)

        labels: dict[str, str] = {}
        for item in record.get("final_behavior", record.get("behavior", [])):
            state = item.get("state")
            if state not in BEHAVIOR_STATES:
                continue
            track_id = item.get("track_id")
            student_id = _clean_id(item.get("student_id"))
            if student_id is None and track_id is not None:
                student_id = track_identity.get(int(track_id))
            # Unknown tracks remain distinct and therefore cannot accidentally
            # receive credit for a known ground-truth student.
            if student_id is None and track_id is not None:
                student_id = f"__unknown_track_{int(track_id)}"
            if student_id is not None:
                labels[student_id] = str(state)
                if not student_id.startswith("__unknown_track_"):
                    attendees.add(student_id)

        tracks = record.get("tracks", [])
        person_count = len(tracks) if isinstance(tracks, list) else len(labels)
        output[frame_index] = _Frame(frame_index, timestamp, labels, attendees, person_count)
    return output


def _normalize_ground_truth(records: Sequence[dict[str, Any]], fps: Optional[float]) -> dict[int, _Frame]:
    output: dict[int, _Frame] = {}
    for position, record in enumerate(records):
        frame_index = int(record.get("frame_index", position + 1))
        timestamp = _timestamp(record, frame_index, fps)
        items = record.get("students")
        if items is None:
            items = record.get("final_behavior", record.get("behavior", []))
        labels: dict[str, str] = {}
        attendees: set[str] = set()
        for item in items:
            student_id = _clean_id(item.get("student_id"))
            if student_id is None:
                continue
            attendees.add(student_id)
            state = item.get("state")
            if state in BEHAVIOR_STATES:
                labels[student_id] = str(state)
        attendees.update(
            sid for sid in (_clean_id(value) for value in record.get("attendance", [])) if sid
        )
        person_count = int(record.get("person_count", len(items)))
        output[frame_index] = _Frame(frame_index, timestamp, labels, attendees, person_count)
    return output


def _attendance_metrics(predictions: Sequence[_Frame], ground_truth: Sequence[_Frame]) -> dict[str, Any]:
    predicted = set().union(*(frame.attendees for frame in predictions))
    expected = set().union(*(frame.attendees for frame in ground_truth))
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    return {
        **_prf(tp, fp, fn),
        "predicted_students": sorted(predicted),
        "ground_truth_students": sorted(expected),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def _behavior_metrics(predictions: Sequence[_Frame], ground_truth: Sequence[_Frame]) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    for state in BEHAVIOR_STATES:
        tp = fp = fn = 0
        for pred, truth in zip(predictions, ground_truth):
            student_ids = set(pred.labels) | set(truth.labels)
            for student_id in student_ids:
                pred_positive = pred.labels.get(student_id) == state
                true_positive = truth.labels.get(student_id) == state
                tp += int(pred_positive and true_positive)
                fp += int(pred_positive and not true_positive)
                fn += int(true_positive and not pred_positive)
        per_class[state] = {**_prf(tp, fp, fn), "support": tp + fn}

    supported = [item for item in per_class.values() if item["support"] > 0]
    macro_f1 = _mean([item["f1"] for item in supported])
    return {"macro_f1": macro_f1, "per_class": per_class, "supported_classes": len(supported)}


def _event_metrics(
    predictions: Sequence[_Frame],
    ground_truth: Sequence[_Frame],
    frame_seconds: float,
    iou_threshold: float,
) -> dict[str, Any]:
    pred_events = _extract_events(predictions, frame_seconds)
    gt_events = _extract_events(ground_truth, frame_seconds)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(pred_events):
        for gt_index, truth in enumerate(gt_events):
            if (pred.student_id, pred.state) != (truth.student_id, truth.state):
                continue
            iou = _interval_iou(pred.start, pred.end, truth.start, truth.end)
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gt_index))
    for _, pred_index, gt_index in sorted(candidates, reverse=True):
        if pred_index not in matched_pred and gt_index not in matched_gt:
            matched_pred.add(pred_index)
            matched_gt.add(gt_index)

    tp = len(matched_pred)
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    duration_keys = {(event.student_id, event.state) for event in pred_events + gt_events}
    pred_duration = _durations(pred_events)
    gt_duration = _durations(gt_events)
    duration_errors = [
        abs(pred_duration.get(key, 0.0) - gt_duration.get(key, 0.0)) for key in duration_keys
    ]
    return {
        **_prf(tp, fp, fn),
        "duration_mae_seconds": _mean(duration_errors),
        "predicted_event_count": len(pred_events),
        "ground_truth_event_count": len(gt_events),
        "matched_event_count": tp,
    }


def _extract_events(frames: Sequence[_Frame], frame_seconds: float) -> list[_Event]:
    active: dict[tuple[str, str], tuple[float, float]] = {}
    events: list[_Event] = []
    previous_keys: set[tuple[str, str]] = set()
    for frame in frames:
        current = {(student_id, state) for student_id, state in frame.labels.items()}
        for key in previous_keys - current:
            start, end = active.pop(key)
            events.append(_Event(key[0], key[1], start, end + frame_seconds))
        for key in current:
            if key not in active:
                active[key] = (frame.timestamp, frame.timestamp)
            else:
                active[key] = (active[key][0], frame.timestamp)
        previous_keys = current
    for key, (start, end) in active.items():
        events.append(_Event(key[0], key[1], start, end + frame_seconds))
    return events


def _performance_metrics(
    *,
    frame_count: int,
    processing_ms: Sequence[float],
    runtime_seconds: Optional[float],
    video_duration_seconds: Optional[float],
    peak_ram_mb: Optional[float],
    peak_vram_mb: Optional[float],
) -> dict[str, Optional[float]]:
    effective_fps = frame_count / runtime_seconds if runtime_seconds and runtime_seconds > 0 else None
    rtf = runtime_seconds / video_duration_seconds if (
        runtime_seconds is not None and video_duration_seconds and video_duration_seconds > 0
    ) else None
    return {
        "runtime_seconds": runtime_seconds,
        "effective_fps": effective_fps,
        "real_time_factor": rtf,
        "mean_latency_ms": _mean(processing_ms) if processing_ms else None,
        "p95_latency_ms": _percentile(processing_ms, 0.95) if processing_ms else None,
        "peak_ram_mb": peak_ram_mb,
        "peak_vram_mb": peak_vram_mb,
    }


def _recognized_student(recognition: Any) -> Optional[str]:
    if not isinstance(recognition, dict):
        return None
    if recognition.get("matched") is False or recognition.get("recognized") is False:
        return None
    return _clean_id(recognition.get("student_id") or recognition.get("identity"))


def _clean_id(value: Any) -> Optional[str]:
    if value in {None, "", "unknown"}:
        return None
    return str(value)


def _timestamp(record: dict[str, Any], frame_index: int, fps: Optional[float]) -> float:
    if record.get("timestamp") is not None:
        return float(record["timestamp"])
    return (frame_index - 1) / fps if fps and fps > 0 else float(frame_index - 1)


def _joined_timestamp(
    frame_index: int,
    predictions: dict[int, _Frame],
    ground_truth: dict[int, _Frame],
    fps: Optional[float],
) -> float:
    if frame_index in ground_truth:
        return ground_truth[frame_index].timestamp
    if frame_index in predictions:
        return predictions[frame_index].timestamp
    return (frame_index - 1) / fps if fps else float(frame_index - 1)


def _frame_seconds(
    ground_truth: Sequence[_Frame], predictions: Sequence[_Frame], fps: Optional[float]
) -> float:
    if fps and fps > 0:
        return 1.0 / fps
    timestamps = sorted({frame.timestamp for frame in ground_truth or predictions})
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    return median(deltas) if deltas else 1.0


def _durations(events: Sequence[_Event]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = defaultdict(float)
    for event in events:
        result[(event.student_id, event.state)] += max(0.0, event.end - event.start)
    return dict(result)


def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _mean(values: Sequence[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
