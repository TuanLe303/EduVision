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
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    boxes_complete: bool = False
    boxed_labels: tuple[tuple[tuple[float, float, float, float], str], ...] = ()
    count_complete: bool = True
    boxed_labels_complete: bool = True
    identity_complete: bool = True


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
    bbox_iou_threshold: float = 0.5,
    behavior_output: str = "final",
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
    if behavior_output not in {"final", "frame"}:
        raise ValueError("behavior_output must be 'final' or 'frame'")
    pred_by_index = _normalize_predictions(predictions, fps, behavior_output)
    gt_by_index = _normalize_ground_truth(ground_truth, fps)
    # Ground truth is the evaluation mask. A prediction on an unannotated
    # frame is unknown, not a false positive.
    frame_indices = sorted(gt_by_index)
    if not frame_indices:
        raise ValueError("ground-truth input is empty")

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
    identity = _identity_metrics(pred_frames, gt_frames)
    behavior = _behavior_metrics(pred_frames, gt_frames)
    frame_seconds = _frame_seconds(gt_frames, pred_frames, fps)
    dense_ground_truth = len(frame_indices) >= 2 and all(
        current == previous + 1
        for previous, current in zip(frame_indices, frame_indices[1:])
    )
    complete_event_labels = all(
        set(frame.labels) == frame.attendees for frame in gt_frames
    )
    events = (
        {
            "available": True,
            **_event_metrics(
                pred_frames, gt_frames, frame_seconds, event_iou_threshold
            ),
        }
        if dense_ground_truth and complete_event_labels
        else {
            "available": False,
            "reason": (
                "event metrics require a state for every student on at least "
                "two consecutive annotated frames"
            ),
        }
    )
    detection = _detection_metrics(pred_frames, gt_frames, bbox_iou_threshold)
    bbox_behavior = _bbox_behavior_metrics(
        pred_frames, gt_frames, bbox_iou_threshold
    )
    count_errors = [
        abs(pred.person_count - truth.person_count)
        for pred, truth in zip(pred_frames, gt_frames)
        if truth.count_complete
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
        "ignored_unannotated_prediction_frames": len(set(pred_by_index) - set(gt_by_index)),
        "evaluation_scope": "ground_truth_frames_only",
        "attendance": attendance,
        "student_identity": identity,
        "student_behavior": behavior,
        "person_detection": detection,
        "bbox_behavior": bbox_behavior,
        "events": events,
        "student_count_mae": _mean(count_errors) if count_errors else None,
        "student_count": {
            "available": bool(count_errors),
            "mae": _mean(count_errors) if count_errors else None,
            "evaluated_frames": len(count_errors),
        },
        "frame_results": _frame_results(pred_frames, gt_frames, bbox_iou_threshold),
        "performance": performance,
        "settings": {
            "event_iou_threshold": event_iou_threshold,
            "bbox_iou_threshold": bbox_iou_threshold,
            "behavior_output": behavior_output,
            "fps": fps,
        },
    }


def _normalize_predictions(
    records: Sequence[dict[str, Any]], fps: Optional[float], behavior_output: str
) -> dict[int, _Frame]:
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
        boxes = tuple(
            box for item in tracks
            if isinstance(item, dict) and (box := _box(item.get("bbox"))) is not None
        ) if isinstance(tracks, list) else ()
        frame_behavior = record.get("frame_behavior", [])
        if behavior_output == "frame":
            boxed_labels = tuple(
                (box, str(item["state"]))
                for item in frame_behavior
                if isinstance(item, dict)
                and item.get("state") in BEHAVIOR_STATES
                and (box := _box(item.get("bbox"))) is not None
            )
        else:
            final_states_by_track = {
                int(item["track_id"]): str(item["state"])
                for item in record.get("final_behavior", [])
                if item.get("track_id") is not None
                and item.get("state") in BEHAVIOR_STATES
            }
            boxed_labels = tuple(
                (box, final_states_by_track[int(item["track_id"])])
                for item in frame_behavior
                if isinstance(item, dict)
                and item.get("track_id") is not None
                and int(item["track_id"]) in final_states_by_track
                and (box := _box(item.get("bbox"))) is not None
            )
            located_tracks = {
                int(item["track_id"])
                for item in frame_behavior
                if isinstance(item, dict)
                and item.get("track_id") is not None
                and _box(item.get("bbox")) is not None
            }
            boxed_labels += tuple(
                (box, final_states_by_track[int(item["track_id"])])
                for item in tracks
                if isinstance(item, dict)
                and item.get("track_id") is not None
                and int(item["track_id"]) in final_states_by_track
                and int(item["track_id"]) not in located_tracks
                and (box := _box(item.get("bbox"))) is not None
            ) if isinstance(tracks, list) else ()
        output[frame_index] = _Frame(
            frame_index, timestamp, labels, attendees, person_count, boxes,
            boxed_labels=boxed_labels,
        )
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
        boxes = tuple(
            box for item in items
            if isinstance(item, dict) and (box := _box(item.get("bbox"))) is not None
        )
        anonymous_behavior = record.get("annotation_type") == "anonymous_bbox_behavior"
        boxes_complete = len(boxes) == person_count and not anonymous_behavior
        boxed_labels = tuple(
            (box, str(item["state"]))
            for item in items
            if isinstance(item, dict)
            and item.get("state") in BEHAVIOR_STATES
            and (box := _box(item.get("bbox"))) is not None
        )
        output[frame_index] = _Frame(
            frame_index=frame_index,
            timestamp=timestamp,
            labels=labels,
            attendees=attendees,
            person_count=person_count,
            boxes=boxes,
            boxes_complete=boxes_complete,
            boxed_labels=boxed_labels,
            count_complete=bool(record.get("person_count_complete", not anonymous_behavior)),
            boxed_labels_complete=bool(record.get("box_annotation_complete", True)),
            identity_complete=not anonymous_behavior,
        )
    return output


def _attendance_metrics(predictions: Sequence[_Frame], ground_truth: Sequence[_Frame]) -> dict[str, Any]:
    predicted = set().union(*(frame.attendees for frame in predictions))
    expected = set().union(*(frame.attendees for frame in ground_truth))
    if not all(frame.identity_complete for frame in ground_truth):
        return {
            "available": False,
            "reason": "ground truth does not contain student IDs",
        }
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    return {
        "available": bool(expected),
        **_prf(tp, fp, fn),
        "predicted_students": sorted(predicted),
        "ground_truth_students": sorted(expected),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def _identity_metrics(
    predictions: Sequence[_Frame], ground_truth: Sequence[_Frame]
) -> dict[str, Any]:
    if not all(frame.identity_complete for frame in ground_truth):
        return {
            "available": False,
            "reason": "ground truth does not contain student IDs",
        }
    tp = fp = fn = 0
    for pred, truth in zip(predictions, ground_truth):
        tp += len(pred.attendees & truth.attendees)
        fp += len(pred.attendees - truth.attendees)
        fn += len(truth.attendees - pred.attendees)
    return {
        "available": any(frame.attendees for frame in ground_truth),
        **_prf(tp, fp, fn),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "unit": "student-frame",
    }


def _behavior_metrics(predictions: Sequence[_Frame], ground_truth: Sequence[_Frame]) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    for state in BEHAVIOR_STATES:
        tp = fp = fn = 0
        for pred, truth in zip(predictions, ground_truth):
            # A missing GT state means "not annotated", not a negative label.
            for student_id in truth.labels:
                pred_positive = pred.labels.get(student_id) == state
                true_positive = truth.labels.get(student_id) == state
                tp += int(pred_positive and true_positive)
                fp += int(pred_positive and not true_positive)
                fn += int(true_positive and not pred_positive)
        per_class[state] = {**_prf(tp, fp, fn), "support": tp + fn}

    supported = [item for item in per_class.values() if item["support"] > 0]
    macro_f1 = _mean([item["f1"] for item in supported])
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pred, truth in zip(predictions, ground_truth):
        for student_id, expected in truth.labels.items():
            predicted = pred.labels.get(student_id, "__missing__")
            confusion[expected][predicted] += 1
    return {
        "available": bool(supported),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "supported_classes": len(supported),
        "confusion_matrix": {
            expected: dict(sorted(row.items())) for expected, row in sorted(confusion.items())
        },
    }


def _bbox_behavior_metrics(
    predictions: Sequence[_Frame], ground_truth: Sequence[_Frame], iou_threshold: float
) -> dict[str, Any]:
    gt_count = sum(len(frame.boxed_labels) for frame in ground_truth)
    if gt_count == 0 or any(
        len(frame.boxed_labels) != len(frame.boxes) for frame in ground_truth
    ):
        return {
            "available": False,
            "reason": "every ground-truth bbox needs a behavior state",
        }

    totals = {state: {"tp": 0, "fp": 0, "fn": 0} for state in BEHAVIOR_STATES}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    matched_count = correct_count = 0
    for pred, truth in zip(predictions, ground_truth):
        pred_boxes = tuple(box for box, _ in pred.boxed_labels)
        truth_boxes = tuple(box for box, _ in truth.boxed_labels)
        matches = _match_boxes(pred_boxes, truth_boxes, iou_threshold)
        matched_pred = {pred_index for pred_index, _, _ in matches}
        matched_truth = {truth_index for _, truth_index, _ in matches}

        for pred_index, truth_index, _ in matches:
            predicted = pred.boxed_labels[pred_index][1]
            expected = truth.boxed_labels[truth_index][1]
            matched_count += 1
            confusion[expected][predicted] += 1
            if predicted == expected:
                correct_count += 1
                totals[expected]["tp"] += 1
            else:
                totals[predicted]["fp"] += 1
                totals[expected]["fn"] += 1

        for pred_index, (_, predicted) in enumerate(pred.boxed_labels):
            if pred_index not in matched_pred and truth.boxed_labels_complete:
                totals[predicted]["fp"] += 1
                confusion["__none__"][predicted] += 1
        for truth_index, (_, expected) in enumerate(truth.boxed_labels):
            if truth_index not in matched_truth:
                totals[expected]["fn"] += 1
                confusion[expected]["__missing__"] += 1

    per_class: dict[str, dict[str, Any]] = {}
    for state, counts in totals.items():
        per_class[state] = {
            **_prf(counts["tp"], counts["fp"], counts["fn"]),
            "support": counts["tp"] + counts["fn"],
        }
    supported = [item for item in per_class.values() if item["support"] > 0]
    return {
        "available": True,
        "end_to_end_accuracy": correct_count / gt_count,
        "correct_end_to_end_count": correct_count,
        "accuracy_on_matched_boxes": (
            correct_count / matched_count if matched_count else 0.0
        ),
        "macro_f1": _mean([item["f1"] for item in supported]),
        "matched_box_count": matched_count,
        "ground_truth_box_count": gt_count,
        "annotation_scope": (
            "complete_frames"
            if all(frame.boxed_labels_complete for frame in ground_truth)
            else "annotated_boxes_only"
        ),
        "per_class": per_class,
        "confusion_matrix": {
            expected: dict(sorted(row.items()))
            for expected, row in sorted(confusion.items())
        },
        "iou_threshold": iou_threshold,
    }


def _detection_metrics(
    predictions: Sequence[_Frame], ground_truth: Sequence[_Frame], iou_threshold: float
) -> dict[str, Any]:
    # Do not silently score a partially box-annotated data set.
    if not any(frame.boxes for frame in ground_truth) or not all(
        frame.boxes_complete for frame in ground_truth
    ):
        return {
            "available": False,
            "reason": "every ground-truth student needs an xyxy bbox",
        }
    tp = fp = fn = 0
    matched_ious: list[float] = []
    for pred, truth in zip(predictions, ground_truth):
        matches = _match_boxes(pred.boxes, truth.boxes, iou_threshold)
        tp += len(matches)
        fp += len(pred.boxes) - len(matches)
        fn += len(truth.boxes) - len(matches)
        matched_ious.extend(iou for _, _, iou in matches)
    return {
        "available": True,
        **_prf(tp, fp, fn),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "mean_matched_iou": _mean(matched_ious),
        "iou_threshold": iou_threshold,
    }


def _frame_results(
    predictions: Sequence[_Frame], ground_truth: Sequence[_Frame], iou_threshold: float
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for pred, truth in zip(predictions, ground_truth):
        expected_ids = set(truth.attendees)
        predicted_ids = set(pred.attendees)
        behavior_correct: list[dict[str, str]] = []
        behavior_errors: list[dict[str, Optional[str]]] = []
        for student_id in sorted(truth.labels):
            expected = truth.labels[student_id]
            predicted = pred.labels.get(student_id)
            item = {
                "student_id": student_id,
                "expected": expected,
                "predicted": predicted,
            }
            if expected is not None and expected == predicted:
                behavior_correct.append({"student_id": student_id, "state": expected})
            else:
                behavior_errors.append(item)
        item: dict[str, Any] = {
            "frame_index": truth.frame_index,
            "timestamp": truth.timestamp,
            "attendance": (
                {
                    "available": True,
                    "correct": sorted(expected_ids & predicted_ids),
                    "false_positive": sorted(predicted_ids - expected_ids),
                    "false_negative": sorted(expected_ids - predicted_ids),
                }
                if truth.identity_complete
                else {"available": False}
            ),
            "behavior_correct": behavior_correct,
            "behavior_errors": behavior_errors,
            "person_count": (
                {
                    "available": True,
                    "expected": truth.person_count,
                    "predicted": pred.person_count,
                    "absolute_error": abs(pred.person_count - truth.person_count),
                }
                if truth.count_complete
                else {"available": False}
            ),
        }
        if truth.boxes_complete and truth.boxes:
            matches = _match_boxes(pred.boxes, truth.boxes, iou_threshold)
            item["detection"] = {
                "true_positive": len(matches),
                "false_positive": len(pred.boxes) - len(matches),
                "false_negative": len(truth.boxes) - len(matches),
                "matched_ious": [iou for _, _, iou in matches],
            }
        if truth.boxed_labels:
            pred_boxes = tuple(box for box, _ in pred.boxed_labels)
            truth_boxes = tuple(box for box, _ in truth.boxed_labels)
            matches = _match_boxes(pred_boxes, truth_boxes, iou_threshold)
            matched_truth = {truth_index for _, truth_index, _ in matches}
            comparisons = [
                {
                    "expected": truth.boxed_labels[truth_index][1],
                    "predicted": pred.boxed_labels[pred_index][1],
                    "iou": iou,
                    "correct": (
                        truth.boxed_labels[truth_index][1]
                        == pred.boxed_labels[pred_index][1]
                    ),
                }
                for pred_index, truth_index, iou in matches
            ]
            item["bbox_behavior"] = {
                "matched": comparisons,
                "missing_ground_truth": [
                    truth.boxed_labels[index][1]
                    for index in range(len(truth.boxed_labels))
                    if index not in matched_truth
                ],
            }
        results.append(item)
    return results


def _match_boxes(
    predictions: Sequence[tuple[float, float, float, float]],
    ground_truth: Sequence[tuple[float, float, float, float]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        (
            (_bbox_iou(pred_box, truth_box), pred_index, truth_index)
            for pred_index, pred_box in enumerate(predictions)
            for truth_index, truth_box in enumerate(ground_truth)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pred_index, truth_index in candidates:
        if iou < threshold:
            break
        if pred_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(pred_index)
        used_truth.add(truth_index)
        matches.append((pred_index, truth_index, iou))
    return matches


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


def _box(value: Any) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    box = tuple(float(coordinate) for coordinate in value)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


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
