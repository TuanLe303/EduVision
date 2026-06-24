from __future__ import annotations

from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .temporal_aggregator import (
    DEFAULT_PRIORITY,
    TemporalBehaviorAggregator,
    TemporalBehaviorResult,
)


_CONFIG_PATH = (
    Path(__file__).resolve().parents[5]
    / "configs"
    / "services"
    / "behavior_detection"
    / "yolo_behavior.yaml"
)
_VISUAL_STATES = {
    "focused",
    "drowsy",
    "sleeping",
    "using_phone",
    "off_task",
    "side_talking",
    "raising_hand",
}


@dataclass(frozen=True)
class BehaviorTrack:
    track_id: int
    bbox: list[float]
    confidence: float


@dataclass(frozen=True)
class BehaviorDetection:
    track_id: int
    class_id: int
    state: str
    bbox: list[float]
    confidence: float
    association_score: float = 0.0


@dataclass(frozen=True)
class _RawBehaviorDetection:
    class_id: int
    state: str
    bbox: list[float]
    confidence: float


def _load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    import yaml

    with _CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class BehaviorDetector:
    """Associate behavior detections with IDs from the canonical person tracker."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        window_size: Optional[int] = None,
        config: Optional[dict[str, Any]] = None,
        model: Optional[Any] = None,
    ) -> None:
        cfg = {**_load_config(), **(config or {})}
        self._model_path = model_path or cfg.get("model", "weights/behavior_yolo26n.pt")
        self._conf = float(cfg.get("confidence_threshold", 0.30))
        self._iou = float(cfg.get("iou_threshold", 0.50))
        self._input_size = int(cfg.get("input_size", 640))
        self._association_min_score = float(cfg.get("association_min_score", 0.35))
        selected_device = device if device is not None else cfg.get("device", "auto")
        self._device = None if selected_device == "auto" else selected_device
        configured_window = int(cfg.get("window_size", 12))
        selected_window = int(window_size if window_size is not None else configured_window)
        configured_min_history = int(cfg.get("min_history", configured_window))
        configured_min_state_frames = int(cfg.get("min_state_frames", 7))
        if window_size is not None:
            ratio = selected_window / configured_window
            min_history = _scale_window_count(configured_min_history, ratio, selected_window)
            min_state_frames = _scale_window_count(
                configured_min_state_frames, ratio, selected_window
            )
        else:
            min_history = configured_min_history
            min_state_frames = configured_min_state_frames
        self._aggregator = TemporalBehaviorAggregator(
            window_size=selected_window,
            min_history=min_history,
            min_state_frames=min_state_frames,
            enter_threshold=float(cfg.get("enter_threshold", 0.60)),
            switch_margin=float(cfg.get("switch_margin", 0.10)),
            stale_track_frames=int(cfg.get("stale_track_frames", 90)),
            max_detection_gap=int(cfg.get("max_detection_gap", 5)),
            state_thresholds=cfg.get("state_thresholds", {}),
            aliases=cfg.get("label_aliases", {}),
            priority=cfg.get("priority") or DEFAULT_PRIORITY,
        )

        if model is None:
            model_file = Path(self._model_path)
            if not model_file.is_file():
                raise FileNotFoundError(
                    f"Behavior YOLO weight not found: {model_file}. "
                    "Train the behavior detector first or pass --behavior-model."
                )
            from ultralytics import YOLO

            model = YOLO(str(model_file))
        if not callable(getattr(model, "predict", None)):
            raise TypeError("behavior model must provide a callable predict() method")
        model_task = getattr(model, "task", None)
        if model_task not in {None, "detect"}:
            raise ValueError(f"behavior model must be a YOLO detection model, got task={model_task!r}")
        model_names = getattr(model, "names", None)
        if model_names is not None:
            raw_labels = set(
                model_names.values() if isinstance(model_names, dict) else model_names
            )
            try:
                labels = {
                    self._aggregator.normalize_state(label) for label in raw_labels
                }
            except ValueError as exc:
                raise ValueError(
                    "behavior model contains an unsupported label; "
                    f"got {sorted(str(item) for item in raw_labels)}"
                ) from exc
            if labels != _VISUAL_STATES:
                raise ValueError(
                    "behavior model labels must normalize to exactly "
                    f"{sorted(_VISUAL_STATES)}, got {sorted(labels)} "
                    f"from raw labels {sorted(str(item) for item in raw_labels)}"
                )
        self._model = model

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        canonical_tracks: Sequence[Any],
        state_overrides: Optional[Mapping[int, tuple[str, float]]] = None,
    ) -> tuple[list[BehaviorTrack], list[BehaviorDetection], list[TemporalBehaviorResult]]:
        _validate_frame(frame)
        tracks = [_coerce_track(item) for item in canonical_tracks]
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._input_size,
            device=self._device,
            verbose=False,
        )

        raw_detections: list[_RawBehaviorDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            bboxes = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.int().cpu().numpy()
            names = result.names
            for bbox, confidence, class_id in zip(bboxes, confidences, class_ids):
                raw_label = names[int(class_id)]
                state = self._aggregator.normalize_state(raw_label)
                raw_detections.append(
                    _RawBehaviorDetection(
                        class_id=int(class_id),
                        state=state,
                        bbox=[float(value) for value in bbox.tolist()],
                        confidence=float(confidence),
                    )
                )

        detections = _associate_behaviors(
            tracks, raw_detections, min_score=self._association_min_score
        )
        detections_by_track = {item.track_id: item for item in detections}
        tracks_by_id = {item.track_id: item for item in tracks}
        for raw_track_id, (raw_state, confidence) in (state_overrides or {}).items():
            track_id = int(raw_track_id)
            track = tracks_by_id.get(track_id)
            if track is None:
                continue
            state = self._aggregator.normalize_state(raw_state)
            detections_by_track[track_id] = BehaviorDetection(
                track_id=track_id,
                class_id=-1,
                state=state,
                bbox=track.bbox,
                confidence=float(confidence),
                association_score=1.0,
            )
        detections = sorted(detections_by_track.values(), key=lambda item: item.track_id)
        temporal = [
            self._aggregator.update(
                detection.track_id,
                detection.state,
                detection.confidence,
                frame_index,
            )
            for detection in detections
        ]
        observed_track_ids = {detection.track_id for detection in detections}
        for track in tracks:
            if track.track_id in observed_track_ids:
                continue
            retained = self._aggregator.hold(track.track_id, frame_index)
            if retained is not None:
                temporal.append(retained)
        temporal.sort(key=lambda item: item.track_id)
        self._aggregator.finish_frame(
            frame_index,
            active_track_ids=[track.track_id for track in tracks],
            observed_track_ids=[detection.track_id for detection in detections],
        )
        return tracks, detections, temporal

    def reset(self) -> None:
        self._aggregator.reset()


def _scale_window_count(value: int, ratio: float, window_size: int) -> int:
    """Scale an integer evidence count while keeping it inside the new window."""

    scaled = floor(value * ratio + 0.5)
    return max(1, min(window_size, scaled))


def _coerce_track(item: Any) -> BehaviorTrack:
    if isinstance(item, dict):
        track_id = item["track_id"]
        bbox = item["bbox"]
        confidence = item.get("confidence", 1.0)
    else:
        track_id = getattr(item, "track_id")
        bbox = getattr(item, "bbox")
        confidence = getattr(item, "confidence", 1.0)
    return BehaviorTrack(
        track_id=int(track_id),
        bbox=[float(value) for value in bbox],
        confidence=float(confidence),
    )


def _associate_behaviors(
    tracks: Sequence[BehaviorTrack],
    detections: Sequence[_RawBehaviorDetection],
    min_score: float,
) -> list[BehaviorDetection]:
    if not tracks or not detections:
        return []

    scores = [
        [_association_score(track.bbox, detection.bbox) for detection in detections]
        for track in tracks
    ]
    associated: list[BehaviorDetection] = []
    for track_index, detection_index in _hungarian_maximize(scores):
        score = scores[track_index][detection_index]
        if score < min_score:
            continue
        track = tracks[track_index]
        detection = detections[detection_index]
        associated.append(
            BehaviorDetection(
                track_id=track.track_id,
                class_id=detection.class_id,
                state=detection.state,
                bbox=detection.bbox,
                confidence=detection.confidence,
                association_score=score,
            )
        )
    return sorted(associated, key=lambda item: item.track_id)


def _association_score(person_bbox: Sequence[float], behavior_bbox: Sequence[float]) -> float:
    intersection = _intersection_area(person_bbox, behavior_bbox)
    person_area = _area(person_bbox)
    behavior_area = _area(behavior_bbox)
    union = person_area + behavior_area - intersection
    iou = intersection / union if union > 0 else 0.0
    behavior_coverage = intersection / behavior_area if behavior_area > 0 else 0.0

    bx, by = _center(behavior_bbox)
    center_inside = float(
        person_bbox[0] <= bx <= person_bbox[2]
        and person_bbox[1] <= by <= person_bbox[3]
    )
    px, py = _center(person_bbox)
    diagonal = max(
        1.0,
        float(np.hypot(person_bbox[2] - person_bbox[0], person_bbox[3] - person_bbox[1])),
    )
    proximity = max(0.0, 1.0 - float(np.hypot(px - bx, py - by)) / diagonal)
    return 0.35 * iou + 0.35 * behavior_coverage + 0.20 * center_inside + 0.10 * proximity


def _hungarian_maximize(scores: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Maximum-weight Hungarian assignment for a rectangular score matrix."""

    if not scores or not scores[0]:
        return []
    row_count = len(scores)
    column_count = len(scores[0])
    transposed = row_count > column_count
    matrix = (
        [[float(scores[row][column]) for row in range(row_count)] for column in range(column_count)]
        if transposed
        else [[float(value) for value in row] for row in scores]
    )
    n = len(matrix)
    m = len(matrix[0])
    max_value = max(max(row) for row in matrix)
    costs = [[max_value - value for value in row] for row in matrix]
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        min_values = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if current < min_values[j]:
                    min_values[j] = current
                    way[j] = j0
                if min_values[j] < delta:
                    delta = min_values[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    min_values[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            previous = way[j0]
            p[j0] = p[previous]
            j0 = previous
            if j0 == 0:
                break

    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    return [(column, row) for row, column in pairs] if transposed else pairs


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def _area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _center(bbox: Sequence[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("frame must have shape (height, width, 3)")
    if frame.dtype != np.uint8:
        raise TypeError("frame dtype must be uint8")
