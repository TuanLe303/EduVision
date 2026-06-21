from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np


@dataclass
class ObjectDetection:
    label: str
    class_id: int
    bbox: List[float]
    confidence: float
    person_id: Optional[int] = None
    person_bbox: Optional[List[float]] = None
    proximity: Optional[str] = None


_CONFIGS_DIR = Path(__file__).resolve().parents[5] / "configs" / "services" / "object_detection"

_COCO_NAMES = {
    24: "backpack",
    39: "bottle",
    41: "cup",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    63: "laptop",
    67: "cell phone",
    73: "book",
}


def _load_config() -> dict:
    import yaml

    path = _CONFIGS_DIR / "yolo.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


class OffTaskObjectDetector:
    """
    YOLO wrapper for objects that may indicate off-task behavior.

    The detector can optionally associate each object with the closest tracked
    person bbox. Person inputs may be TrackResult-like objects, PersonDetection
    objects, or dictionaries containing `bbox` and optionally `track_id`.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        target_labels: Optional[Sequence[str]] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        input_size: Optional[int] = None,
        device: Optional[str] = None,
        proximity_threshold: Optional[float] = None,
    ) -> None:
        cfg = _load_config()
        self._model_name = model_name or cfg.get("model", "yolo11n")
        self._target_labels = set(target_labels or cfg.get("target_labels", _default_labels()))
        self._target_ids = [
            class_id for class_id, label in _COCO_NAMES.items() if label in self._target_labels
        ]
        self._conf = (
            confidence_threshold
            if confidence_threshold is not None
            else cfg.get("confidence_threshold", 0.35)
        )
        self._iou = iou_threshold if iou_threshold is not None else cfg.get("iou_threshold", 0.5)
        self._input_size = input_size if input_size is not None else cfg.get("input_size", 640)
        self._proximity_threshold = (
            proximity_threshold
            if proximity_threshold is not None
            else cfg.get("proximity_threshold", 0.25)
        )

        selected_device = device if device is not None else cfg.get("device", "auto")
        self._device: Optional[str] = None if selected_device == "auto" else selected_device

        from ultralytics import YOLO

        weight = self._model_name if str(self._model_name).endswith(".pt") else f"{self._model_name}.pt"
        self._model = YOLO(weight)

    def detect(
        self,
        frame: np.ndarray,
        persons: Optional[Sequence[Any]] = None,
    ) -> List[ObjectDetection]:
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._input_size,
            classes=self._target_ids or None,
            device=self._device,
            verbose=False,
        )

        detections: List[ObjectDetection] = []
        person_records = [_person_record(person) for person in (persons or [])]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            bboxes = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.int().cpu().numpy()

            for bbox, conf, class_id in zip(bboxes, confs, class_ids):
                label = _COCO_NAMES.get(int(class_id), str(int(class_id)))
                if self._target_labels and label not in self._target_labels:
                    continue

                person_id, person_bbox, proximity = _associate_person(
                    bbox.tolist(), person_records, self._proximity_threshold
                )
                detections.append(
                    ObjectDetection(
                        label=label,
                        class_id=int(class_id),
                        bbox=bbox.tolist(),
                        confidence=float(conf),
                        person_id=person_id,
                        person_bbox=person_bbox,
                        proximity=proximity,
                    )
                )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def reset(self) -> None:
        """No persistent state to clear; provided for API consistency."""
        pass


def _default_labels() -> list[str]:
    return [
        "cell phone",
        "laptop",
        "book",
        "bottle",
        "backpack",
        "cup",
        "sandwich",
        "pizza",
        "cake",
    ]


def _person_record(person: Any) -> tuple[Optional[int], List[float]]:
    if isinstance(person, dict):
        bbox = person.get("bbox")
        track_id = person.get("track_id") or person.get("person_id")
        return (int(track_id) if track_id is not None else None, list(bbox))

    bbox = getattr(person, "bbox")
    track_id = getattr(person, "track_id", None)
    return (int(track_id) if track_id is not None else None, list(bbox))


def _associate_person(
    object_bbox: List[float],
    persons: Sequence[tuple[Optional[int], List[float]]],
    proximity_threshold: float,
) -> tuple[Optional[int], Optional[List[float]], Optional[str]]:
    if not persons:
        return None, None, None

    object_center = _center(object_bbox)
    best: Optional[tuple[float, Optional[int], List[float], str]] = None

    for person_id, person_bbox in persons:
        relation = _relation(object_bbox, person_bbox, proximity_threshold)
        if relation is None:
            continue
        distance = _distance(object_center, _center(person_bbox))
        if best is None or distance < best[0]:
            best = (distance, person_id, person_bbox, relation)

    if best is None:
        return None, None, None
    return best[1], best[2], best[3]


def _relation(object_bbox: List[float], person_bbox: List[float], threshold: float) -> Optional[str]:
    if _contains_center(object_bbox, person_bbox):
        return "inside_person_bbox"
    if _iou(object_bbox, person_bbox) > 0:
        return "intersects_person_bbox"

    object_center = _center(object_bbox)
    person_center = _center(person_bbox)
    person_diag = max(1.0, _diagonal(person_bbox))
    if _distance(object_center, person_center) / person_diag <= threshold:
        return "near_person_bbox"
    return None


def _center(bbox: List[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _diagonal(bbox: List[float]) -> float:
    return float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))


def _contains_center(object_bbox: List[float], person_bbox: List[float]) -> bool:
    x, y = _center(object_bbox)
    return person_bbox[0] <= x <= person_bbox[2] and person_bbox[1] <= y <= person_bbox[3]


def _iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-6, area_a + area_b - intersection)

