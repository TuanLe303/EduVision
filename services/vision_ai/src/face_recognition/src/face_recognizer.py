from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class RegisteredFace:
    student_id: str
    embedding: List[float]
    name: Optional[str] = None


@dataclass
class RecognitionResult:
    student_id: Optional[str]
    name: Optional[str]
    confidence: float
    distance: float
    matched: bool


_SUPPORTED_BACKENDS = ("insightface",)

_CONFIGS_DIR = Path(__file__).resolve().parents[5] / "configs" / "services" / "face_recognition"


def _load_config(backend: str) -> dict:
    import yaml

    path = _CONFIGS_DIR / f"{backend}.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _normalize(embedding: Iterable[float]) -> np.ndarray:
    vector = np.asarray(list(embedding), dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class FaceRecognizer:
    """
    Face recognition and attendance wrapper.

    Legacy implementation retained only as an import-compatibility shim.
    Public imports resolve to the production implementation in recognizer.py.

    Enrollment storage format:
        {
          "students": [
            {"student_id": "S001", "name": "Nguyen Van A", "embedding": [...]}
          ]
        }
    """

    def __init__(
        self,
        backend: str = "insightface",
        model_name: Optional[str] = None,
        enrollment_path: Optional[str | Path] = None,
        similarity_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {list(_SUPPORTED_BACKENDS)}, got '{backend}'"
            )

        cfg = _load_config(backend)
        self._backend = backend
        self._model_name = model_name or cfg.get("model", "buffalo_s")
        self._threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else cfg.get("similarity_threshold", 0.35)
        )
        selected_device = device if device is not None else cfg.get("device", "cpu")
        self._ctx_id = -1 if selected_device == "cpu" else 0

        self._registry: list[tuple[str, Optional[str], np.ndarray]] = []
        if enrollment_path is not None:
            self.load_enrollments(enrollment_path)

        self._model = self._build_model(cfg)

    def _build_model(self, cfg: dict):
        from insightface.app import FaceAnalysis

        providers = cfg.get("providers")
        model = FaceAnalysis(name=self._model_name, providers=providers)
        input_size = tuple(cfg.get("input_size", [640, 640]))
        model.prepare(ctx_id=self._ctx_id, det_size=input_size)
        return model

    def load_enrollments(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8") as f:
            raw = json.load(f)

        students = raw.get("students", raw if isinstance(raw, list) else [])
        self._registry = []
        for student in students:
            record = RegisteredFace(
                student_id=str(student["student_id"]),
                name=student.get("name"),
                embedding=list(student["embedding"]),
            )
            self.add_enrollment(record)

    def add_enrollment(self, record: RegisteredFace) -> None:
        self._registry.append(
            (record.student_id, record.name, _normalize(record.embedding))
        )

    def encode(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Return one normalized embedding from a face crop.

        If the crop still contains multiple faces, the face with the largest
        detection area is used.
        """
        faces = self._model.get(face_crop)
        if not faces:
            return None

        face = max(faces, key=lambda item: _bbox_area(item.bbox))
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            return None
        return _normalize(embedding)

    def recognize(self, face_crop: np.ndarray) -> RecognitionResult:
        embedding = self.encode(face_crop)
        if embedding is None or not self._registry:
            return RecognitionResult(
                student_id=None,
                name=None,
                confidence=0.0,
                distance=1.0,
                matched=False,
            )

        best_id: Optional[str] = None
        best_name: Optional[str] = None
        best_similarity = -1.0

        for student_id, name, registered_embedding in self._registry:
            similarity = float(np.dot(embedding, registered_embedding))
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = student_id
                best_name = name

        distance = 1.0 - best_similarity
        matched = best_similarity >= self._threshold
        return RecognitionResult(
            student_id=best_id if matched else None,
            name=best_name if matched else None,
            confidence=max(0.0, best_similarity),
            distance=distance,
            matched=matched,
        )

    def reset(self) -> None:
        """No persistent inference state to clear; provided for API consistency."""
        pass


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


# Compatibility for callers that imported this historical module directly.
# The production implementation lives in ``recognizer.py``.
from .recognizer import FaceRecognizer, RecognitionResult, RegisteredFace  # noqa: E402,F401
