from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass
class RegisteredFace:
    """One enrollment template; a student may own multiple templates."""

    student_id: str
    embedding: list[float]
    name: Optional[str] = None


@dataclass
class RecognitionResult:
    student_id: Optional[str]
    name: Optional[str]
    similarity: float
    distance: float
    matched: bool

    @property
    def confidence(self) -> float:
        """Compatibility alias. This is cosine similarity, not probability."""
        return self.similarity


_CONFIGS_DIR = Path(__file__).resolve().parents[5] / "configs" / "services" / "face_recognition"
_SUPPORTED_MODELS = ("buffalo_s", "buffalo_l")
_Registry = dict[str, tuple[Optional[str], list[np.ndarray]]]


def _load_config() -> dict:
    import yaml

    path = _CONFIGS_DIR / "insightface.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Face recognition config not found: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Face recognition config must be a YAML mapping: {path}")
    return config


def _number(name: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number in [{minimum}, {maximum}]")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {result}")
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _input_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("input_size must contain exactly [width, height]")
    return (
        _positive_int("input_size width", value[0]),
        _positive_int("input_size height", value[1]),
    )


def _device(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")
    result = value.strip().lower()
    if result in {"auto", "cpu", "cuda"}:
        return result
    if result.startswith("cuda:") and result[5:].isdigit():
        return result
    raise ValueError(f"unsupported face recognition device: '{value}'")


def _runtime(device: str) -> tuple[str, int, list[Any]]:
    import onnxruntime

    has_cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    if device == "auto":
        device = "cuda" if has_cuda else "cpu"
    if device != "cpu" and not has_cuda:
        raise RuntimeError(
            "CUDA recognition requires onnxruntime-gpu with CUDAExecutionProvider"
        )
    if device == "cpu":
        return device, -1, ["CPUExecutionProvider"]
    index = 0 if device == "cuda" else int(device.split(":", 1)[1])
    return device, index, [
        ("CUDAExecutionProvider", {"device_id": index}),
        "CPUExecutionProvider",
    ]


def _image(value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError("face_image must be a NumPy array")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"face_image must have shape (height, width, 3), got {value.shape}")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("face_image height and width must be greater than zero")
    if value.dtype != np.uint8:
        raise TypeError(f"face_image dtype must be uint8, got {value.dtype}")


def _landmarks(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (5, 2):
        raise ValueError(f"landmarks must have shape (5, 2), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("landmarks must contain only finite values")
    return result


def _embedding(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(value), dtype=np.float32)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("embedding must be a non-empty one-dimensional sequence")
    if not np.isfinite(result).all():
        raise ValueError("embedding must contain only finite values")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError("embedding must not be a zero vector")
    return result / norm


def _student_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("student_id must be a non-empty string")
    return value.strip()


def _name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("name must be a string or None")
    return value.strip() or None


class FaceRecognizer:
    """InsightFace recognition with multi-view enrollment and track stability."""

    def __init__(
        self,
        backend: str = "insightface",
        model_name: Optional[str] = None,
        enrollment_path: Optional[str | Path] = None,
        similarity_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        if backend != "insightface":
            raise ValueError("backend must be 'insightface'")
        config = _load_config()
        self._model_name = model_name or config.get("model", "buffalo_s")
        if self._model_name not in _SUPPORTED_MODELS:
            raise ValueError(f"model_name must be one of {list(_SUPPORTED_MODELS)}")
        self._threshold = _number(
            "similarity_threshold",
            similarity_threshold if similarity_threshold is not None else config.get("similarity_threshold", 0.35),
            0.0,
            1.0,
        )
        requested_device = _device(device if device is not None else config.get("device", "auto"))
        self._device, self._ctx_id, self._providers = _runtime(requested_device)
        self._input_size = _input_size(config.get("input_size", [640, 640]))
        self._face_size = _positive_int("embedding_image_size", config.get("embedding_image_size", 112))
        self._confirmation_hits = _positive_int("confirmation_hits", config.get("confirmation_hits", 3))
        self._history_size = _positive_int("history_size", config.get("history_size", 5))
        if self._confirmation_hits > self._history_size:
            raise ValueError("confirmation_hits must not exceed history_size")
        self._switch_margin = _number("switch_margin", config.get("switch_margin", 0.08), 0.0, 1.0)
        self._cache_frames = _positive_int("identity_cache_frames", config.get("identity_cache_frames", 30))
        self._quality = {
            "min_brightness": _number("min_brightness", config.get("min_brightness", 20), 0, 255),
            "max_brightness": _number("max_brightness", config.get("max_brightness", 235), 0, 255),
            "min_blur_score": _number("min_blur_score", config.get("min_blur_score", 20), 0, 10000),
        }
        if self._quality["min_brightness"] > self._quality["max_brightness"]:
            raise ValueError("min_brightness must not exceed max_brightness")

        self._registry: _Registry = {}
        self._dimension: Optional[int] = None
        self._history: dict[int, deque[tuple[Optional[str], float]]] = {}
        self._confirmed: dict[int, tuple[str, Optional[str], float]] = {}
        self._missing: dict[int, int] = {}
        self._app, self._recognition_model = self._build_model()
        if enrollment_path is not None:
            self.load_enrollments(enrollment_path)

    def _build_model(self):
        from insightface.app import FaceAnalysis

        try:
            app = FaceAnalysis(
                name=self._model_name,
                allowed_modules=["detection", "recognition"],
                providers=self._providers,
            )
            app.prepare(ctx_id=self._ctx_id, det_size=self._input_size)
        except Exception as exc:
            raise RuntimeError(f"Failed to load InsightFace model '{self._model_name}'") from exc
        model = app.models.get("recognition")
        if model is None:
            raise RuntimeError(f"Model pack '{self._model_name}' has no recognition model")
        return app, model

    def load_enrollments(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8") as file:
            raw = json.load(file)
        metadata: Mapping[str, Any] = {}
        if isinstance(raw, list):
            students = raw
        elif isinstance(raw, dict):
            metadata = raw.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("enrollment metadata must be an object")
            if "students" not in raw:
                raise ValueError("enrollment JSON must contain a 'students' list")
            students = raw["students"]
        else:
            raise ValueError("enrollment JSON must be an object or a list")
        if not isinstance(students, list):
            raise ValueError("enrollment 'students' must be a list")
        if metadata.get("model", self._model_name) != self._model_name:
            raise ValueError("enrollment model does not match the active recognition model")

        registry: _Registry = {}
        dimension: Optional[int] = None
        for index, student in enumerate(students):
            if not isinstance(student, dict):
                raise ValueError(f"students[{index}] must be an object")
            sid, name = _student_id(student.get("student_id")), _name(student.get("name"))
            single, multiple = "embedding" in student, "embeddings" in student
            if single == multiple:
                raise ValueError(f"students[{index}] must contain one of embedding/embeddings")
            raw_embeddings = [student["embedding"]] if single else student["embeddings"]
            if not isinstance(raw_embeddings, list) or not raw_embeddings:
                raise ValueError(f"students[{index}] embeddings must be a non-empty list")
            for value in raw_embeddings:
                normalized = _embedding(value)
                dimension = self._check_dimension(normalized, dimension)
                self._add(registry, sid, name, normalized)
        stored_dimension = metadata.get("embedding_dimension")
        if stored_dimension is not None and stored_dimension != dimension:
            raise ValueError("metadata embedding_dimension does not match enrollment data")
        self._registry, self._dimension = registry, dimension
        self.reset()

    def save_enrollments(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {"version": 1, "model": self._model_name, "embedding_dimension": self._dimension},
            "students": [
                {"student_id": sid, "name": name, "embeddings": [item.tolist() for item in items]}
                for sid, (name, items) in sorted(self._registry.items())
            ],
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)

    def add_enrollment(self, record: RegisteredFace) -> None:
        if not isinstance(record, RegisteredFace):
            raise TypeError("record must be a RegisteredFace")
        normalized = _embedding(record.embedding)
        dimension = self._check_dimension(normalized, self._dimension)
        self._add(self._registry, _student_id(record.student_id), _name(record.name), normalized)
        self._dimension = dimension

    def enroll(
        self,
        student_id: str,
        face_image: np.ndarray,
        *,
        name: Optional[str] = None,
        landmarks: Optional[Sequence[Sequence[float]]] = None,
    ) -> RegisteredFace:
        aligned = self._align(face_image, landmarks)
        self._check_quality(aligned)
        record = RegisteredFace(_student_id(student_id), self._encode_aligned(aligned).tolist(), _name(name))
        self.add_enrollment(record)
        return record

    def remove_enrollment(self, student_id: str) -> bool:
        removed = self._registry.pop(_student_id(student_id), None) is not None
        if removed:
            self._dimension = next((item.size for _, items in self._registry.values() for item in items), None)
            self.reset()
        return removed

    def encode(self, face_image: np.ndarray, landmarks: Optional[Sequence[Sequence[float]]] = None) -> np.ndarray:
        return self._encode_aligned(self._align(face_image, landmarks))

    def recognize(
        self,
        face_image: np.ndarray,
        landmarks: Optional[Sequence[Sequence[float]]] = None,
        *,
        track_id: Optional[int] = None,
    ) -> RecognitionResult:
        result = self._match(self.encode(face_image, landmarks))
        if track_id is None:
            return result
        if isinstance(track_id, bool) or not isinstance(track_id, int):
            raise TypeError("track_id must be an integer or None")
        return self._stabilize(track_id, result)

    def update_active_tracks(self, active_track_ids: Iterable[int]) -> None:
        active = set(active_track_ids)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in active):
            raise TypeError("active track IDs must be integers")
        for track_id in set(self._history) | set(self._confirmed) | set(self._missing):
            if track_id in active:
                self._missing[track_id] = 0
            elif self._missing.get(track_id, 0) + 1 > self._cache_frames:
                self.forget_track(track_id)
            else:
                self._missing[track_id] = self._missing.get(track_id, 0) + 1

    def forget_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)
        self._confirmed.pop(track_id, None)
        self._missing.pop(track_id, None)

    def reset(self) -> None:
        self._history.clear()
        self._confirmed.clear()
        self._missing.clear()

    def _align(self, face_image: np.ndarray, landmarks: Optional[Sequence[Sequence[float]]]) -> np.ndarray:
        _image(face_image)
        if landmarks is not None:
            from insightface.utils import face_align

            return face_align.norm_crop(face_image, landmark=_landmarks(landmarks), image_size=self._face_size)
        import cv2

        return cv2.resize(face_image, (self._face_size, self._face_size), interpolation=cv2.INTER_LINEAR)

    def _encode_aligned(self, aligned: np.ndarray) -> np.ndarray:
        normalized = _embedding(np.asarray(self._recognition_model.get_feat(aligned)).reshape(-1))
        self._check_dimension(normalized, self._dimension)
        return normalized

    def _match(self, embedding: np.ndarray) -> RecognitionResult:
        if not self._registry:
            return self._unmatched()
        best_id, best_name, best_score = None, None, -1.0
        for sid, (name, templates) in self._registry.items():
            score = max(float(np.dot(embedding, template)) for template in templates)
            if score > best_score:
                best_id, best_name, best_score = sid, name, score
        score = float(np.clip(best_score, -1, 1))
        matched = score >= self._threshold
        return RecognitionResult(best_id if matched else None, best_name if matched else None, score, 1 - score, matched)

    def _stabilize(self, track_id: int, raw: RecognitionResult) -> RecognitionResult:
        history = self._history.setdefault(track_id, deque(maxlen=self._history_size))
        history.append((raw.student_id if raw.matched else None, raw.similarity))
        self._missing[track_id] = 0
        ids = [sid for sid, _ in history if sid is not None]
        candidate, score = None, -1.0
        if ids:
            candidate, hits = Counter(ids).most_common(1)[0]
            if hits >= self._confirmation_hits:
                score = float(np.mean([value for sid, value in history if sid == candidate]))
            else:
                candidate = None
        confirmed = self._confirmed.get(track_id)
        if confirmed is None and candidate is not None:
            confirmed = (candidate, self._registry[candidate][0], score)
            self._confirmed[track_id] = confirmed
        elif confirmed and candidate and candidate != confirmed[0] and score >= confirmed[2] + self._switch_margin:
            confirmed = (candidate, self._registry[candidate][0], score)
            self._confirmed[track_id] = confirmed
        elif confirmed and raw.student_id == confirmed[0]:
            confirmed = (confirmed[0], confirmed[1], raw.similarity)
            self._confirmed[track_id] = confirmed
        if confirmed is None:
            return self._unmatched(raw.similarity)
        return RecognitionResult(confirmed[0], confirmed[1], confirmed[2], 1 - confirmed[2], True)

    @staticmethod
    def _unmatched(similarity: float = 0.0) -> RecognitionResult:
        similarity = float(np.clip(similarity, -1, 1))
        return RecognitionResult(None, None, similarity, 1 - similarity, False)

    @staticmethod
    def _add(registry: _Registry, sid: str, name: Optional[str], embedding: np.ndarray) -> None:
        existing = registry.get(sid)
        if existing is None:
            registry[sid] = (name, [embedding])
            return
        old_name, templates = existing
        if old_name and name and old_name != name:
            raise ValueError(f"student_id '{sid}' has conflicting names '{old_name}' and '{name}'")
        registry[sid] = (old_name or name, [*templates, embedding])

    @staticmethod
    def _check_dimension(value: np.ndarray, expected: Optional[int]) -> int:
        if expected is not None and value.size != expected:
            raise ValueError(f"embedding dimension must be {expected}, got {value.size}")
        return int(value.size)

    def _check_quality(self, aligned: np.ndarray) -> None:
        import cv2

        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if not self._quality["min_brightness"] <= brightness <= self._quality["max_brightness"]:
            raise ValueError(f"face brightness {brightness:.1f} is outside the accepted range")
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur < self._quality["min_blur_score"]:
            raise ValueError(f"face is too blurry for enrollment (score {blur:.1f})")
