from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Deque, Dict, List, Optional, Sequence


@dataclass
class StudentFrameSignal:
    track_id: int
    student_id: Optional[str] = None
    timestamp: Optional[float] = None
    face_detected: bool = True
    recognized: bool = False
    seated: bool = True
    head_pose: Optional[Any] = None
    pose: Optional[Any] = None
    objects: Sequence[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorResult:
    track_id: int
    student_id: Optional[str]
    state: str
    scores: Dict[str, float]
    events: List[str]
    timestamp: float


_CONFIG_PATH = Path(__file__).resolve().parents[5] / "configs" / "behavior.yaml"
_STATES = ("focused", "drowsy", "using_phone", "off_task", "away_from_seat", "side_talking")
_PRIORITY = ("away_from_seat", "using_phone", "drowsy", "side_talking", "off_task", "focused")


def _load_config() -> dict:
    import yaml

    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open() as f:
        return yaml.safe_load(f) or {}


class BehaviorAnalyzer:
    """Rule-based temporal behavior classifier."""

    def __init__(
        self,
        history_size: Optional[int] = None,
        min_state_frames: Optional[int] = None,
        phone_labels: Optional[Sequence[str]] = None,
        off_task_object_labels: Optional[Sequence[str]] = None,
        yaw_side_threshold: Optional[float] = None,
        pitch_down_threshold: Optional[float] = None,
    ) -> None:
        cfg = _load_config()
        self._history_size = history_size if history_size is not None else cfg.get("history_size", 12)
        self._min_state_frames = (
            min_state_frames if min_state_frames is not None else cfg.get("min_state_frames", 3)
        )
        self._phone_labels = set(phone_labels or cfg.get("phone_labels", ["cell phone"]))
        self._off_task_object_labels = set(
            off_task_object_labels
            or cfg.get("off_task_object_labels", ["laptop", "book", "cup", "bottle", "backpack"])
        )
        self._yaw_side_threshold = (
            yaw_side_threshold if yaw_side_threshold is not None else cfg.get("yaw_side_threshold", 30.0)
        )
        self._pitch_down_threshold = (
            pitch_down_threshold
            if pitch_down_threshold is not None
            else cfg.get("pitch_down_threshold", 25.0)
        )

        self._history: dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=self._history_size))
        self._last_state: dict[int, str] = {}

    def analyze(self, signal: StudentFrameSignal | dict) -> BehaviorResult:
        signal = _coerce_signal(signal)
        raw_state = self._classify_frame(signal)
        history = self._history[signal.track_id]
        history.append(raw_state)

        state = self._smooth_state(history, raw_state)
        scores = self._scores(history)
        events = self._events(signal.track_id, state)

        timestamp = signal.timestamp if signal.timestamp is not None else time()
        return BehaviorResult(
            track_id=signal.track_id,
            student_id=signal.student_id,
            state=state,
            scores=scores,
            events=events,
            timestamp=timestamp,
        )

    def analyze_many(self, signals: Sequence[StudentFrameSignal | dict]) -> List[BehaviorResult]:
        return [self.analyze(signal) for signal in signals]

    def reset(self, track_id: Optional[int] = None) -> None:
        if track_id is None:
            self._history.clear()
            self._last_state.clear()
            return
        self._history.pop(track_id, None)
        self._last_state.pop(track_id, None)

    def _classify_frame(self, signal: StudentFrameSignal) -> str:
        if not signal.seated or signal.metadata.get("away_from_seat", False):
            return "away_from_seat"

        labels = [_object_label(obj) for obj in signal.objects]
        if any(label in self._phone_labels for label in labels):
            return "using_phone"

        if signal.metadata.get("eyes_closed", False):
            return "drowsy"

        pitch = _get_number(signal.head_pose, "pitch")
        if pitch is not None and pitch > self._pitch_down_threshold:
            return "drowsy"

        yaw = _get_number(signal.head_pose, "yaw")
        gaze_direction = _get_text(signal.head_pose, "gaze_direction")
        if yaw is not None and abs(yaw) > self._yaw_side_threshold:
            return "side_talking"
        if gaze_direction in {"left", "right"}:
            return "side_talking"

        if not signal.face_detected:
            return "off_task"
        if any(label in self._off_task_object_labels for label in labels):
            return "off_task"
        if gaze_direction in {"down", "up"}:
            return "off_task"

        return "focused"

    def _smooth_state(self, history: Deque[str], fallback: str) -> str:
        counts = Counter(history)
        for state in _PRIORITY:
            if counts[state] >= self._min_state_frames:
                return state
        return fallback

    def _scores(self, history: Deque[str]) -> Dict[str, float]:
        total = max(1, len(history))
        counts = Counter(history)
        return {state: counts[state] / total for state in _STATES}

    def _events(self, track_id: int, state: str) -> List[str]:
        previous = self._last_state.get(track_id)
        self._last_state[track_id] = state
        if previous is None or previous == state:
            return []
        return [f"behavior_changed:{previous}->{state}"]


def _coerce_signal(signal: StudentFrameSignal | dict) -> StudentFrameSignal:
    if isinstance(signal, StudentFrameSignal):
        return signal
    return StudentFrameSignal(**signal)


def _object_label(obj: Any) -> str:
    if isinstance(obj, dict):
        return str(obj.get("label", ""))
    return str(getattr(obj, "label", ""))


def _get_number(obj: Any, field_name: str) -> Optional[float]:
    if obj is None:
        return None
    value = obj.get(field_name) if isinstance(obj, dict) else getattr(obj, field_name, None)
    if value is None:
        return None
    return float(value)


def _get_text(obj: Any, field_name: str) -> Optional[str]:
    if obj is None:
        return None
    value = obj.get(field_name) if isinstance(obj, dict) else getattr(obj, field_name, None)
    if value is None:
        return None
    return str(value)

