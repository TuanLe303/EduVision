from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Mapping, Optional, Sequence


CANONICAL_STATES = (
    "focused",
    "drowsy",
    "sleeping",
    "using_phone",
    "off_task",
    "side_talking",
    "raising_hand",
    "away_from_seat",
)

DEFAULT_ALIASES = {
    "focus": "focused",
    "side_taliking": "side_talking",
}

DEFAULT_PRIORITY = (
    "away_from_seat",
    "sleeping",
    "using_phone",
    "drowsy",
    "side_talking",
    "off_task",
    "raising_hand",
    "focused",
)


@dataclass(frozen=True)
class FrameBehavior:
    state: str
    confidence: float
    frame_index: int


@dataclass(frozen=True)
class TemporalBehaviorResult:
    track_id: int
    state: Optional[str]
    confidence: float
    scores: dict[str, float]
    frame_count: int
    ready: bool
    changed: bool
    reason: str
    observed: bool = True
    detection_age: int = 0


class TemporalBehaviorAggregator:
    """Aggregate YOLO frame predictions independently for every track ID."""

    def __init__(
        self,
        window_size: int = 12,
        min_history: int = 4,
        min_state_frames: int = 3,
        enter_threshold: float = 0.55,
        switch_margin: float = 0.10,
        stale_track_frames: int = 90,
        max_detection_gap: int = 5,
        state_thresholds: Optional[Mapping[str, float]] = None,
        aliases: Optional[Mapping[str, str]] = None,
        priority: Sequence[str] = DEFAULT_PRIORITY,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")
        if not 1 <= min_history <= window_size:
            raise ValueError("min_history must be between 1 and window_size")
        if not 1 <= min_state_frames <= window_size:
            raise ValueError("min_state_frames must be between 1 and window_size")
        if not 0.0 <= enter_threshold <= 1.0:
            raise ValueError("enter_threshold must be in [0, 1]")
        if not 0.0 <= switch_margin <= 1.0:
            raise ValueError("switch_margin must be in [0, 1]")
        if max_detection_gap < 0:
            raise ValueError("max_detection_gap must be zero or greater")

        self.window_size = window_size
        self.min_history = min_history
        self.min_state_frames = min_state_frames
        self.enter_threshold = enter_threshold
        self.switch_margin = switch_margin
        self.stale_track_frames = stale_track_frames
        self.max_detection_gap = max_detection_gap
        self.state_thresholds = dict(state_thresholds or {})
        self.aliases = {**DEFAULT_ALIASES, **dict(aliases or {})}
        self.priority = tuple(priority)
        self._history: dict[int, Deque[FrameBehavior]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._last_state: dict[int, str] = {}
        self._last_seen: dict[int, int] = {}
        self._last_result: dict[int, TemporalBehaviorResult] = {}

    def update(
        self,
        track_id: int,
        state: str,
        confidence: float,
        frame_index: int,
    ) -> TemporalBehaviorResult:
        state = self.normalize_state(state)
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

        history = self._history[int(track_id)]
        history.append(FrameBehavior(state, confidence, int(frame_index)))
        self._last_seen[int(track_id)] = int(frame_index)

        counts = Counter(item.state for item in history)
        weighted = Counter()
        for item in history:
            weighted[item.state] += item.confidence
        total_weight = sum(weighted.values())
        scores = {
            name: (weighted[name] / total_weight if total_weight else 0.0)
            for name in CANONICAL_STATES
        }
        candidate = max(
            CANONICAL_STATES,
            key=lambda name: (scores[name], -self._priority_index(name)),
        )

        previous = self._last_state.get(int(track_id))
        ready = len(history) >= self.min_history
        threshold = self.state_thresholds.get(candidate, self.enter_threshold)
        enough_evidence = counts[candidate] >= self.min_state_frames and scores[candidate] >= threshold

        if previous is None:
            selected = candidate if ready and enough_evidence else None
            reason = "initial temporal decision" if selected is not None else "collecting temporal evidence"
        elif candidate == previous:
            selected = previous
            reason = "current state remains dominant"
        elif ready and enough_evidence and scores[candidate] >= scores.get(previous, 0.0) + self.switch_margin:
            selected = candidate
            reason = f"{candidate} passed temporal threshold and switch margin"
        else:
            selected = previous
            reason = "hysteresis retained previous state"

        changed = previous is not None and selected is not None and selected != previous
        if selected is not None:
            self._last_state[int(track_id)] = selected
        result = TemporalBehaviorResult(
            track_id=int(track_id),
            state=selected,
            confidence=scores.get(selected, 0.0) if selected is not None else 0.0,
            scores=scores,
            frame_count=len(history),
            ready=ready,
            changed=changed,
            reason=reason,
        )
        self._last_result[int(track_id)] = result
        return result

    def hold(self, track_id: int, frame_index: int) -> Optional[TemporalBehaviorResult]:
        """Return a decayed, unobserved state during a short detection gap."""

        track_id = int(track_id)
        last_seen = self._last_seen.get(track_id)
        previous = self._last_result.get(track_id)
        if last_seen is None or previous is None or previous.state is None:
            return None
        detection_age = int(frame_index) - last_seen
        if detection_age <= 0 or detection_age > self.max_detection_gap:
            return None
        decay = 1.0 - detection_age / (self.max_detection_gap + 1.0)
        return TemporalBehaviorResult(
            track_id=track_id,
            state=previous.state,
            confidence=previous.confidence * decay,
            scores={name: score * decay for name, score in previous.scores.items()},
            frame_count=previous.frame_count,
            ready=previous.ready,
            changed=False,
            reason=f"retained state during detection gap ({detection_age} frame(s))",
            observed=False,
            detection_age=detection_age,
        )

    def finish_frame(
        self,
        frame_index: int,
        active_track_ids: Sequence[int] = (),
        observed_track_ids: Sequence[int] = (),
    ) -> None:
        """Apply missing-detection and stale-track policies after a frame."""

        active = {int(track_id) for track_id in active_track_ids}
        observed = {int(track_id) for track_id in observed_track_ids}
        for track_id in active - observed:
            last_seen = self._last_seen.get(track_id)
            if last_seen is not None and frame_index - last_seen > self.max_detection_gap:
                self.reset(track_id)
        self.cleanup(frame_index, active_track_ids)

    def cleanup(self, frame_index: int, active_track_ids: Sequence[int] = ()) -> None:
        active = {int(track_id) for track_id in active_track_ids}
        stale = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if track_id not in active and frame_index - last_seen > self.stale_track_frames
        ]
        for track_id in stale:
            self.reset(track_id)

    def reset(self, track_id: Optional[int] = None) -> None:
        if track_id is None:
            self._history.clear()
            self._last_state.clear()
            self._last_seen.clear()
            self._last_result.clear()
            return
        self._history.pop(int(track_id), None)
        self._last_state.pop(int(track_id), None)
        self._last_seen.pop(int(track_id), None)
        self._last_result.pop(int(track_id), None)

    def normalize_state(self, state: str) -> str:
        normalized = str(state).strip().lower().replace(" ", "_")
        normalized = self.aliases.get(normalized, normalized)
        if normalized not in CANONICAL_STATES:
            raise ValueError(
                f"unsupported behavior label '{state}'; expected one of {list(CANONICAL_STATES)}"
            )
        return normalized

    def _priority_index(self, state: str) -> int:
        try:
            return self.priority.index(state)
        except ValueError:
            return len(self.priority)
