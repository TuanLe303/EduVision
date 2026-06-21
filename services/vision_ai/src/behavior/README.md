# Behavior Analysis

Rule-based temporal classifier that fuses frame-level signals.

## Usage

```python
from services.vision_ai.src.behavior import BehaviorAnalyzer, StudentFrameSignal

analyzer = BehaviorAnalyzer()
result = analyzer.analyze(
    StudentFrameSignal(
        track_id=1,
        student_id="S001",
        face_detected=True,
        seated=True,
        head_pose={"yaw": 5, "pitch": 3, "gaze_direction": "center"},
        objects=[],
    )
)
```

## States

- `focused`
- `drowsy`
- `using_phone`
- `off_task`
- `away_from_seat`
- `side_talking`

The analyzer keeps short per-track history and smooths state changes using
`configs/behavior.yaml`.

