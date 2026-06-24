# Tracking Module

Multi-object tracking that assigns the canonical persistent `track_id` to each
detected person across frames. Downstream modules reuse this ID for behavior,
identity, object, and seat results.

---

## Implementation

ByteTrack and BoT-SORT are **built into Ultralytics** — no separate installation required. The module wraps `model.track()` from the `ultralytics` package already installed in the virtual environment.

---

## Configuration

Tracker parameters are loaded from:

```
configs/services/tracking/
├── bytetrack.yaml
└── botsort.yaml
```

Edit those files to tune thresholds (`track_high_thresh`, `match_thresh`, `track_buffer`, etc.). The wrapper falls back to Ultralytics built-in defaults if the project config file is missing.

---

## Input / Output

| | Description |
|---|---|
| **Input** | BGR frame (`np.ndarray`, H × W × 3, `uint8`) — same frame passed to person detection |
| **Output** | `List[TrackResult]` — one entry per tracked person |

`TrackResult` fields:

| Field | Type | Description |
|---|---|---|
| `track_id` | `int` | Persistent ID for this person across frames |
| `bbox` | `List[float]` | `[x1, y1, x2, y2]` in pixel coordinates |
| `confidence` | `float` | Detection confidence score |

Returns an empty list on the first frame or when no persons are detected.

---

## Special Requirements

- Call `tracker.update(frame)` **once per frame, in order** — the tracker maintains internal state between calls.
- Call `tracker.reset()` when switching to a new video source or starting a new session, to clear stale track IDs.
- `persist=True` is set internally — do not create a new `Tracker` instance per frame.

---

## Usage

```python
from services.vision_ai.src.tracking import Tracker

# Default: yolo11n + ByteTrack
tracker = Tracker()

# Use yolo11s + BoT-SORT
tracker = Tracker(model_name="yolo11s", tracker="botsort")

# Per-frame loop
for frame in video_stream:
    tracks = tracker.update(frame)
    for t in tracks:
        print(t.track_id, t.bbox, t.confidence)

# Reset between sessions
tracker.reset()
```

The end-to-end pipeline calls `update(frame)` once and uses these IDs for every
downstream task. Behavior YOLO only predicts boxes/classes; Hungarian matching
attaches those predictions to the canonical IDs here.

The tracker is designed to run as part of the end-to-end Vision AI pipeline.
Use `python -m services.vision_ai.src.main` to start that pipeline.

---

## Directory Structure

```
tracking/
├── README.md               ← this file
└── src/
    ├── __init__.py
    └── tracker.py          ← Tracker class, TrackResult dataclass
```
