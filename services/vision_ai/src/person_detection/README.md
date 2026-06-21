# Person Detection

Current first-party wrapper around Ultralytics YOLOv11 for detecting persons
in a video frame.

## Current API

```python
from services.vision_ai.src.person_detection import PersonDetector

detector = PersonDetector()
persons = detector.detect(frame)

for person in persons:
    print(person.bbox, person.confidence, person.class_id)
```

Output is `List[PersonDetection]`, sorted by descending confidence:

```python
@dataclass
class PersonDetection:
    bbox: List[float]  # [x1, y1, x2, y2] pixel coordinates
    confidence: float  # person detection score [0, 1]
    class_id: int = 0  # COCO class 0 = person
```

Config files are at `configs/services/person_detection/`:

- `yolo11n.yaml`
- `yolo11s.yaml`

Switch models with:

```python
detector = PersonDetector(model_name="yolo11s")
```

The older planning notes below are kept for context.

# Original Planning Notes

Detects all persons in each video frame and returns bounding boxes with confidence scores. This is the first stage of the EduVision pipeline — all downstream modules (tracking, face recognition, behavior analysis) operate on the regions produced here.

---

## Model Options

| Option | Model | Parameters | Input Size | Notes |
|---|---|---|---|---|
| **`yolo11n`** *(default)* | YOLOv11-nano | ~2.6 M | 640×640 | Fastest; runs on CPU or low-end GPU. Recommended starting point. |
| **`yolo11s`** | YOLOv11-small | ~9.4 M | 640×640 | Better recall on small/occluded persons; moderate GPU recommended. |

Both variants are provided by [Ultralytics](https://github.com/ultralytics/ultralytics) and share the same API — switching between them requires only a config change.

---

## Input / Output

| | Description |
|---|---|
| **Input** | BGR frame (H × W × 3, `uint8`) from the video stream |
| **Output** | List of detections, each containing: `bbox [x1, y1, x2, y2]`, `confidence (float)`, `class_id (int, always 0 = person)` |

Only the `person` class (COCO class 0) is passed downstream. All other detected classes are discarded at this stage.

---

## Configuration

Configured via `configs/services/person_detection/{model}.yaml`:

```yaml
model: yolo11n          # yolo11n | yolo11s
confidence_threshold: 0.4
iou_threshold: 0.5
input_size: 640
device: auto            # auto | cpu | cuda:0
```

Or overridden at runtime:

```bash
python -m services.vision_ai.src.main --source video.mp4 --detector yolo11n
python -m services.vision_ai.src.main --source video.mp4 --detector yolo11s
```

**`device: auto`** selects CUDA if available, falls back to CPU automatically.

---

## Recommended Settings by Hardware

| Hardware | Recommended Model | Confidence Threshold |
|---|---|---|
| CPU only | `yolo11n` | 0.4 |
| Low-end GPU (≤ 4 GB VRAM) | `yolo11n` | 0.4 |
| Mid-range GPU (6–8 GB VRAM) | `yolo11s` | 0.4 |
| High-end GPU (≥ 10 GB VRAM) | `yolo11s` | 0.35 |

---

## Model Weights

Pre-trained weights are downloaded automatically by Ultralytics on first run and cached in `models/yolo/`:

```
models/
└── yolo/
    ├── yolo11n.pt
    └── yolo11s.pt
```

The `models/` directory is git-ignored. Weights are not committed to the repository.

---

## Integration with Downstream Modules

```
Frame
  └─▶ Person Detection
          └─▶ [bbox list]
                  ├─▶ Multi-object Tracking   (assigns persistent track IDs)
                  ├─▶ Face Detection          (crops person region → find face)
                  └─▶ Pose Estimation         (crops person region → keypoints)
```

The detection output is consumed by the tracker first. The tracker attaches a `track_id` to each box, and only tracked boxes are forwarded to face and pose modules.

---

## Directory Structure

```
person_detection/
├── README.md           ← this file
└── src/
    └── ultralytics/    ← Ultralytics YOLOv11 library source
```
