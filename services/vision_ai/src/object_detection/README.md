# Off-task Object Detection

YOLO wrapper for objects that may support behavior rules.

## Usage

```python
from services.vision_ai.src.object_detection import OffTaskObjectDetector

detector = OffTaskObjectDetector()
objects = detector.detect(frame, persons=tracks)
```

`persons` is optional. When provided, each object is associated with the
nearest/intersecting person bbox.

## Output

```python
@dataclass
class ObjectDetection:
    label: str
    class_id: int
    bbox: List[float]
    confidence: float
    person_id: Optional[int]
    person_bbox: Optional[List[float]]
    proximity: Optional[str]
```

## Notes

The default COCO model supports labels such as `cell phone`, `laptop`, `book`,
`bottle`, and `backpack`. Classes such as tablet or earphone require a custom
fine-tuned detector later.

