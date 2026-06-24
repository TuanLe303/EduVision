import numpy as np

from services.vision_ai.src.behavior_detection import BehaviorDetector


class _AliasLabelModel:
    task = "detect"
    names = {
        0: "focus",
        1: "drowsy",
        2: "sleeping",
        3: "using_phone",
        4: "off_task",
        5: "side_taliking",
        6: "raising_hand",
    }

    def predict(self, **kwargs):
        return []


def test_model_label_aliases_are_normalized_before_validation() -> None:
    detector = BehaviorDetector(
        model=_AliasLabelModel(),
        config={
            "label_aliases": {
                "focus": "focused",
                "side_taliking": "side_talking",
            }
        },
    )

    assert detector._aggregator.normalize_state("focus") == "focused"
    assert detector._aggregator.normalize_state("side_taliking") == "side_talking"


def test_window_override_scales_temporal_evidence_counts() -> None:
    detector = BehaviorDetector(
        model=_AliasLabelModel(),
        window_size=6,
        config={
            "window_size": 12,
            "min_history": 12,
            "min_state_frames": 7,
        },
    )

    assert detector._aggregator.window_size == 6
    assert detector._aggregator.min_history == 6
    assert detector._aggregator.min_state_frames == 4


def test_detector_emits_retained_state_when_behavior_detection_is_missed() -> None:
    detector = BehaviorDetector(
        model=_AliasLabelModel(),
        config={
            "window_size": 2,
            "min_history": 2,
            "min_state_frames": 2,
            "max_detection_gap": 2,
        },
    )
    detector._aggregator.update(3, "focused", 0.9, 1)
    detector._aggregator.update(3, "focused", 0.9, 2)

    _, detections, temporal = detector.update(
        np.zeros((20, 20, 3), dtype=np.uint8),
        frame_index=3,
        canonical_tracks=[{"track_id": 3, "bbox": [0, 0, 10, 10]}],
    )

    assert detections == []
    assert len(temporal) == 1
    assert temporal[0].state == "focused"
    assert temporal[0].observed is False
    assert temporal[0].detection_age == 1
