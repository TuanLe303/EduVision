import pytest

from tools.evaluate.metrics import evaluate_records


def test_perfect_e2e_predictions_score_one() -> None:
    truth = [
        {
            "frame_index": index,
            "timestamp": (index - 1) * 0.04,
            "person_count": 1,
            "students": [{"student_id": "S001", "state": "focused"}],
        }
        for index in range(1, 4)
    ]
    predictions = [
        {
            "frame_index": index,
            "timestamp": (index - 1) * 0.04,
            "tracks": [{"track_id": 7}],
            "recognition": {
                "7": {"student_id": "S001", "matched": True}
            } if index == 1 else {},
            "final_behavior": [{"track_id": 7, "state": "focused"}],
        }
        for index in range(1, 4)
    ]

    result = evaluate_records(predictions, truth, fps=25)

    assert result["attendance"]["f1"] == 1.0
    assert result["student_behavior"]["macro_f1"] == 1.0
    assert result["events"]["f1"] == 1.0
    assert result["events"]["duration_mae_seconds"] == 0.0
    assert result["student_count_mae"] == 0.0


def test_wrong_identity_and_behavior_are_penalized() -> None:
    truth = [{
        "frame_index": 1,
        "students": [{"student_id": "S001", "state": "sleeping"}],
        "person_count": 1,
    }]
    predictions = [{
        "frame_index": 1,
        "tracks": [{"track_id": 2}],
        "recognition": {"2": {"student_id": "S999", "matched": True}},
        "final_behavior": [{"track_id": 2, "state": "focused"}],
    }]

    result = evaluate_records(predictions, truth, fps=25)

    assert result["attendance"]["f1"] == 0.0
    assert result["student_behavior"]["macro_f1"] == 0.0
    assert result["events"]["available"] is False


def test_performance_metrics_are_combined() -> None:
    records = [{
        "frame_index": 1,
        "timestamp": 0.0,
        "tracks": [],
        "recognition": {},
        "final_behavior": [],
        "processing_ms": 20.0,
    }]
    truth = [{"frame_index": 1, "timestamp": 0.0, "students": [], "person_count": 0}]

    result = evaluate_records(
        records,
        truth,
        runtime_seconds=0.02,
        video_duration_seconds=0.04,
        peak_ram_mb=100.0,
    )

    assert result["performance"]["effective_fps"] == pytest.approx(50.0)
    assert result["performance"]["real_time_factor"] == pytest.approx(0.5)
    assert result["performance"]["p95_latency_ms"] == 20.0


def test_unannotated_prediction_frames_are_ignored() -> None:
    predictions = [
        {
            "frame_index": index,
            "tracks": [{"track_id": 1}],
            "recognition": {"1": {"student_id": "S001", "matched": True}},
            "final_behavior": [{"track_id": 1, "state": "focused"}],
        }
        for index in range(1, 11)
    ]
    truth = [{
        "frame_index": 5,
        "students": [{"student_id": "S001", "state": "focused"}],
        "person_count": 1,
    }]

    result = evaluate_records(predictions, truth, fps=25)

    assert result["evaluated_frames"] == 1
    assert result["ignored_unannotated_prediction_frames"] == 9
    assert result["attendance"]["f1"] == 1.0
    assert result["student_behavior"]["macro_f1"] == 1.0
    assert result["student_count_mae"] == 0.0
    assert result["events"]["available"] is False


def test_optional_bbox_detection_metrics() -> None:
    predictions = [{
        "frame_index": 1,
        "tracks": [
            {"track_id": 1, "bbox": [0, 0, 10, 10]},
            {"track_id": 2, "bbox": [20, 20, 30, 30]},
        ],
        "final_behavior": [],
    }]
    truth = [{
        "frame_index": 1,
        "students": [{"student_id": "S001", "bbox": [0, 0, 10, 10]}],
        "person_count": 1,
    }]

    result = evaluate_records(predictions, truth, bbox_iou_threshold=0.5)

    assert result["person_detection"]["available"] is True
    assert result["person_detection"]["true_positive"] == 1
    assert result["person_detection"]["false_positive"] == 1
    assert result["person_detection"]["false_negative"] == 0


def test_missing_ground_truth_state_is_not_scored_as_behavior_error() -> None:
    predictions = [{
        "frame_index": 1,
        "tracks": [{"track_id": 1}],
        "recognition": {"1": {"student_id": "S001", "matched": True}},
        "final_behavior": [{"track_id": 1, "state": "focused"}],
    }]
    truth = [{
        "frame_index": 1,
        "students": [{"student_id": "S001"}],
        "person_count": 1,
    }]

    result = evaluate_records(predictions, truth)

    assert result["attendance"]["f1"] == 1.0
    assert result["student_behavior"]["supported_classes"] == 0
    assert result["student_behavior"]["confusion_matrix"] == {}
    assert result["frame_results"][0]["behavior_errors"] == []


def test_anonymous_partial_yolo_labels_only_score_matched_behavior_boxes() -> None:
    predictions = [{
        "frame_index": 1,
        "tracks": [{"track_id": 1, "bbox": [0, 0, 20, 20]}],
        "frame_behavior": [
            {"track_id": 1, "state": "focused", "bbox": [0, 0, 10, 10]},
            {"track_id": 2, "state": "drowsy", "bbox": [50, 50, 60, 60]},
        ],
        "final_behavior": [],
    }]
    truth = [{
        "frame_index": 1,
        "annotation_type": "anonymous_bbox_behavior",
        "person_count": 1,
        "person_count_complete": False,
        "box_annotation_complete": False,
        "students": [{"state": "focused", "bbox": [0, 0, 10, 10]}],
    }]

    result = evaluate_records(predictions, truth, behavior_output="frame")

    assert result["person_detection"]["available"] is False
    assert result["bbox_behavior"]["available"] is True
    assert result["bbox_behavior"]["annotation_scope"] == "annotated_boxes_only"
    assert result["bbox_behavior"]["accuracy_on_matched_boxes"] == 1.0
    assert result["bbox_behavior"]["per_class"]["drowsy"]["precision"] == 0.0
    assert result["student_count_mae"] is None


def test_bbox_behavior_uses_final_temporal_state_in_e2e_mode() -> None:
    predictions = [{
        "frame_index": 1,
        "tracks": [{"track_id": 1, "bbox": [0, 0, 20, 20]}],
        "frame_behavior": [
            {"track_id": 1, "state": "drowsy", "bbox": [0, 0, 10, 10]}
        ],
        "final_behavior": [{"track_id": 1, "state": "focused"}],
    }]
    truth = [{
        "frame_index": 1,
        "annotation_type": "anonymous_bbox_behavior",
        "person_count": 1,
        "person_count_complete": False,
        "box_annotation_complete": False,
        "students": [{"state": "focused", "bbox": [0, 0, 10, 10]}],
    }]

    result = evaluate_records(predictions, truth, behavior_output="final")

    assert result["bbox_behavior"]["accuracy_on_matched_boxes"] == 1.0
    assert result["bbox_behavior"]["end_to_end_accuracy"] == 1.0
    assert result["settings"]["behavior_output"] == "final"
