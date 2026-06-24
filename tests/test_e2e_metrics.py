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
    assert result["events"]["f1"] == 0.0


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
