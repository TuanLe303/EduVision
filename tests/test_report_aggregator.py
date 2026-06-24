from services.report_generator.src.aggregator import aggregate_frames


def test_aggregates_public_final_behavior_including_sleeping() -> None:
    summary = aggregate_frames(
        [
            {
                "timestamp": 0.0,
                "behavior": [{"track_id": 1, "state": "focused"}],
                "final_behavior": [{"track_id": 1, "state": "sleeping"}],
            },
            {
                "timestamp": 0.04,
                "behavior": [{"track_id": 1, "state": "focused"}],
                "final_behavior": [{"track_id": 1, "state": "sleeping"}],
            },
        ],
        min_present_frames=1,
    )

    assert summary.students[0].dominant_behavior == "sleeping"
    assert summary.students[0].behavior_counts == {"sleeping": 2}
    assert summary.class_stats.duration_seconds == 0.04


def test_ignores_unready_temporal_results_from_legacy_output() -> None:
    summary = aggregate_frames(
        [{"timestamp": 0.0, "behavior": [{"track_id": 1, "state": None}]}],
        min_present_frames=1,
    )

    assert summary.students == []


def test_raising_hand_is_counted_as_positive_attention() -> None:
    summary = aggregate_frames(
        [{"timestamp": 0.0, "final_behavior": [{"track_id": 1, "state": "raising_hand"}]}],
        min_present_frames=1,
    )

    assert summary.students[0].dominant_behavior == "raising_hand"
    assert summary.students[0].attention_score == 1.0
