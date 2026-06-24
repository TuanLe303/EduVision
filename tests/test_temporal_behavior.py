from services.vision_ai.src.behavior_detection import TemporalBehaviorAggregator


def test_normalizes_existing_dataset_typos() -> None:
    aggregator = TemporalBehaviorAggregator(window_size=4, min_history=2)

    assert aggregator.normalize_state("focus") == "focused"
    assert aggregator.normalize_state("side_taliking") == "side_talking"


def test_does_not_publish_state_before_temporal_window_is_ready() -> None:
    aggregator = TemporalBehaviorAggregator(
        window_size=4,
        min_history=4,
        min_state_frames=3,
        enter_threshold=0.5,
    )

    for frame_index in range(1, 4):
        result = aggregator.update(1, "focused", 0.9, frame_index)

    assert result.ready is False
    assert result.state is None

    result = aggregator.update(1, "focused", 0.9, 4)
    assert result.ready is True
    assert result.state == "focused"


def test_history_is_independent_per_track() -> None:
    aggregator = TemporalBehaviorAggregator(
        window_size=4,
        min_history=2,
        min_state_frames=2,
        enter_threshold=0.5,
        switch_margin=0.0,
    )

    aggregator.update(1, "focused", 0.9, 1)
    aggregator.update(2, "sleeping", 0.9, 1)
    focused = aggregator.update(1, "focused", 0.9, 2)
    sleeping = aggregator.update(2, "sleeping", 0.9, 2)

    assert focused.state == "focused"
    assert sleeping.state == "sleeping"
    assert focused.frame_count == sleeping.frame_count == 2


def test_hysteresis_prevents_single_frame_state_flip() -> None:
    aggregator = TemporalBehaviorAggregator(
        window_size=5,
        min_history=3,
        min_state_frames=2,
        enter_threshold=0.5,
        switch_margin=0.1,
    )

    for frame_index in range(1, 4):
        result = aggregator.update(7, "focused", 0.9, frame_index)
    assert result.state == "focused"

    result = aggregator.update(7, "sleeping", 0.99, 4)
    assert result.state == "focused"
    assert result.changed is False


def test_state_changes_after_sustained_evidence() -> None:
    aggregator = TemporalBehaviorAggregator(
        window_size=4,
        min_history=2,
        min_state_frames=2,
        enter_threshold=0.45,
        switch_margin=0.0,
        state_thresholds={"sleeping": 0.45},
    )

    aggregator.update(3, "focused", 0.8, 1)
    aggregator.update(3, "focused", 0.8, 2)
    aggregator.update(3, "sleeping", 0.95, 3)
    result = aggregator.update(3, "sleeping", 0.95, 4)

    assert result.state == "sleeping"
    assert result.changed is True


def test_retains_ready_state_with_decay_during_short_detection_gap() -> None:
    aggregator = TemporalBehaviorAggregator(
        window_size=2,
        min_history=2,
        min_state_frames=2,
        max_detection_gap=2,
    )
    aggregator.update(9, "focused", 0.9, 1)
    observed = aggregator.update(9, "focused", 0.9, 2)

    retained = aggregator.hold(9, 3)

    assert retained is not None
    assert retained.state == "focused"
    assert retained.observed is False
    assert retained.detection_age == 1
    assert 0.0 < retained.confidence < observed.confidence
    assert aggregator.hold(9, 5) is None
