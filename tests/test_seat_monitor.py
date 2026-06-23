from services.vision_ai.src.seat_monitor import SeatMonitor


def _track(track_id: int, bbox: list[float]) -> dict:
    return {"track_id": track_id, "bbox": bbox, "confidence": 0.9}


def _recognition(student_id: str) -> dict:
    return {"student_id": student_id, "recognized": True, "confidence": 0.95}


def _monitor() -> SeatMonitor:
    return SeatMonitor(
        config={
            "calibration_frames": 2,
            "min_calibration_samples": 2,
            "late_calibration_frames": 2,
            "leave_confirm_frames": 2,
            "return_confirm_frames": 2,
            "occlusion_frames": 1,
        }
    )


def test_initial_empty_class_does_not_create_away_assignment() -> None:
    monitor = _monitor()
    monitor.start_session()
    monitor.update([], {}, 1, 1.0)
    results = monitor.update([], {}, 2, 2.0)

    assert monitor.session_state == "active"
    assert results == []
    assert monitor.assignments == {}


def test_late_student_is_calibrated_without_teacher_confirmation() -> None:
    monitor = _monitor()
    monitor.start_session()
    monitor.update([], {}, 1, 1.0)
    monitor.update([], {}, 2, 2.0)
    student = _track(7, [10, 10, 110, 110])
    monitor.update([student], {7: _recognition("S007")}, 3, 3.0)
    results = monitor.update([student], {7: _recognition("S007")}, 4, 4.0)

    assignment = monitor.assignments["S007"]
    assert assignment.attendance_status == "late"
    assert assignment.assignment_status == "provisional"
    assert results[0].state == "seated"


def test_missing_detection_is_not_treated_as_away() -> None:
    monitor = _monitor()
    monitor.start_session()
    student = _track(1, [0, 0, 100, 100])
    monitor.update([student], {1: _recognition("S001")}, 1, 1.0)
    monitor.update([student], {1: _recognition("S001")}, 2, 2.0)

    first = monitor.update([], {}, 3, 3.0)[0]
    second = monitor.update([], {}, 4, 4.0)[0]

    assert first.state == "temporarily_occluded"
    assert second.state == "missing"


def test_away_confidence_increases_with_confirmed_outside_frames() -> None:
    monitor = _monitor()
    monitor.start_session()
    seated = _track(1, [0, 0, 100, 100])
    monitor.update([seated], {1: _recognition("S001")}, 1, 1.0)
    monitor.update([seated], {1: _recognition("S001")}, 2, 2.0)

    outside = _track(1, [200, 0, 300, 100])
    candidate = monitor.update([outside], {1: _recognition("S001")}, 3, 3.0)[0]
    confirmed = monitor.update([outside], {}, 4, 4.0)[0]

    assert candidate.state == "candidate_away"
    assert 0.0 < candidate.confidence < 1.0
    assert confirmed.state == "away_from_seat"
    assert confirmed.spatial_score == 1.0
    assert confirmed.temporal_score == 1.0
    assert confirmed.confidence == 1.0


def test_outside_track_without_face_confirmation_stays_candidate() -> None:
    monitor = _monitor()
    monitor.start_session()
    seated = _track(1, [0, 0, 100, 100])
    monitor.update([seated], {1: _recognition("S001")}, 1, 1.0)
    monitor.update([seated], {1: _recognition("S001")}, 2, 2.0)

    outside = _track(1, [200, 0, 300, 100])
    monitor.update([outside], {}, 3, 3.0)
    result = monitor.update([outside], {}, 4, 4.0)[0]

    assert result.state == "candidate_away"
    assert result.confidence == 0.0


def test_occupied_seat_blocks_away_confirmation() -> None:
    monitor = _monitor()
    monitor.start_session()
    seated = _track(1, [0, 0, 100, 100])
    monitor.update([seated], {1: _recognition("S001")}, 1, 1.0)
    monitor.update([seated], {1: _recognition("S001")}, 2, 2.0)

    outside = _track(1, [200, 0, 300, 100])
    occupant = _track(8, [0, 0, 100, 100])
    monitor.update(
        [outside, occupant],
        {1: _recognition("S001"), 8: _recognition("S008")},
        3,
        3.0,
    )
    result = monitor.update([outside, occupant], {}, 4, 4.0)[0]

    assert result.state == "candidate_away"
    assert "seat empty for 0 frame(s)" in result.reason


def test_recognized_new_track_replaces_old_binding_but_keeps_assignment() -> None:
    monitor = _monitor()
    monitor.start_session()
    seated = _track(1, [0, 0, 100, 100])
    monitor.update([seated], {1: _recognition("S001")}, 1, 1.0)
    monitor.update([seated], {1: _recognition("S001")}, 2, 2.0)
    original_seat_id = monitor.assignments["S001"].seat_id

    monitor.update([], {}, 3, 3.0)
    returned = _track(7, [0, 0, 100, 100])
    result = monitor.update([returned], {7: _recognition("S001")}, 4, 4.0)[0]

    assert result.track_id == 7
    assert result.seat_id == original_seat_id
    assert monitor._track_to_student == {7: "S001"}
