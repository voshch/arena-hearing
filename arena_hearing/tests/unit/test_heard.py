from arena_hearing.heard import PositionTracker


def test_a_position_counts_once_per_cluster_of_listening_points() -> None:
    tracker = PositionTracker(1.0)
    tracker.hear((0.0, 0.0))
    tracker.hear((0.5, 0.0))
    tracker.hear((3.0, 0.0))
    tracker.hear((3.4, 0.2))
    assert tracker.count == 2


def test_a_point_at_the_separation_is_the_same_position() -> None:
    tracker = PositionTracker(1.0)
    tracker.hear((0.0, 0.0))
    tracker.hear((1.0, 0.0))
    assert tracker.count == 1


def test_reset_forgets_the_positions() -> None:
    tracker = PositionTracker(1.0)
    tracker.hear((0.0, 0.0))
    tracker.reset()
    assert tracker.count == 0
