from __future__ import annotations

import math

from arena_hearing.search.snap import snap_bearing

ROBOT = (1.0, 1.0)
DOORS = [(4.0, 1.0), (3.0, 3.0), (1.0, 5.0)]


def test_snap_picks_the_door_nearest_the_heard_direction() -> None:
    assert snap_bearing(ROBOT, math.radians(50.0), DOORS, max_error_rad=math.radians(20.0)) == 1


def test_snap_leaves_a_bearing_far_from_every_door() -> None:
    assert snap_bearing(ROBOT, math.radians(180.0), DOORS, max_error_rad=math.radians(20.0)) is None


def test_snap_wraps_the_bearing_around_pi() -> None:
    doors = [(0.0, 1.01)]
    assert snap_bearing(ROBOT, math.radians(-179.0), doors, max_error_rad=math.radians(5.0)) == 0
