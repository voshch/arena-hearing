from __future__ import annotations

import math

import numpy as np

from arena_hearing.corners import MapFeatures, door_pairs, visibility_gaps

RESOLUTION = 0.05
ORIGIN = (0.0, 0.0)


def _room() -> np.ndarray:
    """A 10 m by 6 m room as a blocked array, a wall at x = 5 m with a 1.2 m door at y in [2.4, 3.6]."""
    blocked = np.zeros((120, 200), dtype=bool)
    blocked[0, :] = blocked[-1, :] = True
    blocked[:, 0] = blocked[:, -1] = True
    blocked[:, 99:101] = True
    blocked[48:72, 99:101] = False
    return blocked


def _corridor_with_branch() -> np.ndarray:
    """A corridor x in [0.5, 9.5], y in [0.5, 2.0] with a side branch x in [5.0, 7.0], y in [2.0, 8.0]."""
    centers = (np.arange(200) + 0.5) * RESOLUTION
    xx, yy = np.meshgrid(centers, centers)
    corridor = (xx >= 0.5) & (xx <= 9.5) & (yy >= 0.5) & (yy <= 2.0)
    branch = (xx >= 5.0) & (xx <= 7.0) & (yy >= 2.0) & (yy <= 8.0)
    return ~(corridor | branch)


def test_visibility_gaps_put_a_corner_on_each_side_of_the_door() -> None:
    gaps = visibility_gaps(_room(), ORIGIN, RESOLUTION, (3.0, 1.0))

    corners = [gap.corner_xy for gap in gaps]
    assert any(abs(x - 5.0) < 0.2 and abs(y - 2.4) < 0.2 for x, y in corners)
    assert any(abs(x - 5.0) < 0.2 and abs(y - 3.6) < 0.2 for x, y in corners)
    assert all(gap.jump_m > 1.0 for gap in gaps)


def test_door_pairs_join_the_two_corners_that_flank_an_aperture() -> None:
    gaps = visibility_gaps(_room(), ORIGIN, RESOLUTION, (3.0, 1.0))

    pairs = door_pairs(gaps, 1.5)

    assert len(pairs) == 1
    first, second = (gaps[k] for k in pairs[0])
    assert first.far_ccw and not second.far_ccw
    assert 0.9 <= math.dist(first.corner_xy, second.corner_xy) <= 1.5


def test_a_door_wider_than_the_limit_is_two_corners() -> None:
    features = MapFeatures.at(_room(), ORIGIN, RESOLUTION, (3.0, 1.0), range_m=12.0, door_max_width_m=1.0)

    assert len(features.doors_xy) == 0
    assert len(features.corners_xy) >= 2


def test_features_report_the_door_center_and_no_corner_for_its_frame() -> None:
    features = MapFeatures.at(_room(), ORIGIN, RESOLUTION, (3.0, 1.0), range_m=12.0, door_max_width_m=1.5)

    assert len(features.doors_xy) == 1
    assert abs(features.doors_xy[0][0] - 5.0) < 0.2
    assert abs(features.doors_xy[0][1] - 3.0) < 0.2
    assert not any(abs(x - 5.0) < 0.2 and 2.2 < y < 3.8 for x, y in features.corners_xy)


def test_a_side_branch_gives_one_occluding_corner_at_its_near_edge() -> None:
    gaps = visibility_gaps(_corridor_with_branch(), ORIGIN, RESOLUTION, (3.5, 1.25))

    assert len(gaps) == 1
    assert abs(gaps[0].corner_xy[0] - 5.0) < 0.2
    assert abs(gaps[0].corner_xy[1] - 2.0) < 0.2
