"""Snap a heard bearing onto the opening it most likely came through."""

from __future__ import annotations

import math
from collections.abc import Sequence


def snap_bearing(robot_xy: tuple[float, float], bearing_rad: float, candidates_xy: Sequence[tuple[float, float]], max_error_rad: float) -> int | None:
    """Index of the candidate whose map-frame bearing from the robot is closest to ``bearing_rad``, None beyond ``max_error_rad``."""
    best, best_error = None, max_error_rad
    for i, (x, y) in enumerate(candidates_xy):
        error = abs((math.atan2(y - robot_xy[1], x - robot_xy[0]) - bearing_rad + math.pi) % (2.0 * math.pi) - math.pi)
        if error <= best_error:
            best, best_error = i, error
    return best
