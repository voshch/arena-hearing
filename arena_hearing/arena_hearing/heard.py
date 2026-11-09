"""Distinct robot positions calls were heard from."""

from __future__ import annotations

import math


class PositionTracker:
    """A position counts once per cluster of listening points less than separation_m apart."""

    def __init__(self, separation_m: float) -> None:
        self.separation_m = separation_m
        self._positions: list[tuple[float, float]] = []

    @property
    def count(self) -> int:
        return len(self._positions)

    def reset(self) -> None:
        self._positions.clear()

    def hear(self, robot_xy: tuple[float, float]) -> None:
        if all(math.dist(robot_xy, seen) > self.separation_m for seen in self._positions):
            self._positions.append(robot_xy)
