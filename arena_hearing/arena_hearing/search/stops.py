"""Stop rules: whether the search halts to listen on this tick."""

from __future__ import annotations

import math
from typing import Protocol

import attrs
import numpy as np

from arena_hearing.corners import find_blind_bend
from arena_hearing.search.frontiers import MapGrid


@attrs.frozen
class StopView:
    robot_xy: tuple[float, float]
    goal_changed: bool
    calls: int  # calls heard since the last stop ended
    plan_xy: np.ndarray | None
    grid: MapGrid


class StopRule(Protocol):
    def should_stop(self, view: StopView) -> bool: ...

    def reset(self) -> None:
        """Forget per-episode state."""
        ...


class NeverStop:
    def should_stop(self, view: StopView) -> bool:
        return False

    def reset(self) -> None:
        pass


class EveryGoalStop:
    def should_stop(self, view: StopView) -> bool:
        return view.goal_changed

    def reset(self) -> None:
        pass


class OnCallStop:
    def should_stop(self, view: StopView) -> bool:
        return view.calls > 0

    def reset(self) -> None:
        pass


@attrs.define(kw_only=True)
class FixedBendStop:
    """Stops once ``stop_before_m`` along the plan before each blind bend, a bend within ``hysteresis_m`` of a stopped-at one excluded."""

    lookahead_m: float
    stop_before_m: float
    hysteresis_m: float
    stopped_at: list[tuple[float, float]] = attrs.Factory(list)

    def should_stop(self, view: StopView) -> bool:
        if view.plan_xy is None:
            return False
        grid = view.grid
        bend = find_blind_bend(view.plan_xy, grid.occupied, grid.origin, grid.resolution, view.robot_xy, lookahead_m=self.lookahead_m, approach_m=self.stop_before_m, hold_len_m=0.0, hold_offset_m=0.0)
        if bend is None or any(math.dist(bend.bend_xy, xy) <= self.hysteresis_m for xy in self.stopped_at):
            return False
        if not 0.0 < bend.dist_from(view.plan_xy, view.robot_xy) <= self.stop_before_m:
            return False
        self.stopped_at.append(bend.bend_xy)
        return True

    def reset(self) -> None:
        self.stopped_at.clear()
