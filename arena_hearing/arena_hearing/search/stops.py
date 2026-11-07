"""Stop rules: whether the search halts to listen on this tick."""

from __future__ import annotations

import math
from typing import Protocol

import attrs
import numpy as np

from arena_hearing.corners import MapFeatures, find_blind_bend
from arena_hearing.search.frontiers import MapGrid


@attrs.frozen
class StopView:
    robot_xy: tuple[float, float]
    goal_changed: bool
    calls: int  # calls heard since the last stop ended
    plan_xy: np.ndarray | None
    grid: MapGrid
    heard_positions: int = 0


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


SERVED_MATCH_M = 0.5
PATH_STEP_M = 0.1
DOOR_RANGE_M = 8.0


def arclength(path: np.ndarray) -> np.ndarray:
    if len(path) == 0:
        return np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])


def resample(path: np.ndarray, step_m: float) -> np.ndarray:
    if len(path) < 2:
        return path
    s = arclength(path)
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    path, s = path[keep], s[keep]
    if len(path) < 2:
        return path
    grid = np.linspace(0.0, s[-1], max(math.ceil(s[-1] / step_m), 1) + 1)
    return np.stack([np.interp(grid, s, path[:, 0]), np.interp(grid, s, path[:, 1])], axis=1)


def path_ahead(plan_xy: np.ndarray, robot_xy: tuple[float, float], horizon_m: float) -> np.ndarray:
    """The plan from the vertex nearest the robot onward, cut at horizon_m and resampled."""
    start = int(np.argmin(np.hypot(plan_xy[:, 0] - robot_xy[0], plan_xy[:, 1] - robot_xy[1])))
    dense = resample(np.concatenate([[robot_xy], plan_xy[start:]], axis=0), PATH_STEP_M)
    return dense[arclength(dense) <= horizon_m + 1e-9]


def features_ahead(features_xy: np.ndarray, path: np.ndarray, lateral_m: float) -> list[tuple[int, float]]:
    """Index and along-path distance of every feature within lateral_m of the path and strictly ahead of its start."""
    if len(features_xy) == 0 or len(path) < 2:
        return []
    s = arclength(path)
    d = np.hypot(features_xy[:, None, 0] - path[None, :, 0], features_xy[:, None, 1] - path[None, :, 1])
    nearest = d.argmin(axis=1)
    return [(i, float(s[j])) for i, j in enumerate(nearest) if j > 0 and d[i, j] <= lateral_m]


def map_features(view: StopView, range_m: float, door_max_width_m: float) -> MapFeatures:
    grid = view.grid
    return MapFeatures.at(grid.occupied, grid.origin, grid.resolution, view.robot_xy, range_m=range_m, door_max_width_m=door_max_width_m)


def _served(served: list[tuple[float, float]], xy: np.ndarray) -> bool:
    return any(math.dist((float(xy[0]), float(xy[1])), p) <= SERVED_MATCH_M for p in served)


@attrs.define(kw_only=True)
class FixedStop:
    """Stops once per corner or door of the live map within ``lateral_m`` of the next ``path_horizon_m`` of the plan, ``trigger_m`` of path before it, again only after the robot was more than ``rearm_m`` away from it."""

    trigger_m: float
    lateral_m: float
    path_horizon_m: float
    rearm_m: float
    door_max_width_m: float
    served: list[tuple[float, float]] = attrs.Factory(list)

    def should_stop(self, view: StopView) -> bool:
        self.served = [p for p in self.served if math.dist(p, view.robot_xy) <= self.rearm_m]
        if view.plan_xy is None or len(view.plan_xy) < 2:
            return False
        features = map_features(view, self.path_horizon_m, self.door_max_width_m)
        points = np.concatenate([features.corners_xy, features.doors_xy], axis=0)
        path = path_ahead(view.plan_xy, view.robot_xy, self.path_horizon_m)
        due = [(along, index) for index, along in features_ahead(points, path, self.lateral_m) if along <= self.trigger_m and not _served(self.served, points[index])]
        if not due:
            return False
        chosen = points[min(due)[1]]
        self.served.append((float(chosen[0]), float(chosen[1])))
        return True

    def reset(self) -> None:
        self.served.clear()


@attrs.define(kw_only=True)
class RoomEntryStop:
    """Stops once per door of the live map the robot comes within ``radius_m`` of, again only after the robot was more than ``rearm_m`` away from it."""

    radius_m: float
    rearm_m: float
    door_max_width_m: float
    served: list[tuple[float, float]] = attrs.Factory(list)

    def should_stop(self, view: StopView) -> bool:
        self.served = [p for p in self.served if math.dist(p, view.robot_xy) <= self.rearm_m]
        for door in map_features(view, DOOR_RANGE_M, self.door_max_width_m).doors_xy:
            if math.dist((float(door[0]), float(door[1])), view.robot_xy) > self.radius_m or _served(self.served, door):
                continue
            self.served.append((float(door[0]), float(door[1])))
            return True
        return False

    def reset(self) -> None:
        self.served.clear()


@attrs.define(kw_only=True)
class ThreePositionsStop:
    """Stops at each new goal until calls were heard from ``count`` distinct positions."""

    count: int

    def should_stop(self, view: StopView) -> bool:
        return view.goal_changed and view.heard_positions < self.count

    def reset(self) -> None:
        pass
