"""Search state machine: select a goal, drive to it, halt to listen when the stop rule fires, reselect on a new call or when stuck."""

from __future__ import annotations

import enum
import math

import attrs
import numpy as np

from arena_hearing.search.frontiers import MapGrid, away_from, find_frontiers, find_openings
from arena_hearing.search.goals import BeliefMap, GoalRule, GoalView, Heard
from arena_hearing.search.stops import StopRule, StopView


class Phase(enum.StrEnum):
    SELECT = "select"
    DRIVE = "drive"
    LISTEN = "listen"
    DONE = "done"  # no frontier on the latest map, re-checked every tick


class NavStatus(enum.StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@attrs.frozen
class Send:
    xy: tuple[float, float]


@attrs.frozen
class Cancel:
    pass


@attrs.frozen
class Tick:
    now: float
    robot_xy: tuple[float, float]
    grid: MapGrid | None
    belief: BeliefMap | None
    heard: Heard | None
    calls: int  # calls heard since the driver started, cumulative
    plan_xy: np.ndarray | None
    status: NavStatus  # of the last goal sent
    limited: bool = False


@attrs.frozen(kw_only=True)
class DriverConfig:
    listen_s: float
    min_frontier_cells: int
    reached_radius_m: float
    opening_min_width_m: float
    opening_max_width_m: float
    opening_depth_m: float
    replan_on_call: bool
    stuck_s: float  # 0 = never give a goal up for lack of progress


@attrs.define
class SearchDriver:
    goal_rule: GoalRule
    stop_rule: StopRule
    config: DriverConfig
    phase: Phase = Phase.SELECT
    goal: tuple[float, float] | None = None
    fresh_goal: bool = False
    blocked: list[tuple[float, float]] = attrs.Factory(list)
    calls_mark: int = 0
    replan_mark: int = 0
    progress_xy: tuple[float, float] = (math.nan, math.nan)
    progress_t: float = 0.0
    listen_start: float = 0.0
    stops: int = 0
    listened_s: float = 0.0
    goals_sent: int = 0
    failed: int = 0

    def reset(self) -> None:
        self.phase, self.goal, self.fresh_goal = Phase.SELECT, None, False
        self.blocked.clear()
        self.calls_mark = self.replan_mark = self.stops = self.goals_sent = self.failed = 0
        self.progress_xy, self.progress_t = (math.nan, math.nan), 0.0
        self.listened_s = 0.0
        self.stop_rule.reset()

    def step(self, tick: Tick) -> Send | Cancel | None:
        if tick.grid is None:
            return None
        if self.phase is Phase.LISTEN:
            if tick.now - self.listen_start < self.config.listen_s:
                return None
            self.listened_s += tick.now - self.listen_start
            self.calls_mark = tick.calls
            self.phase = Phase.SELECT
        if self.phase is Phase.DRIVE and self.goal is not None:
            reached = tick.status is NavStatus.SUCCEEDED or math.dist(tick.robot_xy, self.goal) <= self.config.reached_radius_m
            failed = tick.status is NavStatus.FAILED or self._stuck(tick)
            if reached or failed:
                self.failed += failed and not reached
                self.blocked.append(self.goal)
                self.phase = Phase.SELECT
            else:
                view = StopView(robot_xy=tick.robot_xy, goal_changed=self.fresh_goal, calls=tick.calls - self.calls_mark, plan_xy=tick.plan_xy, grid=tick.grid)
                self.fresh_goal = False
                if self.stop_rule.should_stop(view):
                    self.phase, self.listen_start = Phase.LISTEN, tick.now
                    self.stops += 1
                    return Cancel()
                if self.config.replan_on_call and tick.heard is not None and tick.calls > self.replan_mark:
                    self.replan_mark = tick.calls
                    goal = self._choose(tick, tick.grid)
                    if goal is not None and math.dist(goal, self.goal) > self.config.reached_radius_m:
                        return self._send(tick, goal)
                if tick.status is NavStatus.IDLE:
                    return Send(self.goal)
                return None
        return self._select(tick, tick.grid)

    def _stuck(self, tick: Tick) -> bool:
        """True once the robot has stayed within reached_radius_m of one spot for stuck_s while driving under no speed limit."""
        if tick.limited or not math.dist(tick.robot_xy, self.progress_xy) <= self.config.reached_radius_m:
            self.progress_xy, self.progress_t = tick.robot_xy, tick.now
            return False
        return self.config.stuck_s > 0.0 and tick.now - self.progress_t >= self.config.stuck_s

    def _choose(self, tick: Tick, grid: MapGrid) -> tuple[float, float] | None:
        c = self.config
        unblocked = [f for f in find_frontiers(grid, c.min_frontier_cells) if not self._is_blocked(f.goal_xy)]
        frontiers = away_from(unblocked, tick.robot_xy, c.reached_radius_m, c.min_frontier_cells)
        openings = [o for o in find_openings(grid, c.opening_min_width_m, c.opening_max_width_m, c.opening_depth_m) if not self._is_blocked(o.xy)]
        return self.goal_rule.choose(GoalView(robot_xy=tick.robot_xy, grid=grid, frontiers=frontiers, openings=openings, belief=tick.belief, heard=tick.heard, blocked=tuple(self.blocked), blocked_radius_m=c.reached_radius_m))

    def _select(self, tick: Tick, grid: MapGrid) -> Send | None:
        goal = self._choose(tick, grid)
        if goal is None:
            self.phase = Phase.DONE
            return None
        return self._send(tick, goal)

    def _send(self, tick: Tick, goal: tuple[float, float]) -> Send:
        self.fresh_goal = self.goal is None or math.dist(goal, self.goal) > self.config.reached_radius_m
        self.goal, self.phase = goal, Phase.DRIVE
        self.goals_sent += 1
        self.replan_mark = tick.calls
        self.progress_xy, self.progress_t = tick.robot_xy, tick.now
        return Send(goal)

    def _is_blocked(self, xy: tuple[float, float]) -> bool:
        return any(math.dist(xy, b) <= self.config.reached_radius_m for b in self.blocked)

    def state(self, now: float) -> dict[str, object]:
        """Counters for scoring, published as JSON."""
        listening = now - self.listen_start if self.phase is Phase.LISTEN else 0.0
        return {
            "phase": self.phase.value,
            "goal": None if self.goal is None else [round(self.goal[0], 2), round(self.goal[1], 2)],
            "stops": self.stops,
            "listened_s": round(self.listened_s + listening, 3),
            "goals_sent": self.goals_sent,
            "goals_failed": self.failed,
        }
