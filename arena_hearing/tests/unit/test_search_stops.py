from __future__ import annotations

import numpy as np

from arena_hearing.search.frontiers import FREE, OCCUPIED, MapGrid
from arena_hearing.search.stops import EveryGoalStop, FixedBendStop, NeverStop, OnCallStop, StopView

RESOLUTION = 0.1


def _l_corridor() -> MapGrid:
    xs = (np.arange(200) + 0.5) * RESOLUTION
    xx, yy = np.meshgrid(xs, xs)
    data = np.full((200, 200), OCCUPIED, dtype=np.int16)
    data[((xx >= 0.5) & (xx <= 10.0) & (yy >= 0.4) & (yy <= 1.6)) | ((yy >= 1.0) & (yy <= 10.0) & (xx >= 9.4) & (xx <= 10.6))] = FREE
    return MapGrid(data=data, origin=(0.0, 0.0), resolution=RESOLUTION, occupied_threshold=50)


def _l_plan(start_x: float) -> np.ndarray:
    xs = np.arange(start_x, 10.0 + 1e-9, 0.1)
    ys = np.arange(1.1, 9.5 + 1e-9, 0.1)
    return np.concatenate([np.stack([xs, np.full_like(xs, 1.0)], axis=1), np.stack([np.full_like(ys, 10.0), ys], axis=1)])


def _view(robot_x: float = 7.0, goal_changed: bool = False, calls: int = 0) -> StopView:
    return StopView(robot_xy=(robot_x, 1.0), goal_changed=goal_changed, calls=calls, plan_xy=_l_plan(robot_x), grid=_l_corridor())


def _fixed() -> FixedBendStop:
    return FixedBendStop(lookahead_m=4.0, stop_before_m=4.0, hysteresis_m=1.0)


def test_never_rule_never_stops() -> None:
    rule = NeverStop()
    assert not any(rule.should_stop(view) for view in (_view(), _view(goal_changed=True), _view(calls=3)))


def test_every_goal_rule_stops_on_a_new_goal_only() -> None:
    rule = EveryGoalStop()
    assert rule.should_stop(_view(goal_changed=True))
    assert not rule.should_stop(_view())


def test_on_call_rule_stops_after_a_heard_call() -> None:
    rule = OnCallStop()
    assert rule.should_stop(_view(calls=1))
    assert not rule.should_stop(_view())


def test_fixed_stop_fires_before_a_blind_bend() -> None:
    assert _fixed().should_stop(_view(robot_x=7.0))


def test_fixed_stop_waits_while_the_bend_is_far() -> None:
    assert not _fixed().should_stop(_view(robot_x=1.0))


def test_fixed_stop_fires_once_per_bend() -> None:
    rule = _fixed()
    assert rule.should_stop(_view(robot_x=7.0))
    assert not rule.should_stop(_view(robot_x=7.5))


def test_fixed_stop_fires_again_after_a_reset() -> None:
    rule = _fixed()
    rule.should_stop(_view(robot_x=7.0))
    rule.reset()
    assert rule.should_stop(_view(robot_x=7.5))
