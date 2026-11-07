from __future__ import annotations

import numpy as np

from arena_hearing.search.frontiers import FREE, OCCUPIED, MapGrid
from arena_hearing.search.stops import EveryGoalStop, FixedBendStop, FixedStop, NeverStop, OnCallStop, RoomEntryStop, StopView, ThreePositionsStop

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


def _room() -> MapGrid:
    blocked = np.zeros((120, 200), dtype=bool)
    blocked[0, :] = blocked[-1, :] = True
    blocked[:, 0] = blocked[:, -1] = True
    blocked[:, 99:101] = True
    blocked[48:72, 99:101] = False
    return MapGrid(data=np.where(blocked, OCCUPIED, FREE).astype(np.int16), origin=(0.0, 0.0), resolution=0.05, occupied_threshold=50)


def _room_view(robot_x: float, with_plan: bool = True, **fields) -> StopView:
    xs = np.arange(robot_x, 9.0 + 1e-9, 0.1)
    plan = np.stack([xs, np.full_like(xs, 3.0)], axis=1) if with_plan else None
    return StopView(robot_xy=(robot_x, 3.0), goal_changed=False, calls=0, plan_xy=plan, grid=_room(), **fields)


def _door() -> FixedStop:
    return FixedStop(trigger_m=2.5, lateral_m=1.5, path_horizon_m=8.0, rearm_m=3.0, door_max_width_m=1.5)


def test_fixed_stop_fires_before_a_door_on_the_plan() -> None:
    assert _door().should_stop(_room_view(3.0))


def test_fixed_stop_waits_while_the_door_is_beyond_the_trigger_distance() -> None:
    assert not _door().should_stop(_room_view(1.0))


def test_fixed_stop_without_a_plan_never_fires() -> None:
    assert not _door().should_stop(_room_view(3.0, with_plan=False))


def test_fixed_stop_fires_once_per_door_until_the_robot_moved_away() -> None:
    rule = _door()
    assert rule.should_stop(_room_view(3.0))
    assert not rule.should_stop(_room_view(3.2))
    assert not rule.should_stop(_room_view(0.6))
    assert rule.should_stop(_room_view(3.0))


def test_fixed_stop_fires_again_after_a_reset() -> None:
    rule = _door()
    rule.should_stop(_room_view(3.0))
    rule.reset()
    assert rule.should_stop(_room_view(3.2))


def _entry() -> RoomEntryStop:
    return RoomEntryStop(radius_m=0.6, rearm_m=3.0, door_max_width_m=1.5)


def test_room_entry_stop_fires_at_a_door_and_not_before() -> None:
    assert not _entry().should_stop(_room_view(3.0))
    assert _entry().should_stop(_room_view(4.8))


def test_room_entry_stop_fires_once_per_door_until_the_robot_moved_away() -> None:
    rule = _entry()
    assert rule.should_stop(_room_view(4.8))
    assert not rule.should_stop(_room_view(4.7))
    assert not rule.should_stop(_room_view(1.0))
    assert rule.should_stop(_room_view(4.8))


def test_three_positions_stop_fires_on_a_new_goal_until_enough_positions_heard() -> None:
    rule = ThreePositionsStop(count=3)
    assert rule.should_stop(_view(goal_changed=True))
    assert rule.should_stop(StopView(robot_xy=(7.0, 1.0), goal_changed=True, calls=0, plan_xy=None, grid=_l_corridor(), heard_positions=2))
    assert not rule.should_stop(StopView(robot_xy=(7.0, 1.0), goal_changed=True, calls=0, plan_xy=None, grid=_l_corridor(), heard_positions=3))
    assert not rule.should_stop(StopView(robot_xy=(7.0, 1.0), goal_changed=False, calls=0, plan_xy=None, grid=_l_corridor(), heard_positions=0))
