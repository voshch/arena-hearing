from __future__ import annotations

import math

import numpy as np

from arena_hearing.search.driver import Cancel, DriverConfig, NavStatus, Phase, SearchDriver, Send, Tick
from arena_hearing.search.frontiers import FREE, UNKNOWN, MapGrid
from arena_hearing.search.goals import BeliefGoal, BeliefMap, GoalRule, Heard, NearestGoal
from arena_hearing.search.stops import EveryGoalStop, NeverStop, StopRule

RESOLUTION = 0.1
ROBOT = (2.0, 2.0)
LISTEN_S = 3.0
STUCK_S = 15.0


def _grid(left_unknown: bool = True, right_unknown: bool = True) -> MapGrid:
    data = np.full((40, 60), FREE, dtype=np.int16)
    if left_unknown:
        data[:, :5] = UNKNOWN
    if right_unknown:
        data[:, 55:] = UNKNOWN
    return MapGrid(data=data, origin=(0.0, 0.0), resolution=RESOLUTION, occupied_threshold=50)


def _driver(stop: StopRule | None = None, goal: GoalRule | None = None, replan_on_call: bool = True, stuck_s: float = STUCK_S) -> SearchDriver:
    config = DriverConfig(listen_s=LISTEN_S, min_frontier_cells=5, reached_radius_m=0.5, opening_min_width_m=0.6, opening_max_width_m=1.6, opening_depth_m=0.6, replan_on_call=replan_on_call, stuck_s=stuck_s)
    return SearchDriver(goal or NearestGoal(), stop or NeverStop(), config)


def _snap() -> BeliefGoal:
    return BeliefGoal(radius_m=1.5, min_mass=0.5, locate_threshold=2.0, distance_scale_m=5.0, snap=True, snap_max_error_rad=math.radians(20.0), standoff_m=1.0)


def _tick(now: float, status: NavStatus = NavStatus.ACTIVE, grid: MapGrid | None = None, robot_xy: tuple[float, float] = ROBOT, calls: int = 0, heard: Heard | None = None, belief: BeliefMap | None = None, limited: bool = False) -> Tick:
    return Tick(now=now, robot_xy=robot_xy, grid=grid or _grid(), belief=belief, heard=heard, calls=calls, plan_xy=None, status=status, limited=limited)


EAST = Heard(robot_xy=ROBOT, bearing_rad=0.0)


def test_the_driver_sends_the_goal_its_rule_chooses() -> None:
    action = _driver().step(_tick(0.0, NavStatus.IDLE))
    assert isinstance(action, Send)
    assert action.xy[0] < 1.0


def test_a_stop_cancels_the_goal_and_listens_for_listen_s() -> None:
    driver = _driver(stop=EveryGoalStop())
    assert isinstance(driver.step(_tick(0.0, NavStatus.IDLE)), Send)
    assert isinstance(driver.step(_tick(0.1)), Cancel)
    assert driver.phase is Phase.LISTEN
    assert driver.step(_tick(0.1 + LISTEN_S - 0.05, NavStatus.IDLE)) is None
    assert isinstance(driver.step(_tick(0.1 + LISTEN_S, NavStatus.IDLE)), Send)
    assert driver.step(_tick(0.2 + LISTEN_S)) is None
    assert driver.phase is Phase.DRIVE


def test_a_failed_goal_is_not_chosen_again() -> None:
    driver = _driver()
    first = driver.step(_tick(0.0, NavStatus.IDLE))
    second = driver.step(_tick(1.0, NavStatus.FAILED))
    assert isinstance(first, Send)
    assert isinstance(second, Send)
    assert second.xy[0] > 5.0


def test_a_reached_goal_selects_the_next_frontier() -> None:
    driver = _driver()
    first = driver.step(_tick(0.0, NavStatus.IDLE))
    assert isinstance(first, Send)
    second = driver.step(_tick(1.0, NavStatus.SUCCEEDED, robot_xy=first.xy))
    assert isinstance(second, Send)
    assert second.xy[0] > 5.0


def test_a_goal_lost_while_driving_is_sent_again() -> None:
    driver = _driver()
    first = driver.step(_tick(0.0, NavStatus.IDLE))
    again = driver.step(_tick(1.0, NavStatus.IDLE))
    assert isinstance(first, Send)
    assert isinstance(again, Send)
    assert again.xy == first.xy


def test_the_driver_is_done_when_no_frontier_is_left() -> None:
    driver = _driver()
    assert driver.step(_tick(0.0, NavStatus.IDLE, grid=_grid(left_unknown=False, right_unknown=False))) is None
    assert driver.phase is Phase.DONE


def test_the_driver_resumes_once_a_frontier_appears() -> None:
    driver = _driver()
    empty = MapGrid(data=np.zeros((0, 0), dtype=np.int16), origin=(0.0, 0.0), resolution=RESOLUTION, occupied_threshold=50)
    assert driver.step(_tick(0.0, NavStatus.IDLE, grid=empty)) is None
    assert driver.phase is Phase.DONE
    assert isinstance(driver.step(_tick(0.5, NavStatus.IDLE)), Send)
    assert driver.phase is Phase.DRIVE


def test_the_state_counts_stops_and_listening_time() -> None:
    driver = _driver(stop=EveryGoalStop())
    driver.step(_tick(0.0, NavStatus.IDLE))
    driver.step(_tick(0.1))
    driver.step(_tick(0.1 + LISTEN_S, NavStatus.IDLE))
    state = driver.state(0.1 + LISTEN_S)
    assert state["stops"] == 1
    assert abs(state["listened_s"] - LISTEN_S) < 1e-6


def test_a_reset_forgets_failed_goals_and_stops() -> None:
    driver = _driver(stop=EveryGoalStop())
    driver.step(_tick(0.0, NavStatus.IDLE))
    driver.step(_tick(0.1))
    driver.reset()
    assert driver.state(0.2)["stops"] == 0
    action = driver.step(_tick(0.2, NavStatus.IDLE))
    assert isinstance(action, Send)
    assert action.xy[0] < 1.0


def test_a_new_call_while_driving_sends_the_goal_it_points_at() -> None:
    driver = _driver(goal=_snap())
    first = driver.step(_tick(0.0, NavStatus.IDLE))
    assert isinstance(first, Send)
    assert first.xy[0] < 1.0
    turn = driver.step(_tick(0.5, calls=1, heard=EAST))
    assert isinstance(turn, Send)
    assert turn.xy[0] > 5.0
    assert driver.state(0.5)["goals_sent"] == 2


def test_a_call_that_points_at_the_current_goal_sends_nothing() -> None:
    driver = _driver(goal=_snap())
    first = driver.step(_tick(0.0, NavStatus.IDLE, calls=1, heard=EAST))
    assert isinstance(first, Send)
    assert first.xy[0] > 5.0
    assert driver.step(_tick(0.5, calls=2, heard=EAST)) is None


def test_without_replan_on_call_the_driver_keeps_its_goal() -> None:
    driver = _driver(goal=_snap(), replan_on_call=False)
    first = driver.step(_tick(0.0, NavStatus.IDLE))
    assert isinstance(first, Send)
    assert driver.step(_tick(0.5, calls=1, heard=EAST)) is None


def test_a_robot_that_does_not_move_for_stuck_s_gives_its_goal_up() -> None:
    driver = _driver()
    first = driver.step(_tick(0.0, NavStatus.IDLE))
    assert isinstance(first, Send)
    assert driver.step(_tick(STUCK_S - 0.1)) is None
    other = driver.step(_tick(STUCK_S))
    assert isinstance(other, Send)
    assert other.xy[0] > 5.0
    assert driver.state(STUCK_S)["goals_failed"] == 1


def test_a_robot_that_moves_is_not_stuck() -> None:
    driver = _driver()
    driver.step(_tick(0.0, NavStatus.IDLE))
    assert driver.step(_tick(STUCK_S - 1.0, robot_xy=(1.4, 2.0))) is None
    assert driver.step(_tick(STUCK_S + 1.0, robot_xy=(1.4, 2.0))) is None
    assert driver.state(STUCK_S + 1.0)["goals_failed"] == 0


def test_stuck_s_zero_never_gives_a_goal_up() -> None:
    driver = _driver(stuck_s=0.0)
    driver.step(_tick(0.0, NavStatus.IDLE))
    assert driver.step(_tick(10.0 * STUCK_S)) is None


def _located(xy: tuple[float, float]) -> BeliefMap:
    data = np.zeros((40, 60), dtype=np.float32)
    data[int(xy[1] / RESOLUTION), int(xy[0] / RESOLUTION)] = 1.0
    return BeliefMap(data=data, origin=(0.0, 0.0), resolution=RESOLUTION)


def test_a_reached_standoff_of_a_peak_nobody_is_at_is_not_sent_again() -> None:
    peak = _located((3.05, 2.05))
    driver = _driver(goal=BeliefGoal(radius_m=1.5, min_mass=0.5, locate_threshold=0.9, distance_scale_m=5.0, snap=False, snap_max_error_rad=0.0, standoff_m=1.0))
    first = driver.step(_tick(0.0, NavStatus.IDLE, belief=peak))
    assert isinstance(first, Send)
    second = driver.step(_tick(1.0, NavStatus.SUCCEEDED, robot_xy=first.xy, belief=peak))
    assert isinstance(second, Send)
    assert math.dist(second.xy, first.xy) > 0.5


def _creep(driver: SearchDriver, limited: bool) -> None:
    for step in range(1, 101):
        t = 0.5 * step
        driver.step(_tick(t, robot_xy=(ROBOT[0], ROBOT[1] + 0.03 * t), limited=limited))


def test_a_creep_under_a_speed_limit_keeps_its_goal() -> None:
    driver = _driver()
    assert isinstance(driver.step(_tick(0.0, NavStatus.IDLE)), Send)
    _creep(driver, limited=True)
    assert driver.state(50.0)["goals_failed"] == 0


def test_the_same_creep_without_a_speed_limit_gives_its_goal_up() -> None:
    driver = _driver()
    assert isinstance(driver.step(_tick(0.0, NavStatus.IDLE)), Send)
    _creep(driver, limited=False)
    assert driver.state(50.0)["goals_failed"] >= 1


def test_a_frontier_through_the_robot_gets_a_goal_off_the_robot() -> None:
    grid = _grid(left_unknown=False)
    robot = (5.45, 1.95)
    driver = _driver()
    first = driver.step(_tick(0.0, NavStatus.IDLE, grid=grid, robot_xy=robot))
    assert isinstance(first, Send)
    assert math.dist(first.xy, robot) > 0.5
    assert driver.phase is Phase.DRIVE
