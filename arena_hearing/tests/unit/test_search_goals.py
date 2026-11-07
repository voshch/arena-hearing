from __future__ import annotations

import math

import numpy as np

from arena_hearing.search.frontiers import FREE, UNKNOWN, Frontier, MapGrid, Opening
from arena_hearing.search.goals import BeliefGoal, BeliefMap, GoalView, Heard, NearestGoal

RESOLUTION = 0.1
ROBOT = (2.0, 2.0)
NEAR = (1.0, 4.0)
FAR = (6.0, 4.0)


def _grid() -> MapGrid:
    data = np.full((60, 80), UNKNOWN, dtype=np.int16)
    data[:40, :] = FREE
    return MapGrid(data=data, origin=(0.0, 0.0), resolution=RESOLUTION, occupied_threshold=50)


def _frontier(xy: tuple[float, float]) -> Frontier:
    return Frontier(cells_xy=np.array([xy]), goal_xy=xy)


def _belief(*peaks: tuple[float, float]) -> BeliefMap:
    data = np.zeros((60, 80), dtype=np.float32)
    for x, y in peaks:
        data[int(y / RESOLUTION), int(x / RESOLUTION)] = 1.0
    return BeliefMap(data=data, origin=(0.0, 0.0), resolution=RESOLUTION)


def _view(belief: BeliefMap | None = None, heard: Heard | None = None, openings: list[Opening] | None = None) -> GoalView:
    return GoalView(robot_xy=ROBOT, grid=_grid(), frontiers=[_frontier(NEAR), _frontier(FAR)], openings=openings or [], belief=belief, heard=heard)


def _rule(snap: bool = False) -> BeliefGoal:
    return BeliefGoal(radius_m=1.5, min_mass=0.5, locate_threshold=2.0, distance_scale_m=5.0, snap=snap, snap_max_error_rad=math.radians(20.0), standoff_m=1.0)


def test_nearest_rule_takes_the_closest_frontier() -> None:
    assert NearestGoal().choose(_view()) == NEAR


def test_belief_rule_prefers_the_frontier_facing_the_call() -> None:
    assert _rule().choose(_view(belief=_belief((6.5, 4.5)))) == FAR


def test_belief_rule_without_belief_mass_takes_the_nearest_frontier() -> None:
    assert _rule().choose(_view(belief=_belief())) == NEAR


def _locate_rule(standoff_m: float = 1.0) -> BeliefGoal:
    return BeliefGoal(radius_m=1.5, min_mass=0.5, locate_threshold=0.9, distance_scale_m=5.0, snap=False, snap_max_error_rad=0.0, standoff_m=standoff_m)


def test_belief_rule_stops_short_of_a_located_person_on_the_robot_side() -> None:
    person = (3.05, 1.05)
    goal = _locate_rule().choose(_view(belief=_belief(person)))
    assert goal is not None
    assert 1.0 <= math.dist(goal, person) <= 1.15
    assert math.dist(goal, ROBOT) < math.dist(person, ROBOT) - 0.5


def test_a_located_person_the_map_marks_occupied_gets_a_goal_beside_them() -> None:
    grid = _grid()
    grid.data[8:13, 28:33] = 100
    view = GoalView(robot_xy=ROBOT, grid=grid, frontiers=[_frontier(NEAR), _frontier(FAR)], openings=[], belief=_belief((3.05, 1.05)), heard=None)
    goal = _locate_rule().choose(view)
    assert goal is not None
    assert 1.0 <= math.dist(goal, (3.05, 1.05)) <= 1.15


def test_belief_rule_clamps_a_person_far_in_unknown_space_to_the_nearest_frontier() -> None:
    assert _locate_rule(standoff_m=0.8).choose(_view(belief=_belief((6.05, 5.85)))) == FAR


def test_snap_sends_the_belief_rule_through_the_opening_the_call_came_through() -> None:
    door = Opening(xy=(2.0, 3.5))
    heard = Heard(robot_xy=ROBOT, bearing_rad=math.radians(95.0))
    assert _rule(snap=True).choose(_view(belief=_belief((6.5, 4.5)), heard=heard, openings=[door])) == door.xy


def test_without_snap_the_heard_bearing_is_ignored() -> None:
    heard = Heard(robot_xy=ROBOT, bearing_rad=math.radians(95.0))
    assert _rule().choose(_view(belief=_belief((6.5, 4.5)), heard=heard, openings=[Opening(xy=(2.0, 3.5))])) == FAR


def test_a_reached_standoff_is_not_chosen_again() -> None:
    person = (3.05, 1.05)
    first = _locate_rule().choose(_view(belief=_belief(person)))
    assert first is not None
    view = GoalView(robot_xy=first, grid=_grid(), frontiers=[_frontier(NEAR), _frontier(FAR)], openings=[], belief=_belief(person), heard=None, blocked=(first,), blocked_radius_m=0.5)
    second = _locate_rule().choose(view)
    assert second is not None
    assert math.dist(second, first) > 0.5
    assert 1.0 <= math.dist(second, person) <= 1.15


def test_a_located_person_with_every_standoff_reached_falls_back_to_the_nearest_frontier() -> None:
    person = (3.05, 1.05)
    ring = [(person[0] + 1.05 * math.cos(a), person[1] + 1.05 * math.sin(a)) for a in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)]
    view = GoalView(robot_xy=ROBOT, grid=_grid(), frontiers=[_frontier(NEAR), _frontier(FAR)], openings=[], belief=_belief(person), heard=None, blocked=tuple(ring), blocked_radius_m=0.5)
    assert _locate_rule().choose(view) == NEAR


def _walled_person(person: tuple[float, float], clear_m: float) -> GoalView:
    grid = _grid()
    rows, cols = np.indices(grid.data.shape)
    cx, cy = person
    inside = np.hypot((cols + 0.5) * RESOLUTION - cx, (rows + 0.5) * RESOLUTION - cy) < clear_m
    grid.data[inside & (grid.data == FREE)] = 100
    return GoalView(robot_xy=ROBOT, grid=grid, frontiers=[_frontier(NEAR), _frontier(FAR)], openings=[], belief=_belief(person), heard=None)


def test_a_located_person_within_the_approach_reach_gets_a_free_cell_beside_them() -> None:
    person = (3.05, 2.05)
    goal = BeliefGoal(radius_m=1.5, min_mass=0.5, locate_threshold=0.9, distance_scale_m=5.0, snap=False, snap_max_error_rad=0.0, standoff_m=1.0, approach_reach=2.0).choose(_walled_person(person, 1.5))
    assert goal is not None
    assert 1.5 <= math.dist(goal, person) <= 1.65


def test_a_located_person_beyond_the_approach_reach_is_approached_through_the_nearest_frontier() -> None:
    person = (3.05, 2.05)
    goal = BeliefGoal(radius_m=1.5, min_mass=0.5, locate_threshold=0.9, distance_scale_m=5.0, snap=False, snap_max_error_rad=0.0, standoff_m=1.0, approach_reach=1.2).choose(_walled_person(person, 1.5))
    assert goal == NEAR
