from __future__ import annotations

import numpy as np

from arena_hearing.search.frontiers import FREE, OCCUPIED, UNKNOWN, MapGrid, away_from, find_frontiers, find_openings

RESOLUTION = 0.1


def _grid(data: np.ndarray) -> MapGrid:
    return MapGrid(data=data, origin=(0.0, 0.0), resolution=RESOLUTION, occupied_threshold=50)


def _walled(height: int, width: int) -> np.ndarray:
    data = np.full((height, width), FREE, dtype=np.int16)
    data[[0, -1], :] = OCCUPIED
    data[:, [0, -1]] = OCCUPIED
    return data


def _two_rooms(gap_rows: slice) -> np.ndarray:
    data = _walled(40, 61)
    data[:, 30] = OCCUPIED
    data[gap_rows, 30] = FREE
    return data


def test_an_opening_is_the_door_between_two_rooms() -> None:
    openings = find_openings(_grid(_two_rooms(slice(15, 25))), min_width_m=0.6, max_width_m=1.6, depth_m=0.6)
    assert len(openings) == 1
    x, y = openings[0].xy
    assert abs(x - 3.05) < 0.11
    assert abs(y - 2.0) < 0.11


def test_a_gap_wider_than_a_door_is_no_opening() -> None:
    assert find_openings(_grid(_two_rooms(slice(5, 35))), min_width_m=0.6, max_width_m=1.6, depth_m=0.6) == []


def test_an_empty_map_has_no_frontier_and_no_opening() -> None:
    grid = _grid(np.zeros((0, 0), dtype=np.int16))
    assert find_frontiers(grid, min_cells=5) == []
    assert find_openings(grid, min_width_m=0.6, max_width_m=1.6, depth_m=0.6) == []


def test_a_map_without_free_cells_has_no_opening() -> None:
    grid = _grid(np.full((20, 20), UNKNOWN, dtype=np.int16))
    assert find_openings(grid, min_width_m=0.6, max_width_m=1.6, depth_m=0.6) == []


def test_a_narrow_corridor_is_no_opening() -> None:
    data = _walled(12, 80)
    assert find_openings(_grid(data), min_width_m=0.6, max_width_m=1.6, depth_m=0.6) == []


def test_a_frontier_lies_between_free_and_unknown() -> None:
    data = np.full((40, 60), UNKNOWN, dtype=np.int16)
    data[:, :30] = FREE
    frontiers = find_frontiers(_grid(data), min_cells=5)
    assert len(frontiers) == 1
    cols = (frontiers[0].cells_xy[:, 0] / RESOLUTION).astype(int)
    assert set(cols.tolist()) == {29}
    assert frontiers[0].n_cells == 40


def test_a_wall_between_free_and_unknown_is_no_frontier() -> None:
    data = np.full((40, 60), UNKNOWN, dtype=np.int16)
    data[:, :30] = FREE
    data[:, 29] = OCCUPIED
    assert find_frontiers(_grid(data), min_cells=5) == []


def test_frontiers_smaller_than_min_cells_are_dropped() -> None:
    data = np.full((40, 60), FREE, dtype=np.int16)
    data[0:40, 59] = UNKNOWN
    data[10:13, 20] = UNKNOWN
    frontiers = find_frontiers(_grid(data), min_cells=10)
    assert len(frontiers) == 1
    assert frontiers[0].n_cells == 40


def test_the_frontier_goal_is_a_frontier_cell() -> None:
    data = np.full((40, 60), UNKNOWN, dtype=np.int16)
    data[10:30, 10:30] = FREE
    frontier = find_frontiers(_grid(data), min_cells=5)[0]
    goal = np.asarray(frontier.goal_xy)
    assert np.min(np.linalg.norm(frontier.cells_xy - goal, axis=1)) < 1e-9


def _edge() -> np.ndarray:
    data = np.full((40, 60), FREE, dtype=np.int16)
    data[:, 55:] = UNKNOWN
    return data


def test_a_frontier_away_from_a_point_loses_its_cells_there_and_moves_its_goal() -> None:
    (frontier,) = find_frontiers(_grid(_edge()), min_cells=5)
    robot = frontier.goal_xy
    (rest,) = away_from([frontier], robot, 0.5, 5)
    assert rest.n_cells < frontier.n_cells
    assert np.min(np.linalg.norm(rest.cells_xy - np.asarray(robot), axis=1)) > 0.5
    assert np.linalg.norm(np.asarray(rest.goal_xy) - np.asarray(robot)) > 0.5


def test_a_frontier_with_too_few_cells_left_is_dropped() -> None:
    (frontier,) = find_frontiers(_grid(_edge()), min_cells=5)
    assert away_from([frontier], frontier.goal_xy, 3.0, 5) == []
