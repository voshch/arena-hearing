"""Frontiers on an occupancy grid: free cells next to unknown ones, grouped by 8-connectivity."""

from __future__ import annotations

import typing

import attrs
import numpy as np
from scipy import ndimage

UNKNOWN = -1
FREE = 0
OCCUPIED = 100

_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_SQUARE = np.ones((3, 3), dtype=bool)


def _label(mask: np.ndarray, structure: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components of ``mask`` and their count, labels from 1."""
    return typing.cast(tuple[np.ndarray, int], ndimage.label(mask, structure=structure))


@attrs.frozen
class MapGrid:
    """An ``OccupancyGrid`` payload as a (height, width) array in the map frame."""

    data: np.ndarray
    origin: tuple[float, float]
    resolution: float
    occupied_threshold: int

    @property
    def unknown(self) -> np.ndarray:
        return self.data < 0

    @property
    def occupied(self) -> np.ndarray:
        return self.data >= self.occupied_threshold

    @property
    def free(self) -> np.ndarray:
        return ~self.unknown & ~self.occupied

    def cells_to_world(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """(n, 2) map-frame centers of the given cells."""
        return np.stack([self.origin[0] + (cols + 0.5) * self.resolution, self.origin[1] + (rows + 0.5) * self.resolution], axis=1)


@attrs.frozen
class Frontier:
    cells_xy: np.ndarray  # (n, 2) cell centers
    goal_xy: tuple[float, float]  # the frontier cell nearest the centroid

    @property
    def n_cells(self) -> int:
        return len(self.cells_xy)


@attrs.frozen
class Opening:
    xy: tuple[float, float]


def _runs(free: np.ndarray, occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per cell, the length of its free run along axis 0 and whether occupied cells bound both ends."""
    if free.size == 0:
        return np.zeros(free.shape, dtype=int), np.zeros(free.shape, dtype=bool)
    labels, n = _label(free, np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool))
    index = np.arange(1, n + 1)
    lengths = np.bincount(labels.ravel(), minlength=n + 1)
    rows = np.broadcast_to(np.arange(free.shape[0])[:, None], free.shape)
    cols = np.broadcast_to(np.arange(free.shape[1])[None, :], free.shape)
    top = np.asarray(ndimage.minimum(rows, labels, index), dtype=int) - 1
    bottom = np.asarray(ndimage.maximum(rows, labels, index), dtype=int) + 1
    col = np.asarray(ndimage.minimum(cols, labels, index), dtype=int)
    padded = np.pad(occupied, ((1, 1), (0, 0)))
    bounded = np.concatenate([[False], padded[top + 1, col] & padded[bottom + 1, col]])
    return lengths[labels] * free, bounded[labels] & free


def _openings_across(free: np.ndarray, occupied: np.ndarray, min_cells: int, max_cells: int, depth: int) -> np.ndarray:
    """Cells of a gap along axis 0 with wider space within ``depth`` cells on both sides along axis 1."""
    length, bounded = _runs(free, occupied)
    narrow = bounded & (length >= min_cells) & (length <= max_cells)
    wide = free & ~(bounded & (length <= max_cells))
    before = np.zeros_like(narrow)
    after = np.zeros_like(narrow)
    for d in range(1, depth + 1):
        before[:, d:] |= wide[:, :-d]
        after[:, :-d] |= wide[:, d:]
    return narrow & before & after


def find_openings(grid: MapGrid, min_width_m: float, max_width_m: float, depth_m: float) -> list[Opening]:
    """Door-sized gaps between occupied cells, axis-aligned, with wider free space within ``depth_m`` on both sides."""
    cells = (int(round(min_width_m / grid.resolution)), int(round(max_width_m / grid.resolution)), max(int(round(depth_m / grid.resolution)), 1))
    free, occupied = grid.free, grid.occupied
    mask = _openings_across(free, occupied, *cells) | _openings_across(free.T, occupied.T, *cells).T
    labels, n = _label(mask, _SQUARE)
    openings: list[Opening] = []
    for label in range(1, n + 1):
        rows, cols = np.nonzero(labels == label)
        x, y = grid.cells_to_world(rows, cols).mean(axis=0)
        openings.append(Opening(xy=(float(x), float(y))))
    return openings


def find_frontiers(grid: MapGrid, min_cells: int) -> list[Frontier]:
    """Frontiers of at least ``min_cells`` cells, largest first."""
    near_unknown = ndimage.binary_dilation(grid.unknown, structure=_CROSS)
    labels, n = _label(grid.free & near_unknown, _SQUARE)
    frontiers: list[Frontier] = []
    for label in range(1, n + 1):
        rows, cols = np.nonzero(labels == label)
        if len(rows) < min_cells:
            continue
        cells = grid.cells_to_world(rows, cols)
        frontiers.append(_frontier(cells))
    return sorted(frontiers, key=lambda f: -f.n_cells)


def _frontier(cells: np.ndarray) -> Frontier:
    nearest = int(np.argmin(np.linalg.norm(cells - cells.mean(axis=0), axis=1)))
    return Frontier(cells_xy=cells, goal_xy=(float(cells[nearest, 0]), float(cells[nearest, 1])))


def away_from(frontiers: list[Frontier], xy: tuple[float, float], radius_m: float, min_cells: int) -> list[Frontier]:
    """The frontiers without their cells within ``radius_m`` of ``xy``, goals recomputed, those left under ``min_cells`` dropped."""
    kept = [_frontier(cells) for f in frontiers if len(cells := f.cells_xy[np.linalg.norm(f.cells_xy - np.asarray(xy), axis=1) > radius_m]) >= min_cells]
    return sorted(kept, key=lambda f: -f.n_cells)
