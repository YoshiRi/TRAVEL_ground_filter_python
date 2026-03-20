"""Tests for seed.py — find_dominant_subcells."""

import numpy as np
import pytest

from travel_py.grid import Grid, GridSpec
from travel_py.seed import find_dominant_subcells
from travel_py.types import SubCell, SubCellIndex, CellState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_with_ground_subcells(entries: list[tuple[int, int, int, float]]) -> Grid:
    """
    Build a Grid with specific GROUND SubCells.

    entries: list of (ix, iy, tri, weight)
    """
    spec = GridSpec(resolution=2.0)
    grid = Grid(spec)
    for ix, iy, tri, weight in entries:
        cell = grid.get_or_create_cell((ix, iy))
        sub = SubCell(points=np.zeros((1, 3)), label=CellState.GROUND, weight=weight)
        cell.subcells[tri] = sub
    return grid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindDominantSubcells:
    def test_empty_grid_returns_empty(self):
        spec = GridSpec(resolution=2.0)
        grid = Grid(spec)
        result = find_dominant_subcells(grid, top_k=5)
        assert result == []

    def test_no_ground_subcells_returns_empty(self):
        spec = GridSpec(resolution=2.0)
        grid = Grid(spec)
        cell = grid.get_or_create_cell((0, 0))
        cell.subcells[0] = SubCell(points=np.zeros((1, 3)), label=CellState.NON_GROUND)
        result = find_dominant_subcells(grid, top_k=5)
        assert result == []

    def test_single_ground_returns_it(self):
        grid = _grid_with_ground_subcells([(0, 0, 1, 3.0)])
        result = find_dominant_subcells(grid, top_k=1)
        assert len(result) == 1
        assert result[0] == SubCellIndex(0, 0, 1)

    def test_top_k_limits_results(self):
        entries = [(i, 0, 0, float(i)) for i in range(10)]
        grid = _grid_with_ground_subcells(entries)
        result = find_dominant_subcells(grid, top_k=3)
        assert len(result) == 3

    def test_top_k_larger_than_candidates(self):
        entries = [(i, 0, 0, float(i)) for i in range(4)]
        grid = _grid_with_ground_subcells(entries)
        result = find_dominant_subcells(grid, top_k=10)
        assert len(result) == 4

    def test_sorted_by_weight_descending(self):
        entries = [
            (0, 0, 0, 1.0),
            (1, 0, 0, 5.0),
            (2, 0, 0, 3.0),
        ]
        grid = _grid_with_ground_subcells(entries)
        result = find_dominant_subcells(grid, top_k=3)
        weights = []
        for idx in result:
            cell = grid.cells[(idx.i, idx.j)]
            weights.append(cell.subcells[idx.tri].weight)
        assert weights == sorted(weights, reverse=True)

    def test_default_top_k_is_one(self):
        entries = [(i, 0, 0, float(i)) for i in range(5)]
        grid = _grid_with_ground_subcells(entries)
        result = find_dominant_subcells(grid)
        assert len(result) == 1
        # Should be the highest weight (ix=4)
        assert result[0].i == 4

    def test_returns_subcell_indices(self):
        grid = _grid_with_ground_subcells([(3, 7, 2, 9.0)])
        result = find_dominant_subcells(grid, top_k=1)
        assert isinstance(result[0], SubCellIndex)
        assert result[0].i == 3
        assert result[0].j == 7
        assert result[0].tri == 2
