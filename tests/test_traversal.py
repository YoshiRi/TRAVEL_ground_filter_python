"""Tests for traversal.py — run_subcell_traversal."""

import numpy as np
import pytest

from travel_py.grid import Grid, GridSpec
from travel_py.graph import TraversabilityGraph
from travel_py.traversal import run_subcell_traversal
from travel_py.types import SubCell, SubCellIndex, CellState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_with_cells(positions: list[tuple[int, int]], resolution: float = 2.0) -> Grid:
    spec = GridSpec(resolution=resolution)
    grid = Grid(spec)
    for ix, iy in positions:
        cell = grid.get_or_create_cell((ix, iy))
        for t in range(4):
            cell.subcells[t] = SubCell(points=np.zeros((1, 3)), label=CellState.GROUND)
    return grid


def _accept_all(src, dst) -> bool:
    return True


def _accept_none(src, dst) -> bool:
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunSubcellTraversal:
    def test_empty_start_returns_empty(self):
        grid = _grid_with_cells([(0, 0)])
        graph = TraversabilityGraph(grid)
        visited, rejected, max_depth = run_subcell_traversal(
            graph=graph, start_nodes=[], accept_fn=_accept_all
        )
        assert len(visited) == 0
        assert rejected == 0
        assert max_depth == 0

    def test_single_seed_no_expansion_reject_all(self):
        grid = _grid_with_cells([(0, 0)])
        graph = TraversabilityGraph(grid)
        start = [SubCellIndex(0, 0, 0)]
        visited, rejected, _ = run_subcell_traversal(
            graph=graph, start_nodes=start, accept_fn=_accept_none
        )
        # Seed is always in visited; no expansion
        assert SubCellIndex(0, 0, 0) in visited
        assert rejected > 0

    def test_accept_all_visits_all_internal(self):
        """With accept_all and a single cell, all 4 tris should be visited."""
        grid = _grid_with_cells([(0, 0)])
        graph = TraversabilityGraph(grid)
        start = [SubCellIndex(0, 0, 0)]
        visited, _, _ = run_subcell_traversal(
            graph=graph, start_nodes=start, accept_fn=_accept_all
        )
        for t in range(4):
            assert SubCellIndex(0, 0, t) in visited

    def test_traversal_expands_to_adjacent_cell(self):
        """Tri 0 of (0,0) should expand to tri 2 of (1,0) with accept_all."""
        grid = _grid_with_cells([(0, 0), (1, 0)])
        graph = TraversabilityGraph(grid)
        start = [SubCellIndex(0, 0, 0)]
        visited, _, _ = run_subcell_traversal(
            graph=graph, start_nodes=start, accept_fn=_accept_all
        )
        assert SubCellIndex(1, 0, 2) in visited

    def test_multiple_seeds_union(self):
        """Two seeds in disconnected cells still visit all their tris."""
        grid = _grid_with_cells([(0, 0), (5, 5)])
        graph = TraversabilityGraph(grid)
        start = [SubCellIndex(0, 0, 0), SubCellIndex(5, 5, 0)]
        visited, _, _ = run_subcell_traversal(
            graph=graph, start_nodes=start, accept_fn=_accept_all
        )
        for t in range(4):
            assert SubCellIndex(0, 0, t) in visited
            assert SubCellIndex(5, 5, t) in visited

    def test_rejected_count_non_zero_when_blocking(self):
        grid = _grid_with_cells([(0, 0)])
        graph = TraversabilityGraph(grid)
        start = [SubCellIndex(0, 0, 0)]
        _, rejected, _ = run_subcell_traversal(
            graph=graph, start_nodes=start, accept_fn=_accept_none
        )
        # 3 internal neighbors are rejected
        assert rejected == 3

    def test_max_depth_zero_for_single_seed_no_expansion(self):
        grid = _grid_with_cells([(0, 0)])
        graph = TraversabilityGraph(grid)
        start = [SubCellIndex(0, 0, 0)]
        _, _, max_depth = run_subcell_traversal(
            graph=graph, start_nodes=start, accept_fn=_accept_none
        )
        assert max_depth == 0

    def test_max_depth_increases_with_expansion(self):
        """Chain of 3 cells: (0,0)→(1,0)→(2,0), starting from (0,0).tri=0."""
        grid = _grid_with_cells([(0, 0), (1, 0), (2, 0)])
        graph = TraversabilityGraph(grid)
        start = [SubCellIndex(0, 0, 0)]
        _, _, max_depth = run_subcell_traversal(
            graph=graph, start_nodes=start, accept_fn=_accept_all
        )
        assert max_depth >= 1

    def test_no_duplicate_visits(self):
        """Visited set should not contain duplicates (guaranteed by set semantics)."""
        grid = _grid_with_cells([(0, 0), (1, 0)])
        graph = TraversabilityGraph(grid)
        start = [SubCellIndex(0, 0, 0), SubCellIndex(0, 0, 1)]
        visited, _, _ = run_subcell_traversal(
            graph=graph, start_nodes=start, accept_fn=_accept_all
        )
        # A set cannot have duplicates by definition; just verify it's a set
        assert isinstance(visited, set)
