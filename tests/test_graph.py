"""Tests for graph.py — TraversabilityGraph strict TGS adjacency."""

import numpy as np
import pytest

from travel_py.graph import TraversabilityGraph, _EXTERNAL_NEIGHBOR
from travel_py.grid import Grid, GridSpec
from travel_py.types import SubCellIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_cell_grid(ix: int = 0, iy: int = 0, resolution: float = 2.0) -> Grid:
    """Grid containing exactly one cell at (ix, iy)."""
    spec = GridSpec(resolution=resolution)
    grid = Grid(spec)
    from travel_py.types import Cell, SubCell
    cell = grid.get_or_create_cell((ix, iy))
    for t in range(4):
        cell.subcells[t] = SubCell(points=np.zeros((1, 3)))
    return grid


def _two_adjacent_cells(dx: int, dy: int, resolution: float = 2.0) -> Grid:
    """Grid with cells at (0,0) and (dx, dy)."""
    spec = GridSpec(resolution=resolution)
    grid = Grid(spec)
    from travel_py.types import Cell, SubCell
    for pos in [(0, 0), (dx, dy)]:
        cell = grid.get_or_create_cell(pos)
        for t in range(4):
            cell.subcells[t] = SubCell(points=np.zeros((1, 3)))
    return grid


# ---------------------------------------------------------------------------
# _EXTERNAL_NEIGHBOR table
# ---------------------------------------------------------------------------

class TestExternalNeighborTable:
    def test_all_tris_covered(self):
        assert set(_EXTERNAL_NEIGHBOR.keys()) == {0, 1, 2, 3}

    def test_symmetry(self):
        """If tri A points to (di, dj, B) then B must point to (-di, -dj, A)."""
        for tri_a, (di, dj, tri_b) in _EXTERNAL_NEIGHBOR.items():
            di2, dj2, tri_a2 = _EXTERNAL_NEIGHBOR[tri_b]
            assert (di2, dj2, tri_a2) == (-di, -dj, tri_a), (
                f"Symmetry broken for tri {tri_a}: expected inverse from tri {tri_b}"
            )


# ---------------------------------------------------------------------------
# Internal neighbors
# ---------------------------------------------------------------------------

class TestInternalNeighbors:
    def test_returns_three_internal_neighbors(self):
        grid = _single_cell_grid()
        g = TraversabilityGraph(grid)
        for t in range(4):
            idx = SubCellIndex(0, 0, t)
            neighbors = g.get_neighbor_subcells(idx)
            internal = [n for n in neighbors if n.i == 0 and n.j == 0]
            assert len(internal) == 3
            assert all(n.tri != t for n in internal)

    def test_no_external_neighbor_when_grid_has_single_cell(self):
        grid = _single_cell_grid()
        g = TraversabilityGraph(grid)
        for t in range(4):
            idx = SubCellIndex(0, 0, t)
            neighbors = g.get_neighbor_subcells(idx)
            external = [n for n in neighbors if n.i != 0 or n.j != 0]
            assert len(external) == 0


# ---------------------------------------------------------------------------
# External neighbors
# ---------------------------------------------------------------------------

class TestExternalNeighbors:
    @pytest.mark.parametrize("tri,dx,dy,exp_tri", [
        (0, +1,  0, 2),   # East tri → right cell's West tri
        (1,  0, +1, 3),   # North tri → top cell's South tri
        (2, -1,  0, 0),   # West tri → left cell's East tri
        (3,  0, -1, 1),   # South tri → bottom cell's North tri
    ])
    def test_external_neighbor_correct_tri(self, tri, dx, dy, exp_tri):
        grid = _two_adjacent_cells(dx, dy)
        g = TraversabilityGraph(grid)
        idx = SubCellIndex(0, 0, tri)
        neighbors = g.get_neighbor_subcells(idx)
        external = [n for n in neighbors if (n.i, n.j) != (0, 0)]
        assert len(external) == 1
        assert external[0].i == dx
        assert external[0].j == dy
        assert external[0].tri == exp_tri

    def test_no_external_neighbor_if_adjacent_cell_missing(self):
        """Tri 0 (East) should have no external neighbor if (1,0) doesn't exist."""
        grid = _single_cell_grid(0, 0)
        g = TraversabilityGraph(grid)
        idx = SubCellIndex(0, 0, 0)
        neighbors = g.get_neighbor_subcells(idx)
        external = [n for n in neighbors if n.i != 0]
        assert len(external) == 0

    def test_exactly_one_external_neighbor_per_tri(self):
        """With an adjacent cell present, exactly 1 external neighbor."""
        for tri, (dx, dy, _) in _EXTERNAL_NEIGHBOR.items():
            grid = _two_adjacent_cells(dx, dy)
            g = TraversabilityGraph(grid)
            idx = SubCellIndex(0, 0, tri)
            neighbors = g.get_neighbor_subcells(idx)
            external = [n for n in neighbors if (n.i, n.j) != (0, 0)]
            assert len(external) == 1, f"tri={tri} should have exactly 1 external neighbor"
