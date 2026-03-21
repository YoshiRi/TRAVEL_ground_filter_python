"""Tests for traversal.py — run_subcell_traversal and run_traversal."""

import numpy as np
import pytest

from travel_py.grid import Grid, GridSpec
from travel_py.graph import TraversabilityGraph
from travel_py.traversal import run_traversal, run_subcell_traversal
from travel_py.types import Cell, SubCell, SubCellIndex, CellState, RejectReason


# ===========================================================================
# Helpers
# ===========================================================================

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


def _cell_at(ix: int, iy: int, min_z: float = 0.0) -> Cell:
    c = Cell(index=(ix, iy))
    c.min_z = min_z
    c.mean_z = min_z
    return c


# ===========================================================================
# run_subcell_traversal
# ===========================================================================

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
        assert SubCellIndex(0, 0, 0) in visited
        assert rejected > 0

    def test_accept_all_visits_all_internal_tris(self):
        grid = _grid_with_cells([(0, 0)])
        graph = TraversabilityGraph(grid)
        start = [SubCellIndex(0, 0, 0)]
        visited, _, _ = run_subcell_traversal(
            graph=graph, start_nodes=start, accept_fn=_accept_all
        )
        for t in range(4):
            assert SubCellIndex(0, 0, t) in visited

    def test_traversal_expands_to_adjacent_cell(self):
        grid = _grid_with_cells([(0, 0), (1, 0)])
        graph = TraversabilityGraph(grid)
        start = [SubCellIndex(0, 0, 0)]
        visited, _, _ = run_subcell_traversal(
            graph=graph, start_nodes=start, accept_fn=_accept_all
        )
        assert SubCellIndex(1, 0, 2) in visited

    def test_multiple_seeds_union(self):
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
        assert rejected == 3  # 3 internal neighbors rejected

    def test_max_depth_zero_when_no_expansion(self):
        grid = _grid_with_cells([(0, 0)])
        graph = TraversabilityGraph(grid)
        _, _, max_depth = run_subcell_traversal(
            graph=graph, start_nodes=[SubCellIndex(0, 0, 0)], accept_fn=_accept_none
        )
        assert max_depth == 0

    def test_max_depth_increases_with_chain_expansion(self):
        grid = _grid_with_cells([(0, 0), (1, 0), (2, 0)])
        graph = TraversabilityGraph(grid)
        _, _, max_depth = run_subcell_traversal(
            graph=graph, start_nodes=[SubCellIndex(0, 0, 0)], accept_fn=_accept_all
        )
        assert max_depth >= 1

    def test_visited_is_a_set_no_duplicates(self):
        grid = _grid_with_cells([(0, 0), (1, 0)])
        graph = TraversabilityGraph(grid)
        start = [SubCellIndex(0, 0, 0), SubCellIndex(0, 0, 1)]
        visited, _, _ = run_subcell_traversal(
            graph=graph, start_nodes=start, accept_fn=_accept_all
        )
        assert isinstance(visited, set)


# ===========================================================================
# run_traversal (Cell-based BFS)
# ===========================================================================

class TestRunTraversal:
    def _cells_map(self, cells):
        return {c.index: c for c in cells}

    def test_empty_seed_returns_empty_state(self):
        cells = self._cells_map([_cell_at(0, 0)])
        state = run_traversal(seed_cells=[], cells=cells)
        assert state.num_ground == 0
        assert state.num_rejected == 0

    def test_single_seed_no_neighbors(self):
        c = _cell_at(0, 0)
        state = run_traversal(seed_cells=[c], cells={c.index: c})
        assert state.num_ground == 1
        assert c.state == CellState.GROUND
        assert c.visited is True

    def test_seed_propagates_to_all_connected_cells(self):
        """4-connectivity BFS from center reaches all 9 cells in a 3x3 grid."""
        cells = [_cell_at(i, j) for i in range(-1, 2) for j in range(-1, 2)]
        cells_map = self._cells_map(cells)
        center = cells_map[(0, 0)]
        state = run_traversal(seed_cells=[center], cells=cells_map, connectivity=4)
        # BFS chains through direct neighbors to corners → all 9 reachable
        assert state.num_ground == 9

    def test_seed_one_hop_only_with_isolated_grid(self):
        """With only the center and its 4 direct neighbors (no diagonal cells),
        4-connectivity BFS visits exactly 5 cells."""
        cells = [_cell_at(0, 0), _cell_at(1, 0), _cell_at(-1, 0),
                 _cell_at(0, 1), _cell_at(0, -1)]
        cells_map = self._cells_map(cells)
        seed = cells_map[(0, 0)]
        state = run_traversal(seed_cells=[seed], cells=cells_map, connectivity=4)
        assert state.num_ground == 5

    def test_accept_fn_blocks_expansion(self):
        cells = [_cell_at(0, 0), _cell_at(1, 0), _cell_at(0, 1)]
        cells_map = self._cells_map(cells)
        seed = cells_map[(0, 0)]
        state = run_traversal(
            seed_cells=[seed], cells=cells_map,
            connectivity=4, accept_fn=lambda c, n: False
        )
        assert state.num_ground == 1  # only seed
        assert state.num_rejected >= 1

    def test_accept_fn_exception_marks_invalid_feature(self):
        def bad_accept(c, n):
            raise ValueError("deliberate error")

        cells = [_cell_at(0, 0), _cell_at(1, 0)]
        cells_map = self._cells_map(cells)
        seed = cells_map[(0, 0)]
        state = run_traversal(
            seed_cells=[seed], cells=cells_map,
            connectivity=4, accept_fn=bad_accept
        )
        neighbor = cells_map[(1, 0)]
        assert neighbor.reject_reason == RejectReason.INVALID_FEATURE

    def test_already_visited_neighbors_skipped(self):
        """A node marked visited before BFS should not be re-processed."""
        cells = [_cell_at(0, 0), _cell_at(1, 0)]
        cells_map = self._cells_map(cells)
        cells_map[(1, 0)].visited = True  # pre-mark as visited
        seed = cells_map[(0, 0)]
        state = run_traversal(seed_cells=[seed], cells=cells_map, connectivity=4)
        assert state.num_ground == 1  # neighbor was skipped

    def test_multiple_seeds_propagate_independently(self):
        # Two isolated pairs of cells
        cells = [_cell_at(0, 0), _cell_at(1, 0), _cell_at(10, 10), _cell_at(11, 10)]
        cells_map = self._cells_map(cells)
        seeds = [cells_map[(0, 0)], cells_map[(10, 10)]]
        state = run_traversal(seed_cells=seeds, cells=cells_map, connectivity=4)
        assert state.num_ground == 4

    def test_iteration_history_recorded(self):
        cells = [_cell_at(0, 0), _cell_at(1, 0)]
        cells_map = self._cells_map(cells)
        seed = cells_map[(0, 0)]
        state = run_traversal(seed_cells=[seed], cells=cells_map, connectivity=4)
        # iter 0 should have the seed
        assert (0, 0) in state.iteration_history.get(0, [])

    def test_accepted_neighbor_has_no_reject_reason(self):
        cells = [_cell_at(0, 0), _cell_at(1, 0)]
        cells_map = self._cells_map(cells)
        seed = cells_map[(0, 0)]
        state = run_traversal(seed_cells=[seed], cells=cells_map, connectivity=4)
        neighbor = cells_map[(1, 0)]
        assert neighbor.reject_reason == RejectReason.NONE
        assert neighbor.state == CellState.GROUND

    def test_rejected_neighbor_gets_height_diff_reason(self):
        cells = [_cell_at(0, 0), _cell_at(1, 0)]
        cells_map = self._cells_map(cells)
        seed = cells_map[(0, 0)]
        state = run_traversal(
            seed_cells=[seed], cells=cells_map,
            connectivity=4, accept_fn=lambda c, n: False
        )
        neighbor = cells_map[(1, 0)]
        assert neighbor.reject_reason == RejectReason.HEIGHT_DIFF_TOO_LARGE
        assert neighbor.state == CellState.REJECTED
