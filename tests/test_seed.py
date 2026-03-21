"""Tests for seed.py — SeedCriterion, SeedSelector, find_dominant_subcells."""

import math

import numpy as np
import pytest

from travel_py.grid import Grid, GridSpec
from travel_py.seed import (
    SeedCriterion,
    SeedSelector,
    MinPointCount,
    SmallHeightRange,
    LowMeanHeight,
    LowMinHeight,
    find_dominant_subcells,
)
from travel_py.types import Cell, SubCell, SubCellIndex, CellState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell(min_z=None, mean_z=None, max_z=None, height_range=None, n_points=0):
    c = Cell(index=(0, 0))
    c.min_z = min_z
    c.mean_z = mean_z
    c.max_z = max_z
    c.height_range = height_range
    c.point_indices = list(range(n_points))
    return c


def _grid_with_ground_subcells(entries: list[tuple[int, int, int, float]]) -> Grid:
    spec = GridSpec(resolution=2.0)
    grid = Grid(spec)
    for ix, iy, tri, weight in entries:
        cell = grid.get_or_create_cell((ix, iy))
        sub = SubCell(points=np.zeros((1, 3)), label=CellState.GROUND, weight=weight)
        cell.subcells[tri] = sub
    return grid


# ---------------------------------------------------------------------------
# SeedCriterion base class
# ---------------------------------------------------------------------------

class TestSeedCriterionBase:
    def test_score_raises_not_implemented(self):
        crit = SeedCriterion()
        with pytest.raises(NotImplementedError):
            crit.score(_cell())


# ---------------------------------------------------------------------------
# MinPointCount
# ---------------------------------------------------------------------------

class TestMinPointCount:
    def test_meets_minimum_returns_zero(self):
        crit = MinPointCount(min_points=3)
        assert crit.score(_cell(n_points=3)) == 0.0
        assert crit.score(_cell(n_points=10)) == 0.0

    def test_below_minimum_returns_neg_inf(self):
        crit = MinPointCount(min_points=3)
        assert crit.score(_cell(n_points=0)) == -math.inf
        assert crit.score(_cell(n_points=2)) == -math.inf

    def test_exactly_minimum_ok(self):
        crit = MinPointCount(min_points=1)
        assert crit.score(_cell(n_points=1)) == 0.0


# ---------------------------------------------------------------------------
# SmallHeightRange
# ---------------------------------------------------------------------------

class TestSmallHeightRange:
    def test_small_range_scores_higher_than_large(self):
        crit = SmallHeightRange()
        flat = _cell(height_range=0.1)
        bumpy = _cell(height_range=2.0)
        assert crit.score(flat) > crit.score(bumpy)

    def test_none_height_range_returns_neg_inf(self):
        crit = SmallHeightRange()
        assert crit.score(_cell(height_range=None)) == -math.inf

    def test_score_is_negative_range(self):
        crit = SmallHeightRange()
        assert crit.score(_cell(height_range=0.5)) == pytest.approx(-0.5)
        assert crit.score(_cell(height_range=1.3)) == pytest.approx(-1.3)


# ---------------------------------------------------------------------------
# LowMeanHeight
# ---------------------------------------------------------------------------

class TestLowMeanHeight:
    def test_lower_mean_scores_higher(self):
        crit = LowMeanHeight()
        low = _cell(mean_z=-1.0)
        high = _cell(mean_z=5.0)
        assert crit.score(low) > crit.score(high)

    def test_none_mean_z_returns_neg_inf(self):
        crit = LowMeanHeight()
        assert crit.score(_cell(mean_z=None)) == -math.inf

    def test_score_is_negative_mean_z(self):
        crit = LowMeanHeight()
        assert crit.score(_cell(mean_z=2.0)) == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# LowMinHeight
# ---------------------------------------------------------------------------

class TestLowMinHeight:
    def test_lower_min_scores_higher(self):
        crit = LowMinHeight()
        low = _cell(min_z=-2.0)
        high = _cell(min_z=3.0)
        assert crit.score(low) > crit.score(high)

    def test_none_min_z_returns_neg_inf(self):
        crit = LowMinHeight()
        assert crit.score(_cell(min_z=None)) == -math.inf

    def test_score_is_negative_min_z(self):
        crit = LowMinHeight()
        assert crit.score(_cell(min_z=0.5)) == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# SeedSelector
# ---------------------------------------------------------------------------

class TestSeedSelector:
    def test_selects_top_k_cells(self):
        cells = [_cell(mean_z=float(i)) for i in range(5)]
        for i, c in enumerate(cells):
            c.index = (i, 0)
        selector = SeedSelector(criteria=[LowMeanHeight()], top_k=2)
        result = selector.select(cells)
        assert len(result) == 2
        # Lowest mean_z wins: mean_z=0 and mean_z=1
        result_mean_z = sorted(c.mean_z for c in result)
        assert result_mean_z == [0.0, 1.0]

    def test_invalid_cell_excluded_by_neg_inf(self):
        """A cell with None mean_z gets -inf → excluded."""
        cells = [_cell(mean_z=None), _cell(mean_z=0.5), _cell(mean_z=1.0)]
        selector = SeedSelector(criteria=[LowMeanHeight()], top_k=3)
        result = selector.select(cells)
        # Only 2 valid cells
        assert len(result) == 2

    def test_multi_criteria_combined(self):
        """Two criteria: min_points + low_mean_z."""
        c_few = _cell(n_points=1, mean_z=0.0)  # fails min_points=3
        c_ok1 = _cell(n_points=5, mean_z=0.1)
        c_ok2 = _cell(n_points=5, mean_z=2.0)
        selector = SeedSelector(criteria=[MinPointCount(3), LowMeanHeight()], top_k=2)
        result = selector.select([c_few, c_ok1, c_ok2])
        assert c_few not in result
        assert c_ok1 in result

    def test_top_k_exceeds_valid_cells(self):
        cells = [_cell(mean_z=0.0), _cell(mean_z=1.0)]
        selector = SeedSelector(criteria=[LowMeanHeight()], top_k=10)
        result = selector.select(cells)
        assert len(result) == 2

    def test_empty_cells_returns_empty(self):
        selector = SeedSelector(criteria=[LowMeanHeight()], top_k=3)
        result = selector.select([])
        assert result == []

    def test_all_invalid_returns_empty(self):
        cells = [_cell(mean_z=None) for _ in range(5)]
        selector = SeedSelector(criteria=[LowMeanHeight()], top_k=3)
        result = selector.select(cells)
        assert result == []

    def test_first_neg_inf_criterion_short_circuits(self):
        """Once a criterion returns -inf, subsequent criteria are skipped."""
        class CountingCriterion(SeedCriterion):
            def __init__(self):
                self.call_count = 0
            def score(self, cell):
                self.call_count += 1
                return 0.0

        block = MinPointCount(min_points=999)  # always -inf
        counter = CountingCriterion()
        selector = SeedSelector(criteria=[block, counter], top_k=1)
        selector.select([_cell(n_points=0)])
        # counter should never be called because block short-circuits
        assert counter.call_count == 0


# ---------------------------------------------------------------------------
# find_dominant_subcells
# ---------------------------------------------------------------------------

class TestFindDominantSubcells:
    def test_empty_grid_returns_empty(self):
        grid = Grid(GridSpec(resolution=2.0))
        assert find_dominant_subcells(grid, top_k=5) == []

    def test_no_ground_subcells_returns_empty(self):
        grid = Grid(GridSpec(resolution=2.0))
        cell = grid.get_or_create_cell((0, 0))
        cell.subcells[0] = SubCell(points=np.zeros((1, 3)), label=CellState.NON_GROUND)
        assert find_dominant_subcells(grid, top_k=5) == []

    def test_single_ground_returns_it(self):
        grid = _grid_with_ground_subcells([(0, 0, 1, 3.0)])
        result = find_dominant_subcells(grid, top_k=1)
        assert len(result) == 1
        assert result[0] == SubCellIndex(0, 0, 1)

    def test_top_k_limits_results(self):
        entries = [(i, 0, 0, float(i)) for i in range(10)]
        assert len(find_dominant_subcells(_grid_with_ground_subcells(entries), top_k=3)) == 3

    def test_top_k_larger_than_candidates(self):
        entries = [(i, 0, 0, float(i)) for i in range(4)]
        assert len(find_dominant_subcells(_grid_with_ground_subcells(entries), top_k=10)) == 4

    def test_sorted_by_weight_descending(self):
        entries = [(0, 0, 0, 1.0), (1, 0, 0, 5.0), (2, 0, 0, 3.0)]
        grid = _grid_with_ground_subcells(entries)
        result = find_dominant_subcells(grid, top_k=3)
        weights = [grid.cells[(idx.i, idx.j)].subcells[idx.tri].weight for idx in result]
        assert weights == sorted(weights, reverse=True)

    def test_default_top_k_is_one(self):
        entries = [(i, 0, 0, float(i)) for i in range(5)]
        result = find_dominant_subcells(_grid_with_ground_subcells(entries))
        assert len(result) == 1
        assert result[0].i == 4  # highest weight

    def test_returns_subcell_index_type(self):
        grid = _grid_with_ground_subcells([(3, 7, 2, 9.0)])
        result = find_dominant_subcells(grid, top_k=1)
        assert isinstance(result[0], SubCellIndex)
        assert result[0] == SubCellIndex(3, 7, 2)
