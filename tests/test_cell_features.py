"""Tests for cell_features.py module."""

import numpy as np
import pytest

from travel_py.cell_features import compute_all_cell_features, compute_cell_height_features
from travel_py.types import Cell


class TestComputeCellHeightFeatures:
    """Tests for compute_cell_height_features function."""

    def test_single_point(self):
        """Test with a single point in the cell."""
        cell = Cell(index=(0, 0), point_indices=[0])
        points = np.array([[1.0, 2.0, 3.0]])
        
        compute_cell_height_features(cell, points)
        
        assert cell.min_z == 3.0
        assert cell.max_z == 3.0
        assert cell.mean_z == 3.0
        assert cell.height_range == 0.0

    def test_multiple_points(self):
        """Test with multiple points with different z values."""
        cell = Cell(index=(0, 0), point_indices=[0, 1, 2, 3])
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 4.0],
        ])
        
        compute_cell_height_features(cell, points)
        
        assert cell.min_z == 1.0
        assert cell.max_z == 4.0
        assert cell.mean_z == 2.5
        assert cell.height_range == 3.0

    def test_empty_cell(self):
        """Test with empty cell (no points)."""
        cell = Cell(index=(0, 0), point_indices=[])
        points = np.array([[1.0, 2.0, 3.0]])
        
        compute_cell_height_features(cell, points)
        
        # Features should remain None for empty cells
        assert cell.min_z is None
        assert cell.max_z is None
        assert cell.mean_z is None
        assert cell.height_range is None

    def test_all_same_z_values(self):
        """Test with all points having the same z value."""
        cell = Cell(index=(0, 0), point_indices=[0, 1, 2])
        points = np.array([
            [0.0, 0.0, 5.0],
            [0.1, 0.1, 5.0],
            [0.2, 0.2, 5.0],
        ])
        
        compute_cell_height_features(cell, points)
        
        assert cell.min_z == 5.0
        assert cell.max_z == 5.0
        assert cell.mean_z == 5.0
        assert cell.height_range == 0.0

    def test_negative_z_values(self):
        """Test with negative z values."""
        cell = Cell(index=(0, 0), point_indices=[0, 1, 2])
        points = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -2.0],
            [0.0, 0.0, -0.5],
        ])
        
        compute_cell_height_features(cell, points)
        
        assert cell.min_z == -2.0
        assert cell.max_z == -0.5
        assert cell.mean_z == pytest.approx(-1.1666666666666667)
        assert cell.height_range == 1.5

    def test_mixed_positive_negative_z(self):
        """Test with mixed positive and negative z values."""
        cell = Cell(index=(0, 0), point_indices=[0, 1, 2])
        points = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        compute_cell_height_features(cell, points)
        
        assert cell.min_z == -1.0
        assert cell.max_z == 1.0
        assert cell.mean_z == 0.0
        assert cell.height_range == 2.0

    def test_points_with_extra_dimensions(self):
        """Test with points having more than 3 dimensions."""
        cell = Cell(index=(0, 0), point_indices=[0, 1])
        points = np.array([
            [0.0, 0.0, 1.0, 0.5, 0.5],  # 5D point
            [0.0, 0.0, 2.0, 0.6, 0.6],
        ])
        
        compute_cell_height_features(cell, points)
        
        assert cell.min_z == 1.0
        assert cell.max_z == 2.0
        assert cell.mean_z == 1.5
        assert cell.height_range == 1.0

    def test_non_sequential_point_indices(self):
        """Test with non-sequential point indices."""
        cell = Cell(index=(0, 0), point_indices=[0, 5, 10])
        points = np.array([
            [0.0, 0.0, 1.0],  # index 0
            [1.0, 1.0, 1.0],  # index 1
            [2.0, 2.0, 1.0],  # index 2
            [3.0, 3.0, 1.0],  # index 3
            [4.0, 4.0, 1.0],  # index 4
            [5.0, 5.0, 5.0],  # index 5
            [6.0, 6.0, 1.0],  # index 6
            [7.0, 7.0, 1.0],  # index 7
            [8.0, 8.0, 1.0],  # index 8
            [9.0, 9.0, 1.0],  # index 9
            [10.0, 10.0, 10.0],  # index 10
        ])
        
        compute_cell_height_features(cell, points)
        
        assert cell.min_z == 1.0
        assert cell.max_z == 10.0
        assert cell.mean_z == pytest.approx((1.0 + 5.0 + 10.0) / 3.0)
        assert cell.height_range == 9.0

    def test_large_range(self):
        """Test with large z value range."""
        cell = Cell(index=(0, 0), point_indices=[0, 1, 2])
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0],
            [0.0, 0.0, 200.0],
        ])
        
        compute_cell_height_features(cell, points)
        
        assert cell.min_z == 0.0
        assert cell.max_z == 200.0
        assert cell.mean_z == 100.0
        assert cell.height_range == 200.0

    def test_small_range(self):
        """Test with very small z value differences."""
        cell = Cell(index=(0, 0), point_indices=[0, 1, 2])
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0001],
            [0.0, 0.0, 1.0002],
        ])
        
        compute_cell_height_features(cell, points)
        
        assert cell.min_z == 1.0
        assert cell.max_z == 1.0002
        assert cell.height_range == pytest.approx(0.0002)

    def test_preserves_other_cell_attributes(self):
        """Test that other cell attributes are not modified."""
        from travel_py.types import CellState, RejectReason
        
        cell = Cell(
            index=(5, 10),
            point_indices=[0, 1],
            min_z=None,
            mean_z=None,
            max_z=None,
            height_range=None,
            slope=0.5,
            state=CellState.GROUND,
            reject_reason=RejectReason.NONE,
            visited=True,
            iteration=1,
            parent=(4, 9),
        )
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ])
        
        compute_cell_height_features(cell, points)
        
        # Check that height features are computed
        assert cell.min_z == 1.0
        assert cell.max_z == 2.0
        assert cell.mean_z == 1.5
        assert cell.height_range == 1.0
        
        # Check that other attributes are preserved
        assert cell.index == (5, 10)
        assert cell.slope == 0.5
        assert cell.state == CellState.GROUND
        assert cell.reject_reason == RejectReason.NONE
        assert cell.visited is True
        assert cell.iteration == 1
        assert cell.parent == (4, 9)

    def test_overwrites_existing_features(self):
        """Test that existing feature values are overwritten."""
        cell = Cell(
            index=(0, 0),
            point_indices=[0, 1],
            min_z=999.0,
            max_z=999.0,
            mean_z=999.0,
            height_range=999.0,
        )
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ])
        
        compute_cell_height_features(cell, points)
        
        # Old values should be overwritten
        assert cell.min_z == 1.0
        assert cell.max_z == 2.0
        assert cell.mean_z == 1.5
        assert cell.height_range == 1.0

    def test_many_points(self):
        """Test with many points to ensure it handles larger datasets."""
        n_points = 1000
        cell = Cell(index=(0, 0), point_indices=list(range(n_points)))
        z_values = np.linspace(0.0, 100.0, n_points)
        points = np.column_stack([
            np.zeros(n_points),
            np.zeros(n_points),
            z_values,
        ])
        
        compute_cell_height_features(cell, points)
        
        assert cell.min_z == 0.0
        assert cell.max_z == 100.0
        assert cell.mean_z == pytest.approx(50.0, rel=1e-10)
        assert cell.height_range == 100.0


class TestComputeAllCellFeatures:
    """Tests for compute_all_cell_features function."""

    def test_single_cell(self):
        """Test with a single cell."""
        cell = Cell(index=(0, 0), point_indices=[0, 1])
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ])
        
        compute_all_cell_features([cell], points)
        
        assert cell.min_z == 1.0
        assert cell.max_z == 2.0
        assert cell.mean_z == 1.5
        assert cell.height_range == 1.0

    def test_multiple_cells(self):
        """Test with multiple cells."""
        cell1 = Cell(index=(0, 0), point_indices=[0, 1])
        cell2 = Cell(index=(1, 0), point_indices=[2, 3])
        cell3 = Cell(index=(0, 1), point_indices=[4])
        
        points = np.array([
            [0.0, 0.0, 1.0],  # cell1
            [0.0, 0.0, 2.0],  # cell1
            [1.0, 0.0, 3.0],  # cell2
            [1.0, 0.0, 4.0],  # cell2
            [0.0, 1.0, 5.0],  # cell3
        ])
        
        compute_all_cell_features([cell1, cell2, cell3], points)
        
        # Check cell1
        assert cell1.min_z == 1.0
        assert cell1.max_z == 2.0
        assert cell1.mean_z == 1.5
        assert cell1.height_range == 1.0
        
        # Check cell2
        assert cell2.min_z == 3.0
        assert cell2.max_z == 4.0
        assert cell2.mean_z == 3.5
        assert cell2.height_range == 1.0
        
        # Check cell3
        assert cell3.min_z == 5.0
        assert cell3.max_z == 5.0
        assert cell3.mean_z == 5.0
        assert cell3.height_range == 0.0

    def test_mixed_empty_and_populated_cells(self):
        """Test with mix of empty and populated cells."""
        cell1 = Cell(index=(0, 0), point_indices=[])  # empty
        cell2 = Cell(index=(1, 0), point_indices=[0, 1])  # populated
        cell3 = Cell(index=(0, 1), point_indices=[])  # empty
        
        points = np.array([
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
        ])
        
        compute_all_cell_features([cell1, cell2, cell3], points)
        
        # Empty cells should remain None
        assert cell1.min_z is None
        assert cell1.max_z is None
        assert cell1.mean_z is None
        assert cell1.height_range is None
        
        # Populated cell should have features
        assert cell2.min_z == 1.0
        assert cell2.max_z == 2.0
        assert cell2.mean_z == 1.5
        assert cell2.height_range == 1.0
        
        # Empty cell should remain None
        assert cell3.min_z is None
        assert cell3.max_z is None
        assert cell3.mean_z is None
        assert cell3.height_range is None

    def test_empty_iterable(self):
        """Test with empty iterable of cells."""
        points = np.array([[0.0, 0.0, 1.0]])
        
        # Should not raise an error
        compute_all_cell_features([], points)

    def test_cells_with_overlapping_point_indices(self):
        """Test with cells that share point indices (edge case)."""
        cell1 = Cell(index=(0, 0), point_indices=[0, 1])
        cell2 = Cell(index=(1, 0), point_indices=[1, 2])
        
        points = np.array([
            [0.0, 0.0, 1.0],  # index 0
            [0.5, 0.0, 2.0],  # index 1 (shared)
            [1.0, 0.0, 3.0],  # index 2
        ])
        
        compute_all_cell_features([cell1, cell2], points)
        
        # cell1 should use indices [0, 1]
        assert cell1.min_z == 1.0
        assert cell1.max_z == 2.0
        assert cell1.mean_z == 1.5
        
        # cell2 should use indices [1, 2]
        assert cell2.min_z == 2.0
        assert cell2.max_z == 3.0
        assert cell2.mean_z == 2.5

    def test_preserves_cell_state(self):
        """Test that cell state and other attributes are preserved."""
        from travel_py.types import CellState, RejectReason
        
        cell1 = Cell(
            index=(0, 0),
            point_indices=[0],
            state=CellState.GROUND,
            visited=True,
        )
        cell2 = Cell(
            index=(1, 0),
            point_indices=[1],
            state=CellState.NON_GROUND,
            visited=False,
        )
        
        points = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
        ])
        
        compute_all_cell_features([cell1, cell2], points)
        
        # Features should be computed
        assert cell1.min_z == 1.0
        assert cell2.min_z == 2.0
        
        # State should be preserved
        assert cell1.state == CellState.GROUND
        assert cell1.visited is True
        assert cell2.state == CellState.NON_GROUND
        assert cell2.visited is False

    def test_with_grid_iter_cells_pattern(self):
        """Test using the pattern that would be used with Grid.iter_cells()."""
        from travel_py.grid import Grid, GridSpec
        
        # Create a grid and populate it
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 2.0],
            [1.0, 1.0, 3.0],
        ])
        spec = GridSpec(resolution=1.0)
        grid = Grid.from_points(points, spec)
        
        # Compute features for all cells
        compute_all_cell_features(grid.iter_cells(), points)
        
        # Check that features were computed
        cell_00 = grid.get_cell((0, 0))
        assert cell_00 is not None
        assert cell_00.min_z is not None
        assert cell_00.max_z is not None
        assert cell_00.mean_z is not None
        assert cell_00.height_range is not None

