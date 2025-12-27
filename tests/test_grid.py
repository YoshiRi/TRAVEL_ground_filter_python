"""Tests for grid.py module."""

import numpy as np
import pytest

from travel_py.grid import Grid, GridSpec
from travel_py.types import Cell


class TestGridSpec:
    """Tests for GridSpec dataclass."""

    def test_default_origin(self):
        """Test GridSpec with default origin."""
        spec = GridSpec(resolution=0.5)
        assert spec.resolution == 0.5
        assert spec.origin_xy == (0.0, 0.0)
        assert spec.size_xy is None

    def test_custom_origin(self):
        """Test GridSpec with custom origin."""
        spec = GridSpec(resolution=1.0, origin_xy=(10.0, 20.0))
        assert spec.resolution == 1.0
        assert spec.origin_xy == (10.0, 20.0)
        assert spec.size_xy is None

    def test_with_size(self):
        """Test GridSpec with size specified."""
        spec = GridSpec(resolution=0.5, size_xy=(100, 200))
        assert spec.resolution == 0.5
        assert spec.size_xy == (100, 200)


class TestGrid:
    """Tests for Grid class."""

    def test_init_valid(self):
        """Test Grid initialization with valid spec."""
        spec = GridSpec(resolution=0.5)
        grid = Grid(spec)
        assert grid.spec == spec
        assert grid.cells == {}
        assert grid.num_points_in_grid == 0
        assert grid.dropped_point_indices == []

    def test_init_invalid_resolution_zero(self):
        """Test Grid initialization with zero resolution."""
        spec = GridSpec(resolution=0.0)
        with pytest.raises(ValueError, match="resolution must be > 0"):
            Grid(spec)

    def test_init_invalid_resolution_negative(self):
        """Test Grid initialization with negative resolution."""
        spec = GridSpec(resolution=-1.0)
        with pytest.raises(ValueError, match="resolution must be > 0"):
            Grid(spec)

    def test_world_to_grid_index_default_origin(self):
        """Test world to grid index conversion with default origin."""
        spec = GridSpec(resolution=1.0)
        grid = Grid(spec)
        
        # Test basic conversion
        assert grid.world_to_grid_index(0.0, 0.0) == (0, 0)
        assert grid.world_to_grid_index(1.0, 1.0) == (1, 1)
        assert grid.world_to_grid_index(2.5, 3.7) == (2, 3)
        
        # Test negative coordinates
        assert grid.world_to_grid_index(-0.5, -0.5) == (-1, -1)
        assert grid.world_to_grid_index(-1.0, -2.0) == (-1, -2)

    def test_world_to_grid_index_custom_origin(self):
        """Test world to grid index conversion with custom origin."""
        spec = GridSpec(resolution=0.5, origin_xy=(10.0, 20.0))
        grid = Grid(spec)
        
        assert grid.world_to_grid_index(10.0, 20.0) == (0, 0)
        assert grid.world_to_grid_index(10.5, 20.5) == (1, 1)
        assert grid.world_to_grid_index(11.0, 21.0) == (2, 2)
        assert grid.world_to_grid_index(9.5, 19.5) == (-1, -1)

    def test_world_to_grid_index_floor_behavior(self):
        """Test that floor behavior is correct for boundary values."""
        spec = GridSpec(resolution=1.0)
        grid = Grid(spec)
        
        # Points exactly on cell boundaries
        assert grid.world_to_grid_index(1.0, 1.0) == (1, 1)
        assert grid.world_to_grid_index(0.999, 0.999) == (0, 0)
        assert grid.world_to_grid_index(-0.001, -0.001) == (-1, -1)

    def test_grid_index_to_world_default_origin(self):
        """Test grid index to world conversion with default origin."""
        spec = GridSpec(resolution=1.0)
        grid = Grid(spec)
        
        x, y = grid.grid_index_to_world(0, 0)
        assert x == 0.0
        assert y == 0.0
        
        x, y = grid.grid_index_to_world(1, 1)
        assert x == 1.0
        assert y == 1.0
        
        x, y = grid.grid_index_to_world(5, 10)
        assert x == 5.0
        assert y == 10.0

    def test_grid_index_to_world_custom_origin(self):
        """Test grid index to world conversion with custom origin."""
        spec = GridSpec(resolution=0.5, origin_xy=(10.0, 20.0))
        grid = Grid(spec)
        
        x, y = grid.grid_index_to_world(0, 0)
        assert x == 10.0
        assert y == 20.0
        
        x, y = grid.grid_index_to_world(1, 1)
        assert x == 10.5
        assert y == 20.5
        
        x, y = grid.grid_index_to_world(2, 3)
        assert x == 11.0
        assert y == 21.5

    def test_in_bounds_no_size(self):
        """Test _in_bounds when size is not specified."""
        spec = GridSpec(resolution=1.0)
        grid = Grid(spec)
        
        # Should always return True when size is not specified
        assert grid._in_bounds(0, 0) is True
        assert grid._in_bounds(100, 100) is True
        assert grid._in_bounds(-100, -100) is True

    def test_in_bounds_with_size(self):
        """Test _in_bounds when size is specified."""
        spec = GridSpec(resolution=1.0, size_xy=(10, 20))
        grid = Grid(spec)
        
        # Valid indices
        assert grid._in_bounds(0, 0) is True
        assert grid._in_bounds(9, 19) is True
        assert grid._in_bounds(5, 10) is True
        
        # Invalid indices
        assert grid._in_bounds(10, 0) is False  # x out of bounds
        assert grid._in_bounds(0, 20) is False  # y out of bounds
        assert grid._in_bounds(-1, 0) is False  # negative x
        assert grid._in_bounds(0, -1) is False  # negative y
        assert grid._in_bounds(10, 20) is False  # both out of bounds

    def test_get_cell_nonexistent(self):
        """Test get_cell for non-existent cell."""
        spec = GridSpec(resolution=1.0)
        grid = Grid(spec)
        
        assert grid.get_cell((0, 0)) is None
        assert grid.get_cell((5, 5)) is None

    def test_get_cell_existing(self):
        """Test get_cell for existing cell."""
        spec = GridSpec(resolution=1.0)
        grid = Grid(spec)
        
        # Create a cell manually
        cell = Cell(index=(0, 0))
        grid.cells[(0, 0)] = cell
        
        assert grid.get_cell((0, 0)) == cell

    def test_get_or_create_cell_new(self):
        """Test get_or_create_cell creates new cell."""
        spec = GridSpec(resolution=1.0)
        grid = Grid(spec)
        
        cell = grid.get_or_create_cell((0, 0))
        assert isinstance(cell, Cell)
        assert cell.index == (0, 0)
        assert (0, 0) in grid.cells
        assert grid.cells[(0, 0)] == cell

    def test_get_or_create_cell_existing(self):
        """Test get_or_create_cell returns existing cell."""
        spec = GridSpec(resolution=1.0)
        grid = Grid(spec)
        
        # Create cell first
        cell1 = grid.get_or_create_cell((0, 0))
        cell1.point_indices.append(42)
        
        # Get same cell
        cell2 = grid.get_or_create_cell((0, 0))
        assert cell1 is cell2
        assert cell2.point_indices == [42]

    def test_iter_cells(self):
        """Test iter_cells returns all cells."""
        spec = GridSpec(resolution=1.0)
        grid = Grid(spec)
        
        # Initially empty
        assert list(grid.iter_cells()) == []
        
        # Add some cells
        cell1 = grid.get_or_create_cell((0, 0))
        cell2 = grid.get_or_create_cell((1, 1))
        cell3 = grid.get_or_create_cell((2, 2))
        
        cells = list(grid.iter_cells())
        assert len(cells) == 3
        assert cell1 in cells
        assert cell2 in cells
        assert cell3 in cells


class TestGridFromPoints:
    """Tests for Grid.from_points class method."""

    def test_from_points_basic(self):
        """Test from_points with basic point cloud."""
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 2.0],
            [1.0, 1.0, 3.0],
        ])
        spec = GridSpec(resolution=1.0)
        grid = Grid.from_points(points, spec)
        
        assert grid.num_points_in_grid == 3
        assert len(grid.dropped_point_indices) == 0
        
        # Check that points are in correct cells
        cell_00 = grid.get_cell((0, 0))
        assert cell_00 is not None
        assert len(cell_00.point_indices) == 2  # (0,0) and (0.5, 0.5) both in (0,0)
        
        cell_11 = grid.get_cell((1, 1))
        assert cell_11 is not None
        assert len(cell_11.point_indices) == 1  # (1.0, 1.0) in (1,1)

    def test_from_points_custom_origin(self):
        """Test from_points with custom origin."""
        points = np.array([
            [10.0, 20.0, 1.0],
            [10.5, 20.5, 2.0],
            [11.0, 21.0, 3.0],
        ])
        spec = GridSpec(resolution=0.5, origin_xy=(10.0, 20.0))
        grid = Grid.from_points(points, spec)
        
        assert grid.num_points_in_grid == 3
        
        # Check cell indices
        cell_00 = grid.get_cell((0, 0))
        assert cell_00 is not None
        assert len(cell_00.point_indices) == 1  # (10.0, 20.0) -> (0, 0)
        
        cell_11 = grid.get_cell((1, 1))
        assert cell_11 is not None
        assert len(cell_11.point_indices) == 1  # (10.5, 20.5) -> (1, 1)
        
        cell_22 = grid.get_cell((2, 2))
        assert cell_22 is not None
        assert len(cell_22.point_indices) == 1  # (11.0, 21.0) -> (2, 2)

    def test_from_points_with_size_in_bounds(self):
        """Test from_points with size, all points in bounds."""
        points = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 2.0],
            [2.0, 2.0, 3.0],
        ])
        spec = GridSpec(resolution=1.0, size_xy=(5, 5))
        grid = Grid.from_points(points, spec)
        
        assert grid.num_points_in_grid == 3
        assert len(grid.dropped_point_indices) == 0

    def test_from_points_with_size_out_of_bounds(self):
        """Test from_points with size, some points out of bounds."""
        points = np.array([
            [0.0, 0.0, 1.0],      # in bounds
            [1.0, 1.0, 2.0],      # in bounds
            [10.0, 10.0, 3.0],    # out of bounds (x)
            [1.0, 10.0, 4.0],     # out of bounds (y)
            [-1.0, -1.0, 5.0],    # out of bounds (negative)
        ])
        spec = GridSpec(resolution=1.0, size_xy=(5, 5))
        grid = Grid.from_points(points, spec, keep_dropped_indices=True)
        
        assert grid.num_points_in_grid == 2
        assert len(grid.dropped_point_indices) == 3
        assert set(grid.dropped_point_indices) == {2, 3, 4}
        
        # Check that only in-bounds points are in cells
        cell_00 = grid.get_cell((0, 0))
        assert cell_00 is not None
        assert 0 in cell_00.point_indices
        
        cell_11 = grid.get_cell((1, 1))
        assert cell_11 is not None
        assert 1 in cell_11.point_indices

    def test_from_points_keep_dropped_false(self):
        """Test from_points with keep_dropped_indices=False."""
        points = np.array([
            [0.0, 0.0, 1.0],
            [10.0, 10.0, 2.0],  # out of bounds
        ])
        spec = GridSpec(resolution=1.0, size_xy=(5, 5))
        grid = Grid.from_points(points, spec, keep_dropped_indices=False)
        
        assert grid.num_points_in_grid == 1
        assert len(grid.dropped_point_indices) == 0

    def test_from_points_empty(self):
        """Test from_points with empty point cloud."""
        points = np.empty((0, 3))
        spec = GridSpec(resolution=1.0)
        grid = Grid.from_points(points, spec)
        
        assert grid.num_points_in_grid == 0
        assert len(grid.dropped_point_indices) == 0
        assert len(grid.cells) == 0

    def test_from_points_multiple_points_same_cell(self):
        """Test from_points with multiple points in same cell."""
        points = np.array([
            [0.1, 0.1, 1.0],
            [0.2, 0.2, 2.0],
            [0.3, 0.3, 3.0],
            [0.4, 0.4, 4.0],
        ])
        spec = GridSpec(resolution=1.0)
        grid = Grid.from_points(points, spec)
        
        assert grid.num_points_in_grid == 4
        cell_00 = grid.get_cell((0, 0))
        assert cell_00 is not None
        assert len(cell_00.point_indices) == 4
        assert set(cell_00.point_indices) == {0, 1, 2, 3}

    def test_from_points_extra_dimensions(self):
        """Test from_points with points having more than 3 dimensions."""
        points = np.array([
            [0.0, 0.0, 1.0, 0.5, 0.5],  # 5D point
            [1.0, 1.0, 2.0, 0.6, 0.6],
        ])
        spec = GridSpec(resolution=1.0)
        grid = Grid.from_points(points, spec)
        
        assert grid.num_points_in_grid == 2
        cell_00 = grid.get_cell((0, 0))
        assert cell_00 is not None
        assert 0 in cell_00.point_indices

    def test_from_points_invalid_shape_1d(self):
        """Test from_points with invalid 1D array."""
        points = np.array([0.0, 0.0, 1.0])
        spec = GridSpec(resolution=1.0)
        
        with pytest.raises(ValueError, match="points_xyz must be shape"):
            Grid.from_points(points, spec)

    def test_from_points_invalid_shape_insufficient_columns(self):
        """Test from_points with insufficient columns."""
        points = np.array([
            [0.0, 0.0],  # Only 2 columns
        ])
        spec = GridSpec(resolution=1.0)
        
        with pytest.raises(ValueError, match="points_xyz must be shape"):
            Grid.from_points(points, spec)

    def test_from_points_negative_coordinates(self):
        """Test from_points with negative coordinates."""
        points = np.array([
            [-1.0, -1.0, 1.0],
            [-0.5, -0.5, 2.0],
            [-2.0, -2.0, 3.0],
        ])
        spec = GridSpec(resolution=1.0)
        grid = Grid.from_points(points, spec)
        
        assert grid.num_points_in_grid == 3
        
        cell_neg1_neg1 = grid.get_cell((-1, -1))
        assert cell_neg1_neg1 is not None
        assert len(cell_neg1_neg1.point_indices) == 2  # (-1.0, -1.0) and (-0.5, -0.5)
        
        cell_neg2_neg2 = grid.get_cell((-2, -2))
        assert cell_neg2_neg2 is not None
        assert len(cell_neg2_neg2.point_indices) == 1  # (-2.0, -2.0)

    def test_from_points_boundary_values(self):
        """Test from_points with points on cell boundaries."""
        points = np.array([
            [0.0, 0.0, 1.0],    # Exactly on origin
            [1.0, 1.0, 2.0],    # Exactly on cell boundary
            [0.999, 0.999, 3.0],  # Just below boundary
            [1.001, 1.001, 4.0],  # Just above boundary
        ])
        spec = GridSpec(resolution=1.0)
        grid = Grid.from_points(points, spec)
        
        assert grid.num_points_in_grid == 4
        
        cell_00 = grid.get_cell((0, 0))
        assert cell_00 is not None
        assert len(cell_00.point_indices) == 2  # (0.0, 0.0) and (0.999, 0.999)
        
        cell_11 = grid.get_cell((1, 1))
        assert cell_11 is not None
        assert len(cell_11.point_indices) == 2  # (1.0, 1.0) and (1.001, 1.001)

