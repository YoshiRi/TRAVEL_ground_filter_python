"""Tests for adjacency.py module."""

import pytest

from travel_py.adjacency import (
    EIGHT_NEIGHBOR_OFFSETS,
    FOUR_NEIGHBOR_OFFSETS,
    abs_height_diff,
    height_diff,
    iter_existing_neighbors,
    iter_neighbor_indices,
    mean_height_diff,
)
from travel_py.types import Cell


class TestNeighborOffsets:
    """Tests for neighbor offset constants."""

    def test_four_neighbor_offsets(self):
        """Test that FOUR_NEIGHBOR_OFFSETS contains correct offsets."""
        expected = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        assert set(FOUR_NEIGHBOR_OFFSETS) == set(expected)
        assert len(FOUR_NEIGHBOR_OFFSETS) == 4

    def test_eight_neighbor_offsets(self):
        """Test that EIGHT_NEIGHBOR_OFFSETS contains correct offsets."""
        expected = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]
        assert set(EIGHT_NEIGHBOR_OFFSETS) == set(expected)
        assert len(EIGHT_NEIGHBOR_OFFSETS) == 8

    def test_four_offsets_subset_of_eight(self):
        """Test that four-neighbor offsets are a subset of eight-neighbor offsets."""
        assert set(FOUR_NEIGHBOR_OFFSETS).issubset(set(EIGHT_NEIGHBOR_OFFSETS))


class TestIterNeighborIndices:
    """Tests for iter_neighbor_indices function."""

    def test_four_connectivity_center(self):
        """Test 4-connectivity from center cell."""
        index = (5, 5)
        neighbors = list(iter_neighbor_indices(index, connectivity=4))
        
        expected = [(6, 5), (4, 5), (5, 6), (5, 4)]
        assert set(neighbors) == set(expected)
        assert len(neighbors) == 4

    def test_four_connectivity_origin(self):
        """Test 4-connectivity from origin."""
        index = (0, 0)
        neighbors = list(iter_neighbor_indices(index, connectivity=4))
        
        expected = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        assert set(neighbors) == set(expected)
        assert len(neighbors) == 4

    def test_four_connectivity_negative(self):
        """Test 4-connectivity with negative coordinates."""
        index = (-5, -5)
        neighbors = list(iter_neighbor_indices(index, connectivity=4))
        
        expected = [(-4, -5), (-6, -5), (-5, -4), (-5, -6)]
        assert set(neighbors) == set(expected)
        assert len(neighbors) == 4

    def test_eight_connectivity_center(self):
        """Test 8-connectivity from center cell."""
        index = (5, 5)
        neighbors = list(iter_neighbor_indices(index, connectivity=8))
        
        expected = [
            (6, 5), (4, 5), (5, 6), (5, 4),  # 4-connectivity
            (6, 6), (6, 4), (4, 6), (4, 4),  # diagonal
        ]
        assert set(neighbors) == set(expected)
        assert len(neighbors) == 8

    def test_eight_connectivity_origin(self):
        """Test 8-connectivity from origin."""
        index = (0, 0)
        neighbors = list(iter_neighbor_indices(index, connectivity=8))
        
        expected = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # 4-connectivity
            (1, 1), (1, -1), (-1, 1), (-1, -1),  # diagonal
        ]
        assert set(neighbors) == set(expected)
        assert len(neighbors) == 8

    def test_eight_connectivity_negative(self):
        """Test 8-connectivity with negative coordinates."""
        index = (-2, -3)
        neighbors = list(iter_neighbor_indices(index, connectivity=8))
        
        expected = [
            (-1, -3), (-3, -3), (-2, -2), (-2, -4),  # 4-connectivity
            (-1, -2), (-1, -4), (-3, -2), (-3, -4),  # diagonal
        ]
        assert set(neighbors) == set(expected)
        assert len(neighbors) == 8

    def test_invalid_connectivity(self):
        """Test that invalid connectivity raises ValueError."""
        index = (0, 0)
        
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            list(iter_neighbor_indices(index, connectivity=0))
        
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            list(iter_neighbor_indices(index, connectivity=6))
        
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            list(iter_neighbor_indices(index, connectivity=16))

    def test_iter_neighbor_indices_is_iterator(self):
        """Test that iter_neighbor_indices returns an iterator."""
        index = (0, 0)
        neighbors = iter_neighbor_indices(index, connectivity=4)
        
        # Should be an iterator
        assert hasattr(neighbors, '__iter__')
        assert hasattr(neighbors, '__next__')
        
        # Should be consumable
        first = next(neighbors)
        assert isinstance(first, tuple)
        assert len(first) == 2


class TestIterExistingNeighbors:
    """Tests for iter_existing_neighbors function."""

    def test_all_neighbors_exist_four_connectivity(self):
        """Test when all 4 neighbors exist."""
        cell = Cell(index=(5, 5))
        cells = {
            (5, 5): cell,
            (6, 5): Cell(index=(6, 5)),
            (4, 5): Cell(index=(4, 5)),
            (5, 6): Cell(index=(5, 6)),
            (5, 4): Cell(index=(5, 4)),
        }
        
        neighbors = list(iter_existing_neighbors(cell, cells, connectivity=4))
        assert len(neighbors) == 4
        assert all(isinstance(n, Cell) for n in neighbors)
        assert set(n.index for n in neighbors) == {(6, 5), (4, 5), (5, 6), (5, 4)}

    def test_some_neighbors_exist_four_connectivity(self):
        """Test when only some 4 neighbors exist."""
        cell = Cell(index=(5, 5))
        cells = {
            (5, 5): cell,
            (6, 5): Cell(index=(6, 5)),
            (5, 6): Cell(index=(5, 6)),
            # (4, 5) and (5, 4) don't exist
        }
        
        neighbors = list(iter_existing_neighbors(cell, cells, connectivity=4))
        assert len(neighbors) == 2
        assert set(n.index for n in neighbors) == {(6, 5), (5, 6)}

    def test_no_neighbors_exist(self):
        """Test when no neighbors exist."""
        cell = Cell(index=(5, 5))
        cells = {
            (5, 5): cell,
        }
        
        neighbors = list(iter_existing_neighbors(cell, cells, connectivity=4))
        assert len(neighbors) == 0

    def test_all_neighbors_exist_eight_connectivity(self):
        """Test when all 8 neighbors exist."""
        cell = Cell(index=(5, 5))
        cells = {
            (5, 5): cell,
            (6, 5): Cell(index=(6, 5)),
            (4, 5): Cell(index=(4, 5)),
            (5, 6): Cell(index=(5, 6)),
            (5, 4): Cell(index=(5, 4)),
            (6, 6): Cell(index=(6, 6)),
            (6, 4): Cell(index=(6, 4)),
            (4, 6): Cell(index=(4, 6)),
            (4, 4): Cell(index=(4, 4)),
        }
        
        neighbors = list(iter_existing_neighbors(cell, cells, connectivity=8))
        assert len(neighbors) == 8
        expected_indices = {
            (6, 5), (4, 5), (5, 6), (5, 4),
            (6, 6), (6, 4), (4, 6), (4, 4),
        }
        assert set(n.index for n in neighbors) == expected_indices

    def test_some_neighbors_exist_eight_connectivity(self):
        """Test when only some 8 neighbors exist."""
        cell = Cell(index=(5, 5))
        cells = {
            (5, 5): cell,
            (6, 5): Cell(index=(6, 5)),
            (5, 6): Cell(index=(5, 6)),
            (6, 6): Cell(index=(6, 6)),
            # Others don't exist
        }
        
        neighbors = list(iter_existing_neighbors(cell, cells, connectivity=8))
        assert len(neighbors) == 3
        assert set(n.index for n in neighbors) == {(6, 5), (5, 6), (6, 6)}

    def test_negative_coordinates(self):
        """Test with negative cell coordinates."""
        cell = Cell(index=(-5, -5))
        cells = {
            (-5, -5): cell,
            (-4, -5): Cell(index=(-4, -5)),
            (-5, -4): Cell(index=(-5, -4)),
        }
        
        neighbors = list(iter_existing_neighbors(cell, cells, connectivity=4))
        assert len(neighbors) == 2
        assert set(n.index for n in neighbors) == {(-4, -5), (-5, -4)}

    def test_iter_existing_neighbors_is_iterator(self):
        """Test that iter_existing_neighbors returns an iterator."""
        cell = Cell(index=(0, 0))
        cells = {
            (0, 0): cell,
            (1, 0): Cell(index=(1, 0)),
        }
        
        neighbors = iter_existing_neighbors(cell, cells, connectivity=4)
        
        # Should be an iterator
        assert hasattr(neighbors, '__iter__')
        assert hasattr(neighbors, '__next__')
        
        # Should be consumable
        first = next(neighbors)
        assert isinstance(first, Cell)

    def test_invalid_connectivity_propagates(self):
        """Test that invalid connectivity in iter_existing_neighbors raises ValueError."""
        cell = Cell(index=(0, 0))
        cells = {(0, 0): cell}
        
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            list(iter_existing_neighbors(cell, cells, connectivity=6))


class TestHeightDiff:
    """Tests for height_diff function."""

    def test_positive_diff(self):
        """Test height difference when neighbor is higher."""
        cell = Cell(index=(0, 0), min_z=1.0)
        neighbor = Cell(index=(1, 0), min_z=3.0)
        
        diff = height_diff(cell, neighbor)
        assert diff == 2.0

    def test_negative_diff(self):
        """Test height difference when neighbor is lower."""
        cell = Cell(index=(0, 0), min_z=5.0)
        neighbor = Cell(index=(1, 0), min_z=2.0)
        
        diff = height_diff(cell, neighbor)
        assert diff == -3.0

    def test_zero_diff(self):
        """Test height difference when cells are at same height."""
        cell = Cell(index=(0, 0), min_z=5.0)
        neighbor = Cell(index=(1, 0), min_z=5.0)
        
        diff = height_diff(cell, neighbor)
        assert diff == 0.0

    def test_negative_heights(self):
        """Test height difference with negative z values."""
        cell = Cell(index=(0, 0), min_z=-5.0)
        neighbor = Cell(index=(1, 0), min_z=-2.0)
        
        diff = height_diff(cell, neighbor)
        assert diff == 3.0

    def test_mixed_positive_negative(self):
        """Test height difference with mixed positive and negative."""
        cell = Cell(index=(0, 0), min_z=-1.0)
        neighbor = Cell(index=(1, 0), min_z=1.0)
        
        diff = height_diff(cell, neighbor)
        assert diff == 2.0

    def test_cell_min_z_none(self):
        """Test that missing cell.min_z raises ValueError."""
        cell = Cell(index=(0, 0), min_z=None)
        neighbor = Cell(index=(1, 0), min_z=1.0)
        
        with pytest.raises(ValueError, match="Cell height features are not computed"):
            height_diff(cell, neighbor)

    def test_neighbor_min_z_none(self):
        """Test that missing neighbor.min_z raises ValueError."""
        cell = Cell(index=(0, 0), min_z=1.0)
        neighbor = Cell(index=(1, 0), min_z=None)
        
        with pytest.raises(ValueError, match="Cell height features are not computed"):
            height_diff(cell, neighbor)

    def test_both_min_z_none(self):
        """Test that missing both min_z raises ValueError."""
        cell = Cell(index=(0, 0), min_z=None)
        neighbor = Cell(index=(1, 0), min_z=None)
        
        with pytest.raises(ValueError, match="Cell height features are not computed"):
            height_diff(cell, neighbor)

    def test_asymmetry(self):
        """Test that height_diff is asymmetric (neighbor - cell)."""
        cell = Cell(index=(0, 0), min_z=1.0)
        neighbor = Cell(index=(1, 0), min_z=3.0)
        
        diff1 = height_diff(cell, neighbor)
        diff2 = height_diff(neighbor, cell)
        
        assert diff1 == 2.0
        assert diff2 == -2.0
        assert diff1 == -diff2


class TestAbsHeightDiff:
    """Tests for abs_height_diff function."""

    def test_positive_diff(self):
        """Test absolute height difference when neighbor is higher."""
        cell = Cell(index=(0, 0), min_z=1.0)
        neighbor = Cell(index=(1, 0), min_z=3.0)
        
        diff = abs_height_diff(cell, neighbor)
        assert diff == 2.0

    def test_negative_diff(self):
        """Test absolute height difference when neighbor is lower."""
        cell = Cell(index=(0, 0), min_z=5.0)
        neighbor = Cell(index=(1, 0), min_z=2.0)
        
        diff = abs_height_diff(cell, neighbor)
        assert diff == 3.0  # Should be absolute value

    def test_zero_diff(self):
        """Test absolute height difference when cells are at same height."""
        cell = Cell(index=(0, 0), min_z=5.0)
        neighbor = Cell(index=(1, 0), min_z=5.0)
        
        diff = abs_height_diff(cell, neighbor)
        assert diff == 0.0

    def test_symmetry(self):
        """Test that abs_height_diff is symmetric."""
        cell = Cell(index=(0, 0), min_z=1.0)
        neighbor = Cell(index=(1, 0), min_z=3.0)
        
        diff1 = abs_height_diff(cell, neighbor)
        diff2 = abs_height_diff(neighbor, cell)
        
        assert diff1 == diff2
        assert diff1 == 2.0

    def test_negative_heights(self):
        """Test absolute height difference with negative z values."""
        cell = Cell(index=(0, 0), min_z=-5.0)
        neighbor = Cell(index=(1, 0), min_z=-2.0)
        
        diff = abs_height_diff(cell, neighbor)
        assert diff == 3.0

    def test_missing_features_propagates(self):
        """Test that missing features in abs_height_diff raises ValueError."""
        cell = Cell(index=(0, 0), min_z=None)
        neighbor = Cell(index=(1, 0), min_z=1.0)
        
        with pytest.raises(ValueError, match="Cell height features are not computed"):
            abs_height_diff(cell, neighbor)


class TestMeanHeightDiff:
    """Tests for mean_height_diff function."""

    def test_positive_diff(self):
        """Test mean height difference when neighbor is higher."""
        cell = Cell(index=(0, 0), mean_z=1.0)
        neighbor = Cell(index=(1, 0), mean_z=3.0)
        
        diff = mean_height_diff(cell, neighbor)
        assert diff == 2.0

    def test_negative_diff(self):
        """Test mean height difference when neighbor is lower."""
        cell = Cell(index=(0, 0), mean_z=5.0)
        neighbor = Cell(index=(1, 0), mean_z=2.0)
        
        diff = mean_height_diff(cell, neighbor)
        assert diff == -3.0

    def test_zero_diff(self):
        """Test mean height difference when cells are at same height."""
        cell = Cell(index=(0, 0), mean_z=5.0)
        neighbor = Cell(index=(1, 0), mean_z=5.0)
        
        diff = mean_height_diff(cell, neighbor)
        assert diff == 0.0

    def test_negative_heights(self):
        """Test mean height difference with negative z values."""
        cell = Cell(index=(0, 0), mean_z=-5.0)
        neighbor = Cell(index=(1, 0), mean_z=-2.0)
        
        diff = mean_height_diff(cell, neighbor)
        assert diff == 3.0

    def test_cell_mean_z_none(self):
        """Test that missing cell.mean_z raises ValueError."""
        cell = Cell(index=(0, 0), mean_z=None)
        neighbor = Cell(index=(1, 0), mean_z=1.0)
        
        with pytest.raises(ValueError, match="Cell height features are not computed"):
            mean_height_diff(cell, neighbor)

    def test_neighbor_mean_z_none(self):
        """Test that missing neighbor.mean_z raises ValueError."""
        cell = Cell(index=(0, 0), mean_z=1.0)
        neighbor = Cell(index=(1, 0), mean_z=None)
        
        with pytest.raises(ValueError, match="Cell height features are not computed"):
            mean_height_diff(cell, neighbor)

    def test_both_mean_z_none(self):
        """Test that missing both mean_z raises ValueError."""
        cell = Cell(index=(0, 0), mean_z=None)
        neighbor = Cell(index=(1, 0), mean_z=None)
        
        with pytest.raises(ValueError, match="Cell height features are not computed"):
            mean_height_diff(cell, neighbor)

    def test_asymmetry(self):
        """Test that mean_height_diff is asymmetric (neighbor - cell)."""
        cell = Cell(index=(0, 0), mean_z=1.0)
        neighbor = Cell(index=(1, 0), mean_z=3.0)
        
        diff1 = mean_height_diff(cell, neighbor)
        diff2 = mean_height_diff(neighbor, cell)
        
        assert diff1 == 2.0
        assert diff2 == -2.0
        assert diff1 == -diff2

    def test_difference_from_min_z(self):
        """Test that mean_height_diff can differ from height_diff (min_z)."""
        cell = Cell(index=(0, 0), min_z=1.0, mean_z=2.0)
        neighbor = Cell(index=(1, 0), min_z=3.0, mean_z=4.0)
        
        min_diff = height_diff(cell, neighbor)  # Uses min_z
        mean_diff = mean_height_diff(cell, neighbor)  # Uses mean_z
        
        assert min_diff == 2.0
        assert mean_diff == 2.0
        # In this case they're the same, but conceptually different

    def test_different_min_and_mean(self):
        """Test mean_height_diff when min_z and mean_z differ."""
        cell = Cell(index=(0, 0), min_z=1.0, mean_z=3.0)
        neighbor = Cell(index=(1, 0), min_z=2.0, mean_z=5.0)
        
        min_diff = height_diff(cell, neighbor)  # 2.0 - 1.0 = 1.0
        mean_diff = mean_height_diff(cell, neighbor)  # 5.0 - 3.0 = 2.0
        
        assert min_diff == 1.0
        assert mean_diff == 2.0
        assert min_diff != mean_diff

