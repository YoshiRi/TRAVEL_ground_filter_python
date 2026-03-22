"""Tests for labeling.py — label_points_from_cells (fine-grained + fallback)."""

import numpy as np
import pytest

from travel_py.labeling import (
    label_points_from_cells,
    POINT_LABEL_GROUND,
    POINT_LABEL_NON_GROUND,
    POINT_LABEL_UNKNOWN,
)
from travel_py.types import Cell, SubCell, CellState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell(ix: int, iy: int, point_indices: list[int], state: CellState = CellState.UNKNOWN) -> Cell:
    c = Cell(index=(ix, iy), state=state, point_indices=point_indices)
    return c


def _subcell(label: CellState, point_indices: list[int]) -> SubCell:
    pts = np.zeros((len(point_indices), 3))
    return SubCell(points=pts, label=label, point_indices=point_indices)


def _ground_subcell_with_plane(
    point_indices: list[int],
    point_z: list[float],
    plane_z: float = 0.0,
) -> SubCell:
    """GROUND SubCell with a horizontal plane at z=plane_z and points at specified heights."""
    # Horizontal plane: normal=(0,0,1), d=-plane_z
    pts = np.zeros((len(point_indices), 3))
    pts[:, 2] = point_z
    sub = SubCell(points=pts, label=CellState.GROUND, point_indices=point_indices)
    sub.normal = np.array([0.0, 0.0, 1.0])
    sub.mean = np.array([0.0, 0.0, plane_z])
    sub.d = -plane_z
    return sub


# ---------------------------------------------------------------------------
# Fine-grained path (SubCell.point_indices populated)
# ---------------------------------------------------------------------------

class TestFineGrainedPropagation:
    def test_all_ground_subcells(self):
        cell = _cell(0, 0, [0, 1, 2, 3])
        cell.subcells[0] = _subcell(CellState.GROUND, [0, 1])
        cell.subcells[1] = _subcell(CellState.GROUND, [2, 3])
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=4)
        assert np.all(labels == POINT_LABEL_GROUND)

    def test_mixed_subcells_in_same_cell(self):
        """Half GROUND, half NON_GROUND → different labels within same cell."""
        cell = _cell(0, 0, [0, 1, 2, 3])
        cell.subcells[0] = _subcell(CellState.GROUND, [0, 1])
        cell.subcells[1] = _subcell(CellState.NON_GROUND, [2, 3])
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=4)
        assert labels[0] == POINT_LABEL_GROUND
        assert labels[1] == POINT_LABEL_GROUND
        assert labels[2] == POINT_LABEL_NON_GROUND
        assert labels[3] == POINT_LABEL_NON_GROUND

    def test_unknown_subcell_stays_unknown(self):
        cell = _cell(0, 0, [0])
        cell.subcells[0] = _subcell(CellState.UNKNOWN, [0])
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=1)
        assert labels[0] == POINT_LABEL_UNKNOWN

    def test_rejected_subcell_mapped_to_non_ground(self):
        cell = _cell(0, 0, [0])
        cell.subcells[0] = _subcell(CellState.REJECTED, [0])
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=1)
        assert labels[0] == POINT_LABEL_NON_GROUND

    def test_empty_subcell_point_indices_skipped(self):
        """SubCell with no point_indices should not write any labels."""
        cell = _cell(0, 0, [0, 1])
        cell.subcells[0] = _subcell(CellState.GROUND, [0, 1])
        cell.subcells[1] = _subcell(CellState.NON_GROUND, [])  # empty
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=2)
        assert labels[0] == POINT_LABEL_GROUND
        assert labels[1] == POINT_LABEL_GROUND  # not overwritten

    def test_multiple_cells_fine_grained(self):
        cell_a = _cell(0, 0, [0, 1])
        cell_a.subcells[0] = _subcell(CellState.GROUND, [0, 1])

        cell_b = _cell(1, 0, [2, 3])
        cell_b.subcells[0] = _subcell(CellState.NON_GROUND, [2, 3])

        labels = label_points_from_cells(
            cells={(0, 0): cell_a, (1, 0): cell_b}, num_points=4
        )
        assert labels[0] == POINT_LABEL_GROUND
        assert labels[1] == POINT_LABEL_GROUND
        assert labels[2] == POINT_LABEL_NON_GROUND
        assert labels[3] == POINT_LABEL_NON_GROUND


# ---------------------------------------------------------------------------
# Coarse fallback (no SubCell.point_indices)
# ---------------------------------------------------------------------------

class TestCoarseFallback:
    def test_ground_cell_labels_all_points_ground(self):
        cell = _cell(0, 0, [0, 1, 2], state=CellState.GROUND)
        # No subcells with point_indices → coarse fallback
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=3)
        assert np.all(labels == POINT_LABEL_GROUND)

    def test_non_ground_cell_labels_all_non_ground(self):
        cell = _cell(0, 0, [0, 1], state=CellState.NON_GROUND)
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=2)
        assert np.all(labels == POINT_LABEL_NON_GROUND)

    def test_rejected_cell_mapped_to_non_ground(self):
        cell = _cell(0, 0, [0], state=CellState.REJECTED)
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=1)
        assert labels[0] == POINT_LABEL_NON_GROUND

    def test_unknown_cell_stays_unknown(self):
        cell = _cell(0, 0, [0], state=CellState.UNKNOWN)
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=1)
        assert labels[0] == POINT_LABEL_UNKNOWN

    def test_empty_cell_no_points_written(self):
        cell = _cell(0, 0, [], state=CellState.GROUND)
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=3)
        # All stay at UNKNOWN (default fill value)
        assert np.all(labels == POINT_LABEL_UNKNOWN)


# ---------------------------------------------------------------------------
# Default fill value and output shape
# ---------------------------------------------------------------------------

class TestOutputProperties:
    def test_output_shape(self):
        labels = label_points_from_cells(cells={}, num_points=10)
        assert labels.shape == (10,)

    def test_unlabeled_points_default_to_unknown(self):
        """Points not covered by any cell should remain UNKNOWN."""
        labels = label_points_from_cells(cells={}, num_points=5)
        assert np.all(labels == POINT_LABEL_UNKNOWN)

    def test_output_dtype_is_int8(self):
        labels = label_points_from_cells(cells={}, num_points=4)
        assert labels.dtype == np.int8

    def test_labels_contain_only_valid_values(self):
        cell = _cell(0, 0, [0, 1, 2])
        cell.subcells[0] = _subcell(CellState.GROUND, [0])
        cell.subcells[1] = _subcell(CellState.NON_GROUND, [1])
        cell.subcells[2] = _subcell(CellState.UNKNOWN, [2])
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=3)
        valid = {POINT_LABEL_GROUND, POINT_LABEL_NON_GROUND, POINT_LABEL_UNKNOWN}
        assert set(labels.tolist()).issubset(valid)


# ---------------------------------------------------------------------------
# Plane-distance point classification (GROUND subcell with fitted plane)
# ---------------------------------------------------------------------------

class TestPlaneDistanceClassification:
    def test_points_on_plane_labeled_ground(self):
        """Points exactly on the fitted plane (dist=0) → GROUND."""
        cell = _cell(0, 0, [0, 1, 2])
        cell.subcells[0] = _ground_subcell_with_plane([0, 1, 2], [0.0, 0.0, 0.0], plane_z=0.0)
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=3)
        assert np.all(labels == POINT_LABEL_GROUND)

    def test_points_within_threshold_labeled_ground(self):
        """Points within th_dist_from_plane → GROUND."""
        cell = _cell(0, 0, [0, 1])
        cell.subcells[0] = _ground_subcell_with_plane([0, 1], [0.1, -0.1], plane_z=0.0)
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=2, th_dist_from_plane=0.2)
        assert np.all(labels == POINT_LABEL_GROUND)

    def test_points_above_threshold_labeled_non_ground(self):
        """Points farther than th_dist_from_plane above the plane → NON_GROUND (obstacles)."""
        cell = _cell(0, 0, [0, 1])
        # point 0 at z=0.0 (on plane), point 1 at z=1.0 (wall, 1m above)
        cell.subcells[0] = _ground_subcell_with_plane([0, 1], [0.0, 1.0], plane_z=0.0)
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=2, th_dist_from_plane=0.2)
        assert labels[0] == POINT_LABEL_GROUND
        assert labels[1] == POINT_LABEL_NON_GROUND

    def test_ground_and_wall_in_same_subcell_separated(self):
        """Ground points near z=0 and wall points at z=1.5 within the same GROUND subcell."""
        cell = _cell(0, 0, list(range(6)))
        ground_z = [0.0, 0.05, -0.05]   # close to plane
        wall_z   = [1.0, 1.5, 2.0]      # far above plane
        cell.subcells[0] = _ground_subcell_with_plane(
            list(range(6)), ground_z + wall_z, plane_z=0.0
        )
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=6, th_dist_from_plane=0.2)
        assert np.all(labels[:3] == POINT_LABEL_GROUND)
        assert np.all(labels[3:] == POINT_LABEL_NON_GROUND)

    def test_no_plane_falls_back_to_state_label(self):
        """GROUND subcell with normal=None uses state-based label (no per-point distance)."""
        cell = _cell(0, 0, [0, 1])
        # Use _subcell (no normal) instead of _ground_subcell_with_plane
        cell.subcells[0] = _subcell(CellState.GROUND, [0, 1])
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=2)
        assert np.all(labels == POINT_LABEL_GROUND)

    def test_th_dist_from_plane_default_is_0_2(self):
        """Default threshold is 0.2m — point at z=0.25 should be NON_GROUND."""
        cell = _cell(0, 0, [0, 1])
        cell.subcells[0] = _ground_subcell_with_plane([0, 1], [0.0, 0.25], plane_z=0.0)
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=2)
        assert labels[0] == POINT_LABEL_GROUND
        assert labels[1] == POINT_LABEL_NON_GROUND


# ---------------------------------------------------------------------------
# Fine-grained vs coarse: fine-grained wins when mixed within cell
# ---------------------------------------------------------------------------

class TestFineGrainedPriority:
    def test_cell_state_ground_but_subcells_have_indices(self):
        """When subcells have point_indices, Cell.state is ignored."""
        cell = _cell(0, 0, [0, 1], state=CellState.GROUND)
        # subcell overrides to NON_GROUND
        cell.subcells[0] = _subcell(CellState.NON_GROUND, [0, 1])
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=2)
        assert np.all(labels == POINT_LABEL_NON_GROUND)

    def test_mixed_subcells_some_with_some_without_indices(self):
        """Only subcells WITH point_indices trigger fine-grained path."""
        cell = _cell(0, 0, [0, 1, 2], state=CellState.GROUND)
        cell.subcells[0] = _subcell(CellState.GROUND, [0])        # has indices
        cell.subcells[1] = _subcell(CellState.NON_GROUND, [])     # empty indices
        cell.subcells[2] = _subcell(CellState.NON_GROUND, [1, 2]) # has indices
        labels = label_points_from_cells(cells={(0, 0): cell}, num_points=3)
        assert labels[0] == POINT_LABEL_GROUND
        assert labels[1] == POINT_LABEL_NON_GROUND
        assert labels[2] == POINT_LABEL_NON_GROUND
