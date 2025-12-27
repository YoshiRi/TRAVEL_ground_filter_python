from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .types import Cell


def compute_cell_height_features(
    cell: Cell,
    points_xyz: np.ndarray,
) -> None:
    """
    Compute basic height-related features for a single cell.

    This function MUTATES the given Cell instance by filling:
      - min_z
      - max_z
      - mean_z
      - height_range

    Parameters
    ----------
    cell:
        Cell whose point_indices are already populated.
    points_xyz:
        Original point cloud array of shape (N, 3) or (N, >=3).
    """
    if not cell.point_indices:
        # No points in this cell → leave features as None
        return

    z_values = points_xyz[cell.point_indices, 2]

    # Defensive check (NaN / inf handling can be extended later)
    if z_values.size == 0:
        return

    min_z = float(np.min(z_values))
    max_z = float(np.max(z_values))
    mean_z = float(np.mean(z_values))

    cell.min_z = min_z
    cell.max_z = max_z
    cell.mean_z = mean_z
    cell.height_range = max_z - min_z


def compute_all_cell_features(
    cells: Iterable[Cell],
    points_xyz: np.ndarray,
) -> None:
    """
    Compute height-related features for all cells.

    Parameters
    ----------
    cells:
        Iterable of Cell objects (e.g., grid.iter_cells()).
    points_xyz:
        Original point cloud array.
    """
    for cell in cells:
        compute_cell_height_features(cell, points_xyz)
