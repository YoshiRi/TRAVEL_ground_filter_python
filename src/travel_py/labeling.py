from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .types import Cell, CellState

Index2D = Tuple[int, int]


# =========================
# Label definition
# =========================

POINT_LABEL_GROUND = 1
POINT_LABEL_NON_GROUND = 0
POINT_LABEL_UNKNOWN = -1


# =========================
# Label propagation
# =========================

def label_points_from_cells(
    *,
    cells: Dict[Index2D, Cell],
    num_points: int,
) -> np.ndarray:
    """
    Propagate cell labels to point-wise labels.

    Parameters
    ----------
    cells:
        Mapping from grid index to Cell.
    num_points:
        Total number of points in the original point cloud.

    Returns
    -------
    labels:
        np.ndarray of shape (num_points,)
        Ground: 1
        Non-ground: 0
        Unknown: -1
    """
    labels = np.full(
        shape=(num_points,),
        fill_value=POINT_LABEL_UNKNOWN,
        dtype=np.int8,
    )

    for cell in cells.values():
        if cell.state == CellState.GROUND:
            label = POINT_LABEL_GROUND
        elif cell.state in (CellState.NON_GROUND, CellState.REJECTED):
            label = POINT_LABEL_NON_GROUND
        else:
            label = POINT_LABEL_UNKNOWN

        for pi in cell.point_indices:
            labels[pi] = label

    return labels
