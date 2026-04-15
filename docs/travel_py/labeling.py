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
    th_dist_from_plane: float = 0.2,
) -> np.ndarray:
    """
    Propagate labels to point-wise labels at SubCell granularity.

    For GROUND subcells that have a fitted plane (normal + d), each point is
    classified individually by its signed distance from the plane:
        |dist| <= th_dist_from_plane  →  GROUND
        |dist|  > th_dist_from_plane  →  NON_GROUND  (obstacle above the plane)
    This eliminates wall points that happen to share a GROUND subcell.

    For NON_GROUND / UNKNOWN subcells, the subcell label is applied uniformly
    to all points.

    Fallback: if a cell has no subcells with point_indices (e.g. a manually
    constructed Cell), Cell.state is used for all points in the cell.

    Labels:
        Ground      →  1
        Non-ground  →  0
        Unknown     → -1
    """
    labels = np.full(
        shape=(num_points,),
        fill_value=POINT_LABEL_UNKNOWN,
        dtype=np.int8,
    )

    _state_to_label = {
        CellState.GROUND: POINT_LABEL_GROUND,
        CellState.NON_GROUND: POINT_LABEL_NON_GROUND,
        CellState.REJECTED: POINT_LABEL_NON_GROUND,
        CellState.UNKNOWN: POINT_LABEL_UNKNOWN,
    }

    for cell in cells.values():
        has_subcell_indices = any(
            bool(sub.point_indices) for sub in cell.subcells.values()
        )

        if has_subcell_indices:
            for sub in cell.subcells.values():
                if not sub.point_indices:
                    continue
                indices = np.array(sub.point_indices, dtype=np.int64)

                if sub.label == CellState.GROUND and sub.normal is not None:
                    # Point-level classification: distance from the fitted ground plane.
                    # Points within th_dist_from_plane of the plane → GROUND,
                    # points farther away (obstacles, noise) → NON_GROUND.
                    dists = sub.points @ sub.normal + sub.d  # signed, shape (K,)
                    ground_mask = np.abs(dists) <= th_dist_from_plane
                    labels[indices[ground_mask]] = POINT_LABEL_GROUND
                    labels[indices[~ground_mask]] = POINT_LABEL_NON_GROUND
                else:
                    lbl = _state_to_label.get(sub.label, POINT_LABEL_UNKNOWN)
                    labels[indices] = lbl
        else:
            # Coarse fallback: use Cell.state for all points in the cell.
            lbl = _state_to_label.get(cell.state, POINT_LABEL_UNKNOWN)
            for pi in cell.point_indices:
                labels[pi] = lbl

    return labels
