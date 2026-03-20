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
    Propagate labels to point-wise labels at SubCell granularity.

    When a SubCell has point_indices populated (the normal case when the grid
    was built via Grid.from_points), each point is labeled according to its
    *own* SubCell's label rather than the coarser Cell.state.  This allows
    one part of a cell to be GROUND while another is NON_GROUND.

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
        # Check whether any subcell carries fine-grained index information.
        has_subcell_indices = any(
            bool(sub.point_indices) for sub in cell.subcells.values()
        )

        if has_subcell_indices:
            # Fine-grained: label each point based on its own subcell.
            for sub in cell.subcells.values():
                lbl = _state_to_label.get(sub.label, POINT_LABEL_UNKNOWN)
                for pi in sub.point_indices:
                    labels[pi] = lbl
        else:
            # Coarse fallback: use Cell.state for all points in the cell.
            lbl = _state_to_label.get(cell.state, POINT_LABEL_UNKNOWN)
            for pi in cell.point_indices:
                labels[pi] = lbl

    return labels
