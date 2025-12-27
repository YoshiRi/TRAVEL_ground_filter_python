from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# =========================
# Enums
# =========================

class CellState(Enum):
    """
    State of a grid cell during Travel traversal.
    """
    UNKNOWN = auto()
    GROUND = auto()
    NON_GROUND = auto()
    REJECTED = auto()


class RejectReason(Enum):
    """
    Reason why a cell was rejected during traversal.
    Used for debugging and visualization.
    """
    NONE = auto()

    HEIGHT_DIFF_TOO_LARGE = auto()
    SLOPE_TOO_STEEP = auto()
    NO_VALID_NEIGHBOR = auto()
    INVALID_FEATURE = auto()

    VISITED = auto()
    OUT_OF_BOUNDS = auto()


# =========================
# Core Data Structures
# =========================

@dataclass
class Cell:
    """
    Grid cell representation.
    This object holds *state*, not logic.
    """
    index: Tuple[int, int]  # (ix, iy)

    # Feature values (computed in cell_features.py)
    min_z: Optional[float] = None
    mean_z: Optional[float] = None
    max_z: Optional[float] = None
    height_range: Optional[float] = None
    slope: Optional[float] = None

    # Travel-related state
    state: CellState = CellState.UNKNOWN
    reject_reason: RejectReason = RejectReason.NONE

    # Traversal metadata
    visited: bool = False
    iteration: Optional[int] = None
    parent: Optional[Tuple[int, int]] = None

    # Indices of original points belonging to this cell
    point_indices: List[int] = field(default_factory=list)


@dataclass
class TraversalState:
    """
    Global traversal state for one Travel run.
    Used to track progress and enable debugging.
    """
    iteration: int = 0

    # Frontier and visited cells
    queue: List[Tuple[int, int]] = field(default_factory=list)
    visited: Dict[Tuple[int, int], Cell] = field(default_factory=dict)

    # Statistics
    num_ground: int = 0
    num_rejected: int = 0

    # History (optional, but useful for visualization)
    iteration_history: Dict[int, List[Tuple[int, int]]] = field(
        default_factory=dict
    )
