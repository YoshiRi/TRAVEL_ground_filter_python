from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .adjacency import height_diff
from .types import Cell, RejectReason


@dataclass
class TraversalAcceptConfig:
    max_height_diff: float
    max_slope: float | None = None
    cell_size: float | None = None

def accept_height_only(
    current: Cell,
    neighbor: Cell,
    *,
    config: TraversalAcceptConfig,
) -> Tuple[bool, RejectReason]:
    """
    Accept based only on height difference (min_z).

    Returns
    -------
    (accepted, reject_reason)
    """
    try:
        dz = height_diff(current, neighbor)
    except ValueError:
        return False, RejectReason.INVALID_FEATURE

    if abs(dz) > config.max_height_diff:
        return False, RejectReason.HEIGHT_DIFF_TOO_LARGE

    return True, RejectReason.NONE


def accept_height_and_slope(
    current: Cell,
    neighbor: Cell,
    *,
    config: TraversalAcceptConfig,
) -> Tuple[bool, RejectReason]:
    """
    Accept based on height difference and slope.
    """
    accepted, reason = accept_height_only(
        current, neighbor, config=config
    )
    if not accepted:
        return False, reason

    if config.max_slope is None:
        return True, RejectReason.NONE

    if config.cell_size is None:
        raise ValueError("cell_size must be set when using slope")

    # slope = |dz| / distance
    dz = abs(height_diff(current, neighbor))
    slope = dz / config.cell_size

    if slope > config.max_slope:
        return False, RejectReason.SLOPE_TOO_STEEP

    return True, RejectReason.NONE
