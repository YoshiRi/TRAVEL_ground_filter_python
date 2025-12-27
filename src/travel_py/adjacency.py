from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple

from .types import Cell


Index2D = Tuple[int, int]


# =========================
# Neighbor definitions
# =========================

FOUR_NEIGHBOR_OFFSETS: List[Index2D] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
]

EIGHT_NEIGHBOR_OFFSETS: List[Index2D] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]


# =========================
# Adjacency utilities
# =========================

def iter_neighbor_indices(
    index: Index2D,
    *,
    connectivity: int = 4,
) -> Iterator[Index2D]:
    """
    Yield neighbor indices around a given cell index.

    Parameters
    ----------
    index:
        (ix, iy) of the reference cell.
    connectivity:
        4 or 8.

    Yields
    ------
    (ix, iy) of neighbor cells.
    """
    if connectivity == 4:
        offsets = FOUR_NEIGHBOR_OFFSETS
    elif connectivity == 8:
        offsets = EIGHT_NEIGHBOR_OFFSETS
    else:
        raise ValueError("connectivity must be 4 or 8")

    ix, iy = index
    for dx, dy in offsets:
        yield (ix + dx, iy + dy)


def iter_existing_neighbors(
    cell: Cell,
    cells: Dict[Index2D, Cell],
    *,
    connectivity: int = 4,
) -> Iterator[Cell]:
    """
    Yield neighboring Cell objects that actually exist in the grid.

    Parameters
    ----------
    cell:
        Reference cell.
    cells:
        Mapping from index to Cell.
    connectivity:
        4 or 8.
    """
    for n_idx in iter_neighbor_indices(cell.index, connectivity=connectivity):
        neighbor = cells.get(n_idx)
        if neighbor is not None:
            yield neighbor


# =========================
# Difference computation
# =========================

def height_diff(
    cell: Cell,
    neighbor: Cell,
) -> float:
    """
    Compute height difference between two cells.

    Definition:
        neighbor.min_z - cell.min_z

    This asymmetry is intentional and traversal-side semantics
    should decide how to interpret it.
    """
    if cell.min_z is None or neighbor.min_z is None:
        raise ValueError("Cell height features are not computed")

    return neighbor.min_z - cell.min_z


def abs_height_diff(
    cell: Cell,
    neighbor: Cell,
) -> float:
    """
    Absolute height difference based on min_z.
    """
    return abs(height_diff(cell, neighbor))


def mean_height_diff(
    cell: Cell,
    neighbor: Cell,
) -> float:
    """
    Difference based on mean_z instead of min_z.
    """
    if cell.mean_z is None or neighbor.mean_z is None:
        raise ValueError("Cell height features are not computed")

    return neighbor.mean_z - cell.mean_z
