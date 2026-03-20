from __future__ import annotations

from typing import List

from .grid import Grid
from .types import SubCellIndex


# Strict TGS edge-sharing connectivity:
# Each tri connects externally only to the one tri of the adjacent cell
# that shares the same cell-boundary edge.
#
# Triangle assignments (from grid.py, based on arctan2 angle from cell center):
#   tri 0 – East  [−π/4,  π/4)  → right cell (i+1, j), tri 2 (West)
#   tri 1 – North [ π/4, 3π/4)  → top   cell (i, j+1), tri 3 (South)
#   tri 2 – West  (rest)         → left  cell (i−1, j), tri 0 (East)
#   tri 3 – South [−3π/4,−π/4)  → bottom cell (i, j−1), tri 1 (North)
#
_EXTERNAL_NEIGHBOR: dict[int, tuple[int, int, int]] = {
    0: (+1,  0, 2),
    1: ( 0, +1, 3),
    2: (-1,  0, 0),
    3: ( 0, -1, 1),
}


class TraversabilityGraph:
    def __init__(self, grid: Grid):
        self.grid = grid

    def get_neighbor_subcells(self, idx: SubCellIndex) -> List[SubCellIndex]:
        """
        Return strict TGS neighbors of *idx*.

        Internal (same cell, center-sharing):
            All 3 other triangles in the same cell.

        External (edge-sharing across cell boundary):
            Exactly one triangle in the directly adjacent cell,
            determined by which cell edge the triangle faces.
        """
        neighbors: List[SubCellIndex] = []

        # 1. Internal: the other 3 tris in the same cell share the center vertex.
        for t in range(4):
            if t != idx.tri:
                neighbors.append(SubCellIndex(idx.i, idx.j, t))

        # 2. External: the single adjacent tri that shares the boundary edge.
        di, dj, ext_tri = _EXTERNAL_NEIGHBOR[idx.tri]
        ni, nj = idx.i + di, idx.j + dj

        if self.grid._in_bounds(ni, nj) and (ni, nj) in self.grid.cells:
            neighbors.append(SubCellIndex(ni, nj, ext_tri))

        return neighbors
