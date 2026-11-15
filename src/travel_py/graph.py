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

# 14-neighbor external connectivity from official TRAVEL (tgs.hpp L680-770 searchNeighborNodes):
# Connects not only to the edge-sharing adjacent triangle, but also to adjacent
# triangles across vertices/corners (total 11 external triangles per subcell).
_EXTERNAL_14_NEIGHBORS: dict[int, tuple[tuple[int, int, int], ...]] = {
    0: (
        (+1, +1, 2), (+1, +1, 3),
        (+1,  0, 1), (+1,  0, 2), (+1,  0, 3),
        (+1, -1, 1), (+1, -1, 2),
        ( 0, +1, 0), ( 0, +1, 3),
        ( 0, -1, 0), ( 0, -1, 1),
    ),
    1: (
        (+1, +1, 2), (+1, +1, 3),
        (+1,  0, 1), (+1,  0, 2),
        ( 0, +1, 0), ( 0, +1, 2), ( 0, +1, 3),
        (-1, +1, 0), (-1, +1, 3),
        (-1,  0, 0), (-1,  0, 1),
    ),
    2: (
        ( 0, +1, 2), ( 0, +1, 3),
        ( 0, -1, 1), ( 0, -1, 2),
        (-1, +1, 0), (-1, +1, 3),
        (-1,  0, 0), (-1,  0, 1), (-1,  0, 3),
        (-1, -1, 0), (-1, -1, 1),
    ),
    3: (
        (+1,  0, 2), (+1,  0, 3),
        (+1, -1, 1), (+1, -1, 2),
        ( 0, -1, 0), ( 0, -1, 1), ( 0, -1, 2),
        (-1,  0, 0), (-1,  0, 3),
        (-1, -1, 0), (-1, -1, 1),
    ),
}


class TraversabilityGraph:
    def __init__(self, grid: Grid, use_14_neighbors: bool = False):
        self.grid = grid
        self.use_14_neighbors = use_14_neighbors

    def get_neighbor_subcells(self, idx: SubCellIndex) -> List[SubCellIndex]:
        """
        Return neighbors of *idx*.

        Internal (same cell, center-sharing):
            All 3 other triangles in the same cell.

        External:
            - If use_14_neighbors=False (strict 4-neighbors):
              Exactly one triangle in the directly adjacent cell sharing the boundary edge.
            - If use_14_neighbors=True (official 14-neighbors):
              Up to 11 triangles in edge- and corner-adjacent cells.
        """
        neighbors: List[SubCellIndex] = []

        # 1. Internal: the other 3 tris in the same cell share the center vertex.
        for t in range(4):
            if t != idx.tri:
                neighbors.append(SubCellIndex(idx.i, idx.j, t))

        # 2. External neighbors
        if self.use_14_neighbors:
            for di, dj, ext_tri in _EXTERNAL_14_NEIGHBORS[idx.tri]:
                ni, nj = idx.i + di, idx.j + dj
                if self.grid._in_bounds(ni, nj) and (ni, nj) in self.grid.cells:
                    neighbors.append(SubCellIndex(ni, nj, ext_tri))
        else:
            di, dj, ext_tri = _EXTERNAL_NEIGHBOR[idx.tri]
            ni, nj = idx.i + di, idx.j + dj
            if self.grid._in_bounds(ni, nj) and (ni, nj) in self.grid.cells:
                neighbors.append(SubCellIndex(ni, nj, ext_tri))

        return neighbors
