from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .types import Cell, SubCell


Index2D = Tuple[int, int]


@dataclass
class GridSpec:
    """
    Grid definition.
    - resolution: cell size [m]
    - origin_xy: world coordinate of grid index (0,0) cell's origin [m]
      (i.e., world_x = origin_x + ix * resolution)
    - size_xy: number of cells in x/y direction (nx, ny)
      If specified, points outside the grid are dropped.
    """
    resolution: float
    origin_xy: Tuple[float, float] = (0.0, 0.0)
    size_xy: Optional[Tuple[int, int]] = None  # (nx, ny)


# =========================
# Point -> cell / triangle assignment (shared with Grid.from_points)
# =========================

def assign_cells_and_tris(
    points_xyz: np.ndarray, spec: GridSpec
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized cell index and TGS triangle assignment.

    Returns
    -------
    in_indices : (K,) int64   original indices of points inside the grid (ascending)
    ix, iy     : (K,) int64   cell indices of those points
    tri        : (K,) int64   triangle id 0..3 (see graph.py for the convention)
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError("points_xyz must be shape (N, 3) or (N, >=3)")
    if spec.resolution <= 0:
        raise ValueError("resolution must be > 0")

    ox, oy = spec.origin_xy
    r = spec.resolution
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    ix = np.floor((x - ox) / r).astype(np.int64)
    iy = np.floor((y - oy) / r).astype(np.int64)

    if spec.size_xy is not None:
        nx, ny = spec.size_xy
        in_mask = (0 <= ix) & (ix < nx) & (0 <= iy) & (iy < ny)
        in_indices = np.nonzero(in_mask)[0]
        ix, iy = ix[in_indices], iy[in_indices]
        x, y = x[in_indices], y[in_indices]
    else:
        in_indices = np.arange(points_xyz.shape[0], dtype=np.int64)

    # Angle from the cell center (identical arithmetic to the reference).
    center_x = (ox + ix * r) + r * 0.5
    center_y = (oy + iy * r) + r * 0.5
    angles = np.arctan2(y - center_y, x - center_x)

    q = np.pi / 4
    tri = np.full(angles.shape, 2, dtype=np.int64)
    tri[(angles >= q) & (angles < 3 * q)] = 1
    tri[(angles >= -q) & (angles < q)] = 0
    tri[(angles >= -3 * q) & (angles < -q)] = 3
    return in_indices, ix, iy, tri



class Grid:
    """
    Holds mapping from (ix, iy) -> Cell.

    Responsibility:
      - world (x,y) -> grid index (ix,iy)
      - bucket point indices per cell
      - provide stable data structure for downstream stages

    Non-responsibility:
      - ground / non-ground logic
      - cell feature computation (min_z, etc.)
    """

    def __init__(self, spec: GridSpec) -> None:
        if spec.resolution <= 0:
            raise ValueError("resolution must be > 0")
        self.spec = spec
        self.cells: Dict[Index2D, Cell] = {}

        # For quick access / debugging
        self.num_points_in_grid: int = 0
        self.dropped_point_indices: List[int] = []

    # -------------------------
    # Coordinate conversion
    # -------------------------

    def world_to_grid_index(self, x: float, y: float) -> Index2D:
        """
        Convert world coordinates to grid indices (ix, iy).
        """
        ox, oy = self.spec.origin_xy
        r = self.spec.resolution
        ix = int(np.floor((x - ox) / r))
        iy = int(np.floor((y - oy) / r))
        return (ix, iy)

    def grid_index_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """
        Convert grid indices to world coordinates of the cell origin.
        """
        ox, oy = self.spec.origin_xy
        r = self.spec.resolution
        return (ox + ix * r, oy + iy * r)

    def _in_bounds(self, ix: int, iy: int) -> bool:
        """
        Check whether (ix, iy) is inside grid size if size is specified.
        """
        if self.spec.size_xy is None:
            return True
        nx, ny = self.spec.size_xy
        return (0 <= ix < nx) and (0 <= iy < ny)

    # -------------------------
    # Cell access
    # -------------------------

    def get_cell(self, index: Index2D) -> Optional[Cell]:
        return self.cells.get(index)

    def get_or_create_cell(self, index: Index2D) -> Cell:
        cell = self.cells.get(index)
        if cell is None:
            cell = Cell(index=index)
            self.cells[index] = cell
        return cell

    def iter_cells(self) -> Iterable[Cell]:
        return self.cells.values()

    # -------------------------
    # Build from point cloud
    # -------------------------

    @classmethod
    def from_points(
        cls,
        points_xyz: np.ndarray,
        spec: GridSpec,
        *,
        keep_dropped_indices: bool = True,
    ) -> "Grid":
        """
        Build a Grid from Nx3 (or Nx>=3) numpy array.
        Points are bucketed by (ix, iy) into Cell.point_indices.

        Parameters
        ----------
        points_xyz:
            numpy array shaped (N, 3) or (N, >=3). Uses [:,0]=x, [:,1]=y, [:,2]=z.
        spec:
            GridSpec
        keep_dropped_indices:
            If True, store indices of points outside bounds in dropped_point_indices.

        Returns
        -------
        Grid
        """
        if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
            raise ValueError("points_xyz must be shape (N, 3) or (N, >=3)")

        grid = cls(spec)
        n_total = points_xyz.shape[0]

        in_indices, ix, iy, tri = assign_cells_and_tris(points_xyz, spec)

        if keep_dropped_indices and in_indices.size < n_total:
            grid.dropped_point_indices = np.setdiff1d(
                np.arange(n_total, dtype=np.int64), in_indices, assume_unique=True
            ).tolist()

        grid.num_points_in_grid = int(in_indices.size)
        if in_indices.size == 0:
            return grid

        # Vectorized bucketing: group points by (cell, tri) with a *stable* sort so
        # that indices inside each Cell / SubCell stay in ascending original order.
        ix0, iy0 = int(ix.min()), int(iy.min())
        ny = int(iy.max()) - iy0 + 1
        cell_key = (ix - ix0) * ny + (iy - iy0)

        # Cells are inserted in first-appearance order (stable dict order).
        ucell, first_pos = np.unique(cell_key, return_index=True)
        for c in ucell[np.argsort(first_pos)].tolist():
            grid.get_or_create_cell((c // ny + ix0, c % ny + iy0))

        # Cell.point_indices
        order_c = np.argsort(cell_key, kind="stable")
        _, start_c, count_c = np.unique(cell_key[order_c], return_index=True, return_counts=True)
        idx_c = in_indices[order_c].tolist()
        for c, s, n in zip(ucell.tolist(), start_c.tolist(), count_c.tolist()):
            grid.cells[(c // ny + ix0, c % ny + iy0)].point_indices = idx_c[s:s + n]

        # TGS SubCells (always 4 per cell, possibly empty)
        empty = points_xyz[:0]
        for cell in grid.iter_cells():
            for t_id in range(4):
                cell.subcells[t_id] = SubCell(points=empty.copy(), point_indices=[])

        key = cell_key * 4 + tri
        order_s = np.argsort(key, kind="stable")
        uniq_s, start_s, count_s = np.unique(key[order_s], return_index=True, return_counts=True)
        sorted_idx = in_indices[order_s]
        sorted_pts = points_xyz[sorted_idx]
        sorted_idx_l = sorted_idx.tolist()
        for k, s, n in zip(uniq_s.tolist(), start_s.tolist(), count_s.tolist()):
            c, t_id = divmod(k, 4)
            cell = grid.cells[(c // ny + ix0, c % ny + iy0)]
            cell.subcells[t_id] = SubCell(
                points=sorted_pts[s:s + n], point_indices=sorted_idx_l[s:s + n]
            )

        return grid
