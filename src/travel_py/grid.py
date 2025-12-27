from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .types import Cell


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

        x = points_xyz[:, 0]
        y = points_xyz[:, 1]

        ox, oy = spec.origin_xy
        r = spec.resolution

        # Vectorized index computation
        ix = np.floor((x - ox) / r).astype(np.int64)
        iy = np.floor((y - oy) / r).astype(np.int64)

        if spec.size_xy is not None:
            nx, ny = spec.size_xy
            in_mask = (0 <= ix) & (ix < nx) & (0 <= iy) & (iy < ny)
            in_indices = np.nonzero(in_mask)[0]
            out_indices = np.nonzero(~in_mask)[0]
        else:
            in_indices = np.arange(points_xyz.shape[0], dtype=np.int64)
            out_indices = np.array([], dtype=np.int64)

        if keep_dropped_indices and out_indices.size > 0:
            grid.dropped_point_indices = out_indices.tolist()

        # Bucket point indices into cells
        # Approach: iterate over in-grid points; this is clear and sufficient for experimentation.
        # If needed, later optimize by sorting unique (ix,iy).
        for pi in in_indices.tolist():
            idx = (int(ix[pi]), int(iy[pi]))
            cell = grid.get_or_create_cell(idx)
            cell.point_indices.append(int(pi))

        grid.num_points_in_grid = int(in_indices.size)
        return grid
