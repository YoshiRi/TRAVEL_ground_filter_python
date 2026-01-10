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
        
        # Pre-compute cell centers for subcell assignment
        # We need to do this per point or per cell?
        # Per point is easier here since we iterate points.
        
        for pi in in_indices.tolist():
            idx = (int(ix[pi]), int(iy[pi]))
            cell = grid.get_or_create_cell(idx)
            cell.point_indices.append(int(pi))

        grid.num_points_in_grid = int(in_indices.size)
        
        # TGS: Assign points to SubCells
        # Now that we have points in each cell, we can distribute them to subcells.
        # This is more efficient than doing it in the loop above if we want to batch convert to numpy.
        
        for cell in grid.iter_cells():
            if not cell.point_indices:
                continue
                
            # Get points for this cell
            cell_points = points_xyz[cell.point_indices]
            
            # Cell center in world coordinates
            cx, cy = grid.grid_index_to_world(cell.index[0], cell.index[1])
            # grid_index_to_world returns origin (bottom-left), we need center?
            # "origin_xy: world coordinate of grid index (0,0) cell's origin"
            # "world_x = origin_x + ix * resolution"
            # Usually origin is corner. Center is origin + resolution/2.
            # Let's verify grid_index_to_world implementation.
            # It returns (ox + ix*r, oy + iy*r). This is likely the corner.
            # TGS usually uses center for angle calculation.
            
            center_x = cx + spec.resolution * 0.5
            center_y = cy + spec.resolution * 0.5
            
            # Assign to subcells
            # We can vectorize this for the cell points
            dx = cell_points[:, 0] - center_x
            dy = cell_points[:, 1] - center_y
            angles = np.arctan2(dy, dx)
            
            # 0: up (-pi/4 <= angle < pi/4)  <- WAIT, y is up? 
            # User spec:
            # if  np.pi/4 <= angle < 3*np.pi/4: return 1  # left
            # elif -np.pi/4 <= angle < np.pi/4: return 0  # up
            # elif -3*np.pi/4 <= angle < -np.pi/4: return 3  # right
            # else: return 2  # down
            
            # Note: standard arctan2(y, x)
            # 0 is East (x+). pi/2 is North (y+).
            # User map:
            # 0 (up?): [-pi/4, pi/4) -> This is East (x+) direction actually. 
            # Maybe "up" means "forward" in vehicle frame (x+)? 
            # Let's stick to the ANGLE ranges provided by user, ignoring the comment names if they are confusing.
            # Range [-pi/4, pi/4) is usually "Front" if x is forward.
            
            tri_ids = np.full(angles.shape, 2, dtype=int) # Default 2 (else case)
            
            # 1: [pi/4, 3pi/4)
            mask1 = (angles >= np.pi/4) & (angles < 3*np.pi/4)
            tri_ids[mask1] = 1
            
            # 0: [-pi/4, pi/4)
            mask0 = (angles >= -np.pi/4) & (angles < np.pi/4)
            tri_ids[mask0] = 0
            
            # 3: [-3pi/4, -pi/4)
            mask3 = (angles >= -3*np.pi/4) & (angles < -np.pi/4)
            tri_ids[mask3] = 3
            
            # 2: else (includes [3pi/4, pi] and [-pi, -3pi/4))
            
            # Create SubCells
            for t_id in range(4):
                mask_t = (tri_ids == t_id)
                sub_points = cell_points[mask_t]
                # We store a COPY of points in SubCell for now (easier than indices)
                # User requested: cell.subcells[tri].points.append(point)
                # But bulk init is better.
                
                # Initialize SubCell even if empty? 
                # "cell.subcells[tri].points.append(point)" implies existence.
                # Let's create it.
                
                from .types import SubCell # Import inside to avoid circular if any
                
                cell.subcells[t_id] = SubCell(points=sub_points)

        return grid
