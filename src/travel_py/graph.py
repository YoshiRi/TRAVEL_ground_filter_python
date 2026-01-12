from __future__ import annotations

from typing import List, Optional

import numpy as np

from .grid import Grid
from .types import SubCellIndex, CellState

class TraversabilityGraph:
    def __init__(self, grid: Grid):
        self.grid = grid

    def get_neighbor_subcells(self, idx: SubCellIndex) -> List[SubCellIndex]:
        """
        Get neighbors: 3 internal + neighbors in adjacent cells.
        """
        neighbors = []
        
        # 1. Internal neighbors (same cell, different tri)
        # All other 3 triangles in the same cell are neighbors in the "center" sense?
        # Or strictly sharing an edge?
        # TGS: "4 triangles share the center". So they are all connected.
        for t in range(4):
            if t != idx.tri:
                neighbors.append(SubCellIndex(idx.i, idx.j, t))
                
        # 2. External neighbors (adjacent cells)
        # We need to know WHICH triangle connects to WHICH neighbor cell.
        # Tri definitions (from grid.py):
        # 0: Right (East) [-pi/4, pi/4]  -- connects to (i+1, j)
        # 1: Up (North) [pi/4, 3pi/4]    -- connects to (i, j+1)
        # 2: Down (South) [else] -> [-pi, -3pi/4] U [3pi/4, pi] ?? 
        #    Wait, let's re-read grid.py implementation carefully.
        #    mask1 = [pi/4, 3pi/4) -> Up/North (y+)
        #    mask0 = [-pi/4, pi/4) -> Right/East (x+)
        #    mask3 = [-3pi/4, -pi/4) -> Down/South (y-)
        #    mask2 = else -> Left/West (x-)
        
        # Let's check grid.py again to be sure about 2 and 3.
        # mask3 = [-3pi/4, -pi/4) -> this is South-West? No, -pi/2 is South.
        # -3pi/4 is -135 deg. -pi/4 is -45 deg.
        # So mask3 covers South (y-).
        # mask2 is the rest: [3pi/4, pi] and [-pi, -3pi/4). This is West (x-).
        
        # So:
        # 0: East (+x) -> Neighbor (i+1, j)
        # 1: North (+y) -> Neighbor (i, j+1)
        # 3: South (-y) -> Neighbor (i, j-1)
        # 2: West (-x) -> Neighbor (i-1, j)
        
        # But the user said: "8-neighbor + all tri is sufficient".
        # So we don't need strict edge matching yet?
        # "⚠️ 这里では TGS の searchNeighborNodes を「完全再現」しなくてよい → 最初は 8-neighbor + 全 tri で十分"
        # Okay, so we can just look at all 8 neighbors and their triangles?
        # Or just the specific ones?
        # If we are loose, we might over-connect.
        # Let's try to be slightly specific but not perfect.
        # Connect to ALL triangles of the 8 neighbors? That's 8*4 = 32 edges. A bit much.
        # Let's implement the "Spatial Neighbor" logic roughly.
        
        # If I am Tri 0 (East), I mainly connect to (i+1, j)'s Tri 2 (West).
        # And maybe (i+1, j+1)'s Tri ...?
        # Let's stick to 4-connectivity for the grid cells for now to be safe, 
        # or 8 if requested. User said "8 neighbor".
        
        # Let's iterate 8 neighbors of the grid cell (ix, iy).
        # For each neighbor cell, if it exists, add ALL its subcells?
        # That seems easiest for "Step 2".
        
        ix, iy = idx.i, idx.j
        
        # 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = ix + dx, iy + dy
                
                # Check bounds/existence
                if not self.grid._in_bounds(nx, ny):
                    continue
                    
                # If cell doesn't exist in sparse map, skip
                if (nx, ny) not in self.grid.cells:
                    continue
                    
                # Add all 4 subcells of the neighbor
                # (This is the "loose" implementation requested)
                for t in range(4):
                    neighbors.append(SubCellIndex(nx, ny, t))
                    
        return neighbors

    def find_dominant_subcells(self) -> List[SubCellIndex]:
        """
        Find seed subcells: GROUND label, high weight, near center.
        """
        # Grid center
        ox, oy = self.grid.spec.origin_xy
        w, h = 0.0, 0.0
        if self.grid.spec.size_xy:
            nx, ny = self.grid.spec.size_xy
            w = nx * self.grid.spec.resolution
            h = ny * self.grid.spec.resolution
        
        center_x = ox + w * 0.5
        center_y = oy + h * 0.5
        
        # Search for best seed
        best_idx = None
        best_weight = -1.0
        
        # Scan all cells? Or just a window?
        # Scanning all is fine for now.
        candidate_count = 0
        for cell in self.grid.iter_cells():
            for t, sub in cell.subcells.items():
                if sub.label != CellState.GROUND:
                    continue
                    
                candidate_count += 1
                # Check distance to center (optional, but requested "ego near")
                # We don't have subcell center easily available without recomputing.
                # Use grid cell center.
                cx, cy = self.grid.grid_index_to_world(cell.index[0], cell.index[1])
                dist = (cx - center_x)**2 + (cy - center_y)**2
                
                # Simple heuristic: weight / (1 + dist) ? 
                # Or just "in center region" AND max weight.
                # Let's pick max weight within some radius?
                # Or just max weight globally?
                # User: "grid 中央付近で weight 最大の GROUND SubCell を 1 つ返す"
                
                # Let's define "near" as within 10m?
                if dist < 10000.0: # 100m^2 -> 10m radius? No wait 100^2 = 10000. 
                                   # 100.0 was 10m radius (10^2).
                                   # Let's make it huge to catch anything in sample.
                    if sub.weight > best_weight:
                        best_weight = sub.weight
                        best_idx = SubCellIndex(cell.index[0], cell.index[1], t)
        
        print(f"DEBUG: Grid Center: ({center_x:.2f}, {center_y:.2f})")
        print(f"DEBUG: Ground Candidates: {candidate_count}")
        if best_idx:
            return [best_idx]
        return []


