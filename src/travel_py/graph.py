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
        
        NOTE: This is a LOOSE adjacency implementation (connects to all subcells of 8 neighbors).
        It is intended for early traversal validation and graph wiring tests.
        It does NOT yet implement the strict TGS edge-sharing logic.
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



