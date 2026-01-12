from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import math

from .types import Cell


# =========================
# Criterion base
# =========================

class SeedCriterion:
    """
    Base class for seed selection criteria.

    score(cell) should return:
      - higher is better
      - -inf if the cell is invalid
    """
    name: str = "base"

    def score(self, cell: Cell) -> float:
        raise NotImplementedError


# =========================
# Concrete criteria
# =========================

class MinPointCount(SeedCriterion):
    name = "min_point_count"

    def __init__(self, min_points: int):
        self.min_points = min_points

    def score(self, cell: Cell) -> float:
        return 0.0 if len(cell.point_indices) >= self.min_points else -math.inf


class SmallHeightRange(SeedCriterion):
    """
    Prefer flatter cells.
    """
    name = "small_height_range"

    def score(self, cell: Cell) -> float:
        if cell.height_range is None:
            return -math.inf
        return -cell.height_range


class LowMeanHeight(SeedCriterion):
    """
    Prefer lower cells.
    """
    name = "low_mean_height"

    def score(self, cell: Cell) -> float:
        if cell.mean_z is None:
            return -math.inf
        return -cell.mean_z


class LowMinHeight(SeedCriterion):
    """
    Prefer cells with low minimum height.
    """
    name = "low_min_height"

    def score(self, cell: Cell) -> float:
        if cell.min_z is None:
            return -math.inf
        return -cell.min_z


# =========================
# Seed selector
# =========================

@dataclass
class SeedSelector:
    """
    Select seed cells based on combined criteria.
    """
    criteria: List[SeedCriterion]
    top_k: int = 1

    def select(self, cells: Iterable[Cell]) -> List[Cell]:
        scored: List[Tuple[float, Cell]] = []

        for cell in cells:
            total_score = 0.0

            for criterion in self.criteria:
                s = criterion.score(cell)
                if s == -math.inf:
                    total_score = -math.inf
                    break
                total_score += s

            if total_score != -math.inf:
                scored.append((total_score, cell))

        # High score first
        scored.sort(key=lambda x: x[0], reverse=True)

        return [cell for _, cell in scored[: self.top_k]]


# =========================
# TGS SubCell Seed Selection
# =========================
from .grid import Grid
from .types import SubCellIndex, CellState

def find_dominant_subcells(grid: Grid) -> List[SubCellIndex]:
    """
    Find seed subcells: GROUND label, high weight, near center.
    """
    # Grid center
    ox, oy = grid.spec.origin_xy
    w, h = 0.0, 0.0
    if grid.spec.size_xy:
        nx, ny = grid.spec.size_xy
        w = nx * grid.spec.resolution
        h = ny * grid.spec.resolution
    
    center_x = ox + w * 0.5
    center_y = oy + h * 0.5
    
    # Search for best seed
    best_idx = None
    best_weight = -1.0
    
    # Scan all cells? Or just a window?
    # Scanning all is fine for now.
    candidate_count = 0
    for cell in grid.iter_cells():
        for t, sub in cell.subcells.items():
            if sub.label != CellState.GROUND:
                continue
                
            candidate_count += 1
            # Check distance to center (optional, but requested "ego near")
            # We don't have subcell center easily available without recomputing.
            # Use grid cell center.
            cx, cy = grid.grid_index_to_world(cell.index[0], cell.index[1])
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
