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
from typing import Optional, Set
from .grid import Grid
from .types import SubCellIndex, CellState


def find_dominant_subcells(
    grid: Grid,
    top_k: int = 1,
    already_visited: Optional[Set[SubCellIndex]] = None,
    prefer_center: bool = False,
) -> List[SubCellIndex]:
    """
    Find seed subcells: GROUND label with highest planarity weight.

    Parameters
    ----------
    grid:
        Grid to search for seed subcells.
    top_k:
        Maximum number of seeds to return.
    already_visited:
        SubCellIndices already covered by previous BFS iterations.
        Candidates in this set are skipped so each seed explores new area.
    prefer_center:
        When True, score = weight / (1 + dist_to_origin_xy), where
        dist is the cell-centre distance from (0, 0) in the world frame.
        This prioritises subcells near the sensor / base-link origin.

    Returns
    -------
    List of up to *top_k* SubCellIndex objects, sorted by score descending.
    """
    spec = grid.spec
    candidates: list[tuple[float, SubCellIndex]] = []

    for cell in grid.iter_cells():
        for t, sub in cell.subcells.items():
            if sub.label != CellState.GROUND:
                continue

            score = sub.weight

            if prefer_center:
                cx = spec.origin_xy[0] + (cell.index[0] + 0.5) * spec.resolution
                cy = spec.origin_xy[1] + (cell.index[1] + 0.5) * spec.resolution
                dist = math.sqrt(cx * cx + cy * cy)
                score = score / (1.0 + dist)

            candidates.append((score, SubCellIndex(cell.index[0], cell.index[1], t)))

    candidates.sort(key=lambda x: x[0], reverse=True)

    result: List[SubCellIndex] = []
    for _, idx in candidates:
        if already_visited and idx in already_visited:
            continue
        result.append(idx)
        if len(result) >= top_k:
            break

    return result
