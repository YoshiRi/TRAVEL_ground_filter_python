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
