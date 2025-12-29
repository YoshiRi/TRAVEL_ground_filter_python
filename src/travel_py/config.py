from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional


# =========================
# Grid configuration
# =========================

@dataclass
class GridConfig:
    """
    Grid definition parameters.
    """
    resolution: float = 0.5
    origin_xy: Tuple[float, float] = (-50.0, -50.0)
    size_xy: Tuple[int, int] = (200, 200)


# =========================
# Seed selection
# =========================

@dataclass
class SeedConfig:
    """
    Seed selection parameters.
    """
    min_points: int = 1
    use_height_range: bool = False
    use_mean_height: bool = True
    top_k: int = 5


# =========================
# Traversal / accept
# =========================

@dataclass
class AcceptConfig:
    """
    Traversal acceptance parameters.
    """
    max_height_diff: float = 0.5
    max_slope: Optional[float] = None  # None = disable slope check


# =========================
# Debug / visualization
# =========================

@dataclass
class DebugConfig:
    """
    Debug / visualization flags.
    """
    enable_viz: bool = False
