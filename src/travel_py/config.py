from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple, Optional
import yaml
from pathlib import Path
from typing import Any

# =========================
# Grid configuration
# =========================

@dataclass
class GridConfig:
    """
    Grid definition parameters.
    """
    resolution: float = 5
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

# =========================
# Global configuration
# =========================
@dataclass
class GlobalConfig:
    grid: GridConfig = GridConfig()
    seed: SeedConfig = SeedConfig()
    accept: AcceptConfig = AcceptConfig()
    debug: DebugConfig = DebugConfig()


# =========================
# Loader 
# =========================

def load_config(path: Path | None = None) -> GlobalConfig:
    cfg = GlobalConfig()

    if path is None:
        return cfg

    with open(path, "r") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    if "grid" in data:
        cfg.grid = replace(cfg.grid, **data["grid"])

    if "seed" in data:
        cfg.seed = replace(cfg.seed, **data["seed"])

    if "accept" in data:
        cfg.accept = replace(cfg.accept, **data["accept"])

    if "debug" in data:
        cfg.debug = replace(cfg.debug, **data["debug"])

    return cfg

