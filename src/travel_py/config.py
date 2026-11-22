from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple, Optional, Any
from pathlib import Path

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

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
    prefer_center: bool = False  # score by weight / (1 + dist_to_origin) to favour near-sensor seeds
    re_explore: bool = False     # re-explore unvisited GROUND candidate subcells when queue is empty (official TRAVEL)
    max_components: int = 100    # maximum ground components to explore when re_explore=True


# =========================
# Traversal / accept
# =========================

@dataclass
class AcceptConfig:
    """
    Traversal acceptance parameters.
    """
    max_height_diff: float = 0.5       # th_dist for is_traversable_lcc
    th_normal: float = 0.9             # minimum normal[2] for both planes and LCC similarity
    max_slope: Optional[float] = None  # reserved; currently unused
    th_seeds: float = 0.5              # height band above LPR for ground candidate selection
    use_adaptive_lcc: bool = False     # Use TRAVEL official adaptive LCC thresholds based on inter-mean distance
    th_lcc_normal_similarity: float = 0.03  # angular coefficient for normal similarity: 1 - sin(d * theta)
    th_lcc_planar_dist: float = 0.2         # angular coefficient for plane distance: d * sin(theta)
    use_14_neighbors: bool = False          # Use TRAVEL official 14-neighbor connectivity instead of strict 4-neighbors


# =========================
# Reachability / Traversability Verification (Extension beyond official TRAVEL)
# =========================

@dataclass
class ReachabilityConfig:
    """
    Layer 2 extension: Reachability-aware multi-level ground verification (B+D+E).

    NOTE — Deviation from official TRAVEL (darrenjkt/TRAVEL):
      Pure TRAVEL (Layer 1) performs flat ground detection + unconstrained re-exploration.
      When enabled, this Layer 2 post-filter separates disconnected flat components into:
        - PRIMARY_GROUND (reachable directly from ego top_k seeds)
        - BRIDGED_GROUND (disconnected flat components verified as coplanar extensions)
        - OTHER_LEVEL_GROUND (elevated/unreachable flat surfaces such as building terraces,
          car roofs, or overpasses — excluded from final traversable ground).
      Default is enabled=False to preserve 100% compatibility with official TRAVEL.
    """
    enabled: bool = False
    bridge_max_dist: float = 15.0        # Max 2D horizontal distance (m) to bridge across gaps
    bridge_max_plane_dist: float = 0.45  # Max bidirectional plane extrapolation offset (m)
    bridge_min_normal_sim: float = 0.90  # Min cosine similarity between bridged planes
    z_stack_margin: float = 0.30         # Reject components sitting > margin (m) above confirmed ground


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
    grid: GridConfig = field(default_factory=GridConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    accept: AcceptConfig = field(default_factory=AcceptConfig)
    reachability: ReachabilityConfig = field(default_factory=ReachabilityConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


# =========================
# Loader 
# =========================

def load_config(path: Path | None = None) -> GlobalConfig:
    cfg = GlobalConfig()

    if path is None:
        return cfg

    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for load_config(). Install with: pip install pyyaml"
        )

    with open(path, "r") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    if "grid" in data:
        cfg.grid = replace(cfg.grid, **data["grid"])

    if "seed" in data:
        cfg.seed = replace(cfg.seed, **data["seed"])

    if "accept" in data:
        cfg.accept = replace(cfg.accept, **data["accept"])

    if "reachability" in data:
        cfg.reachability = replace(cfg.reachability, **data["reachability"])

    if "debug" in data:
        cfg.debug = replace(cfg.debug, **data["debug"])

    return cfg

