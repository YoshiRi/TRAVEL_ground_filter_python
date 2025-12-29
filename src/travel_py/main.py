from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from travel_py.grid import Grid, GridSpec
from travel_py.cell_features import compute_all_cell_features
from travel_py.seed import (
    SeedSelector,
    MinPointCount,
    SmallHeightRange,
    LowMeanHeight,
)
from travel_py.accept import (
    TraversalAcceptConfig,
    accept_height_and_slope,
)
from travel_py.traversal import run_traversal
from travel_py.labeling import label_points_from_cells
from travel_py.debug_viz import (
    plot_cell_scalar,
    plot_cell_state,
    plot_reject_reason,
    plot_iteration,
)
from travel_py.config import (
    GridConfig,
    SeedConfig,
    AcceptConfig,
    DebugConfig,
)


# =========================
# Utilities
# =========================

def load_points(path: Path) -> np.ndarray:
    """
    Load point cloud from a simple .npy file.
    Shape: (N,3) or (N,>=3)
    """
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Point cloud must be of shape (N,3) or (N,>=3)")
    return points


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Python Travel ground segmentation (reference implementation)"
    )
    parser.add_argument(
        "--points",
        type=Path,
        required=True,
        help="Path to input point cloud (.npy)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable debug visualization",
    )
    args = parser.parse_args()

    # -------------------------
    # Load input
    # -------------------------
    points = load_points(args.points)
    num_points = points.shape[0]

    # -------------------------
    # Grid
    # -------------------------
    grid_cfg = GridConfig() # TODO: parse from args

    grid_spec = GridSpec(
        resolution=grid_cfg.resolution,
        origin_xy=grid_cfg.origin_xy,
        size_xy=grid_cfg.size_xy,
    )

    grid = Grid.from_points(points, grid_spec)

    # -------------------------
    # Cell features
    # -------------------------
    compute_all_cell_features(grid.iter_cells(), points)

    # -------------------------
    # Seed selection
    # -------------------------
    seed_cfg = SeedConfig() # TODO: parse from args

    criteria = [
        MinPointCount(seed_cfg.min_points),
    ]

    if seed_cfg.use_height_range:
        criteria.append(SmallHeightRange())

    if seed_cfg.use_mean_height:
        criteria.append(LowMeanHeight())

    seed_selector = SeedSelector(
        criteria=criteria,
        top_k=seed_cfg.top_k,
    )
    seed_cells = seed_selector.select(grid.iter_cells())

    if not seed_cells:
        raise RuntimeError("No seed cells selected")

    # -------------------------
    # Traversal (accept config)
    # -------------------------
    accept_cfg = AcceptConfig()

    def accept_fn(current, neighbor):
        return accept_height_and_slope(
            current,
            neighbor,
            config=TraversalAcceptConfig(
                max_height_diff=accept_cfg.max_height_diff,
                max_slope=accept_cfg.max_slope,
                cell_size=grid_cfg.resolution,
            ),
        )

    traversal_state = run_traversal(
        seed_cells=seed_cells,
        cells=grid.cells,
        connectivity=4,
        accept_fn=accept_fn,
    )

    # -------------------------
    # Labeling
    # -------------------------
    labels = label_points_from_cells(
        cells=grid.cells,
        num_points=num_points,
    )

    print("Traversal finished:")
    print(f"  ground cells   : {traversal_state.num_ground}")
    print(f"  rejected cells : {traversal_state.num_rejected}")

    # -------------------------
    # Visualization
    # -------------------------
    if args.viz:
        plot_cell_scalar(grid.iter_cells(), value="min_z")
        plot_cell_scalar(grid.iter_cells(), value="height_range")
        plot_cell_state(grid.iter_cells())
        plot_reject_reason(grid.iter_cells())
        plot_iteration(grid.iter_cells())


if __name__ == "__main__":
    main()
