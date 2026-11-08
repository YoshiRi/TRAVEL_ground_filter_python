from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from travel_py.config import load_config
from travel_py.debug_viz import plot_cell_state, plot_filtered_points_xy
from travel_py.pipeline import run_fast, run_reference
from travel_py.types import CellState


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
        description="Python Travel ground segmentation (reference & fast implementation)"
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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Use object-based reference implementation instead of vectorized fast path",
    )
    args = parser.parse_args()

    # -------------------------
    # Load input & config
    # -------------------------
    points = load_points(args.points)
    global_cfg = load_config(args.config)

    mode_name = "Reference (Object-based)" if args.reference else "Fast (Vectorized)"
    print(f"Running Travel Pipeline [{mode_name}] on {args.points.name} ({len(points):,} points)...")

    if args.reference:
        result = run_reference(points, global_cfg)
    else:
        # Keep grid objects only if visualization is requested
        result = run_fast(points, global_cfg, keep_grid=args.viz)

    # -------------------------
    # Report results
    # -------------------------
    print("Node-wise Ground Estimation:")
    stats = result.step1_stats
    print(f"  Total SubCells : {sum(stats.values())}")
    print(f"  Ground         : {stats[CellState.GROUND]}")
    print(f"  Non-Ground     : {stats[CellState.NON_GROUND]}")
    print(f"  Unknown        : {stats[CellState.UNKNOWN]}")

    print("Traversal Summary:")
    print(f"  Seeds          : {result.num_seeds}")
    print(f"  Visited        : {result.visited}")
    print(f"  Rejected (Total): {result.rejected_count}")
    print(f"  Max Depth      : {result.max_depth}")
    print(f"  Final Ground   : {result.final_ground}")
    print("  Rejection Reasons:")
    for reason, count in result.rejection_stats.items():
        if count > 0:
            print(f"    {reason:<20}: {count}")

    labels = result.labels
    ground_pts = int(np.sum(labels == 1))
    nonground_pts = int(np.sum(labels == 0))
    unknown_pts = int(np.sum(labels == -1))
    print("Label Propagation:")
    print(f"  Ground points    : {ground_pts}")
    print(f"  Non-ground points: {nonground_pts}")
    print(f"  Unknown points   : {unknown_pts}")

    if result.timings:
        total_ms = sum(result.timings.values()) * 1000
        print(f"Timing Breakdown (Total: {total_ms:.1f} ms):")
        for stage, duration in result.timings.items():
            ms = duration * 1000
            pct = (duration / sum(result.timings.values())) * 100
            print(f"  {stage:<12}: {ms:6.1f} ms ({pct:4.1f}%)")

    # -------------------------
    # Visualization
    # -------------------------
    if args.viz and result.grid is not None:
        plot_cell_state(result.grid.iter_cells())
        plot_filtered_points_xy(points, labels)


if __name__ == "__main__":
    main()
