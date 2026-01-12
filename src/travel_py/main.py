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
    plot_filtered_points_xy,
)
from travel_py.config import (
    GridConfig,
    SeedConfig,
    AcceptConfig,
    DebugConfig,
    load_config,
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
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config file",
    )
    args = parser.parse_args()

    # -------------------------
    # Load input
    # -------------------------
    points = load_points(args.points)
    num_points = points.shape[0]

    # Load config
    global_cfg = load_config(args.config)

    # -------------------------
    # Grid
    # -------------------------
    grid_cfg = global_cfg.grid

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
    # TGS Step 1: Node-wise Ground Estimation
    # -------------------------
    from travel_py.plane import LocalPlaneEstimator
    from travel_py.types import CellState

    plane_estimator = LocalPlaneEstimator(
        num_lpr=20,
        th_seeds=0.5,
        th_outlier=0.5,
        th_normal=0.9, # > 0.9 means < 25 degrees tilt approx
        min_points=3,
        th_weight=0.0,
    )
    stats = {
        CellState.GROUND: 0,
        CellState.NON_GROUND: 0,
        CellState.UNKNOWN: 0,
    }

    print("Running Node-wise Ground Estimation...")

    for cell in grid.iter_cells():
        for subcell in cell.subcells.values():
            plane_estimator.estimate_and_update(subcell)
            stats[subcell.label] += 1

    print(f"  Total SubCells : {sum(stats.values())}")
    print(f"  Ground         : {stats[CellState.GROUND]}")
    print(f"  Non-Ground     : {stats[CellState.NON_GROUND]}")
    print(f"  Unknown        : {stats[CellState.UNKNOWN]}")
    
    # -------------------------
    # TGS Step 2: Traversal (BFS + LCC)
    # -------------------------
    from travel_py.graph import TraversabilityGraph
    from travel_py.traversal import run_subcell_traversal
    from travel_py.plane import is_traversable_lcc, PlaneModel
    
    print("Running Traversal (BFS + LCC)...")
    graph = TraversabilityGraph(grid)
    start_nodes = graph.find_dominant_subcells()
    print(f"Traversal Seeds: {len(start_nodes)}")
    
    def accept_lcc(src_idx, dst_idx) -> bool:
        # 1. Retrieve SubCells
        src_cell = grid.cells.get((src_idx.i, src_idx.j))
        dst_cell = grid.cells.get((dst_idx.i, dst_idx.j))
        
        if not src_cell or not dst_cell:
            return False
            
        src_sub = src_cell.subcells.get(src_idx.tri)
        dst_sub = dst_cell.subcells.get(dst_idx.tri)
        
        if not src_sub or not dst_sub:
            return False
            
        # 2. Check Candidate Status (Step 1 result)
        if dst_sub.label != CellState.GROUND:
            return False
            
        # 3. Check Data Availability
        if src_sub.normal is None or dst_sub.normal is None:
            return False
            
        # 4. LCC Check
        src_plane = PlaneModel(
            normal=src_sub.normal,
            mean=src_sub.mean,
            d=src_sub.d,
            weight=src_sub.weight,
            label=src_sub.label
        )
        dst_plane = PlaneModel(
            normal=dst_sub.normal,
            mean=dst_sub.mean,
            d=dst_sub.d,
            weight=dst_sub.weight,
            label=dst_sub.label
        )
        
        return is_traversable_lcc(
            src_plane, 
            dst_plane, 
            th_normal=0.9, 
            th_dist=0.5
        )

    visited, rejected_count = run_subcell_traversal(
        graph=graph,
        start_nodes=start_nodes,
        accept_fn=accept_lcc,
    )
    
    # Apply results
    ground_final = 0
    unknown_final = 0
    
    from travel_py.types import SubCellIndex
    
    for cell in grid.iter_cells():
        for t, sub in cell.subcells.items():
            idx = SubCellIndex(cell.index[0], cell.index[1], t)
            
            if idx in visited:
                sub.label = CellState.GROUND
                ground_final += 1
            else:
                # Was ground, but not reachable -> UNKNOWN
                if sub.label == CellState.GROUND:
                    sub.label = CellState.UNKNOWN
                
                if sub.label == CellState.UNKNOWN:
                    unknown_final += 1
                    
    print(f"Traversal Rejected (LCC): {rejected_count}")
    print(f"Final Ground SubCells: {ground_final}")

    # SKIP Original Traversal
    return

    # -------------------------
    # Seed selection
    # -------------------------
    seed_cfg = global_cfg.seed

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
    accept_cfg = global_cfg.accept

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
        plot_filtered_points_xy(points, labels)


if __name__ == "__main__":
    main()
