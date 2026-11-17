#!/usr/bin/env python3
"""
TRAVEL Ground Filter — Large PCD Map Spatial Tiling Processor

Filters massive point cloud maps (PCD/PLY) without out-of-memory (OOM) errors
by dividing the map into overlapping spatial tiles (chunks), running the vectorized
TRAVEL pipeline on each tile with local coordinate normalization, and merging the results.

Key Features:
  - Constant memory footprint regardless of map area (hundreds of km²).
  - Handles huge coordinate offsets (UTM, Plane Rectangular Coordinate Systems) natively.
  - Multi-process parallel execution across CPU cores.
  - Core & Margin architecture: eliminates boundary discontinuity artifacts.
  - Optional voxel grid downsampling pre-processing.
  - Output ground-only, non-ground-only, or labeled point clouds.

Usage:
  # Filter sample PCD map with 100m tiles
  uv run python tools/filter_pcd_map.py data/sample_pcd/pointcloud_map.pcd \\
      --output-ground data/sample_pcd/ground.pcd \\
      --output-nonground data/sample_pcd/nonground.pcd

  # High-resolution 2.5m grid with voxel downsampling and 14-neighbors
  uv run python tools/filter_pcd_map.py input.pcd \\
      --resolution 2.5 --tile-size 100.0 --voxel-size 0.1 \\
      --use-14-neighbors --re-explore --output-ground ground.pcd
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from tqdm import tqdm

from travel_py.config import GlobalConfig, GridConfig, SeedConfig, AcceptConfig
from travel_py.pipeline import run_fast


@dataclass
class TileTask:
    tile_id: int
    tx: int
    ty: int
    core_min_xy: Tuple[float, float]
    core_max_xy: Tuple[float, float]
    expanded_min_xy: Tuple[float, float]
    expanded_max_xy: Tuple[float, float]
    # Indices in the full point array
    point_indices: np.ndarray


def process_single_tile(
    task: TileTask,
    pts_expanded: np.ndarray,
    resolution: float,
    adaptive_lcc: bool,
    use_14_neighbors: bool,
    re_explore: bool,
    max_components: int,
    th_normal: float,
    max_height_diff: float,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Worker function to process a single spatial tile in parallel.
    Normalizes coordinates locally to prevent huge dense grid allocation.
    Returns: (tile_id, core_point_indices, core_labels)
    """
    if len(task.point_indices) == 0:
        return task.tile_id, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)

    # Local tile origin
    ox, oy = task.expanded_min_xy
    ex_w = task.expanded_max_xy[0] - ox
    ex_h = task.expanded_max_xy[1] - oy

    nx = max(int(np.ceil(ex_w / resolution)), 1)
    ny = max(int(np.ceil(ex_h / resolution)), 1)

    cfg = GlobalConfig(
        grid=GridConfig(
            resolution=resolution,
            origin_xy=(ox, oy),
            size_xy=(nx, ny),
        ),
        seed=SeedConfig(
            top_k=5,
            prefer_center=True,
            re_explore=re_explore,
            max_components=max_components,
        ),
        accept=AcceptConfig(
            max_height_diff=max_height_diff,
            th_normal=th_normal,
            use_adaptive_lcc=adaptive_lcc,
            use_14_neighbors=use_14_neighbors,
        ),
    )

    # Run TRAVEL Fast pipeline on the tile points
    res = run_fast(pts_expanded, cfg, keep_grid=False)
    all_labels = res.labels  # 1=GROUND, 0=NON_GROUND, -1=UNKNOWN

    # Filter to only keep core points (ignore margin points to avoid boundary artifacts)
    cx_min, cy_min = task.core_min_xy
    cx_max, cy_max = task.core_max_xy
    in_core = (
        (pts_expanded[:, 0] >= cx_min) & (pts_expanded[:, 0] < cx_max) &
        (pts_expanded[:, 1] >= cy_min) & (pts_expanded[:, 1] < cy_max)
    )

    core_indices = task.point_indices[in_core]
    core_labels = all_labels[in_core]

    return task.tile_id, core_indices, core_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter large PCD/PLY maps with spatial tiling and TRAVEL ground filter."
    )
    parser.add_argument("input_path", type=Path, help="Input PCD or PLY point cloud file")
    parser.add_argument("--output-ground", type=Path, default=None, help="Output path for ground point cloud (.pcd/.ply)")
    parser.add_argument("--output-nonground", type=Path, default=None, help="Output path for non-ground point cloud (.pcd/.ply)")
    parser.add_argument("--output-labeled", type=Path, default=None, help="Output path for labeled points (.pcd/.ply/.npz)")
    parser.add_argument("--voxel-size", type=float, default=None, help="Voxel downsampling size in meters (optional pre-processing)")
    parser.add_argument("--tile-size", type=float, default=100.0, help="Tile size in meters (default: 100.0)")
    parser.add_argument("--overlap", type=float, default=10.0, help="Tile boundary overlap in meters (default: 10.0)")
    parser.add_argument("--resolution", type=float, default=2.5, help="TRAVEL grid cell resolution in meters (default: 2.5)")
    parser.add_argument("--adaptive", action="store_true", default=True, help="Enable distance-dependent adaptive LCC (default: True)")
    parser.add_argument("--no-adaptive", action="store_false", dest="adaptive", help="Disable adaptive LCC")
    parser.add_argument("--use-14-neighbors", action="store_true", default=True, help="Use official 14-neighbor expansion (default: True)")
    parser.add_argument("--no-14-neighbors", action="store_false", dest="use_14_neighbors", help="Use strict 4-neighbors")
    parser.add_argument("--re-explore", action="store_true", default=True, help="Enable multi-component re-exploration (default: True)")
    parser.add_argument("--no-re-explore", action="store_false", dest="re_explore", help="Disable re-exploration")
    parser.add_argument("--max-components", type=int, default=50, help="Max ground components per tile (default: 50)")
    parser.add_argument("--th-normal", type=float, default=0.9, help="Normal threshold for ground (default: 0.9)")
    parser.add_argument("--th-dist", type=float, default=0.5, help="Max planar height step threshold (default: 0.5)")
    parser.add_argument("--jobs", type=int, default=max(os.cpu_count() or 1, 1), help="Number of parallel worker processes")
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    print("=" * 65)
    print("      TRAVEL Ground Filter — Large PCD Map Tiling Processor")
    print("=" * 65)
    print(f"Input File    : {args.input_path}")
    print(f"Tile Size     : {args.tile_size:.1f} m (Overlap: {args.overlap:.1f} m)")
    print(f"Resolution    : {args.resolution:.1f} m")
    print(f"TRAVEL Config : adaptive={args.adaptive}, 14-nbrs={args.use_14_neighbors}, re-explore={args.re_explore}")
    print(f"Parallel Jobs : {args.jobs} processes")
    print("-" * 65)

    # 1. Load point cloud
    t0 = time.perf_counter()
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(args.input_path))
    pts_raw = np.asarray(pcd.points, dtype=np.float64)
    n_points_orig = len(pts_raw)
    print(f"Loaded {n_points_orig:,} points in {time.perf_counter() - t0:.2f} s")

    # 2. Voxel downsampling (optional)
    if args.voxel_size and args.voxel_size > 0:
        t_vox = time.perf_counter()
        print(f"Applying voxel downsampling (voxel_size={args.voxel_size:.2f}m)...")
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        pts_raw = np.asarray(pcd.points, dtype=np.float64)
        print(f"Downsampled to {len(pts_raw):,} points ({len(pts_raw)/n_points_orig*100:.1f}%) in {time.perf_counter() - t_vox:.2f} s")

    N = len(pts_raw)
    min_xyz = pts_raw.min(axis=0)
    max_xyz = pts_raw.max(axis=0)
    extent = max_xyz - min_xyz
    print(f"Map Bounds X  : [{min_xyz[0]:.2f}, {max_xyz[0]:.2f}] (width: {extent[0]:.1f} m)")
    print(f"Map Bounds Y  : [{min_xyz[1]:.2f}, {max_xyz[1]:.2f}] (height: {extent[1]:.1f} m)")
    print(f"Map Bounds Z  : [{min_xyz[2]:.2f}, {max_xyz[2]:.2f}] (elevation: {extent[2]:.1f} m)")

    # 3. Spatial tiling decomposition
    tile_size = args.tile_size
    overlap = args.overlap

    x_min, y_min = min_xyz[0], min_xyz[1]
    x_max, y_max = max_xyz[0], max_xyz[1]

    num_tx = max(int(np.ceil((x_max - x_min) / tile_size)), 1)
    num_ty = max(int(np.ceil((y_max - y_min) / tile_size)), 1)
    total_grid_tiles = num_tx * num_ty
    print(f"Decomposing into {num_tx} x {num_ty} = {total_grid_tiles} potential spatial tiles...")

    # Assign each point to its core tile index
    pt_tx = np.floor((pts_raw[:, 0] - x_min) / tile_size).astype(np.int64)
    pt_ty = np.floor((pts_raw[:, 1] - y_min) / tile_size).astype(np.int64)
    pt_tx = np.clip(pt_tx, 0, num_tx - 1)
    pt_ty = np.clip(pt_ty, 0, num_ty - 1)

    # Build tasks for tiles that contain at least 1 point
    tasks: List[TileTask] = []
    tile_id = 0
    for tx in range(num_tx):
        for ty in range(num_ty):
            core_x0 = x_min + tx * tile_size
            core_x1 = core_x0 + tile_size
            core_y0 = y_min + ty * tile_size
            core_y1 = core_y0 + tile_size

            # Expanded bounds (core + margin)
            exp_x0 = core_x0 - overlap
            exp_x1 = core_x1 + overlap
            exp_y0 = core_y0 - overlap
            exp_y1 = core_y1 + overlap

            # Find points within the expanded tile
            in_tile = (
                (pts_raw[:, 0] >= exp_x0) & (pts_raw[:, 0] < exp_x1) &
                (pts_raw[:, 1] >= exp_y0) & (pts_raw[:, 1] < exp_y1)
            )
            idx_in_tile = np.nonzero(in_tile)[0]

            if len(idx_in_tile) > 0:
                tasks.append(
                    TileTask(
                        tile_id=tile_id,
                        tx=tx,
                        ty=ty,
                        core_min_xy=(core_x0, core_y0),
                        core_max_xy=(core_x1, core_y1),
                        expanded_min_xy=(exp_x0, exp_y0),
                        expanded_max_xy=(exp_x1, exp_y1),
                        point_indices=idx_in_tile,
                    )
                )
                tile_id += 1

    print(f"Active non-empty tiles to process: {len(tasks)} / {total_grid_tiles} (sparsity: {len(tasks)/total_grid_tiles*100:.1f}%)")
    print("-" * 65)

    # 4. Parallel execution
    final_labels = np.full(N, -1, dtype=np.int8)  # default UNKNOWN
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_single_tile,
                task,
                pts_raw[task.point_indices],
                args.resolution,
                args.adaptive,
                args.use_14_neighbors,
                args.re_explore,
                args.max_components,
                args.th_normal,
                args.th_dist,
            ): task
            for task in tasks
        }

        for f in tqdm(as_completed(futures), total=len(tasks), desc="Processing tiles"):
            t_id, core_idx, core_lbls = f.result()
            if len(core_idx) > 0:
                final_labels[core_idx] = core_lbls

    elapsed = time.perf_counter() - t_start
    print("-" * 65)
    print(f"Tiling execution finished in {elapsed:.2f} s ({elapsed/len(tasks)*1000.0:.1f} ms/tile)")

    # 5. Summary metrics
    n_ground = int(np.sum(final_labels == 1))
    n_nonground = int(np.sum(final_labels == 0))
    n_unknown = int(np.sum(final_labels == -1))
    print(f"Total Points Processed : {N:,}")
    print(f"  - Ground Points      : {n_ground:,} ({n_ground/N*100:5.2f}%)")
    print(f"  - Non-Ground Points  : {n_nonground:,} ({n_nonground/N*100:5.2f}%)")
    print(f"  - Unknown Points     : {n_unknown:,} ({n_unknown/N*100:5.2f}%)")
    print("=" * 65)

    # 6. Save outputs
    if args.output_ground:
        args.output_ground.parent.mkdir(parents=True, exist_ok=True)
        ground_pts = pts_raw[final_labels == 1]
        pcd_ground = o3d.geometry.PointCloud()
        pcd_ground.points = o3d.utility.Vector3dVector(ground_pts)
        o3d.io.write_point_cloud(str(args.output_ground), pcd_ground)
        print(f"Saved ground point cloud to: {args.output_ground} ({len(ground_pts):,} points)")

    if args.output_nonground:
        args.output_nonground.parent.mkdir(parents=True, exist_ok=True)
        nonground_pts = pts_raw[final_labels == 0]
        pcd_nonground = o3d.geometry.PointCloud()
        pcd_nonground.points = o3d.utility.Vector3dVector(nonground_pts)
        o3d.io.write_point_cloud(str(args.output_nonground), pcd_nonground)
        print(f"Saved non-ground point cloud to: {args.output_nonground} ({len(nonground_pts):,} points)")

    if args.output_labeled:
        args.output_labeled.parent.mkdir(parents=True, exist_ok=True)
        if args.output_labeled.suffix.lower() == ".npz":
            np.savez_compressed(args.output_labeled, points=pts_raw, labels=final_labels)
        else:
            # Save colored PCD where Green=Ground, Red=NonGround, Gray=Unknown
            colors = np.zeros((N, 3), dtype=np.float64)
            colors[final_labels == 1] = [0.18, 0.80, 0.44]  # Green
            colors[final_labels == 0] = [0.91, 0.30, 0.24]  # Red
            colors[final_labels == -1] = [0.70, 0.70, 0.70] # Gray
            pcd_labeled = o3d.geometry.PointCloud()
            pcd_labeled.points = o3d.utility.Vector3dVector(pts_raw)
            pcd_labeled.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(str(args.output_labeled), pcd_labeled)
        print(f"Saved labeled point cloud to: {args.output_labeled}")


if __name__ == "__main__":
    main()
