#!/usr/bin/env python3
"""
SemanticKITTI Curb & Drivability Ground Truth Generator

Extracts ground truth for curb edges and drivability evaluation from SemanticKITTI:
1. Drivable Ground (Road, Parking, Lane-markings: 40, 44, 60)
2. Elevated Ground / Sidewalk (Sidewalk, Other-ground: 48, 49)
3. Physical Curb Boundary Points:
   Points along the interface between road and sidewalk where the 2D distance
   is within `curb_interface_dist` (default: 0.35m) and vertical step height
   |Δz| is within `[min_step, max_step]` (default: [0.05m, 0.25m]).

Usage:
  python tools/generate_curb_gt.py --sequence 00 --limit 10
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree

ROAD_CLASSES = {40, 44, 60}
SIDEWALK_CLASSES = {48, 49}


def extract_curb_gt_frame(
    pts_xyz: np.ndarray,
    labels: np.ndarray,
    *,
    curb_interface_dist: float = 0.35,
    min_step: float = 0.05,
    max_step: float = 0.25,
) -> Dict[str, np.ndarray]:
    """
    Extract curb edge points and drivable vs elevated masks for a single frame.

    Returns
    -------
    dict with keys:
      'is_drivable': (N,) bool
      'is_elevated': (N,) bool
      'is_curb_road': (N,) bool
      'is_curb_sidewalk': (N,) bool
      'curb_step_heights': (K,) float32 step heights for detected curb pairs
    """
    N = len(pts_xyz)
    is_drivable = np.isin(labels, list(ROAD_CLASSES))
    is_elevated = np.isin(labels, list(SIDEWALK_CLASSES))

    is_curb_road = np.zeros(N, dtype=bool)
    is_curb_sidewalk = np.zeros(N, dtype=bool)

    road_indices = np.nonzero(is_drivable)[0]
    sw_indices = np.nonzero(is_elevated)[0]

    if len(road_indices) == 0 or len(sw_indices) == 0:
        return {
            "is_drivable": is_drivable,
            "is_elevated": is_elevated,
            "is_curb_road": is_curb_road,
            "is_curb_sidewalk": is_curb_sidewalk,
            "curb_step_heights": np.empty(0, dtype=np.float32),
        }

    # 2D KD-Tree search between road and sidewalk points
    road_xy = pts_xyz[road_indices, :2]
    sw_xy = pts_xyz[sw_indices, :2]

    tree_sw = cKDTree(sw_xy)
    dists, nearest_sw = tree_sw.query(road_xy, distance_upper_bound=curb_interface_dist)

    valid = dists < curb_interface_dist
    valid_road_idx = road_indices[valid]
    matched_sw_idx = sw_indices[nearest_sw[valid]]

    # Compute vertical step height: Sidewalk Z - Road Z
    dz = pts_xyz[matched_sw_idx, 2] - pts_xyz[valid_road_idx, 2]

    # Filter by physical curb step bounds (5cm to 25cm)
    is_curb = (dz >= min_step) & (dz <= max_step)

    curb_road_idx = valid_road_idx[is_curb]
    curb_sw_idx = matched_sw_idx[is_curb]

    is_curb_road[curb_road_idx] = True
    is_curb_sidewalk[curb_sw_idx] = True

    return {
        "is_drivable": is_drivable,
        "is_elevated": is_elevated,
        "is_curb_road": is_curb_road,
        "is_curb_sidewalk": is_curb_sidewalk,
        "curb_step_heights": dz[is_curb].astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Curb & Drivability GT")
    parser.add_argument("--sequence", type=str, default="00")
    parser.add_argument("--data-root", type=Path, default=Path("/Users/yoshiri/Documents/geowt/data/semantickitti/sequences"))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    seq_dir = args.data_root / args.sequence
    velodyne_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"

    bin_files = sorted(velodyne_dir.glob("*.bin"))[:args.limit]
    print(f"Processing {len(bin_files)} frames from sequence {args.sequence}...")

    total_curb_pts = 0
    all_step_heights = []

    for bin_file in bin_files:
        lbl_file = labels_dir / (bin_file.stem + ".label")
        pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        labels = np.fromfile(lbl_file, dtype=np.uint32) & 0xFFFF

        gt = extract_curb_gt_frame(pts, labels)
        n_curb = int(np.sum(gt["is_curb_road"]))
        total_curb_pts += n_curb
        all_step_heights.extend(gt["curb_step_heights"].tolist())

        print(f"  Frame {bin_file.stem}: Road={np.sum(gt['is_drivable']):5d}, SW={np.sum(gt['is_elevated']):5d}, Curb Points={n_curb:4d}")

    if all_step_heights:
        sh = np.array(all_step_heights)
        print(f"\nExtracted Ground Truth Curb Step Statistics (N={len(sh)} points):")
        print(f"  Mean step:   {np.mean(sh)*100:.1f} cm")
        print(f"  Median step: {np.median(sh)*100:.1f} cm")
        print(f"  Std:         {np.std(sh)*100:.1f} cm")
        print(f"  25%: {np.percentile(sh, 25)*100:.1f} cm, 75%: {np.percentile(sh, 75)*100:.1f} cm")


if __name__ == "__main__":
    main()
