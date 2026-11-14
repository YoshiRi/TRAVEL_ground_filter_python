#!/usr/bin/env python3
"""
TRAVEL Ground Filter — SemanticKITTI Quantitative Benchmark

Evaluates precision, recall, F1-score, IoU (overall and range-wise),
and processing latency on SemanticKITTI point clouds with ground-truth semantic labels.

Ground truth classes in SemanticKITTI:
  - 40: road
  - 44: parking
  - 48: sidewalk
  - 49: other-ground
  - 60: lane-marking
  - 72: terrain

Usage:
  # Evaluate all frames with default config
  uv run python tools/benchmark_semantickitti.py

  # Evaluate first 20 frames with adaptive LCC
  uv run python tools/benchmark_semantickitti.py --limit 20 --adaptive

  # Compare resolutions
  uv run python tools/benchmark_semantickitti.py --resolution 2.5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from travel_py.config import GlobalConfig, GridConfig, SeedConfig, AcceptConfig
from travel_py.pipeline import run_fast

# Ground truth semantic classes considered "ground"
GROUND_CLASSES = {40, 44, 48, 49, 60, 72}

DEFAULT_DATA_DIR = Path("/Users/yoshiri/Documents/geowt/data/semantickitti/sequences/00")

# Distance bands [min_dist, max_dist, label]
DISTANCE_BANDS = [
    (0.0, 20.0, "Near  (0-20m)"),
    (20.0, 40.0, "Mid   (20-40m)"),
    (40.0, 100.0, "Far   (40m+)"),
]


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou}


def evaluate_frame(
    pts_xyz: np.ndarray,
    gt_ground: np.ndarray,
    cfg: GlobalConfig,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]], float]:
    """
    Run TRAVEL on a single frame and compute confusion matrix components.
    """
    t0 = time.perf_counter()
    res = run_fast(pts_xyz, cfg, keep_grid=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pred_ground = (res.labels == 1)

    # Overall confusion
    tp = int(np.sum(gt_ground & pred_ground))
    fp = int(np.sum((~gt_ground) & pred_ground))
    fn = int(np.sum(gt_ground & (~pred_ground)))
    tn = int(np.sum((~gt_ground) & (~pred_ground)))
    overall = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    # Range-wise confusion
    dist = np.sqrt(pts_xyz[:, 0] ** 2 + pts_xyz[:, 1] ** 2)
    by_range = {}
    for r_min, r_max, name in DISTANCE_BANDS:
        mask = (dist >= r_min) & (dist < r_max)
        by_range[name] = {
            "tp": int(np.sum(gt_ground[mask] & pred_ground[mask])),
            "fp": int(np.sum((~gt_ground[mask]) & pred_ground[mask])),
            "fn": int(np.sum(gt_ground[mask] & (~pred_ground[mask]))),
            "tn": int(np.sum((~gt_ground[mask]) & (~pred_ground[mask]))),
        }

    return overall, by_range, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SemanticKITTI Benchmark for TRAVEL Ground Filter"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to SemanticKITTI sequence directory (containing velodyne/ and labels/)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=5.0,
        help="Grid resolution in meters (default: 5.0)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable TRAVEL official distance-dependent adaptive LCC",
    )
    parser.add_argument(
        "--prefer-center",
        action="store_true",
        default=True,
        help="Prefer seeds near the sensor origin (default: True)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of seed subcells to explore (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of frames to evaluate (default: all)",
    )
    args = parser.parse_args()

    velodyne_dir = args.data_dir / "velodyne"
    labels_dir = args.data_dir / "labels"

    if not velodyne_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Data directory missing velodyne/ or labels/: {args.data_dir}")

    bin_files = sorted(velodyne_dir.glob("*.bin"))
    if args.limit:
        bin_files = bin_files[: args.limit]

    print(f"=== TRAVEL Ground Filter — SemanticKITTI Benchmark ===")
    print(f"Sequence: {args.data_dir.name} ({len(bin_files)} frames)")
    print(f"Config: resolution={args.resolution}m, adaptive_lcc={args.adaptive}, top_k={args.top_k}, prefer_center={args.prefer_center}")
    print()

    # Grid config
    nx = int(np.ceil(170 / args.resolution))
    cfg = GlobalConfig(
        grid=GridConfig(
            resolution=args.resolution,
            origin_xy=(-85.0, -85.0),
            size_xy=(nx, nx),
        ),
        seed=SeedConfig(
            top_k=args.top_k,
            prefer_center=args.prefer_center,
        ),
        accept=AcceptConfig(
            max_height_diff=0.5,
            th_normal=0.9,
            use_adaptive_lcc=args.adaptive,
            th_lcc_normal_similarity=0.03,
            th_lcc_planar_dist=0.2,
        ),
    )

    total_overall = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    total_by_range = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for _, _, name in DISTANCE_BANDS}
    timings: List[float] = []

    for bin_path in tqdm(bin_files, desc="Benchmarking"):
        label_path = labels_dir / f"{bin_path.stem}.label"
        if not label_path.exists():
            continue

        pts_raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        pts_xyz = pts_raw[:, :3].astype(np.float64)
        raw_labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
        gt_ground = np.isin(raw_labels, list(GROUND_CLASSES))

        overall, by_range, elapsed_ms = evaluate_frame(pts_xyz, gt_ground, cfg)
        timings.append(elapsed_ms)

        for k in total_overall:
            total_overall[k] += overall[k]

        for name in total_by_range:
            for k in total_by_range[name]:
                total_by_range[name][k] += by_range[name][k]

    # Metrics computation
    overall_m = compute_metrics(**total_overall)
    avg_latency = float(np.mean(timings)) if timings else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("\n" + "=" * 60)
    print("                    BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Evaluated Frames : {len(timings):,}")
    print(f"Average Processing Time: {avg_latency:5.1f} ms ({fps:4.1f} FPS)")
    print(f"Overall IoU            : {overall_m['iou']*100:5.2f}%")
    print(f"Overall Precision      : {overall_m['precision']*100:5.2f}%")
    print(f"Overall Recall         : {overall_m['recall']*100:5.2f}%")
    print(f"Overall F1-Score       : {overall_m['f1']*100:5.2f}%")

    print("\nRange-wise Performance Breakdown:")
    print("-" * 60)
    print(f"{'Distance Range':<18} | {'IoU (%)':<8} | {'Precision (%)':<14} | {'Recall (%)':<10}")
    print("-" * 60)
    for _, _, name in DISTANCE_BANDS:
        rm = compute_metrics(**total_by_range[name])
        print(f"{name:<18} | {rm['iou']*100:7.2f}% | {rm['precision']*100:13.2f}% | {rm['recall']*100:9.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
