#!/usr/bin/env python3
"""
Layer 3 Drivability & Curb Post-Processor Benchmark

Evaluates:
  1. Synthetic Parametric Sensitivity Curve (Curb 0..20cm x Slope 0..5 deg)
  2. SemanticKITTI Sequence 00 & 08 (Drivable Road IoU, Road Recall, Sidewalk Rejection,
     Physical Curb Edge Recall, and Layer 3 Latency overhead)
  3. Optional qualitative BEV plot comparing Layer 2 ground vs Layer 3 Drivability.

Usage:
  python tools/benchmark_drivability.py --plot-out /path/to/bev.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from travel_py.config import (
    AcceptConfig,
    DrivabilityConfig,
    GlobalConfig,
    GridConfig,
    ReachabilityConfig,
    SeedConfig,
)
from travel_py.drivability import DRIVABLE_CURB, DRIVABLE_ELEVATED, DRIVABLE_ROAD
from travel_py.pipeline import run_fast
from tools.generate_curb_gt import extract_curb_gt_frame
from tools.generate_synthetic_curb import generate_synthetic_road_curb

DATA_ROOT = Path("/Users/yoshiri/Documents/geowt/data/semantickitti/sequences")


def make_cfg(driv_enabled: bool = True) -> GlobalConfig:
    return GlobalConfig(
        grid=GridConfig(resolution=2.5),
        seed=SeedConfig(prefer_center=True, top_k=5, re_explore=True, max_components=50),
        accept=AcceptConfig(use_adaptive_lcc=True, use_14_neighbors=True),
        reachability=ReachabilityConfig(enabled=True),
        drivability=DrivabilityConfig(enabled=driv_enabled),
    )


def run_synthetic_curve() -> None:
    print("=== 1. SYNTHETIC PARAMETRIC SENSITIVITY CURVE ===")
    print(f"{'Slope':>6} | {'Curb':>5} | {'RoadRec':>8} | {'SW_Rej':>8} | {'DrivIoU':>8} | {'L3 Time':>8}")
    print("-" * 58)
    cfg = make_cfg(driv_enabled=True)
    for slope in [0.0, 5.0]:
        for h_cm in [0, 5, 8, 10, 15, 20]:
            np.random.seed(42)
            pts, driv_gt, _ = generate_synthetic_road_curb(
                curb_height=h_cm / 100.0,
                road_slope_deg=slope,
                noise_std=0.005,
            )
            res = run_fast(pts, cfg, keep_grid=False)
            l3_ms = res.timings.get("drivability", 0.0) * 1000.0
            roi = np.hypot(pts[:, 0], pts[:, 1]) < 30.0
            pred_driv = res.drivability_levels == DRIVABLE_ROAD

            tp = int(np.sum(driv_gt[roi] & pred_driv[roi]))
            fp = int(np.sum((~driv_gt[roi]) & pred_driv[roi]))
            fn = int(np.sum(driv_gt[roi] & (~pred_driv[roi])))
            tn = int(np.sum((~driv_gt[roi]) & (~pred_driv[roi])))
            road_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sw_rej = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            print(f"{slope:5.1f}d | {h_cm:3d}cm | {road_rec*100:7.1f}% | {sw_rej*100:7.1f}% | {iou*100:7.1f}% | {l3_ms:6.2f}ms")


def run_kitti_eval(seq: str, limit: int | None = None) -> Dict[str, float]:
    seq_dir = DATA_ROOT / seq
    bin_files = sorted((seq_dir / "velodyne").glob("*.bin"))
    if limit is not None:
        bin_files = bin_files[:limit]

    cfg = make_cfg(driv_enabled=True)

    # Accumulators: Layer 2 baseline (all ground=drivable) vs Layer 3
    base_tp = base_fp = base_fn = base_tn = 0
    l3_tp = l3_fp = l3_fn = l3_tn = 0
    curb_gt_total = 0
    curb_gt_matched = 0
    l3_times: List[float] = []
    tot_times: List[float] = []

    for bf in bin_files:
        lf = seq_dir / "labels" / (bf.stem + ".label")
        pts = np.fromfile(bf, dtype=np.float32).reshape(-1, 4)[:, :3]
        labels = np.fromfile(lf, dtype=np.uint32) & 0xFFFF
        gt = extract_curb_gt_frame(pts, labels, min_step=0.06, max_step=0.25)

        res = run_fast(pts, cfg, keep_grid=False)
        l3_ms = res.timings.get("drivability", 0.0) * 1000.0
        tot_ms = sum(res.timings.values()) * 1000.0
        l3_times.append(l3_ms)
        tot_times.append(tot_ms)

        roi = np.hypot(pts[:, 0], pts[:, 1]) < 30.0
        eval_mask = roi & (gt["is_drivable"] | gt["is_elevated"])
        gt_d = gt["is_drivable"][eval_mask]
        gt_e = gt["is_elevated"][eval_mask]

        # Baseline (Layer 1+2): all confirmed ground treated as drivable
        b_pred = (res.labels == 1)[eval_mask]
        base_tp += int(np.sum(gt_d & b_pred))
        base_fp += int(np.sum(gt_e & b_pred))
        base_fn += int(np.sum(gt_d & (~b_pred)))
        base_tn += int(np.sum(gt_e & (~b_pred)))

        # Layer 3: only DRIVABLE_ROAD treated as drivable
        l3_pred = (res.drivability_levels == DRIVABLE_ROAD)[eval_mask]
        l3_tp += int(np.sum(gt_d & l3_pred))
        l3_fp += int(np.sum(gt_e & l3_pred))
        l3_fn += int(np.sum(gt_d & (~l3_pred)))
        l3_tn += int(np.sum(gt_e & (~l3_pred)))

        # Physical curb edge recall (within 0.8m of detected DRIVABLE_CURB)
        curb_gt_mask = roi & (gt["is_curb_road"] | gt["is_curb_sidewalk"])
        curb_pred_mask = roi & (res.drivability_levels == DRIVABLE_CURB)
        n_cgt = int(np.sum(curb_gt_mask))
        curb_gt_total += n_cgt
        if n_cgt > 0 and np.any(curb_pred_mask):
            tree = cKDTree(pts[curb_pred_mask, :2])
            dists, _ = tree.query(pts[curb_gt_mask, :2], k=1)
            curb_gt_matched += int(np.sum(dists <= 0.80))

    b_rec = base_tp / max(1, base_tp + base_fn)
    b_sw = base_tn / max(1, base_tn + base_fp)
    b_iou = base_tp / max(1, base_tp + base_fp + base_fn)

    l3_rec = l3_tp / max(1, l3_tp + l3_fn)
    l3_sw = l3_tn / max(1, l3_tn + l3_fp)
    l3_iou = l3_tp / max(1, l3_tp + l3_fp + l3_fn)
    curb_rec = curb_gt_matched / max(1, curb_gt_total)

    print(f"\n=== 2. SEMANTICKITTI SEQ {seq} ({len(bin_files)} frames, ROI < 30m) ===")
    print(f"  Layer 1+2 Baseline : RoadRec={b_rec*100:5.1f}%, SW_Rej={b_sw*100:5.1f}%, Drivable IoU={b_iou*100:5.1f}%")
    print(f"  + Layer 3 Polar    : RoadRec={l3_rec*100:5.1f}%, SW_Rej={l3_sw*100:5.1f}%, Drivable IoU={l3_iou*100:5.1f}%")
    print(f"  Physical Curb Recall (>=6cm, tol=0.8m): {curb_rec*100:5.1f}% ({curb_gt_matched}/{curb_gt_total} pts)")
    print(f"  Latency: Layer 3 = {np.mean(l3_times):.2f} ms (p95={np.percentile(l3_times, 95):.2f} ms) | Total = {np.mean(tot_times):.1f} ms")

    return {
        "b_rec": b_rec, "b_sw": b_sw, "b_iou": b_iou,
        "l3_rec": l3_rec, "l3_sw": l3_sw, "l3_iou": l3_iou,
        "curb_rec": curb_rec, "l3_ms": float(np.mean(l3_times)),
        "tot_ms": float(np.mean(tot_times)),
    }


def save_qualitative_plot(out_path: Path, seq: str = "00", frame: str = "000045") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq_dir = DATA_ROOT / seq
    pts = np.fromfile(seq_dir / "velodyne" / f"{frame}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    labels = np.fromfile(seq_dir / "labels" / f"{frame}.label", dtype=np.uint32) & 0xFFFF
    gt = extract_curb_gt_frame(pts, labels, min_step=0.06, max_step=0.25)

    res = run_fast(pts, make_cfg(driv_enabled=True), keep_grid=False)
    dl = res.drivability_levels

    roi = (np.abs(pts[:, 0]) <= 30.0) & (np.abs(pts[:, 1]) <= 30.0)
    p = pts[roi]
    dl_r = dl[roi]
    gt_r = gt["is_drivable"][roi]
    gt_s = gt["is_elevated"][roi]
    gt_c = (gt["is_curb_road"] | gt["is_curb_sidewalk"])[roi]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=150)

    # Panel 1: SemanticKITTI GT (Road vs Sidewalk vs Physical Curb)
    ax = axes[0]
    other = ~(gt_r | gt_s)
    ax.scatter(p[other, 0], p[other, 1], c="#cccccc", s=0.3, alpha=0.4, label="Non-Ground")
    ax.scatter(p[gt_r, 0], p[gt_r, 1], c="#2ecc71", s=0.6, alpha=0.7, label="GT Road (Drivable)")
    ax.scatter(p[gt_s, 0], p[gt_s, 1], c="#e67e22", s=0.6, alpha=0.8, label="GT Sidewalk")
    if np.any(gt_c):
        ax.scatter(p[gt_c, 0], p[gt_c, 1], c="#e74c3c", s=4.0, label="GT Curb (>=6cm)")
    ax.set_title(f"Ground Truth (Seq {seq} Frame {frame})")
    ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); ax.set_aspect("equal")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.legend(loc="upper right", markerscale=4, fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 2: Layer 3 Output
    ax = axes[1]
    m0 = dl_r == 0
    m1 = dl_r == DRIVABLE_ROAD
    m2 = dl_r == DRIVABLE_ELEVATED
    m3 = dl_r == DRIVABLE_CURB
    ax.scatter(p[m0, 0], p[m0, 1], c="#cccccc", s=0.3, alpha=0.4, label="Non-Ground (L1+L2)")
    ax.scatter(p[m1, 0], p[m1, 1], c="#2ecc71", s=0.6, alpha=0.7, label="DRIVABLE_ROAD (L3)")
    ax.scatter(p[m2, 0], p[m2, 1], c="#9b59b6", s=0.6, alpha=0.8, label="DRIVABLE_ELEVATED (L3)")
    ax.scatter(p[m3, 0], p[m3, 1], c="#e74c3c", s=2.5, label="DRIVABLE_CURB (L3)")
    l3_ms = res.timings.get("drivability", 0.0) * 1000.0
    ax.set_title(f"Layer 3 Polar Drivability (+{l3_ms:.2f} ms)")
    ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); ax.set_aspect("equal")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.legend(loc="upper right", markerscale=4, fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved qualitative BEV plot to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Layer 3 Drivability & Curb Post-Processor")
    parser.add_argument("--limit", type=int, default=None, help="Limit KITTI frames per sequence")
    parser.add_argument("--plot-out", type=Path, default=None, help="Optional path to save BEV plot")
    args = parser.parse_args()

    run_synthetic_curve()
    run_kitti_eval("00", limit=args.limit)
    run_kitti_eval("08", limit=args.limit)

    if args.plot_out is not None:
        save_qualitative_plot(args.plot_out)


if __name__ == "__main__":
    main()
