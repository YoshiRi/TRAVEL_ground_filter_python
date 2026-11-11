#!/usr/bin/env python3
"""
Generate benchmark charts and qualitative point cloud evaluation figures.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from travel_py.config import GlobalConfig, GridConfig, SeedConfig, AcceptConfig
from travel_py.pipeline import run_fast

OUTPUT_DIR = Path("docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Quantitative Benchmark Chart
# -------------------------------------------------------------------------
def plot_metrics_chart():
    # Data from 101 frames benchmark
    categories = ["5.0m (Fixed)", "5.0m (Adaptive)", "2.5m (Fixed)", "2.5m (Adaptive)"]
    overall_iou = [82.74, 82.72, 84.74, 83.61]
    overall_prec = [85.61, 85.91, 91.29, 94.47]
    overall_rec = [96.11, 95.70, 92.19, 87.92]
    overall_f1 = [90.56, 90.55, 91.74, 91.07]
    latency_ms = [32.8, 32.6, 37.6, 37.6]

    near_iou = [83.41, 83.40, 86.38, 85.50]
    mid_iou = [73.81, 73.67, 58.44, 52.98]
    far_iou = [39.78, 37.75, 3.16, 1.38]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=200)

    # Subplot 1: Overall Metrics
    x = np.arange(len(categories))
    w = 0.2
    ax1 = axes[0]
    b1 = ax1.bar(x - 1.5*w, overall_iou, w, label="IoU", color="#2b5c8f")
    b2 = ax1.bar(x - 0.5*w, overall_prec, w, label="Precision", color="#4682b4")
    b3 = ax1.bar(x + 0.5*w, overall_rec, w, label="Recall", color="#5cb85c")
    b4 = ax1.bar(x + 1.5*w, overall_f1, w, label="F1-Score", color="#f0ad4e")

    ax1.set_title("Overall Accuracy Metrics (SemanticKITTI 00, 101 frames)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score (%)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9, rotation=15)
    ax1.set_ylim(70, 102)
    ax1.legend(loc="lower right", framealpha=0.9)
    for b in [b1, b2, b3, b4]:
        for bar in b:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    # Subplot 2: Range-wise IoU Breakdown
    ax2 = axes[1]
    ax2.plot(categories, near_iou, marker="o", linewidth=2.5, label="Near (0-20m)", color="#28a745")
    ax2.plot(categories, mid_iou, marker="s", linewidth=2.5, label="Mid (20-40m)", color="#ffc107")
    ax2.plot(categories, far_iou, marker="^", linewidth=2.5, label="Far (40m+)", color="#dc3545")
    ax2.set_title("IoU Breakdown by Distance Range", fontsize=11, fontweight="bold")
    ax2.set_ylabel("IoU (%)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9, rotation=15)
    ax2.set_ylim(-5, 105)
    ax2.legend(loc="center right", framealpha=0.9)
    for cat_i in range(len(categories)):
        ax2.text(cat_i, near_iou[cat_i] + 2.5, f"{near_iou[cat_i]:.1f}%", ha="center", fontsize=8, color="#28a745")
        ax2.text(cat_i, mid_iou[cat_i] + 2.5, f"{mid_iou[cat_i]:.1f}%", ha="center", fontsize=8, color="#b28900")
        ax2.text(cat_i, max(far_iou[cat_i] + 2.5, 4), f"{far_iou[cat_i]:.1f}%", ha="center", fontsize=8, color="#dc3545")

    # Subplot 3: Latency & Frame Rate
    ax3 = axes[2]
    colors = ["#3498db", "#2980b9", "#9b59b6", "#8e44ad"]
    bars = ax3.bar(categories, latency_ms, color=colors, width=0.5)
    ax3.set_title("Average Processing Latency (M3 Max, Python)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Latency (ms / frame)", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=9, rotation=15)
    ax3.set_ylim(0, 50)
    ax3.axhline(100.0, color="gray", linestyle="--", alpha=0.5, label="10 Hz Budget (100ms)")
    for bar in bars:
        h = bar.get_height()
        fps = 1000.0 / h
        ax3.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f} ms\n({fps:.1f} FPS)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "benchmark_metrics.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved: {chart_path}")
    return chart_path


# -------------------------------------------------------------------------
# 2. Qualitative Point Cloud Visualisation (Frame 000000)
# -------------------------------------------------------------------------
def plot_qualitative_eval():
    bin_path = Path("/Users/yoshiri/Documents/geowt/data/semantickitti/sequences/00/velodyne/000000.bin")
    label_path = Path("/Users/yoshiri/Documents/geowt/data/semantickitti/sequences/00/labels/000000.label")

    if not bin_path.exists() or not label_path.exists():
        print("Data files not found, skipping qualitative plot.")
        return None

    pts_raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pts_xyz = pts_raw[:, :3].astype(np.float64)
    raw_labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
    ground_classes = {40, 44, 48, 49, 60, 72}
    gt_ground = np.isin(raw_labels, list(ground_classes))

    # Run TRAVEL with resolution 5.0m
    nx = int(np.ceil(170 / 5.0))
    cfg = GlobalConfig(
        grid=GridConfig(resolution=5.0, origin_xy=(-85.0, -85.0), size_xy=(nx, nx)),
        seed=SeedConfig(top_k=5, prefer_center=True),
        accept=AcceptConfig(max_height_diff=0.5, th_normal=0.9),
    )
    res = run_fast(pts_xyz, cfg, keep_grid=False)
    pred_ground = (res.labels == 1)

    # Confusion categories
    # TP: Ground correctly identified (Green)
    # TN: Non-ground correctly rejected (Light Gray)
    # FP: Non-ground falsely marked as ground (Red)
    # FN: Ground missed (Orange)
    tp = gt_ground & pred_ground
    tn = (~gt_ground) & (~pred_ground)
    fp = (~gt_ground) & pred_ground
    fn = gt_ground & (~pred_ground)

    # Downsample points for clear visualization
    subsample_idx = np.random.choice(len(pts_xyz), size=min(60000, len(pts_xyz)), replace=False)
    x = pts_xyz[subsample_idx, 0]
    y = pts_xyz[subsample_idx, 1]
    z = pts_xyz[subsample_idx, 2]

    gt_sub = gt_ground[subsample_idx]
    pred_sub = pred_ground[subsample_idx]
    tp_sub = tp[subsample_idx]
    tn_sub = tn[subsample_idx]
    fp_sub = fp[subsample_idx]
    fn_sub = fn[subsample_idx]

    # Crop to relevant region (-50m to +50m)
    crop_mask = (np.abs(x) <= 50) & (np.abs(y) <= 50)
    x = x[crop_mask]
    y = y[crop_mask]
    z = z[crop_mask]
    gt_sub = gt_sub[crop_mask]
    pred_sub = pred_sub[crop_mask]
    tp_sub = tp_sub[crop_mask]
    tn_sub = tn_sub[crop_mask]
    fp_sub = fp_sub[crop_mask]
    fn_sub = fn_sub[crop_mask]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), dpi=200)

    # Panel 1: Ground Truth (BEV)
    ax1 = axes[0]
    ax1.scatter(x[~gt_sub], y[~gt_sub], s=0.4, c="#a0a0a0", alpha=0.4, label="Non-Ground (Obstacles/Vegetation)")
    ax1.scatter(x[gt_sub], y[gt_sub], s=0.6, c="#2ecc71", alpha=0.7, label="Ground Truth (Road/Sidewalk/Terrain)")
    ax1.plot(0, 0, marker="^", color="red", markersize=8, label="Ego Sensor")
    ax1.set_title("(1) SemanticKITTI Ground Truth (Frame 000000)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (forward, m)", fontsize=9)
    ax1.set_ylabel("Y (left, m)", fontsize=9)
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-50, 50)
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right", markerscale=6, framealpha=0.9, fontsize=8)

    # Panel 2: TRAVEL Prediction (BEV)
    ax2 = axes[1]
    ax2.scatter(x[~pred_sub], y[~pred_sub], s=0.4, c="#a0a0a0", alpha=0.4, label="Non-Ground")
    ax2.scatter(x[pred_sub], y[pred_sub], s=0.6, c="#3498db", alpha=0.7, label="TRAVEL Ground Filtered")
    ax2.plot(0, 0, marker="^", color="red", markersize=8, label="Ego Sensor")
    ax2.set_title("(2) TRAVEL Fast Prediction (res=5.0m, 33ms)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (forward, m)", fontsize=9)
    ax2.set_ylabel("Y (left, m)", fontsize=9)
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", markerscale=6, framealpha=0.9, fontsize=8)

    # Panel 3: Confusion Map (Error Analysis)
    ax3 = axes[2]
    ax3.scatter(x[tn_sub], y[tn_sub], s=0.3, c="#d0d0d0", alpha=0.3, label="TN: Correct Obstacle")
    ax3.scatter(x[tp_sub], y[tp_sub], s=0.5, c="#27ae60", alpha=0.6, label="TP: Correct Ground (98.6%)")
    ax3.scatter(x[fp_sub], y[fp_sub], s=1.2, c="#e74c3c", alpha=0.9, label="FP: Over-segmented Obstacle (4.8%)")
    ax3.scatter(x[fn_sub], y[fn_sub], s=1.5, c="#f39c12", alpha=0.9, label="FN: Missed Ground (1.4%)")
    ax3.plot(0, 0, marker="^", color="black", markersize=8, label="Ego Sensor")
    ax3.set_title("(3) Confusion Map (Precision: 90.4%, Recall: 98.6%, IoU: 89.3%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X (forward, m)", fontsize=9)
    ax3.set_ylabel("Y (left, m)", fontsize=9)
    ax3.set_xlim(-50, 50)
    ax3.set_ylim(-50, 50)
    ax3.set_aspect("equal")
    ax3.legend(loc="upper right", markerscale=5, framealpha=0.9, fontsize=8)

    plt.tight_layout()
    qual_path = OUTPUT_DIR / "kitti_000000_eval.png"
    plt.savefig(qual_path)
    plt.close()
    print(f"Saved: {qual_path}")
    return qual_path


if __name__ == "__main__":
    plot_metrics_chart()
    plot_qualitative_eval()
