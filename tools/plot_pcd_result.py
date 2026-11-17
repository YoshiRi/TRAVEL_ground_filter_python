#!/usr/bin/env python3
"""
Visualize the extracted ground and non-ground PCD map.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

OUTPUT_DIR = Path("docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    pcd_path = Path("data/sample_pcd/pointcloud_map.pcd")
    ground_path = Path("data/sample_pcd/ground.pcd")
    nonground_path = Path("data/sample_pcd/nonground.pcd")

    if not pcd_path.exists() or not ground_path.exists():
        print("PCD files not found.")
        return

    pts_orig = np.asarray(o3d.io.read_point_cloud(str(pcd_path)).points)
    pts_gnd = np.asarray(o3d.io.read_point_cloud(str(ground_path)).points)
    pts_nongnd = np.asarray(o3d.io.read_point_cloud(str(nonground_path)).points)

    # Normalize to local origin (relative coordinates)
    x0, y0 = pts_orig.min(axis=0)[:2]
    pts_orig_rel = pts_orig - [x0, y0, 0]
    pts_gnd_rel = pts_gnd - [x0, y0, 0]
    pts_nongnd_rel = pts_nongnd - [x0, y0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=200)

    # Panel 1: Original PCD Map
    ax1 = axes[0]
    ax1.scatter(pts_orig_rel[:, 0], pts_orig_rel[:, 1], s=0.3, c=pts_orig_rel[:, 2], cmap="viridis", alpha=0.6)
    ax1.set_title(f"Original PCD Map ({len(pts_orig):,} points)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (East, m)", fontsize=9)
    ax1.set_ylabel("Y (North, m)", fontsize=9)
    ax1.set_aspect("equal")

    # Panel 2: Extracted Ground Map
    ax2 = axes[1]
    ax2.scatter(pts_nongnd_rel[:, 0], pts_nongnd_rel[:, 1], s=0.2, c="#d0d0d0", alpha=0.3, label="Non-Ground (Obstacles/Buildings)")
    ax2.scatter(pts_gnd_rel[:, 0], pts_gnd_rel[:, 1], s=0.6, c="#2ecc71", alpha=0.8, label=f"Extracted Ground ({len(pts_gnd):,} pts, 14.1%)")
    ax2.set_title("TRAVEL Tiling Ground Extraction (2.5m, 14-nbr)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (East, m)", fontsize=9)
    ax2.set_ylabel("Y (North, m)", fontsize=9)
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", markerscale=5, framealpha=0.9, fontsize=8)

    # Panel 3: Elevation colored Ground Map
    ax3 = axes[2]
    sc = ax3.scatter(pts_gnd_rel[:, 0], pts_gnd_rel[:, 1], s=0.6, c=pts_gnd_rel[:, 2], cmap="terrain", alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Elevation Z (m)", fontsize=9)
    ax3.set_title(f"Extracted Ground Elevation Surface", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X (East, m)", fontsize=9)
    ax3.set_ylabel("Y (North, m)", fontsize=9)
    ax3.set_aspect("equal")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "pcd_map_tiling_eval.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
