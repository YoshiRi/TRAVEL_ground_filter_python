#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


# =========================
# Config
# =========================

@dataclass
class CropConfig:
    x_range: float = 80.0
    y_range: float = 80.0
    radius: float | None = None

    auto_center: bool = False
    center_method: str = "median"  # median / mean

    normalize_xy: bool = True


# =========================
# Core functions
# =========================

def load_pcd(pcd_path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise RuntimeError(f"No points loaded from {pcd_path}")
    return points


def compute_center(points: np.ndarray, method: str) -> tuple[float, float]:
    if method == "mean":
        return float(points[:, 0].mean()), float(points[:, 1].mean())
    return float(np.median(points[:, 0])), float(np.median(points[:, 1]))


def crop_relative_xy(
    points: np.ndarray,
    cx: float,
    cy: float,
    cfg: CropConfig,
) -> np.ndarray:
    mask = (
        (points[:, 0] >= cx - cfg.x_range) & (points[:, 0] <= cx + cfg.x_range) &
        (points[:, 1] >= cy - cfg.y_range) & (points[:, 1] <= cy + cfg.y_range)
    )
    return points[mask]


def crop_radius(
    points: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> np.ndarray:
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    return points[(dx * dx + dy * dy) <= radius * radius]


def normalize_xy(points: np.ndarray, cx: float, cy: float) -> np.ndarray:
    out = points.copy()
    out[:, 0] -= cx
    out[:, 1] -= cy
    return out


def visualize_points(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.2, 0.7, 0.9])
    o3d.visualization.draw_geometries([pcd])


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Crop PCD and save as npy (auto / absolute center supported)"
    )
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)

    # crop size
    parser.add_argument("--x-range", type=float, default=CropConfig.x_range)
    parser.add_argument("--y-range", type=float, default=CropConfig.y_range)
    parser.add_argument("--radius", type=float, default=None)

    # center handling
    parser.add_argument(
        "--auto-center",
        action="store_true",
        help="Automatically compute center from point cloud (relative crop)",
    )
    parser.add_argument(
        "--center-method",
        choices=["median", "mean"],
        default=CropConfig.center_method,
    )

    # normalization
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable XY normalization",
    )

    # visualization
    parser.add_argument("--viz", action="store_true")

    args = parser.parse_args()

    cfg = CropConfig(
        x_range=args.x_range,
        y_range=args.y_range,
        radius=args.radius,
        auto_center=args.auto_center,
        center_method=args.center_method,
        normalize_xy=not args.no_normalize,
    )

    points = load_pcd(args.pcd)

    # center 결정
    if cfg.auto_center:
        cx, cy = compute_center(points, cfg.center_method)
        center_desc = f"auto ({cfg.center_method})"
    else:
        cx, cy = 0.0, 0.0
        center_desc = "fixed (0,0)"

    # crop
    if cfg.radius is not None:
        cropped = crop_radius(points, cx, cy, cfg.radius)
        crop_desc = f"radius <= {cfg.radius}"
    else:
        cropped = crop_relative_xy(points, cx, cy, cfg)
        crop_desc = f"x±{cfg.x_range}, y±{cfg.y_range}"

    if cropped.size == 0:
        raise RuntimeError("No points left after cropping")

    # normalize
    if cfg.normalize_xy:
        cropped = normalize_xy(cropped, cx, cy)
        norm_desc = "normalized"
    else:
        norm_desc = "not normalized"

    if args.viz:
        visualize_points(cropped)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, cropped)

    print("=== pcd_to_npy ===")
    print(f"Input        : {args.pcd}")
    print(f"Output       : {args.out}")
    print(f"Center       : {center_desc}")
    print(f"Crop         : {crop_desc}")
    print(f"Normalization: {norm_desc}")
    print(f"Points       : {points.shape[0]} -> {cropped.shape[0]}")


if __name__ == "__main__":
    main()
