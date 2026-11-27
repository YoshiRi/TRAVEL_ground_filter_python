#!/usr/bin/env python3
"""
Synthetic Parametric Road-Curb-Sidewalk Generator

Generates clean, mathematically controlled point clouds of road with curbs:
- Road surface: width W_road, length L_road, slope_angle (pitch)
- Curb step: height h_curb in meters (e.g. 0.05m, 0.10m, 0.15m), bevel width w_curb
- Sidewalk surface: width W_sidewalk, elevated by h_curb
- Points generated with simulated LiDAR angular rings (or uniform density).

Usage:
  python tools/generate_synthetic_curb.py --height 0.10 --slope 0.0
"""
from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np


def generate_synthetic_road_curb(
    *,
    length: float = 40.0,
    road_width: float = 7.0,
    sidewalk_width: float = 3.0,
    curb_height: float = 0.10,
    road_slope_deg: float = 0.0,
    curb_bevel_width: float = 0.05,
    point_density_per_m2: float = 50.0,
    sensor_height: float = 1.73,
    noise_std: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic road, curb, and sidewalk point cloud.

    Returns
    -------
    pts_xyz : (N, 3) float32
    drivable_gt : (N,) bool (True for road, False for sidewalk)
    curb_gt : (N,) bool (True for points within bevel curb strip)
    """
    # X: Forward [-5.0, length - 5.0]
    # Y: Lateral [-road_width/2 - sidewalk_width, road_width/2 + sidewalk_width]
    # Z: -sensor_height on road, -sensor_height + curb_height on sidewalk
    
    total_width = road_width + 2 * sidewalk_width
    area = length * total_width
    num_points = int(area * point_density_per_m2)

    # Uniform random sampling in XY
    x = np.random.uniform(-5.0, length - 5.0, size=num_points).astype(np.float32)
    y = np.random.uniform(-total_width / 2.0, total_width / 2.0, size=num_points).astype(np.float32)

    half_road = road_width / 2.0
    abs_y = np.abs(y)

    # Base elevation with road slope (pitch)
    slope_rad = np.radians(road_slope_deg)
    base_z = -sensor_height + x * np.tan(slope_rad)

    z = np.zeros_like(x)
    drivable_gt = np.zeros(num_points, dtype=bool)
    curb_gt = np.zeros(num_points, dtype=bool)

    # 1. Road points
    road_mask = abs_y < (half_road - curb_bevel_width / 2.0)
    z[road_mask] = base_z[road_mask]
    drivable_gt[road_mask] = True

    # 2. Sidewalk points
    sw_mask = abs_y > (half_road + curb_bevel_width / 2.0)
    z[sw_mask] = base_z[sw_mask] + curb_height
    drivable_gt[sw_mask] = False

    # 3. Curb bevel strip points
    curb_mask = (~road_mask) & (~sw_mask)
    if np.sum(curb_mask) > 0:
        # Interpolate height across the curb bevel
        t = (abs_y[curb_mask] - (half_road - curb_bevel_width / 2.0)) / max(curb_bevel_width, 1e-4)
        z[curb_mask] = base_z[curb_mask] + t * curb_height
        curb_gt[curb_mask] = True
        drivable_gt[curb_mask] = False

    # Add Gaussian measurement noise
    if noise_std > 0:
        z += np.random.normal(0, noise_std, size=num_points).astype(np.float32)

    pts = np.column_stack([x, y, z])
    return pts, drivable_gt, curb_gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Curb Generator")
    parser.add_argument("--height", type=float, default=0.10, help="Curb height in meters")
    parser.add_argument("--slope", type=float, default=0.0, help="Road slope in degrees")
    args = parser.parse_args()

    pts, drivable, curb = generate_synthetic_road_curb(curb_height=args.height, road_slope_deg=args.slope)
    print(f"Generated {len(pts)} synthetic points:")
    print(f"  Road (Drivable): {np.sum(drivable)} points")
    print(f"  Sidewalk:        {np.sum((~drivable) & (~curb))} points")
    print(f"  Curb Bevel:      {np.sum(curb)} points")
    print(f"  Z min/max:       {pts[:, 2].min():.3f} / {pts[:, 2].max():.3f} m")


if __name__ == "__main__":
    main()
