"""
Layer 3 Post-Processor: Lightweight Polar Ray-Marching Curb & Drivability Verification.

==============================================================================
ARCHITECTURE BOUNDARY (POST-PROCESSING ONLY)
==============================================================================
This module is a pure, optional post-processor that operates AFTER:
  - Layer 1 (`fast.py`): Pure TRAVEL ground segmentation
  - Layer 2 (`reachability.py`): B+D+E multi-level ground verification

It does NOT modify Layer 1 or Layer 2 ground classification. Instead, it takes
the confirmed ground point mask and ego road plane normal, and performs an
ultra-lightweight (~2-4 ms) outward polar ray-marching sweep across `n_sectors`
azimuth rays to separate:
  - DRIVABLE_ROAD (1): Ground continuously reachable from ego without crossing
    a positive step discontinuity (curb).
  - DRIVABLE_ELEVATED (2): Confirmed ground lying behind a detected step/curb
    barrier (e.g., sidewalks, raised medians, platforms).
  - DRIVABLE_CURB (3): Points lying directly on the detected step/curb band.
  - DRIVABLE_NONE (0): Non-ground points.

Key properties:
  - No KD-Tree or kNN search (pure C-level `np.bincount` + vectorized ray sweep).
  - Slope-compensated using ego road normal + gentle local radial slope tracking,
    preventing false curb triggers on steep hills (e.g., 10%+ grade).
  - Immune to 1-pixel pinhole flood-fill leakage and far-range LiDAR ring gaps.
==============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .config import DrivabilityConfig

try:
    from scipy.ndimage import median_filter as _median_filter
except ImportError:  # pragma: no cover - pure NumPy fallback
    _median_filter = None


# Drivability level codes (point level)
DRIVABLE_NONE = 0      # Non-ground
DRIVABLE_ROAD = 1      # Traversable road (no step barrier from ego)
DRIVABLE_ELEVATED = 2  # Confirmed ground behind a curb/step (e.g. sidewalk)
DRIVABLE_CURB = 3      # Detected curb / step transition band


@dataclass
class DrivabilityResult:
    """Output of Layer 3 polar ray-marching drivability verification."""
    point_levels: np.ndarray     # (N,) int8 (DRIVABLE_NONE/ROAD/ELEVATED/CURB)
    curb_hit_r: np.ndarray       # (n_sectors,) float64 detected curb radius per azimuth sector
    stats: Dict[str, int] = field(default_factory=dict)


def _circular_median_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Circular 1D median filter with pure NumPy fallback."""
    if window < 3:
        return arr
    if _median_filter is not None:
        return _median_filter(arr, size=window, mode="wrap")
    pad = window // 2
    padded = np.concatenate([arr[-pad:], arr, arr[:pad]])
    # Sliding window view
    n = arr.shape[0]
    out = np.empty_like(arr)
    for i in range(n):
        out[i] = np.median(padded[i : i + window])
    return out


def verify_drivability(
    points_xyz: np.ndarray,
    ground_mask: np.ndarray,
    ego_normal: np.ndarray,
    cfg: DrivabilityConfig,
) -> DrivabilityResult:
    """
    Separate confirmed ground points into DRIVABLE_ROAD, DRIVABLE_ELEVATED
    (e.g., sidewalks behind curbs), and DRIVABLE_CURB via outward polar
    ray-marching with slope compensation.

    Parameters
    ----------
    points_xyz : (N, 3) array
        Input 3D point cloud in sensor/ego frame.
    ground_mask : (N,) bool array
        Confirmed ground points from Layer 1 (+ optional Layer 2).
    ego_normal : (3,) array
        Unit normal vector of the ego-anchored primary road plane (nz > 0).
    cfg : DrivabilityConfig
        Layer 3 polar ray-marching parameters.
    """
    N = int(points_xyz.shape[0])
    n_sectors = int(cfg.n_sectors)
    max_range = float(cfg.max_range)
    min_range = float(cfg.min_range)
    dr = float(cfg.radial_bin_size)

    point_levels = np.zeros(N, dtype=np.int8)
    curb_hit_r = np.full(n_sectors, max_range + 1.0, dtype=np.float64)

    gnd_idx = np.nonzero(ground_mask)[0]
    if gnd_idx.size == 0:
        return DrivabilityResult(
            point_levels=point_levels,
            curb_hit_r=curb_hit_r,
            stats={"road": 0, "elevated": 0, "curb": 0, "sectors_hit": 0},
        )

    # Default confirmed ground outside ROI to DRIVABLE_ROAD
    point_levels[gnd_idx] = DRIVABLE_ROAD

    gx = points_xyz[gnd_idx, 0]
    gy = points_xyz[gnd_idx, 1]
    gz = points_xyz[gnd_idx, 2].astype(np.float64)

    r = np.hypot(gx, gy)
    in_roi = (r >= min_range) & (r < max_range)
    roi_idx = gnd_idx[in_roi]
    if roi_idx.size == 0:
        return DrivabilityResult(
            point_levels=point_levels,
            curb_hit_r=curb_hit_r,
            stats={"road": int(gnd_idx.size), "elevated": 0, "curb": 0, "sectors_hit": 0},
        )

    rx = gx[in_roi]
    ry = gy[in_roi]
    rz = gz[in_roi]
    rr = r[in_roi]

    # 1. Polar binning: (n_sectors, n_rbins)
    n_rbins = max(1, int(np.ceil((max_range - min_range) / dr)))
    theta = np.arctan2(ry, rx)  # [-pi, pi]
    inv_dtheta = n_sectors / (2.0 * np.pi)
    t_bin = ((theta + np.pi) * inv_dtheta).astype(np.int32)
    np.clip(t_bin, 0, n_sectors - 1, out=t_bin)

    r_bin = ((rr - min_range) / dr).astype(np.int32)
    np.clip(r_bin, 0, n_rbins - 1, out=r_bin)

    flat_bin = t_bin * n_rbins + r_bin
    n_bins = n_sectors * n_rbins

    counts = np.bincount(flat_bin, minlength=n_bins)
    valid = counts > 0
    z_sum = np.bincount(flat_bin, weights=rz, minlength=n_bins)
    r_sum = np.bincount(flat_bin, weights=rr, minlength=n_bins)

    inv_c = np.zeros(n_bins, dtype=np.float64)
    inv_c[valid] = 1.0 / counts[valid]
    z_2d = (z_sum * inv_c).reshape(n_sectors, n_rbins)
    r_2d = (r_sum * inv_c).reshape(n_sectors, n_rbins)
    valid_2d = valid.reshape(n_sectors, n_rbins)

    # 2. Expected radial slope along each azimuth sector from ego road normal
    sector_angles = -np.pi + (np.arange(n_sectors) + 0.5) * (2.0 * np.pi / n_sectors)
    nz = max(float(ego_normal[2]), 0.2)
    s_cur = -(
        float(ego_normal[0]) * np.cos(sector_angles)
        + float(ego_normal[1]) * np.sin(sector_angles)
    ) / nz

    # 3. Vectorized outward ray-marching with 3-bin sliding history
    #    Catches both perpendicular curbs (1-bin jump) and shallow grazing curbs (2-3 bins)
    hist_cnt = np.zeros(n_sectors, dtype=np.int32)
    r_p1 = np.zeros(n_sectors, dtype=np.float64)
    z_p1 = np.zeros(n_sectors, dtype=np.float64)
    r_p2 = np.zeros(n_sectors, dtype=np.float64)
    z_p2 = np.zeros(n_sectors, dtype=np.float64)
    r_p3 = np.zeros(n_sectors, dtype=np.float64)
    z_p3 = np.zeros(n_sectors, dtype=np.float64)

    hit_already = np.zeros(n_sectors, dtype=bool)

    step_h = float(cfg.step_height_thresh)
    step_g = float(cfg.step_slope_thresh)
    win_h = float(cfg.window_height_thresh)
    win_g = float(cfg.window_slope_thresh)
    slope_alpha = float(cfg.slope_adapt_alpha)

    for j in range(n_rbins):
        v = valid_2d[:, j] & (~hit_already)
        if not np.any(v):
            continue

        rc = r_2d[:, j]
        zc = z_2d[:, j]

        init1 = v & (hist_cnt == 0)
        if np.any(init1):
            hist_cnt[init1] = 1
            r_p1[init1] = rc[init1]
            z_p1[init1] = zc[init1]

        step_mask = v & (hist_cnt >= 1) & (~init1)
        if np.any(step_mask):
            idx_all = np.nonzero(step_mask)[0]
            rc_s = rc[idx_all]
            zc_s = zc[idx_all]
            sc_s = s_cur[idx_all]

            # 1-bin step check
            dr1 = np.maximum(rc_s - r_p1[idx_all], 0.05)
            dz1_raw = zc_s - z_p1[idx_all]
            dz1 = dz1_raw - sc_s * dr1
            g1 = dz1 / dr1
            hit1 = (dz1 >= step_h) & (g1 >= step_g)

            # 2-bin window check
            has2 = hist_cnt[idx_all] >= 2
            dr2 = np.maximum(rc_s - r_p2[idx_all], 0.05)
            dz2 = (zc_s - z_p2[idx_all]) - sc_s * dr2
            g2 = dz2 / dr2
            hit2 = has2 & (dz2 >= win_h) & (g2 >= win_g)

            # 3-bin window check (catches shallow grazing curbs)
            has3 = hist_cnt[idx_all] >= 3
            dr3 = np.maximum(rc_s - r_p3[idx_all], 0.05)
            dz3 = (zc_s - z_p3[idx_all]) - sc_s * dr3
            g3 = dz3 / dr3
            hit3 = has3 & (dz3 >= win_h) & (g3 >= win_g * 0.85)

            is_curb = hit1 | hit2 | hit3

            hit_idx = idx_all[is_curb]
            if hit_idx.size > 0:
                hit_already[hit_idx] = True
                curb_hit_r[hit_idx] = rc[hit_idx]

            ok_mask = ~is_curb
            if np.any(ok_mask):
                ok_idx = idx_all[ok_mask]
                # Adapt local radial slope ONLY on smooth road (not climbing a grazing curb)
                if slope_alpha > 0.0:
                    smooth = ok_mask & (np.abs(dz1) < 0.02) & (dr1 >= 0.25)
                    if np.any(smooth):
                        sm_idx = idx_all[smooth]
                        inst_slope = dz1_raw[smooth] / dr1[smooth]
                        s_cur[sm_idx] = (1.0 - slope_alpha) * s_cur[sm_idx] + slope_alpha * inst_slope

                r_p3[ok_idx] = r_p2[ok_idx]
                z_p3[ok_idx] = z_p2[ok_idx]
                r_p2[ok_idx] = r_p1[ok_idx]
                z_p2[ok_idx] = z_p1[ok_idx]
                r_p1[ok_idx] = rc_s[ok_mask]
                z_p1[ok_idx] = zc_s[ok_mask]
                hist_cnt[ok_idx] = np.minimum(hist_cnt[ok_idx] + 1, 3)

    # 4. Circular 1D median smoothing across azimuth sectors
    curb_hit_r_smooth = _circular_median_1d(curb_hit_r, int(cfg.median_window))

    # 5. Assign point-level drivability labels
    pt_curb_r = curb_hit_r_smooth[t_bin]
    band_half = float(cfg.curb_band_half_width)
    is_elevated = rr >= pt_curb_r
    is_curb_band = (pt_curb_r <= max_range) & (rr >= pt_curb_r - band_half) & (rr <= pt_curb_r + band_half)

    point_levels[roi_idx[~is_elevated]] = DRIVABLE_ROAD
    point_levels[roi_idx[is_elevated]] = DRIVABLE_ELEVATED
    point_levels[roi_idx[is_curb_band]] = DRIVABLE_CURB

    stats = {
        "road": int(np.sum(point_levels == DRIVABLE_ROAD)),
        "elevated": int(np.sum(point_levels == DRIVABLE_ELEVATED)),
        "curb": int(np.sum(point_levels == DRIVABLE_CURB)),
        "sectors_hit": int(np.sum(curb_hit_r_smooth <= max_range)),
    }
    return DrivabilityResult(
        point_levels=point_levels,
        curb_hit_r=curb_hit_r_smooth,
        stats=stats,
    )
