"""Unit tests for Layer 3 Polar Ray-Marching Drivability & Curb Post-Processor."""
from __future__ import annotations

import numpy as np

from travel_py.config import (
    AcceptConfig,
    DrivabilityConfig,
    GlobalConfig,
    GridConfig,
    ReachabilityConfig,
    SeedConfig,
)
from travel_py.drivability import (
    DRIVABLE_CURB,
    DRIVABLE_ELEVATED,
    DRIVABLE_NONE,
    DRIVABLE_ROAD,
    verify_drivability,
)
from travel_py.pipeline import run_fast


def _make_road_with_curb(
    *,
    curb_height: float = 0.15,
    slope_deg: float = 0.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n_pts = 20000
    x = rng.uniform(-5.0, 28.0, size=n_pts).astype(np.float32)
    y = rng.uniform(-6.5, 6.5, size=n_pts).astype(np.float32)
    slope_rad = np.radians(slope_deg)
    base_z = -1.73 + x * np.tan(slope_rad)

    road_mask = np.abs(y) < 3.45
    sw_mask = np.abs(y) > 3.55
    curb_mask = (~road_mask) & (~sw_mask)

    z = base_z.copy()
    z[sw_mask] += curb_height
    if np.any(curb_mask):
        t = (np.abs(y[curb_mask]) - 3.45) / 0.10
        z[curb_mask] += t * curb_height
    z += rng.normal(0.0, 0.005, size=n_pts).astype(np.float32)

    pts = np.column_stack([x, y, z])
    return pts, road_mask, sw_mask


def _base_cfg(driv_enabled: bool = True) -> GlobalConfig:
    return GlobalConfig(
        grid=GridConfig(resolution=2.5),
        seed=SeedConfig(prefer_center=True, top_k=5, re_explore=True, max_components=50),
        accept=AcceptConfig(use_adaptive_lcc=True, use_14_neighbors=True),
        reachability=ReachabilityConfig(enabled=True),
        drivability=DrivabilityConfig(enabled=driv_enabled),
    )


def test_disabled_by_default_preserves_exact_labels():
    pts, _, _ = _make_road_with_curb(curb_height=0.15)
    cfg_off = _base_cfg(driv_enabled=False)
    cfg_on = _base_cfg(driv_enabled=True)

    res_off = run_fast(pts, cfg_off, keep_grid=False)
    res_on = run_fast(pts, cfg_on, keep_grid=False)

    assert res_off.drivability_levels is None
    assert res_on.drivability_levels is not None
    # Layer 3 is a pure post-processor and must NEVER mutate Layer 1/2 ground labels
    np.testing.assert_array_equal(res_off.labels, res_on.labels)


def test_flat_road_without_curb_no_false_positives():
    pts, _, _ = _make_road_with_curb(curb_height=0.0, slope_deg=0.0)
    res = run_fast(pts, _base_cfg(driv_enabled=True), keep_grid=False)

    gnd = res.labels == 1
    road_frac = np.mean(res.drivability_levels[gnd] == DRIVABLE_ROAD)
    assert road_frac > 0.99


def test_steep_slope_without_curb_no_false_positives():
    # 5 deg (~8.7% grade) continuous hill without any curbs
    pts, _, _ = _make_road_with_curb(curb_height=0.0, slope_deg=5.0)
    res = run_fast(pts, _base_cfg(driv_enabled=True), keep_grid=False)

    gnd = res.labels == 1
    road_frac = np.mean(res.drivability_levels[gnd] == DRIVABLE_ROAD)
    assert road_frac > 0.98


def test_flat_road_with_15cm_curb_separates_sidewalk():
    pts, road_gt, sw_gt = _make_road_with_curb(curb_height=0.15, slope_deg=0.0)
    res = run_fast(pts, _base_cfg(driv_enabled=True), keep_grid=False)

    roi = np.hypot(pts[:, 0], pts[:, 1]) < 25.0
    pred_road = res.drivability_levels == DRIVABLE_ROAD
    pred_non_road = (res.drivability_levels == DRIVABLE_ELEVATED) | (res.drivability_levels == DRIVABLE_CURB)

    road_recall = np.mean(pred_road[roi & road_gt])
    sw_rejection = np.mean(pred_non_road[roi & sw_gt])
    assert road_recall > 0.95
    assert sw_rejection > 0.85


def test_sloped_road_with_15cm_curb_separates_sidewalk():
    pts, road_gt, sw_gt = _make_road_with_curb(curb_height=0.15, slope_deg=5.0)
    res = run_fast(pts, _base_cfg(driv_enabled=True), keep_grid=False)

    roi = np.hypot(pts[:, 0], pts[:, 1]) < 25.0
    pred_road = res.drivability_levels == DRIVABLE_ROAD
    pred_non_road = (res.drivability_levels == DRIVABLE_ELEVATED) | (res.drivability_levels == DRIVABLE_CURB)

    road_recall = np.mean(pred_road[roi & road_gt])
    sw_rejection = np.mean(pred_non_road[roi & sw_gt])
    assert road_recall > 0.95
    assert sw_rejection > 0.85


def test_empty_point_cloud():
    pts = np.empty((0, 3), dtype=np.float32)
    mask = np.empty(0, dtype=bool)
    res = verify_drivability(pts, mask, np.array([0.0, 0.0, 1.0]), DrivabilityConfig(enabled=True))
    assert res.point_levels.shape == (0,)
    assert res.stats["road"] == 0
    assert res.stats["elevated"] == 0
    assert res.stats["curb"] == 0
