"""
Unit tests for Layer 2 Reachability / Traversability Verification (B+D+E).
"""
from __future__ import annotations

import numpy as np
import pytest

from travel_py.config import (
    AcceptConfig,
    GlobalConfig,
    GridConfig,
    ReachabilityConfig,
    SeedConfig,
)
from travel_py.pipeline import run_fast
from travel_py.reachability import (
    LEVEL_BRIDGED,
    LEVEL_NONE,
    LEVEL_OTHER,
    LEVEL_PRIMARY,
)


def _make_patch(x0: float, x1: float, y0: float, y1: float, z_fn, step: float = 0.25) -> np.ndarray:
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys)
    x = gx.ravel()
    y = gy.ravel()
    z = z_fn(x, y)
    return np.column_stack([x, y, z]).astype(np.float64)


def _base_cfg(*, re_explore: bool = True, reach_enabled: bool = False) -> GlobalConfig:
    return GlobalConfig(
        grid=GridConfig(resolution=2.5, origin_xy=(-25.0, -25.0), size_xy=(24, 24)),
        seed=SeedConfig(
            top_k=1,
            prefer_center=True,
            re_explore=re_explore,
            max_components=20,
        ),
        accept=AcceptConfig(
            max_height_diff=0.5,
            th_normal=0.9,
            use_adaptive_lcc=True,
            use_14_neighbors=True,
        ),
        reachability=ReachabilityConfig(enabled=reach_enabled),
    )


def test_default_disabled_preserves_pure_travel():
    road = _make_patch(-4.8, 4.8, -4.8, 4.8, lambda x, y: np.zeros_like(x))
    cfg_default = _base_cfg(re_explore=False, reach_enabled=False)
    res = run_fast(road, cfg_default, keep_grid=False)

    assert res.reachability_levels is None
    assert res.reachability_stats == {}
    assert np.all(res.labels == 1)


def test_elevated_terrace_rejected_as_other_level():
    """
    A flat horizontal building terrace at Z=+1.6m located 10m away from the road
    (elevation angle ~9 deg < 11.5 deg) must be promoted to GROUND under
    unconstrained re-exploration, but rejected as LEVEL_OTHER under B+D+E
    bidirectional coplanarity verification.
    """
    road = _make_patch(-4.8, 4.8, -4.8, 4.8, lambda x, y: np.zeros_like(x))
    terrace = _make_patch(12.7, 17.3, -2.3, 2.3, lambda x, y: np.full_like(x, 1.6))
    pts = np.vstack([road, terrace])
    n_road = len(road)

    # Pure TRAVEL + unconstrained ReEx -> terrace falsely becomes GROUND
    res_unconstrained = run_fast(pts, _base_cfg(re_explore=True, reach_enabled=False), keep_grid=False)
    assert np.all(res_unconstrained.labels[n_road:] == 1)

    # Layer 2 B+D+E enabled -> terrace is cleanly separated as LEVEL_OTHER and excluded from GROUND
    res_bde = run_fast(pts, _base_cfg(re_explore=True, reach_enabled=True), keep_grid=False)
    assert res_bde.reachability_levels is not None
    assert np.all(res_bde.labels[:n_road] == 1)
    assert np.all(res_bde.reachability_levels[:n_road] == LEVEL_PRIMARY)
    assert np.all(res_bde.labels[n_road:] != 1)
    assert np.all(res_bde.reachability_levels[n_road:] == LEVEL_OTHER)
    assert res_bde.reachability_stats["other"] > 0


def test_z_stacking_overhang_rejected():
    """
    An elevated roof/overpass directly adjacent to or above confirmed primary ground
    must be rejected by Check E (Z-stacking) as LEVEL_OTHER.
    """
    road = _make_patch(-4.8, 4.8, -4.8, 4.8, lambda x, y: np.zeros_like(x))
    # Adjacent cell (X in [5.2, 9.8]) elevated at Z=+1.2m
    overhang = _make_patch(5.2, 9.8, -2.3, 2.3, lambda x, y: np.full_like(x, 1.2))
    pts = np.vstack([road, overhang])
    n_road = len(road)

    res_bde = run_fast(pts, _base_cfg(re_explore=True, reach_enabled=True), keep_grid=False)
    assert res_bde.reachability_levels is not None
    assert np.all(res_bde.labels[:n_road] == 1)
    assert np.all(res_bde.labels[n_road:] != 1)
    assert np.all(res_bde.reachability_levels[n_road:] == LEVEL_OTHER)


def test_sloped_road_across_gap_bridged():
    """
    A 10% sloped road (z = 0.1 * x) with a 7m occlusion gap has a >1.0m absolute
    height difference across the gap, yet is geometrically coplanar.
    Bidirectional coplanarity (Check D) must recover it as LEVEL_BRIDGED.
    """
    slope_fn = lambda x, y: 0.10 * x
    near_road = _make_patch(-4.8, 4.8, -4.8, 4.8, slope_fn)
    far_road = _make_patch(12.7, 17.3, -4.8, 4.8, slope_fn)
    pts = np.vstack([near_road, far_road])
    n_near = len(near_road)

    res_bde = run_fast(pts, _base_cfg(re_explore=True, reach_enabled=True), keep_grid=False)
    assert res_bde.reachability_levels is not None
    assert np.all(res_bde.labels[:n_near] == 1)
    assert np.all(res_bde.reachability_levels[:n_near] == LEVEL_PRIMARY)
    assert np.all(res_bde.labels[n_near:] == 1)
    assert np.all(res_bde.reachability_levels[n_near:] == LEVEL_BRIDGED)
    assert res_bde.reachability_stats["bridged"] > 0


def test_flat_road_across_occlusion_bridged():
    """Flat road separated by a 6m occlusion gap is recovered as LEVEL_BRIDGED."""
    near_road = _make_patch(-4.8, 4.8, -4.8, 4.8, lambda x, y: np.zeros_like(x))
    far_road = _make_patch(10.2, 14.8, -4.8, 4.8, lambda x, y: np.zeros_like(x))
    pts = np.vstack([near_road, far_road])
    n_near = len(near_road)

    res_bde = run_fast(pts, _base_cfg(re_explore=True, reach_enabled=True), keep_grid=False)
    assert res_bde.reachability_levels is not None
    assert np.all(res_bde.labels[:n_near] == 1)
    assert np.all(res_bde.labels[n_near:] == 1)
    assert np.all(res_bde.reachability_levels[n_near:] == LEVEL_BRIDGED)


def test_empty_pointcloud_with_reachability():
    pts = np.empty((0, 3), dtype=np.float64)
    res = run_fast(pts, _base_cfg(re_explore=True, reach_enabled=True), keep_grid=False)
    assert res.labels.shape == (0,)
    assert res.reachability_levels is not None
    assert res.reachability_levels.shape == (0,)
    assert res.reachability_stats == {"primary": 0, "bridged": 0, "other": 0}
