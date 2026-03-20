"""Tests for plane.py — LocalPlaneEstimator and is_traversable_lcc."""

import numpy as np
import pytest

from travel_py.plane import LocalPlaneEstimator, PlaneModel, is_traversable_lcc
from travel_py.types import CellState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_ground(n: int = 30, noise: float = 0.01) -> np.ndarray:
    """Perfectly flat point cloud at z=0 with tiny noise."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, (n, 2))
    z = rng.normal(0.0, noise, n)
    return np.column_stack([xy, z])


def _tilted_plane(n: int = 30, slope: float = 0.5) -> np.ndarray:
    """Point cloud on a tilted plane: z = slope * x."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, (n, 2))
    z = slope * xy[:, 0] + rng.normal(0, 0.01, n)
    return np.column_stack([xy, z])


def _make_plane(normal=(0, 0, 1), mean=(0, 0, 0), d=0.0, weight=1.0, label=CellState.GROUND):
    return PlaneModel(
        normal=np.array(normal, dtype=float),
        mean=np.array(mean, dtype=float),
        d=float(d),
        weight=float(weight),
        label=label,
    )


# ---------------------------------------------------------------------------
# LocalPlaneEstimator.estimate
# ---------------------------------------------------------------------------

class TestLocalPlaneEstimatorEstimate:
    def setup_method(self):
        self.est = LocalPlaneEstimator(min_points=6, th_normal=0.9, th_weight=1e-2)

    def test_flat_ground_labeled_ground(self):
        pts = _flat_ground(n=40)
        model = self.est.estimate(pts)
        assert model.label == CellState.GROUND
        assert model.normal[2] > 0.9

    def test_tilted_plane_not_ground(self):
        pts = _tilted_plane(n=40, slope=2.0)
        model = self.est.estimate(pts)
        # Steep slope → NON_GROUND
        assert model.label == CellState.NON_GROUND

    def test_too_few_points_returns_unknown(self):
        pts = _flat_ground(n=3)
        model = self.est.estimate(pts)
        assert model.label == CellState.UNKNOWN
        assert model.weight == 0.0

    def test_normal_points_upward(self):
        pts = _flat_ground(n=30)
        model = self.est.estimate(pts)
        assert model.normal[2] > 0.0, "Normal should point upward"

    def test_weight_positive_for_planar_data(self):
        pts = _flat_ground(n=30)
        model = self.est.estimate(pts)
        assert model.weight > 0.0


# ---------------------------------------------------------------------------
# LocalPlaneEstimator.estimate_and_update
# ---------------------------------------------------------------------------

class TestEstimateAndUpdate:
    def test_updates_subcell_fields(self):
        from travel_py.types import SubCell

        pts = _flat_ground(n=20)
        subcell = SubCell(points=pts)
        est = LocalPlaneEstimator(min_points=6)
        est.estimate_and_update(subcell)

        assert subcell.normal is not None
        assert subcell.mean is not None
        assert subcell.d is not None
        assert subcell.label in (CellState.GROUND, CellState.NON_GROUND, CellState.UNKNOWN)


# ---------------------------------------------------------------------------
# is_traversable_lcc
# ---------------------------------------------------------------------------

class TestIsTraversableLcc:
    def test_identical_planes_accepted(self):
        p = _make_plane()
        ok, reason = is_traversable_lcc(p, p, th_normal=0.9, th_dist=0.5)
        assert ok
        assert reason == "NONE"

    def test_parallel_shifted_within_threshold(self):
        src = _make_plane(mean=(0, 0, 0))
        dst = _make_plane(mean=(1, 0, 0.1))  # small z offset
        ok, _ = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5)
        assert ok

    def test_large_height_diff_rejected(self):
        src = _make_plane(mean=(0, 0, 0))
        dst = _make_plane(mean=(0, 0, 1.0))  # 1 m height diff
        ok, reason = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5)
        assert not ok
        assert reason == "DIST_TOO_LARGE"

    def test_dissimilar_normals_rejected(self):
        src = _make_plane(normal=(0, 0, 1))
        dst = _make_plane(normal=(0, 1, 0))  # wall-like
        ok, reason = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5)
        assert not ok
        assert reason == "SIMILARITY_TOO_LOW"

    def test_boundary_dist_exact_threshold(self):
        # dist_src = dot([0,0,1], [0,0,0.5]) = 0.5, which is == th_dist
        src = _make_plane(mean=(0, 0, 0))
        dst = _make_plane(mean=(0, 0, 0.5))
        # abs(dist_src) = 0.5, condition is > th_dist → 0.5 > 0.5 is False → accept
        ok, _ = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5)
        assert ok

    def test_boundary_dist_just_over_threshold(self):
        src = _make_plane(mean=(0, 0, 0))
        dst = _make_plane(mean=(0, 0, 0.51))
        ok, reason = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5)
        assert not ok
        assert reason == "DIST_TOO_LARGE"
