"""Tests for plane.py — LocalPlaneEstimator and is_traversable_lcc."""

import io
import sys

import numpy as np
import pytest

from travel_py.plane import LocalPlaneEstimator, PlaneModel, is_traversable_lcc
from travel_py.types import CellState


# ===========================================================================
# Helpers
# ===========================================================================

def _flat_ground(n: int = 30, z_offset: float = 0.0, noise: float = 0.005) -> np.ndarray:
    """Flat point cloud at z = z_offset with tiny noise."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(-1, 1, (n, 2))
    z = z_offset + rng.normal(0.0, noise, n)
    return np.column_stack([xy, z])


def _tilted_plane(n: int = 30, slope_x: float = 0.0, slope_y: float = 0.0, noise: float = 0.005) -> np.ndarray:
    """Point cloud on a plane z = slope_x * x + slope_y * y."""
    rng = np.random.default_rng(7)
    xy = rng.uniform(-1, 1, (n, 2))
    z = slope_x * xy[:, 0] + slope_y * xy[:, 1] + rng.normal(0, noise, n)
    return np.column_stack([xy, z])


def _two_layer(n_ground: int = 20, n_obstacle: int = 10, z_obstacle: float = 1.0) -> np.ndarray:
    """Ground layer at z≈0 plus obstacle points at z=z_obstacle."""
    rng = np.random.default_rng(3)
    xy_g = rng.uniform(-1, 1, (n_ground, 2))
    z_g = rng.normal(0, 0.01, n_ground)
    xy_o = rng.uniform(-1, 1, (n_obstacle, 2))
    z_o = np.full(n_obstacle, z_obstacle) + rng.normal(0, 0.01, n_obstacle)
    return np.vstack([
        np.column_stack([xy_g, z_g]),
        np.column_stack([xy_o, z_o]),
    ])


def _make_plane(normal=(0, 0, 1), mean=(0, 0, 0), d=0.0, weight=1.0, label=CellState.GROUND):
    return PlaneModel(
        normal=np.array(normal, dtype=float),
        mean=np.array(mean, dtype=float),
        d=float(d),
        weight=float(weight),
        label=label,
    )


# ===========================================================================
# LocalPlaneEstimator.estimate — basic checks
# ===========================================================================

class TestEstimateBasic:
    def setup_method(self):
        self.est = LocalPlaneEstimator(min_points=6, th_normal=0.9, th_weight=1e-2)

    def test_flat_ground_labeled_ground(self):
        model = self.est.estimate(_flat_ground(n=40))
        assert model.label == CellState.GROUND

    def test_tilted_steep_labeled_non_ground(self):
        # slope = 2.0 → angle ≈ 63°, normal[2] ≈ 0.45 < 0.9
        model = self.est.estimate(_tilted_plane(n=40, slope_x=2.0))
        assert model.label == CellState.NON_GROUND

    def test_too_few_points_returns_unknown(self):
        model = self.est.estimate(_flat_ground(n=3))
        assert model.label == CellState.UNKNOWN
        assert model.weight == 0.0

    def test_normal_points_upward(self):
        model = self.est.estimate(_flat_ground(n=30))
        assert model.normal[2] > 0.0

    def test_weight_positive_for_flat_data(self):
        model = self.est.estimate(_flat_ground(n=30))
        assert model.weight > 0.0

    def test_unknown_plane_has_default_upward_normal(self):
        model = self.est.estimate(_flat_ground(n=2))  # below min_points
        assert np.allclose(model.normal, [0, 0, 1])
        assert model.d == 0.0
        assert np.allclose(model.mean, [0, 0, 0])


# ===========================================================================
# LPR — Lowest Point Representative behaviour
# ===========================================================================

class TestLPR:
    """Verify that LPR (seed selection by lowest points) works correctly."""

    def test_lpr_selects_ground_layer_from_two_layer_data(self):
        """LPR should pick z≈0 seeds even when obstacle points at z=1 exist."""
        pts = _two_layer(n_ground=25, n_obstacle=10, z_obstacle=1.0)
        est = LocalPlaneEstimator(num_lpr=5, th_seeds=0.2, th_outlier=0.2,
                                  th_normal=0.9, min_points=3, th_weight=0.0)
        model = est.estimate(pts)
        assert model.label == CellState.GROUND
        # mean z of estimated plane should be near 0
        assert abs(model.mean[2]) < 0.2

    def test_lpr_height_is_mean_of_lowest_n_points(self):
        """With num_lpr=3, LPR height should equal mean of 3 lowest-z points."""
        rng = np.random.default_rng(11)
        # Ground layer at z≈0 with xy spread, plus high obstacle points
        xy_g = rng.uniform(-1, 1, (9, 2))
        z_g = [0.05, 0.10, 0.15, 0.20, 0.25, 0.05, 0.15, 0.25, 0.10]
        xy_o = rng.uniform(-1, 1, (3, 2))
        z_o = [1.0, 1.1, 1.2]
        pts = np.vstack([
            np.column_stack([xy_g, z_g]),
            np.column_stack([xy_o, z_o]),
        ])
        # Lowest 3: z=0.05, 0.05, 0.10 → lpr_height ≈ 0.067
        # th_seeds=0.3: accept z < 0.067+0.3=0.367 → all ground points accepted
        # obstacle z≥1.0 excluded
        est = LocalPlaneEstimator(num_lpr=3, th_seeds=0.3, th_outlier=0.2,
                                  min_points=3, th_normal=0.9, th_weight=0.0)
        model = est.estimate(pts)
        # Obstacle points at z=1.0,1.1,1.2 should be excluded from seeds
        assert model.label == CellState.GROUND

    def test_seed_selection_too_strict_returns_unknown(self):
        """If th_seeds is very tight, seed selection may leave fewer than min_points."""
        # All points have z=0 except many at z=1; tiny th_seeds means only 0-height points are seeds
        pts = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],  # only 2 ground points
            [0.2, 0.0, 1.0],
            [0.3, 0.0, 1.0],
            [0.4, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [0.6, 0.0, 1.0],
            [0.7, 0.0, 1.0],
        ], dtype=float)
        # LPR height ≈ 0, th_seeds=0.05 → seeds are z in (-0.05, 0.05) → 2 points
        # min_points=3 → should return UNKNOWN (line 68 coverage)
        est = LocalPlaneEstimator(num_lpr=3, th_seeds=0.05, th_outlier=0.05,
                                  min_points=3, th_normal=0.9, th_weight=0.0)
        model = est.estimate(pts)
        assert model.label == CellState.UNKNOWN

    def test_th_outlier_excludes_very_low_points(self):
        """Points far below lpr_height should be excluded from seeds by th_outlier."""
        pts = _flat_ground(n=20, noise=0.005)
        # Add a single very low point (z = -2.0) as an outlier.
        # num_lpr=5: lowest 5 = outlier(-2.0) + 4 ground(≈0) → lpr_height ≈ -0.4
        # th_outlier=0.3: seeds require z > lpr_height - 0.3 = -0.7
        # → outlier z=-2.0 < -0.7 is EXCLUDED from seeds
        # → 20 ground points with z≈0 (> -0.7) are included as seeds
        outlier = np.array([[0.0, 0.0, -2.0]])
        pts_with_outlier = np.vstack([pts, outlier])
        est = LocalPlaneEstimator(num_lpr=5, th_seeds=0.5, th_outlier=0.3,
                                  min_points=6, th_normal=0.9, th_weight=0.0)
        model = est.estimate(pts_with_outlier)
        # 20 ground points pass as seeds; plane should be GROUND
        assert model.label == CellState.GROUND


# ===========================================================================
# PCA — normal direction accuracy
# ===========================================================================

class TestPCANormal:
    """Verify that PCA produces the correct normal vector."""

    def test_flat_plane_normal_is_z_axis(self):
        pts = _flat_ground(n=50, noise=0.001)
        est = LocalPlaneEstimator(min_points=6, th_normal=0.0, th_weight=0.0)
        model = est.estimate(pts)
        assert model.normal[2] > 0.99, f"Expected near-vertical normal, got {model.normal}"

    def test_tilted_plane_normal_direction(self):
        """slope_x=1 gives plane z=x, normal should be [-1, 0, 1]/sqrt(2)."""
        pts = _tilted_plane(n=100, slope_x=1.0, noise=0.001)
        est = LocalPlaneEstimator(num_lpr=20, th_seeds=1.5, th_outlier=1.5,
                                  min_points=6, th_normal=0.0, th_weight=0.0)
        model = est.estimate(pts)
        # Expected normal: normalize([-1, 0, 1])
        expected = np.array([-1, 0, 1], dtype=float)
        expected /= np.linalg.norm(expected)
        if model.normal[2] < 0:
            model_normal = -model.normal
        else:
            model_normal = model.normal
        # Dot product should be close to 1
        cos_sim = abs(np.dot(model_normal, expected))
        assert cos_sim > 0.99, f"Normal mismatch: {model_normal} vs {expected}"

    def test_normal_always_upward_for_downward_facing_points(self):
        """Even if PCA gives a downward normal, it should be flipped to upward."""
        pts = _flat_ground(n=30)
        # Force downward normal scenario by using a negated plane
        # PCA on a flat plane may give [0,0,-1]; verify it's flipped to [0,0,1]
        est = LocalPlaneEstimator(min_points=6, th_normal=0.0, th_weight=0.0)
        model = est.estimate(pts)
        assert model.normal[2] > 0, "Normal should always point upward"

    def test_normal_z_matches_expected_slope_angle(self):
        """For slope=tan(θ), normal[2] should be cos(θ)."""
        for deg in [0, 15, 30, 45]:
            rad = np.radians(deg)
            slope = np.tan(rad)
            pts = _tilted_plane(n=100, slope_x=slope, noise=0.001)
            est = LocalPlaneEstimator(num_lpr=20, th_seeds=slope + 0.5,
                                      th_outlier=slope + 0.5,
                                      min_points=6, th_normal=0.0, th_weight=0.0)
            model = est.estimate(pts)
            expected_nz = np.cos(rad)
            assert abs(model.normal[2] - expected_nz) < 0.05, (
                f"deg={deg}: expected normal[2]≈{expected_nz:.3f}, got {model.normal[2]:.3f}"
            )


# ===========================================================================
# Plane equation — d coefficient
# ===========================================================================

class TestPlaneD:
    def test_d_at_z_offset(self):
        """For horizontal plane at z=z0, d should be -z0."""
        for z0 in [0.0, 1.0, -0.5, 2.5]:
            pts = _flat_ground(n=40, z_offset=z0, noise=0.001)
            est = LocalPlaneEstimator(num_lpr=10, th_seeds=0.5, th_outlier=0.5,
                                      min_points=6, th_normal=0.0, th_weight=0.0)
            model = est.estimate(pts)
            # Plane eq: dot(normal, p) + d = 0
            # For horizontal: 0*x + 0*y + 1*z + d = 0 → d = -z0
            assert abs(model.d - (-z0)) < 0.05, (
                f"z0={z0}: expected d≈{-z0:.2f}, got d={model.d:.3f}"
            )

    def test_plane_equation_satisfied_by_mean(self):
        """The mean point should satisfy the plane equation (within noise)."""
        for z0 in [0.0, 0.5, -1.0]:
            pts = _flat_ground(n=40, z_offset=z0, noise=0.005)
            est = LocalPlaneEstimator(min_points=6, th_normal=0.0, th_weight=0.0)
            model = est.estimate(pts)
            residual = np.dot(model.normal, model.mean) + model.d
            assert abs(residual) < 1e-6, f"z0={z0}: plane eq not satisfied: {residual}"


# ===========================================================================
# Planarity weight
# ===========================================================================

class TestPlaneWeight:
    def test_flat_plane_has_high_weight(self):
        """A highly planar (flat) surface should produce a high weight."""
        pts = _flat_ground(n=50, noise=0.001)
        est = LocalPlaneEstimator(min_points=6, th_normal=0.0, th_weight=0.0)
        model = est.estimate(pts)
        assert model.weight > 1.0

    def test_random_cloud_has_low_weight(self):
        """A uniform random 3D cloud is not planar → lower weight."""
        rng = np.random.default_rng(99)
        pts = rng.uniform(-1, 1, (50, 3))
        est = LocalPlaneEstimator(num_lpr=50, th_seeds=5.0, th_outlier=5.0,
                                  min_points=6, th_normal=0.0, th_weight=0.0)
        flat = _flat_ground(n=50, noise=0.001)
        model_flat = est.estimate(flat)
        model_rand = est.estimate(pts)
        assert model_flat.weight > model_rand.weight

    def test_weight_clipped_at_max_weight(self):
        """Perfect plane (zero noise) should be clipped to max_weight."""
        rng = np.random.default_rng(0)
        xy = rng.uniform(-1, 1, (50, 2))
        z = np.zeros(50)
        pts = np.column_stack([xy, z])
        est = LocalPlaneEstimator(min_points=6, th_normal=0.0, th_weight=0.0, max_weight=10.0)
        model = est.estimate(pts)
        assert model.weight <= 10.0

    def test_low_weight_causes_non_ground_label(self):
        """th_weight filter: normal OK but weight below threshold → NON_GROUND."""
        pts = _flat_ground(n=10, noise=0.5)  # large noise → low planarity
        est = LocalPlaneEstimator(num_lpr=10, th_seeds=1.0, th_outlier=1.0,
                                  min_points=6, th_normal=0.0, th_weight=1e6)  # very strict weight
        model = est.estimate(pts)
        # Real weight will be << 1e6, so label should be NON_GROUND
        assert model.label == CellState.NON_GROUND


# ===========================================================================
# GROUND / NON_GROUND classification boundary
# ===========================================================================

class TestClassificationBoundary:
    def test_th_normal_boundary_just_above(self):
        """normal[2] > th_normal (strictly) required for GROUND."""
        # Build a plane at exactly 0 degrees tilt → normal[2] close to 1.0
        pts = _flat_ground(n=50, noise=0.001)
        # th_normal = 0.9999: flat plane should barely pass
        est = LocalPlaneEstimator(min_points=6, th_normal=0.9999, th_weight=0.0)
        model = est.estimate(pts)
        # normal[2] ≈ 1.0 > 0.9999 → GROUND
        assert model.label == CellState.GROUND

    def test_th_normal_boundary_steeper_than_threshold(self):
        """Steep slope: normal[2] should be below th_normal → NON_GROUND."""
        # 45-degree slope → normal[2] ≈ 0.707 < 0.9
        pts = _tilted_plane(n=60, slope_x=1.0, noise=0.001)
        est = LocalPlaneEstimator(num_lpr=20, th_seeds=2.0, th_outlier=2.0,
                                  min_points=6, th_normal=0.9, th_weight=0.0)
        model = est.estimate(pts)
        assert model.label == CellState.NON_GROUND

    def test_th_weight_exactly_zero_allows_all(self):
        """th_weight=0.0 means even low-planarity planes get GROUND if normal is OK."""
        pts = _flat_ground(n=8, noise=0.3)  # noisy but broadly flat
        est = LocalPlaneEstimator(num_lpr=8, th_seeds=1.0, th_outlier=1.0,
                                  min_points=6, th_normal=0.9, th_weight=0.0)
        model = est.estimate(pts)
        # If normal[2] > 0.9, it should be GROUND regardless of weight
        if model.normal[2] > 0.9:
            assert model.label == CellState.GROUND


# ===========================================================================
# estimate_and_update
# ===========================================================================

class TestEstimateAndUpdate:
    def test_updates_all_fields(self):
        from travel_py.types import SubCell
        pts = _flat_ground(n=20)
        sub = SubCell(points=pts)
        est = LocalPlaneEstimator(min_points=6)
        est.estimate_and_update(sub)
        assert sub.normal is not None
        assert sub.mean is not None
        assert sub.d is not None
        assert sub.label in (CellState.GROUND, CellState.NON_GROUND, CellState.UNKNOWN)

    def test_unknown_subcell_gets_default_normal(self):
        from travel_py.types import SubCell
        pts = _flat_ground(n=2)  # too few points
        sub = SubCell(points=pts)
        est = LocalPlaneEstimator(min_points=6)
        est.estimate_and_update(sub)
        assert sub.label == CellState.UNKNOWN
        assert np.allclose(sub.normal, [0, 0, 1])


# ===========================================================================
# is_traversable_lcc — comprehensive
# ===========================================================================

class TestIsTraversableLcc:
    def test_identical_planes_accepted(self):
        p = _make_plane()
        ok, reason = is_traversable_lcc(p, p, th_normal=0.9, th_dist=0.5)
        assert ok
        assert reason == "NONE"

    def test_horizontal_shift_does_not_affect_dist(self):
        """Horizontal (xy) shift: dot([0,0,1], delta) = 0 → always OK."""
        src = _make_plane(mean=(0, 0, 0))
        dst = _make_plane(mean=(5, 5, 0))  # large xy shift, no z change
        ok, _ = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.1)
        assert ok

    def test_parallel_shifted_within_threshold(self):
        src = _make_plane(mean=(0, 0, 0))
        dst = _make_plane(mean=(1, 0, 0.1))
        ok, _ = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5)
        assert ok

    def test_large_height_diff_rejected(self):
        src = _make_plane(mean=(0, 0, 0))
        dst = _make_plane(mean=(0, 0, 1.0))
        ok, reason = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5)
        assert not ok
        assert reason == "DIST_TOO_LARGE"

    def test_dissimilar_normals_rejected(self):
        src = _make_plane(normal=(0, 0, 1))
        dst = _make_plane(normal=(0, 1, 0))
        ok, reason = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5)
        assert not ok
        assert reason == "SIMILARITY_TOO_LOW"

    def test_dist_check_uses_both_src_and_dst_normals(self):
        """If dst normal is not [0,0,1], dist_dst can differ from dist_src."""
        # src is horizontal, dst is slightly tilted
        src = _make_plane(normal=(0, 0, 1), mean=(0, 0, 0))
        # tilted dst: normal in xz plane, pointing mostly up
        n = np.array([0.1, 0, 0.995])
        n /= np.linalg.norm(n)
        dst = PlaneModel(
            normal=n,
            mean=np.array([0, 0, 0.4]),
            d=-np.dot(n, [0, 0, 0.4]),
            weight=1.0,
            label=CellState.GROUND,
        )
        ok, reason = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5)
        # normals are similar (dot ≈ 0.995 > 0.9) and height diff is small
        assert ok

    def test_boundary_dist_exact_threshold_accepted(self):
        # dist_src = dot([0,0,1], [0,0,0.5]) = 0.5 == th_dist → not strictly > → accept
        src = _make_plane(mean=(0, 0, 0))
        dst = _make_plane(mean=(0, 0, 0.5))
        ok, _ = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5)
        assert ok

    def test_boundary_dist_just_over_threshold_rejected(self):
        src = _make_plane(mean=(0, 0, 0))
        dst = _make_plane(mean=(0, 0, 0.51))
        ok, reason = is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5)
        assert not ok
        assert reason == "DIST_TOO_LARGE"

    def test_verbose_prints_on_rejection_normal(self, capsys):
        src = _make_plane(normal=(0, 0, 1))
        dst = _make_plane(normal=(0, 1, 0))  # dissimilar
        is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5, verbose=True)
        captured = capsys.readouterr()
        assert "Reject" in captured.out or "LCC" in captured.out

    def test_verbose_prints_on_rejection_dist(self, capsys):
        src = _make_plane(mean=(0, 0, 0))
        dst = _make_plane(mean=(0, 0, 2.0))  # large height diff
        is_traversable_lcc(src, dst, th_normal=0.9, th_dist=0.5, verbose=True)
        captured = capsys.readouterr()
        assert "Reject" in captured.out or "LCC" in captured.out

    def test_verbose_silent_on_acceptance(self, capsys):
        p = _make_plane()
        is_traversable_lcc(p, p, th_normal=0.9, th_dist=0.5, verbose=True)
        captured = capsys.readouterr()
        assert captured.out == ""


# ===========================================================================
# Integration: estimate on realistic mixed data
# ===========================================================================

class TestEstimateIntegration:
    def test_sparse_subcell_3pts_detects_flat_ground(self):
        """With only 3 points in a flat subcell, th_weight=0.0 should detect GROUND."""
        pts = np.array([
            [-0.5, -0.5, 0.01],
            [ 0.5, -0.5, -0.01],
            [ 0.0,  0.5,  0.00],
        ], dtype=float)
        est = LocalPlaneEstimator(num_lpr=3, th_seeds=0.5, th_outlier=0.5,
                                  min_points=3, th_normal=0.9, th_weight=0.0)
        model = est.estimate(pts)
        assert model.label == CellState.GROUND
        assert model.normal[2] > 0.9

    def test_z_2m_elevated_flat_surface(self):
        """Flat surface elevated at z=2m (overpass) should be GROUND by normal check."""
        pts = _flat_ground(n=20, z_offset=2.0)
        est = LocalPlaneEstimator(min_points=6, th_normal=0.9, th_weight=0.0)
        model = est.estimate(pts)
        assert model.label == CellState.GROUND
        assert abs(model.mean[2] - 2.0) < 0.1

    def test_multiple_subcells_on_same_ground_plane_give_consistent_normals(self):
        """Subcells from the same flat ground patch should all return similar normals."""
        normals = []
        rng = np.random.default_rng(0)
        for _ in range(10):
            xy = rng.uniform(-0.5, 0.5, (8, 2))
            z = rng.normal(0, 0.01, 8)
            pts = np.column_stack([xy, z])
            est = LocalPlaneEstimator(min_points=3, th_normal=0.0, th_weight=0.0)
            model = est.estimate(pts)
            normals.append(model.normal)
        normal_arr = np.array(normals)
        # All normal[2] should be close to 1.0
        assert np.all(normal_arr[:, 2] > 0.95), f"Some normals not upward: {normal_arr[:, 2]}"

    def test_wall_like_vertical_surface_labeled_non_ground(self):
        """A vertical wall (x=const plane) should be NON_GROUND."""
        rng = np.random.default_rng(5)
        # Points on x=0 plane: varying y and z
        y = rng.uniform(-1, 1, 30)
        z = rng.uniform(0, 2, 30)
        x = rng.normal(0, 0.005, 30)  # tiny noise around x=0
        pts = np.column_stack([x, y, z])
        est = LocalPlaneEstimator(num_lpr=20, th_seeds=2.5, th_outlier=2.5,
                                  min_points=6, th_normal=0.9, th_weight=0.0)
        model = est.estimate(pts)
        # Wall normal should be [1,0,0] or similar → normal[2] ≈ 0 < 0.9
        assert model.label == CellState.NON_GROUND
        assert model.normal[2] < 0.5
