"""
Equivalence tests between reference pipeline and vectorized (fast) pipeline.
Guarantees that run_fast produces the exact same results as run_reference.
"""
from pathlib import Path
import numpy as np
import pytest

from travel_py.config import GlobalConfig, GridConfig, SeedConfig, AcceptConfig
from travel_py.pipeline import run_reference, run_fast


def _make_synthetic_cloud(n_points: int = 5000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Ground plane with slight slope and noise
    xy = rng.uniform(-40.0, 40.0, size=(n_points, 2))
    z = 0.05 * xy[:, 0] - 0.02 * xy[:, 1] + rng.normal(0, 0.02, size=n_points)
    
    # Add some obstacles (vertical walls / objects)
    n_obs = n_points // 5
    obs_idx = rng.choice(n_points, size=n_obs, replace=False)
    z[obs_idx] += rng.uniform(0.5, 3.0, size=n_obs)
    
    return np.column_stack([xy, z])


class TestFastEquivalenceSynthetic:
    @pytest.mark.parametrize("resolution", [5.0, 2.5, 1.0])
    def test_synthetic_equivalence(self, resolution: float):
        points = _make_synthetic_cloud(n_points=3000, seed=123)
        cfg = GlobalConfig(
            grid=GridConfig(resolution=resolution, origin_xy=(-50.0, -50.0), size_xy=(100, 100)),
            seed=SeedConfig(top_k=5),
            accept=AcceptConfig(max_height_diff=0.5, th_normal=0.9),
        )

        res_ref = run_reference(points, cfg)
        res_fast = run_fast(points, cfg, keep_grid=True)

        # 1. Output point labels must match exactly
        mismatches = np.sum(res_ref.labels != res_fast.labels)
        assert mismatches == 0, f"Point label mismatch: {mismatches} / {len(points)}"

        # 2. Pipeline summary stats must match
        assert res_ref.visited == res_fast.visited
        assert res_ref.rejected_count == res_fast.rejected_count
        assert res_ref.max_depth == res_fast.max_depth
        assert res_ref.final_ground == res_fast.final_ground
        assert res_ref.step1_stats == res_fast.step1_stats
        assert res_ref.rejection_stats == res_fast.rejection_stats


class TestFastEquivalenceRealData:
    @pytest.fixture
    def sample_points(self):
        sample_path = Path(__file__).resolve().parent.parent / "data" / "sample.npy"
        if not sample_path.exists():
            pytest.skip("data/sample.npy not found")
        return np.load(sample_path)

    @pytest.mark.parametrize("resolution", [5.0, 1.0])
    def test_real_data_equivalence(self, sample_points: np.ndarray, resolution: float):
        cfg = GlobalConfig(
            grid=GridConfig(
                resolution=resolution,
                origin_xy=(-85.0, -85.0),
                size_xy=(int(np.ceil(170 / resolution)), int(np.ceil(170 / resolution))),
            ),
            seed=SeedConfig(top_k=5),
            accept=AcceptConfig(max_height_diff=0.5, th_normal=0.9),
        )

        res_ref = run_reference(sample_points, cfg)
        res_fast = run_fast(sample_points, cfg)

        mismatches = np.sum(res_ref.labels != res_fast.labels)
        assert mismatches == 0, f"Mismatch on sample.npy at res={resolution}: {mismatches} points"
        assert res_ref.visited == res_fast.visited
        assert res_ref.step1_stats == res_fast.step1_stats
