import numpy as np
from dataclasses import dataclass
from typing import Optional
from .types import CellState


@dataclass
class PlaneModel:
    normal: np.ndarray   # (3,) unit vector
    mean: np.ndarray     # (3,)
    d: float
    weight: float        # planarity / confidence
    label: CellState


class LocalPlaneEstimator:
    def __init__(
        self,
        num_lpr: int = 20,
        th_seeds: float = 0.5,
        th_outlier: float = 0.5,
        th_normal: float = 0.9,   # normal[2] threshold for ground
        th_weight: float = 1e-2,  # minimum reliable planarity
        min_points: int = 6,      # minimum points for stable PCA
        eps: float = 1e-6,
        max_weight: float = 1e3,  # clip to avoid explosion
    ):
        self.num_lpr = num_lpr
        self.th_seeds = th_seeds
        self.th_outlier = th_outlier
        self.th_normal = th_normal
        self.th_weight = th_weight
        self.min_points = min_points
        self.eps = eps
        self.max_weight = max_weight

    def estimate(self, points: np.ndarray) -> PlaneModel:
        """
        Estimate a local plane using LPR seed selection + PCA.
        Designed for Travel / TGS-style cell-based ground estimation.
        """

        # ------------------------------------------------------------------
        # 0. Basic validation
        # ------------------------------------------------------------------
        if points.shape[0] < self.min_points:
            return self._unknown_plane()

        # ------------------------------------------------------------------
        # 1. LPR (Lowest Point Representative)
        # ------------------------------------------------------------------
        sorted_indices = np.argsort(points[:, 2])
        n_lpr = min(self.num_lpr, points.shape[0])
        lpr_points = points[sorted_indices[:n_lpr]]
        lpr_height = np.mean(lpr_points[:, 2])

        # ------------------------------------------------------------------
        # 2. Seed selection
        # ------------------------------------------------------------------
        mask = (
            (points[:, 2] < lpr_height + self.th_seeds) &
            (points[:, 2] > lpr_height - self.th_outlier)
        )
        seeds = points[mask]

        # If not enough seeds, do NOT fallback to all points
        if seeds.shape[0] < self.min_points:
            return self._unknown_plane()

        # ------------------------------------------------------------------
        # 3. PCA
        # ------------------------------------------------------------------
        mean = np.mean(seeds, axis=0)
        centered = seeds - mean

        # Covariance with regularization for numerical stability
        cov = (
            np.dot(centered.T, centered) /
            max(seeds.shape[0] - 1, 1)
        )
        cov += self.eps * np.eye(3)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigenvalues are sorted ascending:
        # s0 <= s1 <= s2

        normal = eigenvectors[:, 0]

        # Ensure upward normal
        if normal[2] < 0.0:
            normal = -normal

        # Normalize normal explicitly
        normal_norm = np.linalg.norm(normal)
        if normal_norm < self.eps:
            return self._unknown_plane()
        normal = normal / normal_norm

        d = -np.dot(normal, mean)

        # ------------------------------------------------------------------
        # 4. Planarity weight
        # ------------------------------------------------------------------
        s0, s1, s2 = eigenvalues

        # Planarity-style weight (TGS-inspired)
        raw_weight = (s0 + s1) * s1 / (s0 * s2 + self.eps)

        # Clip & stabilize
        weight = min(raw_weight, self.max_weight)

        # ------------------------------------------------------------------
        # 5. Labeling
        # ------------------------------------------------------------------
        if normal[2] > self.th_normal and weight > self.th_weight:
            label = CellState.GROUND
        else:
            label = CellState.NON_GROUND

        return PlaneModel(
            normal=normal,
            mean=mean,
            d=d,
            weight=weight,
            label=label,
        )
    
    def estimate_and_update(self, subcell) -> None:
        model = self.estimate(subcell.points)
        subcell.normal = model.normal
        subcell.mean = model.mean
        subcell.d = model.d
        subcell.weight = model.weight
        subcell.label = model.label

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _unknown_plane(self) -> PlaneModel:
        return PlaneModel(
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            mean=np.zeros(3, dtype=float),
            d=0.0,
            weight=0.0,
            label=CellState.UNKNOWN,
        )
