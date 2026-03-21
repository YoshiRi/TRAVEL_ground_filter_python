import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .types import CellState


@dataclass
class PlaneModel:
    normal: np.ndarray   # (3,) unit vector
    mean: np.ndarray     # (3,)
    d: float
    weight: float        # planarity / confidence
    label: CellState


class LocalPlaneEstimator:
    """
    Estimate the local ground plane inside a SubCell using LPR + PCA.

    Algorithm (4 phases):

    Phase 1 – Ground level detection (LPR):
        Sort points by height.  Average the lowest num_lpr points to obtain a
        robust ground-height reference (LPR height).  Averaging over several
        points reduces the effect of individual sensor-noise spikes below ground.

    Phase 2 – Ground candidate selection:
        Accept points whose height lies in
            [lpr_height − th_outlier,  lpr_height + th_seeds].
        Points above th_seeds above the LPR height are likely walls or
        obstacles; points below th_outlier below are anomalous low noise.

        Fallback: when too few candidates survive — which happens when many
        wall points inflate the LPR mean — anchor instead on the cell's
        *minimum* z (the single lowest measured point).  That point is
        always on the ground surface, never on a wall.

    Phase 3 – Plane fitting (PCA):
        Build the 3×3 covariance matrix of the candidates.  The eigenvector
        corresponding to the smallest eigenvalue points in the direction of
        least variance, i.e. perpendicular to the dominant flat surface.
        The normal is flipped to face upward if necessary.

    Phase 4 – Labelling:
        Mark as GROUND when the normal is nearly vertical (normal[2] > th_normal)
        and the surface is sufficiently flat (planarity weight > th_weight).
        Otherwise mark NON_GROUND.
    """

    def __init__(
        self,
        num_lpr: int = 20,
        th_seeds: float = 0.5,
        th_outlier: float = 0.5,
        th_normal: float = 0.9,
        th_weight: float = 1e-2,
        min_points: int = 6,
        eps: float = 1e-6,
        max_weight: float = 1e3,
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
        if points.shape[0] < self.min_points:
            return self._unknown_plane()

        # ------------------------------------------------------------------
        # Phase 1: Ground level detection (LPR)
        # ------------------------------------------------------------------
        sorted_indices = np.argsort(points[:, 2])
        n_lpr = min(self.num_lpr, points.shape[0])
        lpr_height = points[sorted_indices[:n_lpr], 2].mean()
        min_z = points[sorted_indices[0], 2]

        # ------------------------------------------------------------------
        # Phase 2: Ground candidate selection
        # ------------------------------------------------------------------
        # Accept points within [lpr_height − th_outlier, lpr_height + th_seeds].
        seeds = self._select_seeds(points, lpr_height)

        # Wall-contamination recovery:
        # When many tall-obstacle points inflate the LPR mean, the lower bound
        # (lpr_height − th_outlier) may rise above the actual cell minimum,
        # falsely marking true ground points as "anomalously low" outliers.
        # Detection: the seed window's lower bound exceeds the cell minimum
        #   ↔  lpr_height − th_outlier > min_z
        # Recovery: re-anchor on min_z (always a real surface point) and
        # accept its candidate set — but only if enough points exist there.
        # If the fallback window is also sparse, the minimum was itself an
        # isolated noise spike; keep the original LPR-based seeds.
        if lpr_height - self.th_outlier > min_z:
            fallback = self._select_seeds(points, min_z)
            if fallback.shape[0] >= self.min_points:
                seeds = fallback

        if seeds.shape[0] < self.min_points:
            return self._unknown_plane()

        # ------------------------------------------------------------------
        # Phase 3: Plane fitting (PCA)
        # ------------------------------------------------------------------
        mean = seeds.mean(axis=0)
        centered = seeds - mean

        cov = np.dot(centered.T, centered) / max(len(seeds) - 1, 1)
        cov += self.eps * np.eye(3)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # smallest eigenvalue → plane normal

        if normal[2] < 0.0:
            normal = -normal  # ensure upward-facing normal

        normal_norm = np.linalg.norm(normal)
        if normal_norm < self.eps:
            return self._unknown_plane()
        normal = normal / normal_norm

        d = -np.dot(normal, mean)

        # ------------------------------------------------------------------
        # Phase 4: Planarity weight and labelling
        # ------------------------------------------------------------------
        s0, s1, s2 = eigenvalues
        weight = min((s0 + s1) * s1 / (s0 * s2 + self.eps), self.max_weight)

        if normal[2] > self.th_normal and weight > self.th_weight:
            label = CellState.GROUND
        else:
            label = CellState.NON_GROUND

        return PlaneModel(normal=normal, mean=mean, d=d, weight=weight, label=label)

    def _select_seeds(self, points: np.ndarray, ref_z: float) -> np.ndarray:
        """Return points within the height band [ref_z − th_outlier, ref_z + th_seeds]."""
        mask = (
            (points[:, 2] < ref_z + self.th_seeds) &
            (points[:, 2] > ref_z - self.th_outlier)
        )
        return points[mask]

    def estimate_and_update(self, subcell) -> None:
        model = self.estimate(subcell.points)
        subcell.normal = model.normal
        subcell.mean   = model.mean
        subcell.d      = model.d
        subcell.weight = model.weight
        subcell.label  = model.label

    def _unknown_plane(self) -> PlaneModel:
        return PlaneModel(
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            mean=np.zeros(3, dtype=float),
            d=0.0,
            weight=0.0,
            label=CellState.UNKNOWN,
        )


def is_traversable_lcc(
    src: PlaneModel,
    dst: PlaneModel,
    th_normal: float,
    th_dist: float,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """
    Local Convexity / Concavity (LCC) check between two adjacent planes.

    Two planes are traversable when:
      1. Their normals are nearly aligned (similar surface orientation).
      2. Neither plane is far above or below the other (small height step).

    Returns: (is_traversable, rejection_reason)
    """
    # 1. Normal similarity
    sim = np.dot(src.normal, dst.normal)
    if sim < th_normal:
        if verbose:
            print(f"  [LCC Reject] Normal similarity {sim:.3f} < {th_normal}")
        return False, "SIMILARITY_TOO_LOW"

    # 2. Plane distance: project the inter-mean vector onto each plane's normal.
    delta    = dst.mean - src.mean
    dist_src = np.dot(src.normal, delta)
    dist_dst = np.dot(dst.normal, -delta)

    if abs(dist_src) > th_dist or abs(dist_dst) > th_dist:
        if verbose:
            print(f"  [LCC Reject] Dist src={abs(dist_src):.3f}, dst={abs(dist_dst):.3f} > {th_dist}")
        return False, "DIST_TOO_LARGE"

    return True, "NONE"
