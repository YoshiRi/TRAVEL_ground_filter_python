"""
Vectorized (array-based) implementation of the TGS pipeline.

This module is a *fast path* that produces the same results as the object-based
reference implementation (grid.py / plane.py / traversal.py / labeling.py),
without per-point or per-SubCell Python loops.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .grid import Grid, GridSpec, assign_cells_and_tris
from .plane import LocalPlaneEstimator
from .types import Cell, CellState, SubCell, SubCellIndex

# Integer label codes used in the array representation.
LABEL_UNKNOWN = -1
LABEL_NON_GROUND = 0
LABEL_GROUND = 1

_CODE_TO_STATE = {
    LABEL_UNKNOWN: CellState.UNKNOWN,
    LABEL_NON_GROUND: CellState.NON_GROUND,
    LABEL_GROUND: CellState.GROUND,
}

# Same edge-sharing table as graph._EXTERNAL_NEIGHBOR: tri -> (di, dj, neighbor tri)
_EXTERNAL_NEIGHBOR = ((1, 0, 2), (0, 1, 3), (-1, 0, 0), (0, -1, 1))


# =========================
# Flat grid
# =========================

@dataclass
class FlatGrid:
    spec: GridSpec
    num_points: int
    # Index-space bounds actually used for sid encoding.
    ix0: int
    iy0: int
    nx: int
    ny: int
    # Per sorted point (K = number of in-grid points)
    point_indices: np.ndarray   # (K,) original indices
    points: np.ndarray          # (K, 3) xyz, sorted by (sid, z)
    gid: np.ndarray             # (K,) row of the subcell each point belongs to
    # Per non-empty subcell (M rows)
    sid: np.ndarray             # (M,) global subcell id (ascending)
    start: np.ndarray           # (M,)
    count: np.ndarray           # (M,)
    cell_first_point: np.ndarray  # (M,) min original point index in the *cell* (tie-break order)
    dropped_point_indices: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))

    @property
    def num_subcells(self) -> int:
        return int(self.sid.size)

    @property
    def tri(self) -> np.ndarray:
        return self.sid % 4

    @property
    def cell_ij(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cell indices (ix, iy) in the original grid index space."""
        c = self.sid // 4
        return c // self.ny + self.ix0, c % self.ny + self.iy0


def build_flat_grid(points_xyz: np.ndarray, spec: GridSpec) -> FlatGrid:
    points_xyz = np.asarray(points_xyz)
    n_total = points_xyz.shape[0] if points_xyz.ndim == 2 else 0
    in_indices, ix, iy, tri = assign_cells_and_tris(points_xyz, spec)

    if spec.size_xy is not None:
        ix0 = iy0 = 0
        nx, ny = spec.size_xy
    elif in_indices.size > 0:
        ix0, iy0 = int(ix.min()), int(iy.min())
        nx, ny = int(ix.max()) - ix0 + 1, int(iy.max()) - iy0 + 1
    else:
        ix0 = iy0 = 0
        nx = ny = 1

    cell_id = (ix - ix0) * ny + (iy - iy0)
    sid = cell_id * 4 + tri
    z = points_xyz[in_indices, 2].astype(np.float64)

    order = np.lexsort((z, sid))
    sid_sorted = sid[order]
    uniq, start, count = np.unique(sid_sorted, return_index=True, return_counts=True)
    gid = np.repeat(np.arange(uniq.size), count)

    # Reference Grid stores cells in dict insertion order = order of first point.
    cell_first = np.full(nx * ny, n_total, dtype=np.int64)
    np.minimum.at(cell_first, cell_id, in_indices)

    if spec.size_xy is not None:
        dropped = np.setdiff1d(np.arange(n_total, dtype=np.int64), in_indices, assume_unique=True)
    else:
        dropped = np.empty(0, np.int64)

    return FlatGrid(
        spec=spec,
        num_points=n_total,
        ix0=ix0, iy0=iy0, nx=nx, ny=ny,
        point_indices=in_indices[order],
        points=points_xyz[in_indices[order], :3].astype(np.float64),
        gid=gid,
        sid=uniq,
        start=start,
        count=count,
        cell_first_point=cell_first[uniq // 4],
        dropped_point_indices=dropped,
    )


# =========================
# Batched plane estimation
# =========================

@dataclass
class PlaneArrays:
    normal: np.ndarray   # (M, 3)
    mean: np.ndarray     # (M, 3)
    d: np.ndarray        # (M,)
    weight: np.ndarray   # (M,)
    label: np.ndarray    # (M,) int8 codes (LABEL_*)


def estimate_planes(flat: FlatGrid, est: LocalPlaneEstimator) -> PlaneArrays:
    """
    Batched equivalent of LocalPlaneEstimator.estimate() for every subcell.
    Follows the same 4 phases (LPR, seed band + wall fallback, PCA, labelling).
    """
    M = flat.num_subcells
    gid, cnt, P = flat.gid, flat.count, flat.points
    zs = P[:, 2]

    normal = np.tile(np.array([0.0, 0.0, 1.0]), (M, 1))
    mean = np.zeros((M, 3))
    d = np.zeros(M)
    weight = np.zeros(M)
    label = np.full(M, LABEL_UNKNOWN, dtype=np.int8)
    if M == 0:
        return PlaneArrays(normal, mean, d, weight, label)

    # Phase 1: LPR (points are z-sorted within each block)
    rank = np.arange(zs.size) - flat.start[gid]
    n_lpr = np.minimum(cnt, est.num_lpr)
    lpr = np.bincount(gid, zs * (rank < est.num_lpr), M) / np.maximum(n_lpr, 1)
    min_z = zs[flat.start]

    # Phase 2: seed band with wall-contamination fallback
    def band(ref: np.ndarray) -> np.ndarray:
        r = ref[gid]
        return (zs < r + est.th_seeds) & (zs > r - est.th_outlier)

    seeds = band(lpr)
    fallback = band(min_z)
    use_fb = (lpr - est.th_outlier > min_z) & (np.bincount(gid, fallback, M) >= est.min_points)
    seeds = np.where(use_fb[gid], fallback, seeds)
    n_seeds = np.bincount(gid, seeds, M)
    valid = (cnt >= est.min_points) & (n_seeds >= est.min_points)

    # Phase 3: PCA from per-group sums
    w = seeds.astype(np.float64)
    mean_all = np.stack([np.bincount(gid, P[:, k] * w, M) for k in range(3)], axis=1)
    mean_all /= np.maximum(n_seeds, 1)[:, None]
    C = (P - mean_all[gid]) * w[:, None]
    cov = np.empty((M, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            cov[:, a, b] = cov[:, b, a] = np.bincount(gid, C[:, a] * C[:, b], M)
    cov /= np.maximum(n_seeds - 1, 1)[:, None, None]
    cov += est.eps * np.eye(3)

    ev, evec = np.linalg.eigh(cov)
    nrm = evec[:, :, 0].copy()
    nrm[nrm[:, 2] < 0.0] *= -1.0
    nlen = np.linalg.norm(nrm, axis=1)
    valid &= nlen >= est.eps
    nrm /= np.where(nlen > 0, nlen, 1.0)[:, None]

    # Phase 4: weight + label
    s0, s1, s2 = ev[:, 0], ev[:, 1], ev[:, 2]
    wgt = np.minimum((s0 + s1) * s1 / (s0 * s2 + est.eps), est.max_weight)
    is_ground = (nrm[:, 2] > est.th_normal) & (wgt > est.th_weight)

    normal[valid] = nrm[valid]
    mean[valid] = mean_all[valid]
    d[valid] = -np.einsum("ij,ij->i", nrm[valid], mean_all[valid])
    weight[valid] = wgt[valid]
    label[valid] = np.where(is_ground[valid], LABEL_GROUND, LABEL_NON_GROUND)
    return PlaneArrays(normal, mean, d, weight, label)


# =========================
# Traversal
# =========================

REJECTION_KEYS = (
    "NO_SUBCELL",
    "NOT_GROUND_CANDIDATE",
    "NO_PLANE",
    "LOW_NORMAL",
    "SIMILARITY_TOO_LOW",
    "DIST_TOO_LARGE",
)


@dataclass
class TraversalResult:
    visited: np.ndarray            # (M,) bool
    start_seed_rows: List[int]     # rows chosen as seeds
    rejected_count: int
    max_depth: int
    rejection_stats: Dict[str, int]


def traverse_iterative(
    flat: FlatGrid,
    planes: PlaneArrays,
    *,
    top_k: int,
    th_normal: float,
    th_dist: float,
    prefer_center: bool = False,
    adaptive: bool = False,
    th_lcc_normal_similarity: float = 0.03,
    th_lcc_planar_dist: float = 0.2,
) -> TraversalResult:
    """
    Iterative BFS matching seed.find_dominant_subcells(already_visited=visited).
    Picks one seed at a time from GROUND candidates not yet visited, then expands BFS.
    """
    M = flat.num_subcells
    nx, ny = flat.nx, flat.ny
    dense = np.full(nx * ny * 4, -1, dtype=np.int64)
    dense[flat.sid] = np.arange(M)
    cell_exists = np.zeros(nx * ny, dtype=bool)
    cell_exists[flat.sid // 4] = True

    dense_l = dense.tolist()
    cell_l = cell_exists.tolist()
    sid_l = flat.sid.tolist()
    gnd_l = (planes.label == LABEL_GROUND).tolist()
    nrm_l = planes.normal.tolist()
    mean_l = planes.mean.tolist()

    # Precompute scores for seed selection
    scores = planes.weight.copy()
    if prefer_center:
        ci, cj = flat.cell_ij
        res = flat.spec.resolution
        ox, oy = flat.spec.origin_xy
        cx = ox + (ci + 0.5) * res
        cy = oy + (cj + 0.5) * res
        dist = np.sqrt(cx * cx + cy * cy)
        scores /= (1.0 + dist)

    cand = np.nonzero(planes.label == LABEL_GROUND)[0]
    tie = flat.cell_first_point[cand] * 4 + flat.tri[cand]
    # Stable sort by descending score, then tie
    ranked_candidates = cand[np.lexsort((tie, -scores[cand]))].tolist()

    stats = {k: 0 for k in REJECTION_KEYS}
    visited = [False] * M
    start_seeds: List[int] = []
    total_rejected = 0
    max_depth = 0

    cand_ptr = 0
    num_cand = len(ranked_candidates)

    for _ in range(max(top_k, 0)):
        # Find next unvisited ground seed
        next_seed = -1
        while cand_ptr < num_cand:
            c_idx = ranked_candidates[cand_ptr]
            cand_ptr += 1
            if not visited[c_idx]:
                next_seed = c_idx
                break
        if next_seed < 0:
            break

        start_seeds.append(next_seed)
        visited[next_seed] = True
        q: deque = deque([(next_seed, 0)])

        while q:
            u, depth = q.popleft()
            if depth > max_depth:
                max_depth = depth
            g = sid_l[u]
            c, tr = divmod(g, 4)
            ci_val, cj_val = divmod(c, ny)
            nbrs = [c * 4 + t for t in range(4) if t != tr]
            di, dj, et = _EXTERNAL_NEIGHBOR[tr]
            ni, nj = ci_val + di, cj_val + dj
            if 0 <= ni < nx and 0 <= nj < ny and cell_l[ni * ny + nj]:
                nbrs.append((ni * ny + nj) * 4 + et)

            nu = nrm_l[u]
            mu = mean_l[u]
            sin_planar_dist = math.sin(th_lcc_planar_dist) if adaptive else 0.0
            for g2 in nbrs:
                v = dense_l[g2]
                if v >= 0 and visited[v]:
                    continue
                if v < 0 or not gnd_l[v]:
                    stats["NOT_GROUND_CANDIDATE"] += 1
                    total_rejected += 1
                    continue
                nv = nrm_l[v]
                if nu[2] < th_normal or nv[2] < th_normal:
                    stats["LOW_NORMAL"] += 1
                    total_rejected += 1
                    continue
                mv = mean_l[v]
                dx, dy, dz = mv[0] - mu[0], mv[1] - mu[1], mv[2] - mu[2]
                sim = nu[0] * nv[0] + nu[1] * nv[1] + nu[2] * nv[2]

                if adaptive:
                    diff_norm = math.sqrt(dx * dx + dy * dy + dz * dz)
                    th_sim = 1.0 - math.sin(diff_norm * th_lcc_normal_similarity)
                    if sim < th_sim:
                        stats["SIMILARITY_TOO_LOW"] += 1
                        total_rejected += 1
                        continue
                    max_d = diff_norm * sin_planar_dist
                else:
                    if sim < th_normal:
                        stats["SIMILARITY_TOO_LOW"] += 1
                        total_rejected += 1
                        continue
                    max_d = th_dist

                if (abs(nu[0] * dx + nu[1] * dy + nu[2] * dz) > max_d
                        or abs(nv[0] * dx + nv[1] * dy + nv[2] * dz) > max_d):
                    stats["DIST_TOO_LARGE"] += 1
                    total_rejected += 1
                    continue
                visited[v] = True
                q.append((v, depth + 1))

    return TraversalResult(
        visited=np.array(visited, dtype=bool),
        start_seed_rows=start_seeds,
        rejected_count=total_rejected,
        max_depth=max_depth,
        rejection_stats=stats,
    )


def final_subcell_labels(planes: PlaneArrays, visited: np.ndarray) -> np.ndarray:
    """Visited -> GROUND; GROUND but unreachable -> UNKNOWN; others unchanged."""
    out = planes.label.copy()
    out[(out == LABEL_GROUND) & ~visited] = LABEL_UNKNOWN
    out[visited] = LABEL_GROUND
    return out


# =========================
# Point labeling
# =========================

def label_points(
    flat: FlatGrid,
    planes: PlaneArrays,
    subcell_labels: np.ndarray,
    *,
    th_dist_from_plane: float = 0.2,
) -> np.ndarray:
    """
    Equivalent of labeling.label_points_from_cells(): GROUND subcells classify
    each point by its distance from the subcell plane; other subcells apply
    their label uniformly.  Points outside the grid stay UNKNOWN (-1).
    """
    labels = np.full(flat.num_points, LABEL_UNKNOWN, dtype=np.int8)
    if flat.points.shape[0] == 0:
        return labels
    lbl = subcell_labels[flat.gid]
    dist = np.abs(np.einsum("ij,ij->i", flat.points, planes.normal[flat.gid]) + planes.d[flat.gid])
    pl = np.where(lbl == LABEL_GROUND,
                  np.where(dist <= th_dist_from_plane, LABEL_GROUND, LABEL_NON_GROUND),
                  lbl)
    labels[flat.point_indices] = pl
    return labels


# =========================
# Debug view
# =========================

def to_grid(
    flat: FlatGrid,
    planes: Optional[PlaneArrays] = None,
    subcell_labels: Optional[np.ndarray] = None,
) -> Grid:
    """
    Materialize Cell / SubCell objects (same structure as Grid.from_points) for
    debugging and visualization.
    """
    grid = Grid(flat.spec)
    grid.num_points_in_grid = int(flat.point_indices.size)
    grid.dropped_point_indices = flat.dropped_point_indices.tolist()
    if subcell_labels is None and planes is not None:
        subcell_labels = planes.label

    ci, cj = flat.cell_ij
    ci_l, cj_l, tri_l = ci.tolist(), cj.tolist(), flat.tri.tolist()
    for m in np.lexsort((flat.tri, flat.cell_first_point)).tolist():
        cell = grid.get_or_create_cell((ci_l[m], cj_l[m]))
        s, e = flat.start[m], flat.start[m] + flat.count[m]
        o = np.argsort(flat.point_indices[s:e], kind="stable")
        sub = SubCell(points=flat.points[s:e][o], point_indices=flat.point_indices[s:e][o].tolist())
        if planes is not None:
            sub.normal = planes.normal[m].copy()
            sub.mean = planes.mean[m].copy()
            sub.d = float(planes.d[m])
            sub.weight = float(planes.weight[m])
        if subcell_labels is not None:
            sub.label = _CODE_TO_STATE[int(subcell_labels[m])]
        cell.subcells[tri_l[m]] = sub

    for cell in grid.iter_cells():
        for t in range(4):
            if t not in cell.subcells:
                cell.subcells[t] = SubCell(points=np.empty((0, 3)))
        cell.subcells = dict(sorted(cell.subcells.items()))
        cell.point_indices = sorted(i for sub in cell.subcells.values() for i in sub.point_indices)
        labels = [sub.label for sub in cell.subcells.values()]
        if CellState.GROUND in labels:
            cell.state = CellState.GROUND
        elif CellState.NON_GROUND in labels:
            cell.state = CellState.NON_GROUND
    return grid
