"""
Layer 2 Extension: Reachability-Aware Multi-Level Ground Verification (B+D+E).

==============================================================================
DEVIATION FROM OFFICIAL TRAVEL (darrenjkt/TRAVEL)
==============================================================================
Official TRAVEL (`tgs.hpp`) treats ground detection and re-exploration as a
single flat-surface search: when the BFS queue empties, any unvisited flat
subcell is promoted to a new seed. In urban scenes, this can promote elevated
flat structures (building terraces, window ledges, car roofs, trimmed hedges)
to GROUND, causing dangerous high-elevation false positives.

This module intentionally separates **Ground Detection (Layer 1: Pure TRAVEL)**
from **Reachability / Traversability Verification (Layer 2: This Module)**:

  - Layer 1 (`fast.py`) extracts all flat connected components using pure
    TRAVEL (LPR + PCA + LCC BFS) without heuristic distortion.
  - Layer 2 (`reachability.py`) takes those components and classifies them into
    hierarchical reachability levels:
      * LEVEL_PRIMARY (1): Directly reachable from ego-anchored `top_k` seeds.
      * LEVEL_BRIDGED (2): Disconnected flat components verified as coplanar
        extensions of confirmed ground via:
          (E) Z-Stacking / Under-surface rejection: rejects components sitting
              above already-confirmed ground in the same or immediate 8-neighbor
              grid cells (e.g., roofs, terraces, overpasses).
          (D) Bidirectional Coplanarity Bridge: checks both
              |n_u · (μ_v - μ_u)| <= bridge_max_plane_dist AND
              |n_v · (μ_u - μ_v)| <= bridge_max_plane_dist
              within `bridge_max_dist` and normal similarity >= `bridge_min_normal_sim`.
              This handles slopes naturally while rejecting distant horizontal
              elevated surfaces that would fool an angle-only threshold.
      * LEVEL_OTHER (3): Flat surfaces that fail reachability/coplanarity
        (treated as non-traversable / non-ground for safe obstacle avoidance).
==============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config import ReachabilityConfig
from .fast import FlatGrid, PlaneArrays, TraversalResult

try:
    from scipy.spatial import cKDTree as _cKDTree
except ImportError:  # pragma: no cover - Pyodide / minimal NumPy fallback
    _cKDTree = None


# Reachability level codes (subcell & point level)
LEVEL_NONE = 0          # Non-ground or Unknown
LEVEL_PRIMARY = 1       # Directly reachable from ego top_k seeds
LEVEL_BRIDGED = 2       # Recovered via B+D+E geometric coplanarity bridge
LEVEL_OTHER = 3         # Flat surface on an unreachable / different elevation level


@dataclass
class ReachabilityResult:
    """Output of Layer 2 reachability verification."""
    subcell_levels: np.ndarray     # (M,) int8  (LEVEL_NONE/PRIMARY/BRIDGED/OTHER)
    point_levels: np.ndarray       # (N,) int8  (LEVEL_NONE/PRIMARY/BRIDGED/OTHER)
    reachable_visited: np.ndarray  # (M,) bool  (True only for PRIMARY or BRIDGED)
    stats: Dict[str, int] = field(default_factory=dict)


def _nearest_2d(
    ref_xy: np.ndarray,
    query_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_dist_2d, nearest_ref_idx) for each point in query_xy."""
    if _cKDTree is not None:
        tree = _cKDTree(ref_xy)
        dists, idx = tree.query(query_xy, k=1)
        return np.asarray(dists, dtype=np.float64), np.asarray(idx, dtype=np.int64)
    # Pure NumPy fallback (used if scipy is absent, e.g. lightweight Pyodide)
    diff = query_xy[:, None, :] - ref_xy[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    idx = np.argmin(d2, axis=1)
    dists = np.sqrt(d2[np.arange(len(query_xy)), idx])
    return dists, idx


def verify_reachability(
    flat: FlatGrid,
    planes: PlaneArrays,
    trav: TraversalResult,
    cfg: ReachabilityConfig,
    *,
    primary_component_count: int,
) -> ReachabilityResult:
    """
    Verify Layer 1 connected components and classify them into
    PRIMARY_GROUND, BRIDGED_GROUND, and OTHER_LEVEL_GROUND.

    Parameters
    ----------
    flat : FlatGrid
        Layer 1 flat grid representation.
    planes : PlaneArrays
        Layer 1 local plane estimates.
    trav : TraversalResult
        Layer 1 BFS traversal result (must include `component_id`).
    cfg : ReachabilityConfig
        Layer 2 B+D+E verification thresholds.
    primary_component_count : int
        Number of initial ego-anchored seed components (`cfg.seed.top_k`).
        Components with `0 <= component_id < primary_component_count` are
        treated as PRIMARY_GROUND; components with `component_id >= primary_component_count`
        are verified as secondary candidates.
    """
    M = flat.num_subcells
    N = flat.num_points
    subcell_levels = np.zeros(M, dtype=np.int8)

    if M == 0 or trav.component_id.size == 0:
        return ReachabilityResult(
            subcell_levels=subcell_levels,
            point_levels=np.zeros(N, dtype=np.int8),
            reachable_visited=np.zeros(M, dtype=bool),
            stats={"primary": 0, "bridged": 0, "other": 0},
        )

    ci, cj = flat.cell_ij
    nrm_l = planes.normal.tolist()
    mean_l = planes.mean.tolist()
    comp_ids = trav.component_id

    # Group visited subcell rows by component_id
    max_cid = int(comp_ids.max())
    if max_cid < 0:
        return ReachabilityResult(
            subcell_levels=subcell_levels,
            point_levels=np.zeros(N, dtype=np.int8),
            reachable_visited=np.zeros(M, dtype=bool),
            stats={"primary": 0, "bridged": 0, "other": 0},
        )

    primary_rows: List[int] = []
    secondary_components: List[List[int]] = []
    cell_min_gnd_z: Dict[Tuple[int, int], float] = {}

    # Bucket rows by component_id in O(M)
    buckets: List[List[int]] = [[] for _ in range(max_cid + 1)]
    visited_rows = np.nonzero(comp_ids >= 0)[0]
    for row in visited_rows.tolist():
        buckets[int(comp_ids[row])].append(row)

    n_primary = max(int(primary_component_count), 0)
    for cid, rows in enumerate(buckets):
        if not rows:
            continue
        if cid < n_primary:
            primary_rows.extend(rows)
            for idx in rows:
                subcell_levels[idx] = LEVEL_PRIMARY
                k_ij = (int(ci[idx]), int(cj[idx]))
                mz = mean_l[idx][2]
                if k_ij not in cell_min_gnd_z or mz < cell_min_gnd_z[k_ij]:
                    cell_min_gnd_z[k_ij] = mz
        else:
            secondary_components.append(rows)

    # Iteratively verify secondary components against confirmed ground (PRIMARY + BRIDGED)
    confirmed_rows = list(primary_rows)
    remaining_comps = list(secondary_components)

    bridge_max_dist = cfg.bridge_max_dist
    bridge_max_plane_dist = cfg.bridge_max_plane_dist
    bridge_min_normal_sim = cfg.bridge_min_normal_sim
    z_stack_margin = cfg.z_stack_margin

    if confirmed_rows and remaining_comps:
        changed = True
        while changed and remaining_comps:
            changed = False
            conf_arr = np.array(confirmed_rows, dtype=np.int64)
            conf_xy = planes.mean[conf_arr, :2]

            next_remaining: List[List[int]] = []
            for comp in remaining_comps:
                comp_arr = np.array(comp, dtype=np.int64)
                comp_xy = planes.mean[comp_arr, :2]

                dists_2d, nn_indices = _nearest_2d(conf_xy, comp_xy)
                best_local = int(np.argmin(dists_2d))
                min_d2d = float(dists_2d[best_local])

                if min_d2d > bridge_max_dist:
                    next_remaining.append(comp)
                    continue

                v = int(comp_arr[best_local])
                u = int(conf_arr[nn_indices[best_local]])

                # --- Check E: Z-Stacking / Under-surface rejection ---
                z_stack_fail = False
                for idx in comp:
                    ic, jc = int(ci[idx]), int(cj[idx])
                    vz = mean_l[idx][2]
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            ref_z = cell_min_gnd_z.get((ic + di, jc + dj))
                            if ref_z is not None and (vz - ref_z) > z_stack_margin:
                                z_stack_fail = True
                                break
                        if z_stack_fail:
                            break
                    if z_stack_fail:
                        break
                if z_stack_fail:
                    subcell_levels[comp_arr] = LEVEL_OTHER
                    continue

                # --- Check D: Bidirectional Coplanarity Check ---
                nu = nrm_l[u]
                nv = nrm_l[v]
                sim = nu[0] * nv[0] + nu[1] * nv[1] + nu[2] * nv[2]
                if sim < bridge_min_normal_sim:
                    next_remaining.append(comp)
                    continue

                mu = mean_l[u]
                mv = mean_l[v]
                dx, dy, dz = mv[0] - mu[0], mv[1] - mu[1], mv[2] - mu[2]
                d_u_to_v = abs(nu[0] * dx + nu[1] * dy + nu[2] * dz)
                d_v_to_u = abs(nv[0] * dx + nv[1] * dy + nv[2] * dz)

                if d_u_to_v <= bridge_max_plane_dist and d_v_to_u <= bridge_max_plane_dist:
                    subcell_levels[comp_arr] = LEVEL_BRIDGED
                    confirmed_rows.extend(comp)
                    for idx in comp:
                        k_ij = (int(ci[idx]), int(cj[idx]))
                        mz = mean_l[idx][2]
                        if k_ij not in cell_min_gnd_z or mz < cell_min_gnd_z[k_ij]:
                            cell_min_gnd_z[k_ij] = mz
                    changed = True
                else:
                    next_remaining.append(comp)

            remaining_comps = next_remaining

    # Remaining unbridged secondary components -> LEVEL_OTHER
    for comp in remaining_comps:
        subcell_levels[np.array(comp, dtype=np.int64)] = LEVEL_OTHER

    reachable_visited = (subcell_levels == LEVEL_PRIMARY) | (subcell_levels == LEVEL_BRIDGED)

    point_levels = np.zeros(N, dtype=np.int8)
    if flat.points.shape[0] > 0:
        point_levels[flat.point_indices] = subcell_levels[flat.gid]

    stats = {
        "primary": int(np.sum(subcell_levels == LEVEL_PRIMARY)),
        "bridged": int(np.sum(subcell_levels == LEVEL_BRIDGED)),
        "other": int(np.sum(subcell_levels == LEVEL_OTHER)),
    }

    return ReachabilityResult(
        subcell_levels=subcell_levels,
        point_levels=point_levels,
        reachable_visited=reachable_visited,
        stats=stats,
    )
