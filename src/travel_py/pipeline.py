from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, List, Tuple

import numpy as np

from travel_py.cell_features import compute_all_cell_features
from travel_py.config import GlobalConfig
from travel_py.graph import TraversabilityGraph
from travel_py.grid import Grid, GridSpec
from travel_py.labeling import label_points_from_cells
from travel_py.plane import LocalPlaneEstimator, PlaneModel, is_traversable_lcc
from travel_py.seed import find_dominant_subcells
from travel_py.traversal import run_subcell_traversal
from travel_py.types import CellState, SubCellIndex
from travel_py import fast


def make_plane_estimator(accept_cfg) -> LocalPlaneEstimator:
    """Plane estimator parameters used by the pipeline."""
    th_seeds = getattr(accept_cfg, "th_seeds", 0.5)
    return LocalPlaneEstimator(
        num_lpr=20,
        th_seeds=th_seeds,
        th_outlier=0.5,
        th_normal=accept_cfg.th_normal,
        min_points=3,
        th_weight=0.0,
    )


def make_grid_spec(cfg: GlobalConfig) -> GridSpec:
    g = cfg.grid
    return GridSpec(
        resolution=g.resolution,
        origin_xy=tuple(g.origin_xy),
        size_xy=tuple(g.size_xy) if g.size_xy is not None else None,
    )


@dataclass
class PipelineResult:
    """
    Result of running the TRAVEL pipeline.
    Maintains 100% backward compatibility with interactive web demo (docs/)
    while also exposing timing and summary metrics.
    """
    grid: Grid
    grid_spec: GridSpec
    labels: np.ndarray
    start_nodes: list
    step1_labels: dict[tuple, CellState]
    rejection_map: dict[tuple, str]
    visited: set
    # Extra performance & diagnostic fields
    timings: Dict[str, float] = field(default_factory=dict)
    rejected_count: int = 0
    max_depth: int = 0
    rejection_stats: Dict[str, int] = field(default_factory=dict)

    @property
    def num_seeds(self) -> int:
        return len(self.start_nodes)

    @property
    def final_ground(self) -> int:
        return len(self.visited)

    @property
    def step1_stats(self) -> Dict[CellState, int]:
        stats = {CellState.GROUND: 0, CellState.NON_GROUND: 0, CellState.UNKNOWN: 0}
        for s in self.step1_labels.values():
            if s in stats:
                stats[s] += 1
        return stats


class _Timer:
    def __init__(self) -> None:
        self.t = time.perf_counter()
        self.timings: Dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.timings[name] = now - self.t
        self.t = now


# =========================
# Reference Pipeline (Object-based)
# =========================

def run_pipeline(points: np.ndarray, global_cfg: GlobalConfig) -> PipelineResult:
    """
    Run the full TRAVEL reference pipeline and return debug-friendly artifacts.
    This preserves the exact API expected by docs/interactive.html and viz_data.py.
    """
    timer = _Timer()
    grid_spec = make_grid_spec(global_cfg)
    grid = Grid.from_points(points, grid_spec)
    timer.lap("grid")

    compute_all_cell_features(grid.iter_cells(), points)
    timer.lap("features")

    accept_cfg = global_cfg.accept
    plane_estimator = make_plane_estimator(accept_cfg)
    for cell in grid.iter_cells():
        for subcell in cell.subcells.values():
            plane_estimator.estimate_and_update(subcell)

    step1_labels: dict[tuple, CellState] = {}
    for cell in grid.iter_cells():
        for t, sub in cell.subcells.items():
            step1_labels[(cell.index[0], cell.index[1], t)] = sub.label
    timer.lap("plane")

    graph = TraversabilityGraph(grid)
    seed_cfg = global_cfg.seed

    rejection_map: dict[tuple, str] = {}
    rejection_stats: dict[str, int] = {k: 0 for k in fast.REJECTION_KEYS}

    def accept_lcc(src_idx, dst_idx) -> bool:
        src_cell = grid.cells.get((src_idx.i, src_idx.j))
        dst_cell = grid.cells.get((dst_idx.i, dst_idx.j))
        if not src_cell or not dst_cell:
            rejection_stats["NO_SUBCELL"] += 1
            return False

        src_sub = src_cell.subcells.get(src_idx.tri)
        dst_sub = dst_cell.subcells.get(dst_idx.tri)
        if not src_sub or not dst_sub:
            rejection_stats["NO_SUBCELL"] += 1
            return False

        if dst_sub.label != CellState.GROUND:
            rejection_map[(dst_idx.i, dst_idx.j, dst_idx.tri)] = "NOT_GROUND"
            rejection_stats["NOT_GROUND_CANDIDATE"] += 1
            return False

        if src_sub.normal is None or dst_sub.normal is None:
            rejection_stats["NO_PLANE"] += 1
            return False

        if (
            src_sub.normal[2] < accept_cfg.th_normal
            or dst_sub.normal[2] < accept_cfg.th_normal
        ):
            rejection_map[(dst_idx.i, dst_idx.j, dst_idx.tri)] = "LOW_NORMAL"
            rejection_stats["LOW_NORMAL"] += 1
            return False

        src_plane = PlaneModel(
            normal=src_sub.normal,
            mean=src_sub.mean,
            d=src_sub.d,
            weight=src_sub.weight,
            label=src_sub.label,
        )
        dst_plane = PlaneModel(
            normal=dst_sub.normal,
            mean=dst_sub.mean,
            d=dst_sub.d,
            weight=dst_sub.weight,
            label=dst_sub.label,
        )
        use_adaptive = getattr(accept_cfg, "use_adaptive_lcc", False)
        th_norm_sim = getattr(accept_cfg, "th_lcc_normal_similarity", 0.03)
        th_plan_dist = getattr(accept_cfg, "th_lcc_planar_dist", 0.2)
        ok, reason = is_traversable_lcc(
            src_plane,
            dst_plane,
            th_normal=accept_cfg.th_normal,
            th_dist=accept_cfg.max_height_diff,
            adaptive=use_adaptive,
            th_lcc_normal_similarity=th_norm_sim,
            th_lcc_planar_dist=th_plan_dist,
        )
        if not ok:
            rejection_map[(dst_idx.i, dst_idx.j, dst_idx.tri)] = reason
            if reason in rejection_stats:
                rejection_stats[reason] += 1
        return ok

    # Iterative seeding: pick one seed at a time, skip already-covered subcells.
    start_nodes: list = []
    visited: set = set()
    total_rejected = 0
    max_depth = 0

    prefer_center = getattr(seed_cfg, "prefer_center", False)
    for _ in range(seed_cfg.top_k):
        seeds = find_dominant_subcells(
            grid,
            top_k=1,
            already_visited=visited,
            prefer_center=prefer_center,
        )
        if not seeds:
            break
        start_nodes.append(seeds[0])
        visited, rej, d = run_subcell_traversal(
            graph=graph,
            start_nodes=[seeds[0]],
            accept_fn=accept_lcc,
            initial_visited=visited,
        )
        total_rejected += rej
        if d > max_depth:
            max_depth = d

    for idx in visited:
        cell = grid.cells.get((idx.i, idx.j))
        if cell:
            sub = cell.subcells.get(idx.tri)
            if sub:
                sub.label = CellState.GROUND

    visited_tuples = {(v.i, v.j, v.tri) for v in visited}

    for cell in grid.iter_cells():
        for t, sub in cell.subcells.items():
            idx = (cell.index[0], cell.index[1], t)
            if idx not in visited_tuples and sub.label == CellState.GROUND:
                sub.label = CellState.UNKNOWN

    for cell in grid.iter_cells():
        has_ground = any(sub.label == CellState.GROUND for sub in cell.subcells.values())
        has_nonground = any(sub.label == CellState.NON_GROUND for sub in cell.subcells.values())
        if has_ground:
            cell.state = CellState.GROUND
        elif has_nonground:
            cell.state = CellState.NON_GROUND

    timer.lap("traversal")

    labels = label_points_from_cells(cells=grid.cells, num_points=len(points))
    timer.lap("label")

    return PipelineResult(
        grid=grid,
        grid_spec=grid_spec,
        labels=labels,
        start_nodes=start_nodes,
        step1_labels=step1_labels,
        rejection_map=rejection_map,
        visited=visited,
        timings=timer.timings,
        rejected_count=total_rejected,
        max_depth=max_depth,
        rejection_stats=rejection_stats,
    )


# Alias for explicit reference run
run_reference = run_pipeline


# =========================
# Fast Pipeline (Vectorized)
# =========================

def run_fast(points: np.ndarray, global_cfg: GlobalConfig, *, keep_grid: bool = True) -> PipelineResult:
    """
    Vectorized high-performance pipeline.
    Produces equivalent point labels and metrics.
    """
    timer = _Timer()
    points = np.asarray(points)
    grid_spec = make_grid_spec(global_cfg)

    flat = fast.build_flat_grid(points, grid_spec)
    timer.lap("grid")

    planes = fast.estimate_planes(flat, make_plane_estimator(global_cfg.accept))
    timer.lap("plane")

    prefer_center = getattr(global_cfg.seed, "prefer_center", False)
    use_adaptive = getattr(global_cfg.accept, "use_adaptive_lcc", False)
    th_norm_sim = getattr(global_cfg.accept, "th_lcc_normal_similarity", 0.03)
    th_plan_dist = getattr(global_cfg.accept, "th_lcc_planar_dist", 0.2)
    trav = fast.traverse_iterative(
        flat, planes,
        top_k=global_cfg.seed.top_k,
        th_normal=global_cfg.accept.th_normal,
        th_dist=global_cfg.accept.max_height_diff,
        prefer_center=prefer_center,
        adaptive=use_adaptive,
        th_lcc_normal_similarity=th_norm_sim,
        th_lcc_planar_dist=th_plan_dist,
    )
    final = fast.final_subcell_labels(planes, trav.visited)
    timer.lap("traversal")

    labels = fast.label_points(flat, planes, final)
    timer.lap("label")

    # Step-1 labels dict (including empty subcells within cells as UNKNOWN)
    step1_labels = {}
    ci, cj = flat.cell_ij
    for m in range(flat.num_subcells):
        step1_labels[(int(ci[m]), int(cj[m]), int(flat.tri[m]))] = fast._CODE_TO_STATE[int(planes.label[m])]

    cell_tuples = {(int(ci[m]), int(cj[m])) for m in range(flat.num_subcells)}
    for c_i, c_j in cell_tuples:
        for t in range(4):
            if (c_i, c_j, t) not in step1_labels:
                step1_labels[(c_i, c_j, t)] = CellState.UNKNOWN

    visited_set = {
        SubCellIndex(int(ci[m]), int(cj[m]), int(flat.tri[m]))
        for m in np.nonzero(trav.visited)[0]
    }
    start_nodes = [
        SubCellIndex(int(ci[s]), int(cj[s]), int(flat.tri[s]))
        for s in trav.start_seed_rows
    ]

    grid = fast.to_grid(flat, planes, final) if keep_grid else Grid(grid_spec)
    if keep_grid:
        compute_all_cell_features(grid.iter_cells(), points)
        timer.lap("to_grid")

    return PipelineResult(
        grid=grid,
        grid_spec=grid_spec,
        labels=labels,
        start_nodes=start_nodes,
        step1_labels=step1_labels,
        rejection_map={},
        visited=visited_set,
        timings=timer.timings,
        rejected_count=trav.rejected_count,
        max_depth=trav.max_depth,
        rejection_stats=trav.rejection_stats,
    )
