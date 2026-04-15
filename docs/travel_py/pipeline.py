from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from travel_py.cell_features import compute_all_cell_features
from travel_py.graph import TraversabilityGraph
from travel_py.grid import Grid, GridSpec
from travel_py.labeling import label_points_from_cells
from travel_py.plane import LocalPlaneEstimator, PlaneModel, is_traversable_lcc
from travel_py.seed import find_dominant_subcells
from travel_py.traversal import run_subcell_traversal
from travel_py.types import CellState


@dataclass
class PipelineResult:
    grid: Grid
    grid_spec: GridSpec
    labels: np.ndarray
    start_nodes: list
    step1_labels: dict[tuple, CellState]
    rejection_map: dict[tuple, str]
    visited: set



def run_pipeline(points: np.ndarray, global_cfg) -> PipelineResult:
    """Run the full TRAVEL reference pipeline and return debug-friendly artifacts."""
    grid_cfg = global_cfg.grid
    grid_spec = GridSpec(
        resolution=grid_cfg.resolution,
        origin_xy=grid_cfg.origin_xy,
        size_xy=grid_cfg.size_xy,
    )
    grid = Grid.from_points(points, grid_spec)
    compute_all_cell_features(grid.iter_cells(), points)

    accept_cfg = global_cfg.accept
    plane_estimator = LocalPlaneEstimator(
        num_lpr=20,
        th_seeds=accept_cfg.th_seeds,
        th_outlier=0.5,
        th_normal=accept_cfg.th_normal,
        min_points=3,
        th_weight=0.0,
    )
    for cell in grid.iter_cells():
        for subcell in cell.subcells.values():
            plane_estimator.estimate_and_update(subcell)

    step1_labels: dict[tuple, CellState] = {}
    for cell in grid.iter_cells():
        for t, sub in cell.subcells.items():
            step1_labels[(cell.index[0], cell.index[1], t)] = sub.label

    graph = TraversabilityGraph(grid)
    start_nodes = find_dominant_subcells(grid, top_k=global_cfg.seed.top_k)

    rejection_map: dict[tuple, str] = {}

    def accept_lcc(src_idx, dst_idx) -> bool:
        src_cell = grid.cells.get((src_idx.i, src_idx.j))
        dst_cell = grid.cells.get((dst_idx.i, dst_idx.j))
        if not src_cell or not dst_cell:
            return False

        src_sub = src_cell.subcells.get(src_idx.tri)
        dst_sub = dst_cell.subcells.get(dst_idx.tri)
        if not src_sub or not dst_sub:
            return False

        if dst_sub.label != CellState.GROUND:
            rejection_map[(dst_idx.i, dst_idx.j, dst_idx.tri)] = "NOT_GROUND"
            return False

        if src_sub.normal is None or dst_sub.normal is None:
            return False

        if (
            src_sub.normal[2] < accept_cfg.th_normal
            or dst_sub.normal[2] < accept_cfg.th_normal
        ):
            rejection_map[(dst_idx.i, dst_idx.j, dst_idx.tri)] = "LOW_NORMAL"
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
        ok, reason = is_traversable_lcc(
            src_plane,
            dst_plane,
            th_normal=accept_cfg.th_normal,
            th_dist=accept_cfg.max_height_diff,
        )
        if not ok:
            rejection_map[(dst_idx.i, dst_idx.j, dst_idx.tri)] = reason
        return ok

    visited, _, _ = run_subcell_traversal(
        graph=graph,
        start_nodes=start_nodes,
        accept_fn=accept_lcc,
    )

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

    labels = label_points_from_cells(cells=grid.cells, num_points=len(points))

    return PipelineResult(
        grid=grid,
        grid_spec=grid_spec,
        labels=labels,
        start_nodes=start_nodes,
        step1_labels=step1_labels,
        rejection_map=rejection_map,
        visited=visited,
    )
