"""
TRAVEL Ground Filter — Rerun debug visualizer

Usage:
    uv run python tools/rerun_debug.py --points data/sample.npy
    uv run python tools/rerun_debug.py --points data/sample.npy --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from travel_py.config import load_config
from travel_py.grid import Grid, GridSpec
from travel_py.cell_features import compute_all_cell_features
from travel_py.plane import LocalPlaneEstimator, is_traversable_lcc, PlaneModel
from travel_py.graph import TraversabilityGraph
from travel_py.traversal import run_subcell_traversal
from travel_py.seed import find_dominant_subcells
from travel_py.labeling import label_points_from_cells
from travel_py.types import CellState, SubCellIndex


# Colors (RGBA uint8)
COLOR_GROUND = [50, 205, 50, 255]
COLOR_NONGROUND = [220, 60, 60, 255]
COLOR_UNKNOWN = [160, 160, 160, 120]
COLOR_SEED = [255, 215, 0, 255]


def run_pipeline(points: np.ndarray, global_cfg):
    grid_cfg = global_cfg.grid
    grid_spec = GridSpec(
        resolution=grid_cfg.resolution,
        origin_xy=grid_cfg.origin_xy,
        size_xy=grid_cfg.size_xy,
    )
    grid = Grid.from_points(points, grid_spec)
    compute_all_cell_features(grid.iter_cells(), points)

    plane_estimator = LocalPlaneEstimator(
        num_lpr=20, th_seeds=0.5, th_outlier=0.5,
        th_normal=0.9, min_points=3, th_weight=0.0,
    )
    for cell in grid.iter_cells():
        for subcell in cell.subcells.values():
            plane_estimator.estimate_and_update(subcell)

    # Step1 labels (before BFS)
    step1_labels: dict[tuple, CellState] = {}
    for cell in grid.iter_cells():
        for t, sub in cell.subcells.items():
            step1_labels[(cell.index[0], cell.index[1], t)] = sub.label

    accept_cfg = global_cfg.accept
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
        if src_sub.normal[2] < accept_cfg.th_normal or dst_sub.normal[2] < accept_cfg.th_normal:
            rejection_map[(dst_idx.i, dst_idx.j, dst_idx.tri)] = "LOW_NORMAL"
            return False
        src_plane = PlaneModel(normal=src_sub.normal, mean=src_sub.mean, d=src_sub.d, weight=src_sub.weight, label=src_sub.label)
        dst_plane = PlaneModel(normal=dst_sub.normal, mean=dst_sub.mean, d=dst_sub.d, weight=dst_sub.weight, label=dst_sub.label)
        ok, reason = is_traversable_lcc(src_plane, dst_plane, th_normal=accept_cfg.th_normal, th_dist=accept_cfg.max_height_diff)
        if not ok:
            rejection_map[(dst_idx.i, dst_idx.j, dst_idx.tri)] = reason
        return ok

    visited, rejected_count, max_depth = run_subcell_traversal(
        graph=graph, start_nodes=start_nodes, accept_fn=accept_lcc
    )

    # Apply traversal results
    depth_map: dict[tuple, int] = {}
    for idx in visited:
        cell = grid.cells.get((idx.i, idx.j))
        if cell:
            sub = cell.subcells.get(idx.tri)
            if sub:
                sub.label = CellState.GROUND
    for cell in grid.iter_cells():
        for t, sub in cell.subcells.items():
            idx = SubCellIndex(cell.index[0], cell.index[1], t)
            if idx not in visited and sub.label == CellState.GROUND:
                sub.label = CellState.UNKNOWN

    for cell in grid.iter_cells():
        has_ground = any(sub.label == CellState.GROUND for sub in cell.subcells.values())
        has_nonground = any(sub.label == CellState.NON_GROUND for sub in cell.subcells.values())
        if has_ground:
            cell.state = CellState.GROUND
        elif has_nonground:
            cell.state = CellState.NON_GROUND

    labels = label_points_from_cells(cells=grid.cells, num_points=len(points))

    return grid, grid_spec, labels, start_nodes, step1_labels, rejection_map, visited


def log_to_rerun(points, grid, grid_spec, labels, start_nodes, step1_labels, rejection_map, visited):
    res = grid_spec.resolution
    ox, oy = grid_spec.origin_xy

    # ── 1. 分類済み点群 ─────────────────────────────────────────
    for mask_val, path, color in [
        (1,  "points/ground",    COLOR_GROUND),
        (0,  "points/nonground", COLOR_NONGROUND),
        (-1, "points/unknown",   COLOR_UNKNOWN),
    ]:
        mask = labels == mask_val
        if mask.any():
            rr.log(path, rr.Points3D(
                positions=points[mask, :3],
                colors=np.tile(color, (mask.sum(), 1)),
                radii=0.05,
            ))

    # ── 2. Step1 vs Step2 比較（SubCellレベル、XY平面上に色付きボックス）──
    step1_boxes, step1_colors = [], []
    step2_boxes, step2_colors = [], []
    normal_origins, normal_dirs = [], []
    rejection_boxes, rejection_colors = [], []

    rejection_color_map = {
        "NOT_GROUND":        [128, 0, 128, 200],   # purple
        "LOW_NORMAL":        [255, 165, 0, 200],   # orange
        "SIMILARITY_TOO_LOW":[255, 0, 255, 200],   # magenta
        "DIST_TOO_LARGE":    [0, 0, 255, 200],     # blue
    }

    for cell in grid.iter_cells():
        i, j = cell.index
        cx = ox + (i + 0.5) * res
        cy = oy + (j + 0.5) * res

        for t, sub in cell.subcells.items():
            # SubCellの代表XY（セル中心から少しオフセット）
            offset = {0: (0.25, 0), 1: (0, 0.25), 2: (-0.25, 0), 3: (0, -0.25)}
            dx, dy = offset[t]
            sx = cx + dx * res
            sy = cy + dy * res
            sz = sub.mean[2] if sub.mean is not None else 0.0
            half = res * 0.4

            # Step1
            s1 = step1_labels.get((i, j, t), CellState.UNKNOWN)
            c1 = {CellState.GROUND: COLOR_GROUND, CellState.NON_GROUND: COLOR_NONGROUND, CellState.UNKNOWN: COLOR_UNKNOWN}[s1]
            step1_boxes.append([sx, sy, sz])
            step1_colors.append(c1)

            # Step2
            c2 = {CellState.GROUND: COLOR_GROUND, CellState.NON_GROUND: COLOR_NONGROUND, CellState.UNKNOWN: COLOR_UNKNOWN}[sub.label]
            step2_boxes.append([sx, sy, sz])
            step2_colors.append(c2)

            # 法線ベクトル（GROUNDのみ）
            if sub.normal is not None and sub.label == CellState.GROUND:
                normal_origins.append([sx, sy, sz])
                normal_dirs.append(sub.normal * res * 0.8)

            # 拒絶理由
            key = (i, j, t)
            if key in rejection_map:
                reason = rejection_map[key]
                rc = rejection_color_map.get(reason, [128, 128, 128, 200])
                rejection_boxes.append([sx, sy, sz + 0.3])
                rejection_colors.append(rc)

    if step1_boxes:
        rr.log("subcells/step1", rr.Points3D(
            positions=step1_boxes,
            colors=step1_colors,
            radii=res * 0.35,
        ))
    if step2_boxes:
        rr.log("subcells/step2", rr.Points3D(
            positions=step2_boxes,
            colors=step2_colors,
            radii=res * 0.35,
        ))

    # ── 3. 法線ベクトル ─────────────────────────────────────────
    if normal_origins:
        origins = np.array(normal_origins)
        vectors = np.array(normal_dirs)
        rr.log("subcells/normals", rr.Arrows3D(
            origins=origins,
            vectors=vectors,
            colors=[0, 255, 200, 200],
        ))

    # ── 4. 拒絶理由 ─────────────────────────────────────────────
    if rejection_boxes:
        rr.log("debug/rejections", rr.Points3D(
            positions=rejection_boxes,
            colors=rejection_colors,
            radii=res * 0.3,
        ))

    # ── 5. シード位置 ────────────────────────────────────────────
    seed_positions = []
    for idx in start_nodes:
        sx = ox + (idx.i + 0.5) * res
        sy = oy + (idx.j + 0.5) * res
        cell = grid.cells.get((idx.i, idx.j))
        sz = 2.0
        if cell and idx.tri in cell.subcells:
            sub = cell.subcells[idx.tri]
            if sub.mean is not None:
                sz = sub.mean[2] + 2.0
        seed_positions.append([sx, sy, sz])

    if seed_positions:
        rr.log("debug/seeds", rr.Points3D(
            positions=seed_positions,
            colors=np.tile(COLOR_SEED, (len(seed_positions), 1)),
            radii=res * 0.6,
        ))

    print(f"Logged {len(points)} points, {len(step2_boxes)} subcells, {len(normal_origins)} normals, {len(seed_positions)} seeds")


def main():
    parser = argparse.ArgumentParser(description="TRAVEL Rerun debug visualizer")
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    points = np.load(args.points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Point cloud must be (N, 3) or (N, >=3)")
    global_cfg = load_config(args.config)

    print("Running TRAVEL pipeline...")
    grid, grid_spec, labels, start_nodes, step1_labels, rejection_map, visited = run_pipeline(points, global_cfg)

    g = (labels == 1).sum()
    ng = (labels == 0).sum()
    unk = (labels == -1).sum()
    print(f"Ground: {g}, Non-Ground: {ng}, Unknown: {unk}")

    # Rerun 初期化
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="/"),
            rrb.Vertical(
                rrb.Spatial2DView(name="Top-down (Step1)", origin="/subcells/step1"),
                rrb.Spatial2DView(name="Top-down (Step2)", origin="/subcells/step2"),
            ),
        ),
        collapse_panels=True,
    )

    rr.init("travel_debug", spawn=True)
    rr.send_blueprint(blueprint)

    log_to_rerun(points, grid, grid_spec, labels, start_nodes, step1_labels, rejection_map, visited)
    print("Done. Rerun viewer should open automatically.")


if __name__ == "__main__":
    main()
