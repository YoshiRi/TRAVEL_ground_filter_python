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
from travel_py.pipeline import run_pipeline
from travel_py.types import CellState


# Colors (RGBA uint8)
COLOR_GROUND = [50, 205, 50, 255]
COLOR_NONGROUND = [220, 60, 60, 255]
COLOR_UNKNOWN = [160, 160, 160, 120]
COLOR_SEED = [255, 215, 0, 255]




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
    result = run_pipeline(points, global_cfg)
    grid = result.grid
    grid_spec = result.grid_spec
    labels = result.labels
    start_nodes = result.start_nodes
    step1_labels = result.step1_labels
    rejection_map = result.rejection_map
    visited = result.visited

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
