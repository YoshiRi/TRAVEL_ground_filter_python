"""
Extract visualization data from a PipelineResult for web / debug frontends.

Returns plain JSON-serializable dicts so callers have no Plotly/matplotlib
dependency.
"""
from __future__ import annotations

from .pipeline import PipelineResult


def extract_viz_data(result: PipelineResult) -> dict:
    """
    Build columnar arrays for two overlay layers:

    ``grid``   – one entry per *cell* that has height features:
        cx, cy, cz (cell center x/y, min_z), state ('G'/'N'/'U'/'R')

    ``planes`` – one entry per *subcell* that has a fitted plane:
        x, y, z   : subcell mean position
        nx, ny, nz: unit normal vector
        weight    : planarity confidence
        label     : 'G' / 'N' / 'U'
    """
    grid = result.grid
    spec = result.grid_spec
    res  = spec.resolution
    ox, oy = spec.origin_xy

    # ── Cell-level state grid ─────────────────────────────────────
    cx_list, cy_list, cz_list, state_list = [], [], [], []

    for cell in grid.iter_cells():
        if cell.min_z is None:
            continue
        cx = ox + (cell.index[0] + 0.5) * res
        cy = oy + (cell.index[1] + 0.5) * res
        cx_list.append(round(float(cx), 4))
        cy_list.append(round(float(cy), 4))
        cz_list.append(round(float(cell.min_z), 4))
        state_list.append(cell.state.name[0])   # G / N / U / R

    # ── SubCell plane normals ─────────────────────────────────────
    px, py, pz = [], [], []
    pnx, pny, pnz = [], [], []
    pw, pl = [], []

    for cell in grid.iter_cells():
        for sub in cell.subcells.values():
            if sub.mean is None or sub.normal is None:
                continue
            px.append(round(float(sub.mean[0]), 4))
            py.append(round(float(sub.mean[1]), 4))
            pz.append(round(float(sub.mean[2]), 4))
            pnx.append(round(float(sub.normal[0]), 4))
            pny.append(round(float(sub.normal[1]), 4))
            pnz.append(round(float(sub.normal[2]), 4))
            pw.append(round(float(sub.weight), 4))
            pl.append(sub.label.name[0])           # G / N / U

    return {
        "grid": {
            "cx": cx_list, "cy": cy_list, "cz": cz_list,
            "state": state_list,
        },
        "planes": {
            "x": px, "y": py, "z": pz,
            "nx": pnx, "ny": pny, "nz": pnz,
            "weight": pw, "label": pl,
        },
    }
