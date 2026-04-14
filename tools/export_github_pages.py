"""Export a static payload for GitHub Pages demo viewer.

Usage:
    uv run python tools/export_github_pages.py --points data/sample.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from travel_py.config import load_config
from travel_py.pipeline import run_pipeline


def _downsample(points: np.ndarray, labels: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, labels
    idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[idx], labels[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static TRAVEL demo payload for GitHub Pages")
    parser.add_argument("--points", type=Path, required=True, help="Input point cloud (.npy)")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("docs/data/demo_payload.json"))
    parser.add_argument("--max-points", type=int, default=40000)
    args = parser.parse_args()

    points = np.load(args.points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Point cloud must be shape (N,3) or (N,>=3)")

    cfg = load_config(args.config)
    result = run_pipeline(points, cfg)
    ds_points, ds_labels = _downsample(points[:, :3], result.labels, args.max_points)

    payload = {
        "meta": {
            "source": str(args.points),
            "total_points": int(len(points)),
            "shown_points": int(len(ds_points)),
            "ground_count": int((result.labels == 1).sum()),
            "nonground_count": int((result.labels == 0).sum()),
            "unknown_count": int((result.labels == -1).sum()),
        },
        "points": {
            "x": ds_points[:, 0].tolist(),
            "y": ds_points[:, 1].tolist(),
            "z": ds_points[:, 2].tolist(),
            "label": ds_labels.tolist(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
