from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .types import Cell, CellState, RejectReason

Index2D = Tuple[int, int]


# =========================
# Color definitions
# =========================

CELL_STATE_COLORS = {
    CellState.UNKNOWN: "lightgray",
    CellState.GROUND: "green",
    CellState.NON_GROUND: "red",
    CellState.REJECTED: "orange",
}

REJECT_REASON_COLORS = {
    RejectReason.NONE: "lightgray",
    RejectReason.HEIGHT_DIFF_TOO_LARGE: "purple",
    RejectReason.SLOPE_TOO_STEEP: "brown",
    RejectReason.NO_VALID_NEIGHBOR: "blue",
    RejectReason.INVALID_FEATURE: "black",
    RejectReason.VISITED: "cyan",
    RejectReason.OUT_OF_BOUNDS: "yellow",
}


# =========================
# Utility
# =========================

def _extract_indices(cells: Iterable[Cell]) -> Tuple[np.ndarray, np.ndarray]:
    ix = []
    iy = []
    for c in cells:
        ix.append(c.index[0])
        iy.append(c.index[1])
    return np.asarray(ix), np.asarray(iy)


# =========================
# Visualization functions
# =========================

def plot_cell_state(
    cells: Iterable[Cell],
    *,
    title: str = "Cell State",
    show: bool = True,
) -> None:
    """
    Visualize grid cells colored by CellState.
    """
    cells = list(cells)
    if not cells:
        return

    xs, ys = _extract_indices(cells)
    colors = [CELL_STATE_COLORS[c.state] for c in cells]

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c=colors, s=40, marker="s")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("ix")
    plt.ylabel("iy")
    plt.grid(True)

    if show:
        plt.show()


def plot_reject_reason(
    cells: Iterable[Cell],
    *,
    title: str = "Reject Reason",
    show: bool = True,
) -> None:
    """
    Visualize grid cells colored by RejectReason.
    """
    cells = list(cells)
    if not cells:
        return

    xs, ys = _extract_indices(cells)
    colors = [
        REJECT_REASON_COLORS.get(c.reject_reason, "gray")
        for c in cells
    ]

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c=colors, s=40, marker="s")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("ix")
    plt.ylabel("iy")
    plt.grid(True)

    if show:
        plt.show()


def plot_cell_scalar(
    cells: Iterable[Cell],
    *,
    value: str,
    title: Optional[str] = None,
    cmap: str = "viridis",
    show: bool = True,
) -> None:
    """
    Visualize a scalar value stored in Cell (e.g., min_z, height_range).

    Parameters
    ----------
    value:
        Attribute name of Cell (string).
    """
    cells = list(cells)
    if not cells:
        return

    xs, ys = _extract_indices(cells)

    vals = []
    for c in cells:
        v = getattr(c, value, None)
        vals.append(np.nan if v is None else v)

    vals = np.asarray(vals, dtype=float)

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(xs, ys, c=vals, s=40, marker="s", cmap=cmap)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(sc, label=value)

    if title is None:
        title = value

    plt.title(title)
    plt.xlabel("ix")
    plt.ylabel("iy")
    plt.grid(True)

    if show:
        plt.show()


def plot_iteration(
    cells: Iterable[Cell],
    *,
    title: str = "Traversal Iteration",
    show: bool = True,
) -> None:
    """
    Visualize traversal order (iteration index).
    """
    cells = list(cells)
    if not cells:
        return

    xs, ys = _extract_indices(cells)

    iters = []
    for c in cells:
        iters.append(np.nan if c.iteration is None else c.iteration)

    iters = np.asarray(iters, dtype=float)

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(xs, ys, c=iters, s=40, marker="s", cmap="plasma")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(sc, label="iteration")
    plt.title(title)
    plt.xlabel("ix")
    plt.ylabel("iy")
    plt.grid(True)

    if show:
        plt.show()
