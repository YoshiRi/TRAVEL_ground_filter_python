from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Dict, Iterable, Optional, Tuple

from .adjacency import iter_existing_neighbors
from .types import Cell, CellState, RejectReason, TraversalState

Index2D = Tuple[int, int]


# =========================
# Traversal core
# =========================

def run_traversal(
    *,
    seed_cells: Iterable[Cell],
    cells: Dict[Index2D, Cell],
    connectivity: int = 4,
    accept_fn: Optional[Callable[[Cell, Cell], bool]] = None,
) -> TraversalState:
    """
    Run Travel-style traversal (BFS) starting from seed cells.

    Parameters
    ----------
    seed_cells:
        Iterable of initial ground seed cells.
    cells:
        Mapping from index to Cell.
    connectivity:
        4 or 8 neighbor connectivity.
    accept_fn:
        Function (current_cell, neighbor_cell) -> bool.
        If None, all neighbors are accepted.

    Returns
    -------
    TraversalState
    """

    state = TraversalState()
    queue: Deque[Cell] = deque()

    # -------------------------
    # Initialize seeds
    # -------------------------
    for seed in seed_cells:
        seed.state = CellState.GROUND
        seed.visited = True
        seed.iteration = 0
        seed.reject_reason = RejectReason.NONE

        queue.append(seed)
        state.visited[seed.index] = seed

    state.iteration = 0
    state.num_ground = len(state.visited)

    # -------------------------
    # BFS traversal
    # -------------------------
    while queue:
        current = queue.popleft()
        current_iter = current.iteration or 0

        # record history
        state.iteration_history.setdefault(current_iter, []).append(
            current.index
        )

        for neighbor in iter_existing_neighbors(
            current, cells, connectivity=connectivity
        ):
            if neighbor.visited:
                continue

            neighbor.visited = True
            neighbor.parent = current.index
            neighbor.iteration = current_iter + 1

            # -------------------------
            # Acceptance check
            # -------------------------
            accepted = True
            if accept_fn is not None:
                try:
                    accepted = accept_fn(current, neighbor)
                except Exception:
                    accepted = False
                    neighbor.reject_reason = RejectReason.INVALID_FEATURE

            if accepted:
                neighbor.state = CellState.GROUND
                neighbor.reject_reason = RejectReason.NONE

                queue.append(neighbor)
                state.num_ground += 1
            else:
                neighbor.state = CellState.REJECTED
                if neighbor.reject_reason == RejectReason.NONE:
                    neighbor.reject_reason = RejectReason.HEIGHT_DIFF_TOO_LARGE
                state.num_rejected += 1

            state.visited[neighbor.index] = neighbor

        state.iteration += 1

    return state
