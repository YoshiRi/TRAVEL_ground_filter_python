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


# =========================
# SubCell Traversal (TGS)
# =========================
from .types import SubCellIndex, SubCell

def run_subcell_traversal(
    *,
    graph: "TraversabilityGraph",
    start_nodes: Iterable[SubCellIndex],
    accept_fn: Callable[[SubCellIndex, SubCellIndex], bool],
) -> Tuple[Set[SubCellIndex], int]:
    """
    Run BFS on SubCell graph.
    
    Parameters
    ----------
    graph:
        TraversabilityGraph instance providing neighbor access.
    start_nodes:
        Initial SubCellIndices to start BFS from.
    accept_fn:
        Function (src_idx, dst_idx) -> bool.
        MUST decide whether traversal is allowed.
        Contract:
          - Must handle existence checks (return False if cell/subcell missing).
          - Must check plane validity (normals, etc.).
          - Must perform geometric checks (LCC, etc.).
          - Traversal logic relies SOLELY on this function for acceptance.
    
    Returns
    -------
    visited:
        Set of reachable SubCellIndices.
        NOTE: This represents structural reachability in the graph.
        Semantic labeling (e.g. setting CellState.GROUND) should be done
        by the caller based on this set, not inside the traversal.
    rejected_count:
        Number of rejected edges (where accept_fn returned False).
    max_depth:
        Maximum BFS depth reached.
    """
    queue: Deque[Tuple[SubCellIndex, int]] = deque()
    visited: Set[SubCellIndex] = set()
    
    for s in start_nodes:
        queue.append((s, 0))
        visited.add(s)
        
    rejected_count = 0
    max_depth = 0
    
    while queue:
        current_idx, depth = queue.popleft()
        if depth > max_depth:
            max_depth = depth
        
        neighbors = graph.get_neighbor_subcells(current_idx)
        
        for n_idx in neighbors:
            if n_idx in visited:
                continue
                
            # Check acceptance (LCC, label, etc.)
            if accept_fn(current_idx, n_idx):
                visited.add(n_idx)
                queue.append((n_idx, depth + 1))
            else:
                rejected_count += 1
                
    return visited, rejected_count, max_depth

