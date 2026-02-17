"""
Algorithm 2: Computing k_ij for p-Hub-Arc (Section 3.2 Separation Problem)

Given a solution ȳ of the master problem, computes the index k_ij for OD pair (i,j)
used in polynomial separation of Benders cuts.

For hub-arc: arcs are sorted by cost c_ij,uv for each (i,j).
We accumulate y over arcs in cost order until val >= p.
"""

from typing import Dict, List, Tuple


def compute_k_ij(
    i: int,
    j: int,
    y_bar: Dict[Tuple[int, int], float],
    arcs_sorted: List[Tuple[int, int]],
    cost_map: Dict[Tuple[int, int], float],
    p: int,
) -> int:
    """
    Algorithm 2 (hub-arc): Compute k_ij for OD pair (i,j).

    Parameters
    ----------
    i, j : int
        OD pair
    y_bar : dict
        Current master solution, y_bar[(u,v)] = value for arc (u,v)
    arcs_sorted : list
        Arcs sorted by cost c_ij,uv (cheapest first)
    cost_map : dict
        cost_map[(u,v)] = c_ij,uv for this OD pair
    p : int
        Number of hub arcs to open

    Returns
    -------
    k_ij : int
        Index for Benders cut generation.
    """
    k_ij = 0
    r = 0
    val = 0.0
    M = len(arcs_sorted)

    while val < p - 1e-9 and r < M:
        if r + 1 < M:
            a_curr = arcs_sorted[r]
            a_next = arcs_sorted[r + 1]
            c_curr = cost_map.get(a_curr, 0.0)
            c_next = cost_map.get(a_next, 0.0)
            if c_curr < c_next:
                k_ij += 1

        val += y_bar.get(arcs_sorted[r], 0.0)
        r += 1

    return k_ij
