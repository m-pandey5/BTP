"""
Algorithm 1: Separation algorithm for p-Hub-Arc

Generates violated Benders cuts using dual subproblem for each OD pair (i,j).
"""

import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Any


def separation_algorithm_dual(
    y_bar: Dict[Tuple[int, int], float],
    theta_bar: Dict[Tuple[int, int], float],
    C: Dict[Tuple[int, int], List[float]],
    L: Dict[Tuple[int, int], Dict[int, List[Tuple[int, int]]]],
    K: Dict[Tuple[int, int], int],
    od_pairs: List[Tuple[int, int]],
    tol: float = 1e-6,
) -> Tuple[List[Tuple[Tuple[int, int], Dict]], float]:
    """
    Dual-based separation for hub-arc Benders cuts.

    Parameters
    ----------
    y_bar : dict
        Arc solution from master
    theta_bar : dict
        Theta values from master
    C, L, K : dict
        Cost levels, arcs at each level, count (from F3)
    od_pairs : list
        OD pairs (i,j) with K[(i,j)] > 1
    tol : float
        Violation tolerance

    Returns
    -------
    cuts_to_add : list of ((i,j), cut_data)
    UB : float
    """
    cuts_to_add = []
    UB = 0.0

    for (i, j) in od_pairs:
        Kij = K[(i, j)]
        if Kij <= 1:
            UB += C[(i, j)][0] if Kij == 1 else 0.0
            continue

        sp_obj, cut = _solve_dual_subproblem(i, j, y_bar, C, L, K)
        if sp_obj is None:
            continue

        UB += sp_obj
        if theta_bar.get((i, j), 0.0) < sp_obj - tol:
            cuts_to_add.append(((i, j), cut))

    return cuts_to_add, UB


def _solve_dual_subproblem(
    i: int,
    j: int,
    y_bar: Dict[Tuple[int, int], float],
    C: Dict[Tuple[int, int], List[float]],
    L: Dict[Tuple[int, int], Dict[int, List[Tuple[int, int]]]],
    K: Dict[Tuple[int, int], int],
) -> Tuple[float, Dict[str, Any]]:
    """Solve dual subproblem for OD pair (i,j). Return (obj, cut_data)."""
    Kij = K[(i, j)]
    if Kij <= 1:
        return None, {}

    model = gp.Model(f"DSP_{i}_{j}")
    model.Params.OutputFlag = 0

    nu = model.addVars(range(1, Kij + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="nu")
    L1 = L[(i, j)][1]

    # Objective: C^1 + nu1*(1 - sum_L1 y) - sum_{k>=2} nu_k * sum_Lk y
    obj_expr = C[(i, j)][0]
    obj_expr += nu[1] * (1.0 - sum(y_bar.get(a, 0.0) for a in L1))
    for k in range(2, Kij + 1):
        Lk = L[(i, j)].get(k, [])
        obj_expr += nu[k] * (-sum(y_bar.get(a, 0.0) for a in Lk))

    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Dual constraints: nu[k] - nu[k+1] <= C[k] - C[k-1] for k=1..Kij-1
    for k in range(1, Kij):
        model.addConstr(
            nu[k] - nu[k + 1] <= C[(i, j)][k] - C[(i, j)][k - 1],
            name=f"c{k}",
        )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        nu_sol = {k: nu[k].X for k in range(1, Kij + 1)}
        cut = {
            "i": i,
            "j": j,
            "Ki": Kij,
            "nu": nu_sol,
            "C1": C[(i, j)][0],
        }
        for k in range(1, Kij + 1):
            cut[f"L{k}"] = L[(i, j)].get(k, [])
        return model.ObjVal, cut
    return None, {}
