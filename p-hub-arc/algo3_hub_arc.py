"""
Algorithm 3: Phase 1 - Solving the LP relaxation of the master problem
for p-Hub-Arc Benders decomposition.
"""

import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Any

from algo1_hub_arc import separation_algorithm_dual


def rounding_heuristic(
    y_bar: Dict[Tuple[int, int], float],
    p: int,
    H: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], float]:
    """Round fractional y: select p arcs with largest y values."""
    sorted_arcs = sorted(H, key=lambda a: y_bar.get(a, 0.0), reverse=True)
    y_int = {a: 0.0 for a in H}
    for a in sorted_arcs[:p]:
        y_int[a] = 1.0
    return y_int


def compute_allocation_cost(
    i: int,
    j: int,
    y_int: Dict[Tuple[int, int], float],
    cost_map: Dict[Tuple[int, int], float],
    arcs_sorted: List[Tuple[int, int]],
) -> float:
    """Cost for OD (i,j) = cost of first arc in sorted order that is open."""
    for a in arcs_sorted:
        if y_int.get(a, 0.0) > 0.5:
            return cost_map.get(a, float("inf"))
    return float("inf")


def phase1_solve_mp(
    n: int,
    p: int,
    H: List[Tuple[int, int]],
    C: Dict[Tuple[int, int], List[float]],
    L: Dict[Tuple[int, int], Dict[int, List[Tuple[int, int]]]],
    K: Dict[Tuple[int, int], int],
    od_pairs: List[Tuple[int, int]],
    cost_map: Dict[Tuple[int, int], Dict[Tuple[int, int], float]],
    arcs_sorted: Dict[Tuple[int, int], List[Tuple[int, int]]],
    y_heuristic: Dict[Tuple[int, int], float] = None,
    UB_heuristic: float = None,
    max_iter: int = 200,
    verbose: bool = False,
):
    """
    Algorithm 3: Phase 1 - Solve LP relaxation of master problem.

    Returns
    -------
    LB1, y1, UB1, model, y_vars, theta_vars
    """
    model = gp.Model("HubArc_Master_Phase1")
    model.Params.OutputFlag = 1 if verbose else 0

    y_vars = {a: model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"y_{a[0]}_{a[1]}") for a in H}
    theta_vars = {
        (i, j): model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"theta_{i}_{j}")
        for (i, j) in od_pairs
    }

    model.setObjective(
        gp.quicksum(theta_vars[(i, j)] for (i, j) in od_pairs),
        GRB.MINIMIZE,
    )
    model.addConstr(gp.quicksum(y_vars[a] for a in H) == p, name="p_hub_arcs")

    if y_heuristic:
        for a in H:
            y_vars[a].Start = y_heuristic.get(a, 0.0)

    LB1 = -float("inf")
    UB_hat = UB_heuristic if UB_heuristic is not None else float("inf")
    y_hat = y_heuristic

    iter_count = 0
    while iter_count < max_iter:
        model.optimize()
        if model.status != GRB.OPTIMAL:
            if verbose:
                print(f"[Phase1] Master failed: status={model.status}")
            break

        y_bar = {a: y_vars[a].X for a in H}
        theta_bar = {(i, j): theta_vars[(i, j)].X for (i, j) in od_pairs}
        obj_val = model.ObjVal
        LB1 = max(LB1, obj_val)

        cuts, UB = separation_algorithm_dual(
            y_bar, theta_bar, C, L, K, od_pairs
        )

        if not cuts:
            if verbose:
                print(f"[Phase1] No violated cuts. Converged.")
            break

        for ((i, j), cut_data) in cuts:
            Ki = cut_data["Ki"]
            C1 = cut_data["C1"]
            nu = cut_data["nu"]
            expr = C1
            if nu.get(1, 0) != 0 and cut_data.get("L1"):
                expr += nu[1] * (1.0 - gp.quicksum(y_vars[a] for a in cut_data["L1"]))
            for k in range(2, Ki + 1):
                Lk = cut_data.get(f"L{k}", [])
                if nu.get(k, 0) != 0 and Lk:
                    expr -= nu[k] * gp.quicksum(y_vars[a] for a in Lk)
            model.addConstr(theta_vars[(i, j)] >= expr, name=f"cut_{i}_{j}_{iter_count}")

        is_fractional = any(1e-6 < y_bar.get(a, 0) < 1 - 1e-6 for a in H)
        if is_fractional:
            y_round = rounding_heuristic(y_bar, p, H)
            UB_h = 0.0
            for (i, j) in od_pairs:
                UB_h += compute_allocation_cost(
                    i, j, y_round,
                    cost_map.get((i, j), {}),
                    arcs_sorted.get((i, j), []),
                )
            if UB_h < UB_hat:
                UB_hat = UB_h
                y_hat = y_round
                if verbose:
                    print(f"[Phase1] Rounding improved UB to {UB_hat:.4f}")

        iter_count += 1
        if verbose:
            print(f"[Phase1] Iter {iter_count}: LB={LB1:.4f}, cuts={len(cuts)}")

    y1 = y_hat if y_hat else rounding_heuristic({a: y_vars[a].X for a in H}, p, H)
    UB1 = 0.0
    for (i, j) in od_pairs:
        UB1 += compute_allocation_cost(
            i, j, y1,
            cost_map.get((i, j), {}),
            arcs_sorted.get((i, j), []),
        )
    for (i, j) in K:
        if i != j and K[(i, j)] == 1:
            UB1 += C[(i, j)][0]
    if UB_heuristic is not None and UB_heuristic < UB1:
        UB1 = UB_heuristic
        y1 = y_heuristic

    return LB1, y1, UB1, model, y_vars, theta_vars
