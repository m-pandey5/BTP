"""
Benders Decomposition for p-Hub-Arc Problem

Complete implementation using:
- Algorithm 1: Separation algorithm (dual-based)
- Algorithm 2: Computing k_ij (for polynomial path)
- Algorithm 3: Phase 1 - LP relaxation
- Phase 2: Branch-and-Benders-cut
"""

import gurobipy as gp
from gurobipy import GRB
import time
from typing import Dict, List, Tuple, Any

from algo1_hub_arc import separation_algorithm_dual
from algo3_hub_arc import (
    phase1_solve_mp,
    rounding_heuristic,
    compute_allocation_cost,
)


# ============================================================
# Preprocessing: Build C, L, K, cost_map, arcs_sorted
# ============================================================

def preprocess(n: int, W: List[List[float]], D: List[List[float]]):
    """
    Build C, L, K, cost_map, arcs_sorted from W and D.
    Mirrors F3 formulation preprocessing.
    """
    N = range(n)
    H = [(u, v) for u in N for v in N if u != v]

    C: Dict[Tuple[int, int], List[float]] = {}
    K: Dict[Tuple[int, int], int] = {}
    L: Dict[Tuple[int, int], Dict[int, List[Tuple[int, int]]]] = {}
    cost_map: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}
    arcs_sorted: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for i in N:
        for j in N:
            if i == j:
                continue

            costs_ij = []
            cm = {}
            for u in N:
                for v in N:
                    if u == v:
                        continue
                    cost_uv = W[i][j] * (D[i][u] + D[u][v] + D[v][j])
                    costs_ij.append(cost_uv)
                    cm[(u, v)] = cost_uv

            unique_costs = sorted(set(round(c, 8) for c in costs_ij))
            Kij = len(unique_costs)
            C[(i, j)] = unique_costs
            K[(i, j)] = Kij
            cost_map[(i, j)] = cm

            arcs_sorted[(i, j)] = sorted(cm.keys(), key=lambda a: cm[a])

            L[(i, j)] = {}
            for k_idx, target_cost in enumerate(unique_costs, start=1):
                arcs_k = [
                    (u, v) for (u, v), c_val in cm.items()
                    if abs(c_val - target_cost) < 1e-8
                ]
                L[(i, j)][k_idx] = arcs_k

    od_pairs = [(i, j) for (i, j) in K if i != j and K[(i, j)] > 1]

    return H, C, L, K, cost_map, arcs_sorted, od_pairs


# ============================================================
# Phase 2: Benders Callback
# ============================================================

class BendersCutCallback:
    """Lazy constraint callback for Phase 2."""

    def __init__(
        self,
        C: Dict, L: Dict, K: Dict,
        od_pairs: List[Tuple[int, int]],
        y_vars: Dict, theta_vars: Dict,
    ):
        self.C = C
        self.L = L
        self.K = K
        self.od_pairs = od_pairs
        self.y_vars = y_vars
        self.theta_vars = theta_vars

    def __call__(self, model, where):
        if where != GRB.Callback.MIPSOL:
            return

        y_bar = {a: model.cbGetSolution(self.y_vars[a]) for a in self.y_vars}
        theta_bar = {
            (i, j): model.cbGetSolution(self.theta_vars[(i, j)])
            for (i, j) in self.theta_vars
        }

        cuts, _ = separation_algorithm_dual(
            y_bar, theta_bar, self.C, self.L, self.K, self.od_pairs
        )

        for ((i, j), cut_data) in cuts:
            Ki = cut_data["Ki"]
            C1 = cut_data["C1"]
            nu = cut_data["nu"]
            expr = C1
            if nu.get(1, 0) != 0 and cut_data.get("L1"):
                expr += nu[1] * (
                    1.0 - gp.quicksum(self.y_vars[a] for a in cut_data["L1"])
                )
            for k in range(2, Ki + 1):
                Lk = cut_data.get(f"L{k}", [])
                if nu.get(k, 0) != 0 and Lk:
                    expr -= nu[k] * gp.quicksum(self.y_vars[a] for a in Lk)
            model.cbLazy(self.theta_vars[(i, j)] >= expr)


# ============================================================
# Full Benders Decomposition
# ============================================================

def solve_benders_hub_arc(
    n: int,
    p: int,
    W: List[List[float]],
    D: List[List[float]],
    time_limit: float = None,
    verbose: bool = False,
    use_phase1: bool = True,
) -> Dict[str, Any]:
    """
    Solve p-hub-arc using Benders decomposition.

    Returns
    -------
    dict with: objective, selected_arcs, time, status
    """
    start_time = time.time()

    H, C, L, K, cost_map, arcs_sorted, od_pairs = preprocess(n, W, D)

    y1 = None
    if use_phase1:
        y_heuristic = {a: 0.0 for a in H}
        for idx, a in enumerate(H):
            if idx < p:
                y_heuristic[a] = 1.0
        UB_h = 0.0
        for (i, j) in od_pairs:
            UB_h += compute_allocation_cost(
                i, j, y_heuristic,
                cost_map.get((i, j), {}),
                arcs_sorted.get((i, j), []),
            )
        for (i, j) in K:
            if i != j and K[(i, j)] == 1:
                UB_h += C[(i, j)][0]

        if verbose:
            print("Phase 1: Solving LP relaxation...")
        LB1, y1, UB1, _, _, _ = phase1_solve_mp(
            n, p, H, C, L, K, od_pairs,
            cost_map, arcs_sorted,
            y_heuristic=y_heuristic,
            UB_heuristic=UB_h,
            verbose=verbose,
        )
        if verbose:
            print(f"Phase 1 done: LB={LB1:.4f}, UB={UB1:.4f}")

    if verbose:
        print("Phase 2: Branch-and-Benders-cut...")

    model = gp.Model("HubArc_Phase2")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.LazyConstraints = 1
    model.Params.MIPGap = 1e-9
    if time_limit:
        model.Params.TimeLimit = max(0, time_limit - (time.time() - start_time))

    y_vars = {a: model.addVar(vtype=GRB.BINARY, name=f"y_{a[0]}_{a[1]}") for a in H}
    theta_vars = {
        (i, j): model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"theta_{i}_{j}")
        for (i, j) in od_pairs
    }

    model.setObjective(
        gp.quicksum(theta_vars[(i, j)] for (i, j) in od_pairs),
        GRB.MINIMIZE,
    )
    model.addConstr(gp.quicksum(y_vars[a] for a in H) == p, name="p_hub_arcs")

    if y1:
        for a in H:
            y_vars[a].Start = 1.0 if y1.get(a, 0) > 0.5 else 0.0

    callback = BendersCutCallback(C, L, K, od_pairs, y_vars, theta_vars)
    model.optimize(callback)

    elapsed = time.time() - start_time

    if model.status == GRB.OPTIMAL:
        y_sol = {a: y_vars[a].X for a in H}
        selected = [(u, v) for (u, v) in H if y_sol[(u, v)] > 0.5]
        obj = model.ObjVal
        for (i, j) in K:
            if i != j and K[(i, j)] == 1:
                obj += C[(i, j)][0]
        return {
            "objective": obj,
            "selected_arcs": selected,
            "time": elapsed,
            "status": "OPTIMAL",
        }
    return {
        "objective": None,
        "selected_arcs": None,
        "time": elapsed,
        "status": "FAILED",
    }
