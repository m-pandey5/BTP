"""
Benders Decomposition with Pareto-Optimal Cuts for p-Hub-Arc Problem

Extends the two-phase Benders algorithm with Magnanti-Wong strengthened cuts.

For each OD pair (i,j) at every integer node:
  1. Solve the standard dual subproblem at y_bar -> get z*
  2. Solve the secondary (Magnanti-Wong) problem:
       among all dual solutions optimal at y_bar (cut value = z*),
       find the one that maximises the cut value at the core point y_core.
  3. Use that nu to generate a Pareto-optimal (strongest) Benders cut.

Core point: y_core[a] = p / |H| for all arcs a  (uniform, strictly interior).

Usage
-----
    from pareto_benders_hub_arc import solve_benders_pareto_hub_arc
    result = solve_benders_pareto_hub_arc(n, p, W, D)
"""

import gurobipy as gp
from gurobipy import GRB
import time
from typing import Dict, List, Tuple, Any

from new_model_hub_arc import preprocess
from algo3_hub_arc import (
    phase1_solve_mp,
    compute_allocation_cost,
)


# ============================================================
# Pareto-Optimal Separation
# ============================================================

def _solve_dual_subproblem_pareto(
    i: int,
    j: int,
    y_bar: Dict[Tuple[int, int], float],
    y_core: Dict[Tuple[int, int], float],
    C: Dict[Tuple[int, int], List[float]],
    L: Dict[Tuple[int, int], Dict[int, List[Tuple[int, int]]]],
    K: Dict[Tuple[int, int], int],
) -> Tuple[float, Dict[str, Any]]:
    """
    Solve dual subproblem for OD pair (i,j) with Pareto-optimal cut selection.

    Step 1: Solve standard dual subproblem at y_bar -> z*
    Step 2: Solve secondary problem -> nu that maximises cut at y_core
            subject to cut value at y_bar remaining = z*
    """
    Kij = K[(i, j)]
    if Kij <= 1:
        return None, {}

    L1 = L[(i, j)][1]
    C1 = C[(i, j)][0]

    # ------------------------------------------------------------------
    # Step 1: Standard dual subproblem at y_bar
    # ------------------------------------------------------------------
    m1 = gp.Model(f"DSP_{i}_{j}")
    m1.Params.OutputFlag = 0

    nu = m1.addVars(range(1, Kij + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="nu")

    obj1 = C1
    obj1 += nu[1] * (1.0 - sum(y_bar.get(a, 0.0) for a in L1))
    for k in range(2, Kij + 1):
        Lk = L[(i, j)].get(k, [])
        obj1 += nu[k] * (-sum(y_bar.get(a, 0.0) for a in Lk))
    m1.setObjective(obj1, GRB.MAXIMIZE)

    for k in range(1, Kij):
        m1.addConstr(nu[k] - nu[k + 1] <= C[(i, j)][k] - C[(i, j)][k - 1])

    m1.optimize()

    if m1.status != GRB.OPTIMAL:
        return None, {}

    z_star = m1.ObjVal
    fallback_nu = {k: nu[k].X for k in range(1, Kij + 1)}

    # ------------------------------------------------------------------
    # Step 2: Secondary (Magnanti-Wong) problem
    #
    #   max   C1 + nu1*(1 - sum_{L1} y_core) - sum_{k>=2} nuk*sum_{Lk} y_core
    #   s.t.  nuk - nu_{k+1} <= C^{k+1} - C^k          (dual feasibility)
    #         nuk >= 0
    #         C1 + nu1*(1 - sum_{L1} y_bar) - sum_{k>=2} nuk*sum_{Lk} y_bar
    #             = z_star                               (optimality at y_bar)
    # ------------------------------------------------------------------
    m2 = gp.Model(f"DSP_pareto_{i}_{j}")
    m2.Params.OutputFlag = 0

    nu2 = m2.addVars(range(1, Kij + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="nu")

    # Objective: cut value at core point
    obj2 = C1
    obj2 += nu2[1] * (1.0 - sum(y_core.get(a, 0.0) for a in L1))
    for k in range(2, Kij + 1):
        Lk = L[(i, j)].get(k, [])
        obj2 += nu2[k] * (-sum(y_core.get(a, 0.0) for a in Lk))
    m2.setObjective(obj2, GRB.MAXIMIZE)

    # Dual feasibility
    for k in range(1, Kij):
        m2.addConstr(nu2[k] - nu2[k + 1] <= C[(i, j)][k] - C[(i, j)][k - 1])

    # Optimality fixation: cut at y_bar must equal z_star
    opt_expr = C1
    opt_expr += nu2[1] * (1.0 - sum(y_bar.get(a, 0.0) for a in L1))
    for k in range(2, Kij + 1):
        Lk = L[(i, j)].get(k, [])
        opt_expr += nu2[k] * (-sum(y_bar.get(a, 0.0) for a in Lk))
    m2.addConstr(opt_expr == z_star, name="optimality_fix")

    m2.optimize()

    if m2.status == GRB.OPTIMAL:
        nu_sol = {k: nu2[k].X for k in range(1, Kij + 1)}
    else:
        nu_sol = fallback_nu  # fall back to step-1 solution

    cut = {"i": i, "j": j, "Ki": Kij, "nu": nu_sol, "C1": C1}
    for k in range(1, Kij + 1):
        cut[f"L{k}"] = L[(i, j)].get(k, [])

    return z_star, cut


def separation_pareto(
    y_bar: Dict[Tuple[int, int], float],
    theta_bar: Dict[Tuple[int, int], float],
    y_core: Dict[Tuple[int, int], float],
    C: Dict[Tuple[int, int], List[float]],
    L: Dict[Tuple[int, int], Dict[int, List[Tuple[int, int]]]],
    K: Dict[Tuple[int, int], int],
    od_pairs: List[Tuple[int, int]],
    tol: float = 1e-6,
) -> Tuple[List[Tuple[Tuple[int, int], Dict]], float]:
    """
    Pareto-optimal separation: generate violated Benders cuts using
    Magnanti-Wong strengthened dual solutions.
    """
    cuts_to_add = []
    UB = 0.0

    for (i, j) in od_pairs:
        Kij = K[(i, j)]
        if Kij <= 1:
            UB += C[(i, j)][0] if Kij == 1 else 0.0
            continue

        sp_obj, cut = _solve_dual_subproblem_pareto(i, j, y_bar, y_core, C, L, K)
        if sp_obj is None:
            continue

        UB += sp_obj
        if theta_bar.get((i, j), 0.0) < sp_obj - tol:
            cuts_to_add.append(((i, j), cut))

    return cuts_to_add, UB


# ============================================================
# Phase 2: Benders Callback (Pareto)
# ============================================================

class ParetoBendersCutCallback:
    """Lazy callback that adds Pareto-optimal Benders cuts."""

    def __init__(
        self,
        C: Dict, L: Dict, K: Dict,
        od_pairs: List[Tuple[int, int]],
        y_vars: Dict, theta_vars: Dict,
        y_core: Dict[Tuple[int, int], float],
    ):
        self.C = C
        self.L = L
        self.K = K
        self.od_pairs = od_pairs
        self.y_vars = y_vars
        self.theta_vars = theta_vars
        self.y_core = y_core

    def __call__(self, model, where):
        if where != GRB.Callback.MIPSOL:
            return

        y_bar = {a: model.cbGetSolution(self.y_vars[a]) for a in self.y_vars}
        theta_bar = {
            (i, j): model.cbGetSolution(self.theta_vars[(i, j)])
            for (i, j) in self.theta_vars
        }

        cuts, _ = separation_pareto(
            y_bar, theta_bar, self.y_core,
            self.C, self.L, self.K, self.od_pairs,
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
# Full solver
# ============================================================

def solve_benders_pareto_hub_arc(
    n: int,
    p: int,
    W: List[List[float]],
    D: List[List[float]],
    time_limit: float = None,
    verbose: bool = False,
    use_phase1: bool = True,
) -> Dict[str, Any]:
    """
    Solve p-hub-arc using two-phase Benders with Pareto-optimal cuts.

    Parameters
    ----------
    n, p    : problem size and number of hub arcs
    W, D    : flow and distance matrices
    time_limit : seconds (optional)
    verbose : print progress
    use_phase1 : run LP relaxation warmstart before Phase 2

    Returns
    -------
    dict with: objective, selected_arcs, time, status
    """
    start_time = time.time()

    H, C, L, K, cost_map, arcs_sorted, od_pairs = preprocess(n, W, D)

    # Core point: uniform fractional y strictly inside the master feasible region
    # (satisfies sum y_a = p, with each y_a = p/|H| in (0,1))
    y_core = {a: p / len(H) for a in H}

    # ------------------------------------------------------------------
    # Phase 1: LP relaxation warmstart
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Phase 2: Branch-and-Benders-cut with Pareto-optimal cuts
    # ------------------------------------------------------------------
    if verbose:
        print("Phase 2: Branch-and-Benders-cut (Pareto-optimal cuts)...")

    model = gp.Model("HubArc_Phase2_Pareto")
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

    callback = ParetoBendersCutCallback(
        C, L, K, od_pairs, y_vars, theta_vars, y_core
    )
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
