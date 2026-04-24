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
    eps: float = 1e-6,
) -> Tuple[float, Dict[str, Any]]:
    """
    Solve dual subproblem for OD pair (i,j) with Pareto-optimal cut selection.

    Uses the ε-perturbation trick (Magnanti-Wong 1981): instead of two LP
    solves (standard dual → z*, then secondary with z* constraint), we solve
    a SINGLE LP at the shifted point ỹ = y_bar + ε·y_core.

    Mathematical basis
    ------------------
    The combined objective f(ν,ȳ) + ε·f(ν,y⁰) expands to f(ν, ỹ) + ε·C¹,
    where ỹₐ = ȳₐ + ε·y⁰ₐ.  Maximising at ỹ in one LP therefore:
      (a) selects a ν that is primary-optimal at ȳ  (ε small → primary dominates)
      (b) among all such ν, picks the one with highest cut value at y⁰
    That is exactly the Pareto-optimal (Magnanti-Wong) cut.

    After solving, z* is recomputed at the original ȳ (not ỹ) for the
    violation check and cut RHS.
    """
    Kij = K[(i, j)]
    if Kij <= 1:
        return None, {}

    L1 = L[(i, j)][1]
    C1 = C[(i, j)][0]

    # ỹ = ȳ + ε·y⁰  (shifted point used as the single solve target)
    y_eff = {a: y_bar.get(a, 0.0) + eps * y_core.get(a, 0.0) for a in y_bar}

    m = gp.Model(f"DSP_pareto_{i}_{j}")
    m.Params.OutputFlag = 0

    nu = m.addVars(range(1, Kij + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="nu")

    # Objective: f(ν, ỹ)  — single LP, no coupling constraint needed
    obj = C1
    obj += nu[1] * (1.0 - sum(y_eff.get(a, 0.0) for a in L1))
    for k in range(2, Kij + 1):
        Lk = L[(i, j)].get(k, [])
        obj += nu[k] * (-sum(y_eff.get(a, 0.0) for a in Lk))
    m.setObjective(obj, GRB.MAXIMIZE)

    # Dual feasibility: νₖ − νₖ₊₁ ≤ C^{k+1} − C^k
    for k in range(1, Kij):
        m.addConstr(nu[k] - nu[k + 1] <= C[(i, j)][k] - C[(i, j)][k - 1])

    m.optimize()

    if m.status != GRB.OPTIMAL:
        return None, {}

    nu_sol = {k: nu[k].X for k in range(1, Kij + 1)}

    # Recompute true cut value at ȳ (not ỹ) for violation check and cut RHS
    z_star = C1
    z_star += nu_sol[1] * (1.0 - sum(y_bar.get(a, 0.0) for a in L1))
    for k in range(2, Kij + 1):
        Lk = L[(i, j)].get(k, [])
        z_star -= nu_sol[k] * sum(y_bar.get(a, 0.0) for a in Lk)

    cut = {"i": i, "j": j, "Ki": Kij, "nu": nu_sol, "C1": C1}
    for k in range(1, Kij + 1):
        cut[f"L{k}"] = L[(i, j)].get(k, [])

    return z_star, cut


def _solve_dual_subproblem_pareto_two_step(
    i: int,
    j: int,
    y_bar: Dict[Tuple[int, int], float],
    y_core: Dict[Tuple[int, int], float],
    C: Dict[Tuple[int, int], List[float]],
    L: Dict[Tuple[int, int], Dict[int, List[Tuple[int, int]]]],
    K: Dict[Tuple[int, int], int],
    tol: float = 1e-8,
) -> Tuple[float, Dict[str, Any]]:
    """
    Exact two-step Magnanti-Wong separation.

    Step 1: z* = max f(nu, y_bar) over dual-feasible nu.
    Step 2: max f(nu, y_core) subject to dual-feasibility and
            f(nu, y_bar) >= z* - tol.
    """
    Kij = K[(i, j)]
    if Kij <= 1:
        return None, {}

    L1 = L[(i, j)][1]
    C1 = C[(i, j)][0]

    def f_at(nu_vars, y_point: Dict[Tuple[int, int], float]):
        expr = C1
        expr += nu_vars[1] * (1.0 - sum(y_point.get(a, 0.0) for a in L1))
        for kk in range(2, Kij + 1):
            Lk = L[(i, j)].get(kk, [])
            expr += nu_vars[kk] * (-sum(y_point.get(a, 0.0) for a in Lk))
        return expr

    # Step 1: solve dual at y_bar
    m1 = gp.Model(f"DSP_pareto_step1_{i}_{j}")
    m1.Params.OutputFlag = 0
    nu1 = m1.addVars(range(1, Kij + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="nu")
    for kk in range(1, Kij):
        m1.addConstr(nu1[kk] - nu1[kk + 1] <= C[(i, j)][kk] - C[(i, j)][kk - 1])
    m1.setObjective(f_at(nu1, y_bar), GRB.MAXIMIZE)
    m1.optimize()
    if m1.status != GRB.OPTIMAL:
        return None, {}
    z_star = float(m1.ObjVal)

    # Step 2: maximize at core among solutions optimal at y_bar
    m2 = gp.Model(f"DSP_pareto_step2_{i}_{j}")
    m2.Params.OutputFlag = 0
    nu2 = m2.addVars(range(1, Kij + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="nu")
    for kk in range(1, Kij):
        m2.addConstr(nu2[kk] - nu2[kk + 1] <= C[(i, j)][kk] - C[(i, j)][kk - 1])
    # Exact Magnanti-Wong optimality constraint at current incumbent y_bar.
    m2.addConstr(f_at(nu2, y_bar) == z_star, name="mw_optimality_at_ybar")
    m2.setObjective(f_at(nu2, y_core), GRB.MAXIMIZE)
    m2.optimize()

    if m2.status != GRB.OPTIMAL:
        return None, {}

    nu_sol = {kk: nu2[kk].X for kk in range(1, Kij + 1)}

    # Recompute cut value at y_bar for violation check and RHS.
    z_bar = C1
    z_bar += nu_sol[1] * (1.0 - sum(y_bar.get(a, 0.0) for a in L1))
    for kk in range(2, Kij + 1):
        Lk = L[(i, j)].get(kk, [])
        z_bar -= nu_sol[kk] * sum(y_bar.get(a, 0.0) for a in Lk)

    cut = {"i": i, "j": j, "Ki": Kij, "nu": nu_sol, "C1": C1}
    for kk in range(1, Kij + 1):
        cut[f"L{kk}"] = L[(i, j)].get(kk, [])
    return z_bar, cut


def separation_pareto(
    y_bar: Dict[Tuple[int, int], float],
    theta_bar: Dict[Tuple[int, int], float],
    y_core: Dict[Tuple[int, int], float],
    C: Dict[Tuple[int, int], List[float]],
    L: Dict[Tuple[int, int], Dict[int, List[Tuple[int, int]]]],
    K: Dict[Tuple[int, int], int],
    od_pairs: List[Tuple[int, int]],
    tol: float = 1e-6,
    pareto_method: str = "epsilon",
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

        if pareto_method == "two_step":
            sp_obj, cut = _solve_dual_subproblem_pareto_two_step(
                i, j, y_bar, y_core, C, L, K, tol=tol
            )
        else:
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
        pareto_method: str,
    ):
        self.C = C
        self.L = L
        self.K = K
        self.od_pairs = od_pairs
        self.y_vars = y_vars
        self.theta_vars = theta_vars
        self.y_core = y_core
        self.pareto_method = pareto_method

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
            pareto_method=self.pareto_method,
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

def _uniform_y_core(H: List[Tuple[int, int]], p: int) -> Dict[Tuple[int, int], float]:
    """Magnanti–Wong core: interior of {y >= 0, sum y = p, y <= 1} when |H| > p."""
    return {a: p / len(H) for a in H}


def solve_benders_pareto_hub_arc(
    n: int,
    p: int,
    W: List[List[float]],
    D: List[List[float]],
    time_limit: float = None,
    verbose: bool = False,
    use_phase1: bool = True,
    core_lp_blend: float = 0.0,
    pareto_method: str = "epsilon",
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
    core_lp_blend : in [0, 1], convex blend of MW core with Phase 1 final LP y.
        0 (default) = pure uniform p/|H| (classical interior core).
        >0 = y_core = (1-blend)*uniform + blend*y_LP; biases Pareto dual toward
        the Phase 1 relaxation (heuristic; keep < 1 so y_core stays strictly
        positive on every arc). Ignored if use_phase1 is False.
    pareto_method : "epsilon" (single-LP perturbation) or "two_step"
        (exact MW: solve at y_bar first, then maximize at core with
         optimality-at-y_bar constraint).

    Returns
    -------
    dict with: objective, selected_arcs, time, status
    """
    start_time = time.time()

    H, C, L, K, cost_map, arcs_sorted, od_pairs = preprocess(n, W, D)

    y_core = _uniform_y_core(H, p)

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
        LB1, y1, UB1, model_p1, y_vars_p1, _ = phase1_solve_mp(
            n, p, H, C, L, K, od_pairs,
            cost_map, arcs_sorted,
            y_heuristic=y_heuristic,
            UB_heuristic=UB_h,
            verbose=verbose,
        )
        if verbose:
            print(f"Phase 1 done: LB={LB1:.4f}, UB={UB1:.4f}")

        b = min(max(core_lp_blend, 0.0), 1.0)
        if b > 0.0:
            # Keep the Phase-1 model alive while reading solution values.
            # If the model gets garbage-collected, Var proxies may no longer expose X.
            y_lp = {a: float(y_vars_p1[a].X) for a in H}
            u = _uniform_y_core(H, p)
            y_core = {a: (1.0 - b) * u[a] + b * y_lp[a] for a in H}
            if verbose:
                print(
                    f"Pareto core: blend uniform with Phase-1 LP y "
                    f"(core_lp_blend={b:.3f})"
                )

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
        C, L, K, od_pairs, y_vars, theta_vars, y_core, pareto_method
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
