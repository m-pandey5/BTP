"""
McDaniel & Devine Two-Phase Benders Decomposition for p-Hub-Arc Problem

True McDaniel & Devine (1977) implementation:
  Phase 1 — LP relaxation cutting-plane loop.
             Every Benders cut generated is ACCUMULATED (not discarded).
             Produces a strong lower bound, a primal warm-start solution,
             and a set of cuts that already tighten the master.

  Phase 2 — Branch-and-Benders-cut (MIP).
             STARTS with all Phase 1 cuts pre-loaded as regular constraints
             so the master is already tightened before the first node.
             Lazy Benders cuts are added at each integer node as needed.

This is self-contained: no imports from new_model_hub_arc / algo3_hub_arc.
All building blocks (preprocessing, dual subproblem, phase1, phase2) live here.

Usage
-----
    from md_benders_hub_arc import solve_md_benders_hub_arc
    result = solve_md_benders_hub_arc(n, p, W, D, time_limit=300, verbose=True)
    print(result["objective"], result["selected_arcs"])
"""

import time
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Any, Optional


# ============================================================
# 1. Preprocessing
# ============================================================

def _preprocess(n: int, W: List[List[float]], D: List[List[float]]):
    """
    Build all derived structures for the F3-based formulation.

    Returns
    -------
    H           : list of all directed hub arcs (u,v), u != v
    C           : C[(i,j)] = sorted list of unique routing costs for OD pair (i,j)
    L           : L[(i,j)][k] = list of arcs whose cost for (i,j) equals C[(i,j)][k-1]
    K           : K[(i,j)] = number of distinct cost levels for (i,j)
    cost_map    : cost_map[(i,j)][(u,v)] = W[i][j]*(D[i][u]+D[u][v]+D[v][j])
    arcs_sorted : arcs_sorted[(i,j)] = arcs sorted by cost for (i,j)
    od_pairs    : OD pairs with K > 1 (need Benders theta variables)
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
            cm = {}
            for u in N:
                for v in N:
                    if u == v:
                        continue
                    cm[(u, v)] = W[i][j] * (D[i][u] + D[u][v] + D[v][j])

            unique_costs = sorted(set(round(c, 8) for c in cm.values()))
            Kij = len(unique_costs)
            C[(i, j)] = unique_costs
            K[(i, j)] = Kij
            cost_map[(i, j)] = cm
            arcs_sorted[(i, j)] = sorted(cm.keys(), key=lambda a: cm[a])

            L[(i, j)] = {}
            for k_idx, target in enumerate(unique_costs, start=1):
                L[(i, j)][k_idx] = [a for a, c in cm.items() if abs(c - target) < 1e-8]

    od_pairs = [(i, j) for (i, j) in K if i != j and K[(i, j)] > 1]
    return H, C, L, K, cost_map, arcs_sorted, od_pairs


# ============================================================
# 2. Dual Subproblem (standard — no Pareto strengthening)
# ============================================================

def _solve_dual_subproblem(
    i: int,
    j: int,
    y_bar: Dict[Tuple[int, int], float],
    C: Dict,
    L: Dict,
    K: Dict,
) -> Tuple[Optional[float], Dict]:
    """
    Solve the dual subproblem for OD pair (i,j) at master solution y_bar.

    The primal subproblem for (i,j) picks the cheapest open arc.
    Its LP dual yields the Benders optimality cut:

        theta_ij >= C^1 + nu_1*(1 - sum_{L^1} y) - sum_{k>=2} nu_k * sum_{L^k} y

    Returns (objective_value, cut_data) or (None, {}) if infeasible/trivial.
    """
    Kij = K[(i, j)]
    if Kij <= 1:
        return None, {}

    m = gp.Model(f"DSP_{i}_{j}")
    m.Params.OutputFlag = 0

    nu = m.addVars(range(1, Kij + 1), lb=0.0, name="nu")
    L1 = L[(i, j)][1]

    obj = C[(i, j)][0]
    obj += nu[1] * (1.0 - sum(y_bar.get(a, 0.0) for a in L1))
    for k in range(2, Kij + 1):
        Lk = L[(i, j)].get(k, [])
        obj += nu[k] * (-sum(y_bar.get(a, 0.0) for a in Lk))
    m.setObjective(obj, GRB.MAXIMIZE)

    for k in range(1, Kij):
        m.addConstr(nu[k] - nu[k + 1] <= C[(i, j)][k] - C[(i, j)][k - 1])

    m.optimize()

    if m.status != GRB.OPTIMAL:
        return None, {}

    nu_sol = {k: nu[k].X for k in range(1, Kij + 1)}
    cut = {"i": i, "j": j, "Ki": Kij, "nu": nu_sol, "C1": C[(i, j)][0]}
    for k in range(1, Kij + 1):
        cut[f"L{k}"] = L[(i, j)].get(k, [])
    return m.ObjVal, cut


# ============================================================
# 3. Separation (generate violated cuts from a master solution)
# ============================================================

def _separation(
    y_bar: Dict,
    theta_bar: Dict,
    C: Dict,
    L: Dict,
    K: Dict,
    od_pairs: List,
    tol: float = 1e-6,
) -> Tuple[List, float]:
    """
    For each OD pair, solve dual subproblem and return violated cuts.

    Returns
    -------
    cuts_to_add : list of ((i,j), cut_data) where theta_bar < subproblem_obj
    UB          : sum of subproblem objectives (upper bound on true obj)
    """
    cuts_to_add = []
    UB = 0.0
    for (i, j) in od_pairs:
        sp_obj, cut = _solve_dual_subproblem(i, j, y_bar, C, L, K)
        if sp_obj is None:
            UB += C[(i, j)][0] if K[(i, j)] == 1 else 0.0
            continue
        UB += sp_obj
        if theta_bar.get((i, j), 0.0) < sp_obj - tol:
            cuts_to_add.append(((i, j), cut))
    return cuts_to_add, UB


# ============================================================
# 4. Helper: build cut expression for a Gurobi model
# ============================================================

def _cut_expr(cut_data: Dict, y_vars: Dict, theta_vars: Dict):
    """
    Build  theta_ij >= RHS  as a Gurobi linear expression.
    Returns (theta_var, rhs_expr).
    """
    i, j = cut_data["i"], cut_data["j"]
    Ki, C1, nu = cut_data["Ki"], cut_data["C1"], cut_data["nu"]

    expr = C1
    if nu.get(1, 0) != 0 and cut_data.get("L1"):
        expr += nu[1] * (1.0 - gp.quicksum(y_vars[a] for a in cut_data["L1"]))
    for k in range(2, Ki + 1):
        Lk = cut_data.get(f"L{k}", [])
        if nu.get(k, 0) != 0 and Lk:
            expr -= nu[k] * gp.quicksum(y_vars[a] for a in Lk)
    return theta_vars[(i, j)], expr


# ============================================================
# 5. Rounding heuristic & allocation cost
# ============================================================

def _rounding_heuristic(y_bar: Dict, p: int, H: List) -> Dict:
    """Select p arcs with highest fractional y values."""
    top = sorted(H, key=lambda a: y_bar.get(a, 0.0), reverse=True)[:p]
    return {a: (1.0 if a in top else 0.0) for a in H}


def _allocation_cost(i, j, y_int, cost_map, arcs_sorted_ij):
    """Cost for OD (i,j) = cost of cheapest open arc."""
    for a in arcs_sorted_ij:
        if y_int.get(a, 0.0) > 0.5:
            return cost_map.get(a, float("inf"))
    return float("inf")


# ============================================================
# 6. Phase 1 — LP relaxation with cut accumulation
# ============================================================

def _phase1(
    n: int,
    p: int,
    H: List,
    C: Dict,
    L: Dict,
    K: Dict,
    od_pairs: List,
    cost_map: Dict,
    arcs_sorted: Dict,
    time_budget: Optional[float],
    max_iter: int,
    verbose: bool,
) -> Tuple[float, Dict, float, List]:
    """
    McDaniel & Devine Phase 1: iterative LP cutting-plane.

    Key difference from plain warm-start: every cut added to the LP is
    also stored in `accumulated_cuts` so Phase 2 can pre-load them.

    Returns
    -------
    LB          : best LP lower bound
    y_hat       : best integer-feasible warm-start solution found by rounding
    UB          : upper bound corresponding to y_hat
    accumulated_cuts : list of ((i,j), cut_data) — ALL cuts generated here
    """
    t0 = time.time()

    mp = gp.Model("MD_Phase1")
    mp.Params.OutputFlag = 1 if verbose else 0

    y_vars = {a: mp.addVar(lb=0, ub=1, name=f"y_{a[0]}_{a[1]}") for a in H}
    theta_vars = {
        (i, j): mp.addVar(lb=0, name=f"th_{i}_{j}")
        for (i, j) in od_pairs
    }

    mp.setObjective(gp.quicksum(theta_vars[ij] for ij in od_pairs), GRB.MINIMIZE)
    mp.addConstr(gp.quicksum(y_vars[a] for a in H) == p, name="card")

    # Warm-start with a simple heuristic (first p arcs open)
    y_heur = {a: (1.0 if idx < p else 0.0) for idx, a in enumerate(H)}
    UB_hat = sum(
        _allocation_cost(i, j, y_heur, cost_map[(i, j)], arcs_sorted[(i, j)])
        for (i, j) in od_pairs
    ) + sum(C[(i, j)][0] for (i, j) in K if i != j and K[(i, j)] == 1)
    y_hat = y_heur

    LB = -float("inf")
    accumulated_cuts: List = []

    for it in range(max_iter):
        # Respect time budget
        if time_budget is not None:
            remaining = time_budget - (time.time() - t0)
            if remaining <= 0:
                if verbose:
                    print(f"[Phase1] Time budget exhausted at iter {it}.")
                break
            mp.Params.TimeLimit = remaining

        mp.optimize()
        if mp.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or mp.SolCount == 0:
            if verbose:
                print(f"[Phase1] LP solve failed: status={mp.status}")
            break

        y_bar = {a: y_vars[a].X for a in H}
        theta_bar = {ij: theta_vars[ij].X for ij in od_pairs}
        LB = max(LB, mp.ObjVal)

        cuts, _ = _separation(y_bar, theta_bar, C, L, K, od_pairs)
        if not cuts:
            if verbose:
                print(f"[Phase1] Converged at iter {it}: no violated cuts.")
            break

        # Add cuts to LP AND accumulate for Phase 2
        for (ij, cd) in cuts:
            th_var, rhs = _cut_expr(cd, y_vars, theta_vars)
            mp.addConstr(th_var >= rhs, name=f"cut_{ij[0]}_{ij[1]}_{it}")
            accumulated_cuts.append((ij, cd))   # KEY: saved for Phase 2

        # Rounding heuristic to track upper bound
        if any(1e-6 < y_bar[a] < 1 - 1e-6 for a in H):
            y_round = _rounding_heuristic(y_bar, p, H)
            UB_r = sum(
                _allocation_cost(i, j, y_round, cost_map[(i, j)], arcs_sorted[(i, j)])
                for (i, j) in od_pairs
            ) + sum(C[(i, j)][0] for (i, j) in K if i != j and K[(i, j)] == 1)
            if UB_r < UB_hat:
                UB_hat = UB_r
                y_hat = y_round
                if verbose:
                    print(f"[Phase1] Iter {it}: UB improved to {UB_hat:.4f}")

        if verbose:
            print(f"[Phase1] Iter {it}: LB={LB:.4f}, new_cuts={len(cuts)}, total={len(accumulated_cuts)}")

    if verbose:
        print(f"[Phase1] Done: LB={LB:.4f}, UB={UB_hat:.4f}, cuts_accumulated={len(accumulated_cuts)}")

    return LB, y_hat, UB_hat, accumulated_cuts


# ============================================================
# 7. Phase 2 callback — lazy cuts at integer nodes
# ============================================================

class _MDCutCallback:
    """Add lazy Benders cuts at each integer node in Phase 2."""

    def __init__(self, C, L, K, od_pairs, y_vars, theta_vars):
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
        theta_bar = {ij: model.cbGetSolution(self.theta_vars[ij]) for ij in self.theta_vars}

        cuts, _ = _separation(y_bar, theta_bar, self.C, self.L, self.K, self.od_pairs)
        for (_, cd) in cuts:
            th_var, rhs = _cut_expr(cd, self.y_vars, self.theta_vars)
            model.cbLazy(th_var >= rhs)


# ============================================================
# 8. Main solver
# ============================================================

def solve_md_benders_hub_arc(
    n: int,
    p: int,
    W: List[List[float]],
    D: List[List[float]],
    time_limit: Optional[float] = None,
    phase1_fraction: float = 0.3,
    max_iter_phase1: int = 500,
    verbose: bool = False,
    use_phase1: bool = True,
) -> Dict[str, Any]:
    """
    Solve the p-Hub-Arc problem using true McDaniel & Devine two-phase
    Benders decomposition.

    Parameters
    ----------
    n, p            : number of nodes, number of hub arcs to select
    W               : n×n flow matrix
    D               : n×n distance matrix
    time_limit      : total wall-clock budget in seconds (None = unlimited)
    phase1_fraction : fraction of time_limit allocated to Phase 1 (default 0.3)
    max_iter_phase1 : maximum LP cutting-plane iterations in Phase 1
    verbose         : print progress
    use_phase1      : if False, skip Phase 1 (degenerates to plain B&B-Benders)

    Returns
    -------
    dict with keys:
        objective     : optimal (or best found) total routing cost
        selected_arcs : list of (u,v) hub arcs in the solution
        time          : total wall-clock time
        status        : "OPTIMAL" | "FAILED"
        phase1_cuts   : number of cuts pre-loaded into Phase 2
    """
    t_start = time.time()

    H, C, L, K, cost_map, arcs_sorted, od_pairs = _preprocess(n, W, D)

    # ── Phase 1 ───────────────────────────────────────────────────────────
    accumulated_cuts: List = []
    y_warmstart = None

    if use_phase1:
        phase1_budget = (time_limit * phase1_fraction) if time_limit else None
        if verbose:
            print(f"\n{'='*60}")
            print(f"McDaniel & Devine Phase 1  (budget={phase1_budget}s)")
            print(f"{'='*60}")

        _, y_warmstart, _, accumulated_cuts = _phase1(
            n, p, H, C, L, K, od_pairs,
            cost_map, arcs_sorted,
            time_budget=phase1_budget,
            max_iter=max_iter_phase1,
            verbose=verbose,
        )

        if verbose:
            print(f"\nPhase 1 complete: {len(accumulated_cuts)} cuts accumulated.")

    # ── Phase 2: Branch-and-Benders-cut ───────────────────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"McDaniel & Devine Phase 2  (pre-loaded cuts={len(accumulated_cuts)})")
        print(f"{'='*60}")

    mip = gp.Model("MD_Phase2")
    mip.Params.OutputFlag = 1 if verbose else 0
    mip.Params.LazyConstraints = 1
    mip.Params.MIPGap = 1e-9
    if time_limit:
        mip.Params.TimeLimit = max(0.0, time_limit - (time.time() - t_start))

    y_vars = {a: mip.addVar(vtype=GRB.BINARY, name=f"y_{a[0]}_{a[1]}") for a in H}
    theta_vars = {
        (i, j): mip.addVar(lb=0, name=f"th_{i}_{j}")
        for (i, j) in od_pairs
    }

    mip.setObjective(gp.quicksum(theta_vars[ij] for ij in od_pairs), GRB.MINIMIZE)
    mip.addConstr(gp.quicksum(y_vars[a] for a in H) == p, name="card")

    # ── McDaniel & Devine: pre-load ALL Phase 1 cuts as regular constraints.
    # These are NOT lazy — they are already valid and tighten the LP relaxation
    # of the MIP from the very first node, without needing to rediscover them.
    for (_, cd) in accumulated_cuts:
        th_var, rhs = _cut_expr(cd, y_vars, theta_vars)
        mip.addConstr(th_var >= rhs)

    if verbose and accumulated_cuts:
        print(f"  Pre-loaded {len(accumulated_cuts)} Phase 1 cuts as regular constraints.")

    # Warm-start from Phase 1 rounded solution
    if y_warmstart:
        for a in H:
            y_vars[a].Start = 1.0 if y_warmstart.get(a, 0.0) > 0.5 else 0.0

    cb = _MDCutCallback(C, L, K, od_pairs, y_vars, theta_vars)
    mip.optimize(cb)

    elapsed = time.time() - t_start

    if mip.status == GRB.OPTIMAL:
        y_sol = {a: y_vars[a].X for a in H}
        selected = [(u, v) for (u, v) in H if y_sol[(u, v)] > 0.5]
        obj = mip.ObjVal
        # Add fixed costs for OD pairs with only one cost level (no theta var)
        obj += sum(C[(i, j)][0] for (i, j) in K if i != j and K[(i, j)] == 1)
        return {
            "objective": obj,
            "selected_arcs": selected,
            "time": elapsed,
            "status": "OPTIMAL",
            "phase1_cuts": len(accumulated_cuts),
        }

    return {
        "objective": None,
        "selected_arcs": None,
        "time": elapsed,
        "status": "FAILED",
        "phase1_cuts": len(accumulated_cuts),
    }
