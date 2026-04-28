"""
McDaniel & Devine Two-Phase Benders with Pareto-optimal cuts for p-Hub-Arc.

This is a separate variant of `md_benders_hub_arc.py` that keeps the same
McDaniel-Devine structure:
  - Phase 1: LP cutting-plane loop with cut accumulation
  - Phase 2: MIP with all Phase-1 cuts pre-loaded + lazy cuts

Difference:
  - Separation uses Pareto-optimal cuts (Magnanti-Wong style), selectable as:
      * pareto_method="two_step"  (exact two-step MW)
      * pareto_method="epsilon"   (single-LP perturbation)
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from pareto_benders_hub_arc import separation_pareto


def _preprocess(n: int, W: List[List[float]], D: List[List[float]]):
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


def _uniform_y_core(H: List[Tuple[int, int]], p: int) -> Dict[Tuple[int, int], float]:
    return {a: p / len(H) for a in H}


def _cut_expr(cut_data: Dict, y_vars: Dict, theta_vars: Dict):
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


def _rounding_heuristic(y_bar: Dict, p: int, H: List) -> Dict:
    top = sorted(H, key=lambda a: y_bar.get(a, 0.0), reverse=True)[:p]
    return {a: (1.0 if a in top else 0.0) for a in H}


def _allocation_cost(i, j, y_int, cost_map, arcs_sorted_ij):
    for a in arcs_sorted_ij:
        if y_int.get(a, 0.0) > 0.5:
            return cost_map.get(a, float("inf"))
    return float("inf")


def _phase1(
    p: int,
    H: List,
    C: Dict,
    L: Dict,
    K: Dict,
    od_pairs: List,
    cost_map: Dict,
    arcs_sorted: Dict,
    y_core: Dict,
    pareto_method: str,
    time_budget: Optional[float],
    max_iter: int,
    verbose: bool,
) -> Tuple[float, Dict, float, List]:
    t0 = time.time()

    mp = gp.Model("MDP_Phase1_Pareto")
    mp.Params.OutputFlag = 1 if verbose else 0

    y_vars = {a: mp.addVar(lb=0, ub=1, name=f"y_{a[0]}_{a[1]}") for a in H}
    theta_vars = {(i, j): mp.addVar(lb=0, name=f"th_{i}_{j}") for (i, j) in od_pairs}

    mp.setObjective(gp.quicksum(theta_vars[ij] for ij in od_pairs), GRB.MINIMIZE)
    mp.addConstr(gp.quicksum(y_vars[a] for a in H) == p, name="card")

    y_heur = {a: (1.0 if idx < p else 0.0) for idx, a in enumerate(H)}
    UB_hat = sum(
        _allocation_cost(i, j, y_heur, cost_map[(i, j)], arcs_sorted[(i, j)])
        for (i, j) in od_pairs
    ) + sum(C[(i, j)][0] for (i, j) in K if i != j and K[(i, j)] == 1)
    y_hat = y_heur

    LB = -float("inf")
    accumulated_cuts: List = []

    for it in range(max_iter):
        if time_budget is not None:
            remaining = time_budget - (time.time() - t0)
            if remaining <= 0:
                if verbose:
                    print(f"[Phase1-Pareto] Time budget exhausted at iter {it}.")
                break
            mp.Params.TimeLimit = remaining

        mp.optimize()
        if mp.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or mp.SolCount == 0:
            if verbose:
                print(f"[Phase1-Pareto] LP solve failed: status={mp.status}")
            break

        y_bar = {a: y_vars[a].X for a in H}
        theta_bar = {ij: theta_vars[ij].X for ij in od_pairs}
        LB = max(LB, mp.ObjVal)

        cuts, _ = separation_pareto(
            y_bar=y_bar,
            theta_bar=theta_bar,
            y_core=y_core,
            C=C,
            L=L,
            K=K,
            od_pairs=od_pairs,
            pareto_method=pareto_method,
        )
        if not cuts:
            if verbose:
                print(f"[Phase1-Pareto] Converged at iter {it}: no violated cuts.")
            break

        for (_, cd) in cuts:
            th_var, rhs = _cut_expr(cd, y_vars, theta_vars)
            mp.addConstr(th_var >= rhs, name=f"cut_{cd['i']}_{cd['j']}_{it}")
            accumulated_cuts.append(((cd["i"], cd["j"]), cd))

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
                    print(f"[Phase1-Pareto] Iter {it}: UB improved to {UB_hat:.4f}")

        if verbose:
            print(
                f"[Phase1-Pareto] Iter {it}: LB={LB:.4f}, "
                f"new_cuts={len(cuts)}, total={len(accumulated_cuts)}"
            )

    if verbose:
        print(
            f"[Phase1-Pareto] Done: LB={LB:.4f}, UB={UB_hat:.4f}, "
            f"cuts_accumulated={len(accumulated_cuts)}"
        )
    return LB, y_hat, UB_hat, accumulated_cuts


class _MDParetoCutCallback:
    def __init__(self, C, L, K, od_pairs, y_vars, theta_vars, y_core, pareto_method):
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
        theta_bar = {ij: model.cbGetSolution(self.theta_vars[ij]) for ij in self.theta_vars}
        cuts, _ = separation_pareto(
            y_bar=y_bar,
            theta_bar=theta_bar,
            y_core=self.y_core,
            C=self.C,
            L=self.L,
            K=self.K,
            od_pairs=self.od_pairs,
            pareto_method=self.pareto_method,
        )
        for (_, cd) in cuts:
            th_var, rhs = _cut_expr(cd, self.y_vars, self.theta_vars)
            model.cbLazy(th_var >= rhs)


def solve_md_benders_hub_arc_pareto(
    n: int,
    p: int,
    W: List[List[float]],
    D: List[List[float]],
    time_limit: Optional[float] = None,
    phase1_fraction: float = 0.3,
    max_iter_phase1: int = 500,
    verbose: bool = False,
    use_phase1: bool = True,
    pareto_method: str = "two_step",
) -> Dict[str, Any]:
    """
    McDaniel & Devine two-phase Benders with Pareto-optimal cuts.
    """
    t_start = time.time()

    H, C, L, K, cost_map, arcs_sorted, od_pairs = _preprocess(n, W, D)
    y_core = _uniform_y_core(H, p)

    accumulated_cuts: List = []
    y_warmstart = None

    if use_phase1:
        phase1_budget = (time_limit * phase1_fraction) if time_limit else None
        if verbose:
            print("\n" + "=" * 60)
            print(f"MD Pareto Phase 1 (budget={phase1_budget}s, method={pareto_method})")
            print("=" * 60)
        _, y_warmstart, _, accumulated_cuts = _phase1(
            p=p,
            H=H,
            C=C,
            L=L,
            K=K,
            od_pairs=od_pairs,
            cost_map=cost_map,
            arcs_sorted=arcs_sorted,
            y_core=y_core,
            pareto_method=pareto_method,
            time_budget=phase1_budget,
            max_iter=max_iter_phase1,
            verbose=verbose,
        )
        if verbose:
            print(f"\nPhase 1 complete: {len(accumulated_cuts)} Pareto cuts accumulated.")

    if verbose:
        print("\n" + "=" * 60)
        print(f"MD Pareto Phase 2 (pre-loaded cuts={len(accumulated_cuts)})")
        print("=" * 60)

    mip = gp.Model("MDP_Phase2_Pareto")
    mip.Params.OutputFlag = 1 if verbose else 0
    mip.Params.LazyConstraints = 1
    mip.Params.MIPGap = 1e-9
    if time_limit:
        mip.Params.TimeLimit = max(0.0, time_limit - (time.time() - t_start))

    y_vars = {a: mip.addVar(vtype=GRB.BINARY, name=f"y_{a[0]}_{a[1]}") for a in H}
    theta_vars = {(i, j): mip.addVar(lb=0, name=f"th_{i}_{j}") for (i, j) in od_pairs}
    mip.setObjective(gp.quicksum(theta_vars[ij] for ij in od_pairs), GRB.MINIMIZE)
    mip.addConstr(gp.quicksum(y_vars[a] for a in H) == p, name="card")

    for (_, cd) in accumulated_cuts:
        th_var, rhs = _cut_expr(cd, y_vars, theta_vars)
        mip.addConstr(th_var >= rhs)

    if y_warmstart:
        for a in H:
            y_vars[a].Start = 1.0 if y_warmstart.get(a, 0.0) > 0.5 else 0.0

    cb = _MDParetoCutCallback(
        C=C,
        L=L,
        K=K,
        od_pairs=od_pairs,
        y_vars=y_vars,
        theta_vars=theta_vars,
        y_core=y_core,
        pareto_method=pareto_method,
    )
    mip.optimize(cb)

    elapsed = time.time() - t_start
    has_sol = mip.SolCount > 0
    diag = {
        "has_incumbent": has_sol,
        "incumbent_objective": (mip.ObjVal if has_sol else None),
        "obj_bound": (mip.ObjBound if has_sol else None),
        "mip_gap": (mip.MIPGap if has_sol else None),
    }

    if mip.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and has_sol:
        y_sol = {a: y_vars[a].X for a in H}
        selected = [(u, v) for (u, v) in H if y_sol[(u, v)] > 0.5]
        obj = mip.ObjVal + sum(C[(i, j)][0] for (i, j) in K if i != j and K[(i, j)] == 1)
        return {
            "objective": obj,
            "selected_arcs": selected,
            "time": elapsed,
            "status": ("OPTIMAL" if mip.status == GRB.OPTIMAL else "TIME_LIMIT"),
            "phase1_cuts": len(accumulated_cuts),
            "pareto_method": pareto_method,
            **diag,
        }

    return {
        "objective": None,
        "selected_arcs": None,
        "time": elapsed,
        "status": "FAILED",
        "phase1_cuts": len(accumulated_cuts),
        "pareto_method": pareto_method,
        **diag,
    }

