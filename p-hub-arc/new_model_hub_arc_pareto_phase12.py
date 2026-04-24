"""
Benders Decomposition for p-Hub-Arc with Pareto cuts in BOTH phases.

This keeps `new_model_hub_arc.py` unchanged and provides a separate solver
that applies Magnanti-Wong style Pareto separation in:
  - Phase 1 (LP master loop), and
  - Phase 2 (lazy callback at integer incumbents).
"""

import time
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from algo3_hub_arc import compute_allocation_cost, rounding_heuristic
from new_model_hub_arc import preprocess
from pareto_benders_hub_arc import separation_pareto


def _uniform_y_core(H: List[Tuple[int, int]], p: int) -> Dict[Tuple[int, int], float]:
    return {a: p / len(H) for a in H}


def phase1_solve_mp_pareto(
    p: int,
    H: List[Tuple[int, int]],
    C: Dict[Tuple[int, int], List[float]],
    L: Dict[Tuple[int, int], Dict[int, List[Tuple[int, int]]]],
    K: Dict[Tuple[int, int], int],
    od_pairs: List[Tuple[int, int]],
    cost_map: Dict[Tuple[int, int], Dict[Tuple[int, int], float]],
    arcs_sorted: Dict[Tuple[int, int], List[Tuple[int, int]]],
    y_core: Dict[Tuple[int, int], float],
    y_heuristic: Dict[Tuple[int, int], float] = None,
    UB_heuristic: float = None,
    max_iter: int = 200,
    verbose: bool = False,
    pareto_method: str = "two_step",
) -> Tuple[float, Dict[Tuple[int, int], float], float]:
    """
    Phase 1 LP relaxation with Pareto separation.

    Returns
    -------
    LB1, y1, UB1
    """
    model = gp.Model("HubArc_Master_Phase1_Pareto")
    model.Params.OutputFlag = 1 if verbose else 0

    y_vars = {
        a: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{a[0]}_{a[1]}")
        for a in H
    }
    theta_vars = {
        (i, j): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"theta_{i}_{j}")
        for (i, j) in od_pairs
    }

    model.setObjective(gp.quicksum(theta_vars[(i, j)] for (i, j) in od_pairs), GRB.MINIMIZE)
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
                print(f"[Phase1-Pareto] Master failed: status={model.status}")
            break

        y_bar = {a: y_vars[a].X for a in H}
        theta_bar = {(i, j): theta_vars[(i, j)].X for (i, j) in od_pairs}
        LB1 = max(LB1, model.ObjVal)

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
                print("[Phase1-Pareto] No violated cuts. Converged.")
            break

        for ((i, j), cut_data) in cuts:
            Ki = cut_data["Ki"]
            C1 = cut_data["C1"]
            nu = cut_data["nu"]
            expr = C1
            if nu.get(1, 0) != 0 and cut_data.get("L1"):
                expr += nu[1] * (1.0 - gp.quicksum(y_vars[a] for a in cut_data["L1"]))
            for kk in range(2, Ki + 1):
                Lk = cut_data.get(f"L{kk}", [])
                if nu.get(kk, 0) != 0 and Lk:
                    expr -= nu[kk] * gp.quicksum(y_vars[a] for a in Lk)
            model.addConstr(theta_vars[(i, j)] >= expr, name=f"cut_{i}_{j}_{iter_count}")

        is_fractional = any(1e-6 < y_bar.get(a, 0.0) < 1.0 - 1e-6 for a in H)
        if is_fractional:
            y_round = rounding_heuristic(y_bar, p, H)
            UB_h = 0.0
            for (i, j) in od_pairs:
                UB_h += compute_allocation_cost(
                    i, j, y_round, cost_map.get((i, j), {}), arcs_sorted.get((i, j), [])
                )
            if UB_h < UB_hat:
                UB_hat = UB_h
                y_hat = y_round
                if verbose:
                    print(f"[Phase1-Pareto] Rounding improved UB to {UB_hat:.4f}")

        iter_count += 1
        if verbose:
            print(f"[Phase1-Pareto] Iter {iter_count}: LB={LB1:.4f}, cuts={len(cuts)}")

    y1 = y_hat if y_hat else rounding_heuristic({a: y_vars[a].X for a in H}, p, H)
    UB1 = 0.0
    for (i, j) in od_pairs:
        UB1 += compute_allocation_cost(i, j, y1, cost_map.get((i, j), {}), arcs_sorted.get((i, j), []))
    for (i, j) in K:
        if i != j and K[(i, j)] == 1:
            UB1 += C[(i, j)][0]
    if UB_heuristic is not None and UB_heuristic < UB1:
        UB1 = UB_heuristic
        y1 = y_heuristic
    return LB1, y1, UB1


class ParetoPhase2Callback:
    """Lazy callback that adds Pareto-optimal cuts in Phase 2."""

    def __init__(
        self,
        C: Dict,
        L: Dict,
        K: Dict,
        od_pairs: List[Tuple[int, int]],
        y_vars: Dict,
        theta_vars: Dict,
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
        theta_bar = {(i, j): model.cbGetSolution(self.theta_vars[(i, j)]) for (i, j) in self.theta_vars}
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
        for ((i, j), cut_data) in cuts:
            Ki = cut_data["Ki"]
            C1 = cut_data["C1"]
            nu = cut_data["nu"]
            expr = C1
            if nu.get(1, 0) != 0 and cut_data.get("L1"):
                expr += nu[1] * (1.0 - gp.quicksum(self.y_vars[a] for a in cut_data["L1"]))
            for kk in range(2, Ki + 1):
                Lk = cut_data.get(f"L{kk}", [])
                if nu.get(kk, 0) != 0 and Lk:
                    expr -= nu[kk] * gp.quicksum(self.y_vars[a] for a in Lk)
            model.cbLazy(self.theta_vars[(i, j)] >= expr)


def solve_benders_hub_arc_pareto_phase12(
    n: int,
    p: int,
    W: List[List[float]],
    D: List[List[float]],
    time_limit: float = None,
    verbose: bool = False,
    use_phase1: bool = True,
    pareto_method: str = "two_step",
) -> Dict[str, Any]:
    """
    Solve p-Hub-Arc with Pareto cuts in both phases.

    Parameters
    ----------
    pareto_method: "two_step" (exact MW) or "epsilon".
    """
    start_time = time.time()
    H, C, L, K, cost_map, arcs_sorted, od_pairs = preprocess(n, W, D)
    y_core = _uniform_y_core(H, p)

    y1 = None
    if use_phase1:
        y_heuristic = {a: 0.0 for a in H}
        for idx, a in enumerate(H):
            if idx < p:
                y_heuristic[a] = 1.0
        UB_h = 0.0
        for (i, j) in od_pairs:
            UB_h += compute_allocation_cost(
                i, j, y_heuristic, cost_map.get((i, j), {}), arcs_sorted.get((i, j), [])
            )
        for (i, j) in K:
            if i != j and K[(i, j)] == 1:
                UB_h += C[(i, j)][0]

        if verbose:
            print("Phase 1: LP relaxation with Pareto cuts...")
        LB1, y1, UB1 = phase1_solve_mp_pareto(
            p=p,
            H=H,
            C=C,
            L=L,
            K=K,
            od_pairs=od_pairs,
            cost_map=cost_map,
            arcs_sorted=arcs_sorted,
            y_core=y_core,
            y_heuristic=y_heuristic,
            UB_heuristic=UB_h,
            verbose=verbose,
            pareto_method=pareto_method,
        )
        if verbose:
            print(f"Phase 1 done: LB={LB1:.4f}, UB={UB1:.4f}")

    if verbose:
        print("Phase 2: Branch-and-Benders-cut with Pareto cuts...")

    model = gp.Model("HubArc_Phase2_Pareto_Ph12")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.LazyConstraints = 1
    model.Params.MIPGap = 1e-9
    if time_limit:
        model.Params.TimeLimit = max(0, time_limit - (time.time() - start_time))

    y_vars = {a: model.addVar(vtype=GRB.BINARY, name=f"y_{a[0]}_{a[1]}") for a in H}
    theta_vars = {(i, j): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"theta_{i}_{j}") for (i, j) in od_pairs}
    model.setObjective(gp.quicksum(theta_vars[(i, j)] for (i, j) in od_pairs), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(y_vars[a] for a in H) == p, name="p_hub_arcs")

    if y1:
        for a in H:
            y_vars[a].Start = 1.0 if y1.get(a, 0.0) > 0.5 else 0.0

    callback = ParetoPhase2Callback(
        C=C,
        L=L,
        K=K,
        od_pairs=od_pairs,
        y_vars=y_vars,
        theta_vars=theta_vars,
        y_core=y_core,
        pareto_method=pareto_method,
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
        return {"objective": obj, "selected_arcs": selected, "time": elapsed, "status": "OPTIMAL"}

    return {"objective": None, "selected_arcs": None, "time": elapsed, "status": "FAILED"}

