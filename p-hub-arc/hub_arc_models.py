import gurobipy as gp
from gurobipy import GRB
import time
from typing import List, Tuple, Dict, Any


# ======================================================================
#  F3 CANONICAL FORMULATION (DIRECT MIP)
# ======================================================================

def solve_hub_arc_F3(
    n: int,
    p: int,
    W: List[List[float]],
    D: List[List[float]],
    gurobi_output: bool = True,
) -> Dict[str, Any]:
    """
    Canonical F3 formulation for the p-Hub-Arc problem.

    For each OD pair (i,j), let:
      c_{ij,uv} = W_ij * (D_{i,u} + D_{u,v} + D_{v,j})  for each arc (u,v).
    Let C_{ij}^1 < ... < C_{ij}^{Kij} be the sorted distinct values of {c_{ij,uv}}.

    Decision vars:
      y_{uv} ∈ {0,1}          for each arc (u,v)
      z_{ij}^k ≥ 0            for k = 1..Kij-1

    Objective:
      min Σ_{i,j} [ C_{ij}^1 + Σ_{k=1}^{Kij-1} (C_{ij}^{k+1}-C_{ij}^k) z_{ij}^k ]
    """

    N = range(n)
    H = [(u, v) for u in N for v in N if u != v]

    model = gp.Model("HubArc_F3")
    model.Params.OutputFlag = 1 if gurobi_output else 0

    # ---- 1. Precompute cost levels C_{ij}^k and arcs L_{ij}^k ----
    C: Dict[Tuple[int, int], List[float]] = {}
    G: Dict[Tuple[int, int], int] = {}
    L: Dict[Tuple[int, int], Dict[int, List[Tuple[int, int]]]] = {}

    for i in N:
        for j in N:
            if i == j:
                continue

            costs_ij = []
            cost_map = {}
            for u in N:
                for v in N:
                    if u == v:
                        continue
                    cost_uv = W[i][j] * (D[i][u] + D[u][v] + D[v][j])
                    costs_ij.append(cost_uv)
                    cost_map[(u, v)] = cost_uv

            unique_costs = sorted(set(costs_ij))
            Kij = len(unique_costs)
            C[(i, j)] = unique_costs
            G[(i, j)] = Kij

            L[(i, j)] = {}
            for k_idx, target_cost in enumerate(unique_costs, start=1):
                arcs_k = [
                    (u, v)
                    for (u, v), c_val in cost_map.items()
                    if abs(c_val - target_cost) < 1e-8
                ]
                L[(i, j)][k_idx] = arcs_k

    # ---- 2. Variables ----
    y = model.addVars(H, vtype=GRB.BINARY, name="y")

    z: Dict[Tuple[int, int, int], gp.Var] = {}
    for i in N:
        for j in N:
            if i == j or G[(i, j)] <= 1:
                continue
            Kij = G[(i, j)]
            # z_{ij}^k for k = 1..Kij-1
            for k in range(1, Kij):
                z[(i, j, k)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                            name=f"z_{i}_{j}_{k}")

    # ---- 3. Objective ----
    obj = gp.LinExpr()

    for i in N:
        for j in N:
            if i == j or G[(i, j)] == 0:
                continue

            costs_ij = C[(i, j)]
            Kij = G[(i, j)]

            # base term C_{ij}^1
            obj += costs_ij[0]

            # increments (C^{k+1}-C^k) * z_{ij}^k,  k=1..Kij-1
            if Kij > 1:
                for k in range(1, Kij):
                    diff = costs_ij[k] - costs_ij[k - 1]
                    obj += diff * z[(i, j, k)]

    model.setObjective(obj, GRB.MINIMIZE)

    # ---- 4. Constraints ----
    # (1) exactly p hub arcs
    model.addConstr(gp.quicksum(y[a] for a in H) == p, name="p_hub_arcs")

    # (2) & (3) coverage constraints
    for i in N:
        for j in N:
            if i == j or G[(i, j)] <= 1:
                continue
            Kij = G[(i, j)]

            # k=1: z_{ij}^1 + Σ_{a∈L^1} y_a ≥ 1
            arcs_level1 = L[(i, j)][1]
            model.addConstr(
                z[(i, j, 1)] + gp.quicksum(y[a] for a in arcs_level1) >= 1,
                name=f"cover_{i}_{j}_1"
            )

            # k=2..Kij: z_{ij}^k + Σ_{a∈L^k} y_a ≥ z_{ij}^{k-1}
            for k in range(2, Kij):
                arcs_levelk = L[(i, j)][k]
                model.addConstr(
                    z[(i, j, k)] + gp.quicksum(y[a] for a in arcs_levelk)
                    >= z[(i, j, k - 1)],
                    name=f"cover_{i}_{j}_{k}"
                )

    # ---- 5. Solve ----
    model.update()
    start = time.time()
    model.optimize()
    elapsed = time.time() - start

    if model.status == GRB.OPTIMAL:
        selected_arcs = [(u, v) for (u, v) in H if y[(u, v)].X > 0.5]
        z_vals = {(i, j, k): var.X for (i, j, k), var in z.items()}
        return {
            "objective": model.ObjVal,
            "selected_arcs": selected_arcs,
            "z_values": z_vals,
            "time": elapsed,
            "status": "OPTIMAL",
            "model": model,
        }
    else:
        return {
            "objective": None,
            "selected_arcs": None,
            "z_values": None,
            "time": elapsed,
            "status": model.status,
            "model": model,
        }


# ======================================================================
#  BENDERS DECOMPOSITION MATCHING THE SAME MODEL
# ======================================================================

class HubArcBenders:
    def __init__(
        self,
        n: int,
        p: int,
        W: List[List[float]],
        D: List[List[float]],
        verbose: bool = False,
    ):
        """
        Benders decomposition for the same canonical F3 model.

        Master:   y_{uv} (binary hub arcs) + theta_{ij} (continuous, one per OD pair).
        Subprob:  For each (i,j), dual of the z_{ij}^k-subproblem.
        """
        self.n = n
        self.p = p
        self.W = W
        self.D = D
        self.verbose = verbose

        self.N = range(n)
        self.H = [(u, v) for u in self.N for v in self.N if u != v]

        # ---- precompute C_{ij}^k and L_{ij}^k, identical to F3 ----
        self.K: Dict[Tuple[int, int], int] = {}
        self.C: Dict[Tuple[int, int], List[float]] = {}
        self.L: Dict[Tuple[int, int], Dict[int, List[Tuple[int, int]]]] = {}

        for i in self.N:
            for j in self.N:
                if i == j:
                    continue

                costs_ij = []
                cost_map = {}
                for u in self.N:
                    for v in self.N:
                        if u == v:
                            continue
                        cost_uv = W[i][j] * (D[i][u] + D[u][v] + D[v][j])
                        costs_ij.append(cost_uv)
                        cost_map[(u, v)] = cost_uv

                unique_costs = sorted(set(costs_ij))
                Kij = len(unique_costs)
                self.K[(i, j)] = Kij
                self.C[(i, j)] = unique_costs

                self.L[(i, j)] = {}
                for k_idx, target_cost in enumerate(unique_costs, start=1):
                    arcs_k = [
                        (u, v)
                        for (u, v), c_val in cost_map.items()
                        if abs(c_val - target_cost) < 1e-8
                    ]
                    self.L[(i, j)][k_idx] = arcs_k

        self.master_y_vars: Dict[Tuple[int, int], gp.Var] = {}
        self.master_theta_vars: Dict[Tuple[int, int], gp.Var] = {}

    # ------------------------------------------------------------------
    # Dual subproblem for one (i,j), given incumbent y_hat
    # ------------------------------------------------------------------
    def _solve_dual_subproblem(
        self,
        i: int,
        j: int,
        y_hat: Dict[Tuple[int, int], float],
    ) -> Tuple[float, Dict[str, Any]]:

        Kij = self.K[(i, j)]
        if Kij <= 1:
            # only one cost level → allocation cost is constant,
            # no z-variables; contribution handled directly, so skip
            return 0.0, {"Ki": Kij}

        model = gp.Model(f"Dual_SP_{i}_{j}")
        model.Params.OutputFlag = 0

        # ν_k  for k = 1..Kij
        nu = model.addVars(range(1, Kij + 1), lb=0.0,
                           vtype=GRB.CONTINUOUS, name="nu")

        # objective:
        # C^1 + ν1(1 - Σ_{a∈L^1} y_a) - Σ_{k=2}^{Kij} ν_k Σ_{a∈L^k} y_a
        L1 = self.L[(i, j)][1]
        obj_expr = self.C[(i, j)][0]
        obj_expr += nu[1] * (1.0 - sum(y_hat.get(a, 0.0) for a in L1))
        for k in range(2, Kij + 1):
            Lk = self.L[(i, j)][k]
            obj_expr += nu[k] * (-sum(y_hat.get(a, 0.0) for a in Lk))

        model.setObjective(obj_expr, GRB.MAXIMIZE)

        # constraints:
        # ν_k - ν_{k+1} ≤ C^{k+1} - C^k   for k = 1..Kij-1
        for k in range(1, Kij):
            model.addConstr(
                nu[k] - nu[k + 1]
                <= self.C[(i, j)][k] - self.C[(i, j)][k - 1],
                name=f"dual_{k}"
            )
        # ν_k ≥ 0 already from lb=0

        model.optimize()
        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"Dual subproblem ({i},{j}) not optimal, status={model.status}")

        nu_sol = {k: nu[k].X for k in range(1, Kij + 1)}
        sp_obj = model.ObjVal

        cut_data = {
            "i": i,
            "j": j,
            "Ki": Kij,
            "nu": nu_sol,
            "L1": L1,
        }
        for k in range(2, Kij + 1):
            cut_data[f"L{k}"] = self.L[(i, j)][k]

        return sp_obj, cut_data

    # ------------------------------------------------------------------
    # Build master problem
    # ------------------------------------------------------------------
    def _build_master(self) -> gp.Model:
        model = gp.Model("HubArc_Benders_Master")
        model.Params.OutputFlag = 0
        model.Params.LazyConstraints = 1

        # y_{uv}: hub-arc variables
        self.master_y_vars = {
            (u, v): model.addVar(vtype=GRB.BINARY, name=f"y_{u}_{v}")
            for (u, v) in self.H
        }

        # θ_{ij}: one per OD pair with at least 2 distinct levels
        self.master_theta_vars = {
            (i, j): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"theta_{i}_{j}")
            for (i, j) in self.K
            if i != j and self.K[(i, j)] > 1
        }

        # Objective: min Σ_{(i,j)} θ_{ij}
        obj_expr = gp.quicksum(self.master_theta_vars[(i, j)]
                               for (i, j) in self.master_theta_vars)
        model.setObjective(obj_expr, GRB.MINIMIZE)

        # p-hub-arc constraint
        model.addConstr(
            gp.quicksum(self.master_y_vars[a] for a in self.H) == self.p,
            name="p_hub_arcs"
        )

        model.update()
        return model

    # ------------------------------------------------------------------
    # Lazy callback
    # ------------------------------------------------------------------
    def _lazy_callback(self, model: gp.Model, where: int):
        if where != GRB.Callback.MIPSOL:
            return

        y_hat = {
            a: model.cbGetSolution(self.master_y_vars[a])
            for a in self.H
        }
        theta_hat = {
            (i, j): model.cbGetSolution(self.master_theta_vars[(i, j)])
            for (i, j) in self.master_theta_vars
        }

        tol = 1e-6
        cuts_added = 0

        for (i, j) in self.master_theta_vars:
            sp_obj, cut = self._solve_dual_subproblem(i, j, y_hat)
            Ki = cut["Ki"]
            if Ki <= 1:
                continue

            if sp_obj > theta_hat[(i, j)] + tol:
                nu = cut["nu"]

                # θ_{ij} >= C^1 + ν1(1 - Σ_{a∈L1} y_a) - Σ_{k>=2} ν_k Σ_{a∈Lk} y_a
                expr = self.C[(i, j)][0]
                expr += nu[1] * (1.0 - gp.quicksum(self.master_y_vars[a]
                                                   for a in cut["L1"]))
                for k in range(2, Ki + 1):
                    Lk = cut.get(f"L{k}", [])
                    if Lk:
                        expr -= nu[k] * gp.quicksum(self.master_y_vars[a] for a in Lk)

                theta_var = self.master_theta_vars[(i, j)]
                model.cbLazy(theta_var >= expr)
                cuts_added += 1

                if self.verbose:
                    print(f"[CUT] ({i},{j})  SP={sp_obj:.6f}  θ̂={theta_hat[(i,j)]:.6f}")

        if self.verbose and cuts_added:
            print(f"[CALLBACK] Added {cuts_added} cuts.")

    # ------------------------------------------------------------------
    # Public solve
    # ------------------------------------------------------------------
    def solve(self, time_limit: float = None) -> Dict[str, Any]:
        master = self._build_master()
        if time_limit is not None:
            master.Params.TimeLimit = time_limit

        start = time.time()
        master.optimize(self._lazy_callback)
        elapsed = time.time() - start

        status = master.status
        if status == GRB.OPTIMAL:
            y_sol = [(u, v) for (u, v) in self.H
                     if self.master_y_vars[(u, v)].X > 0.5]
            obj_val = master.ObjVal
        else:
            y_sol = None
            obj_val = None

        return {
            "status": status,
            "objective": obj_val,
            "selected_arcs": y_sol,
            "time": elapsed,
            "model": master,
        }


# ======================================================================
#  Test / Comparison
# ======================================================================
if __name__ == "__main__":
    n = 3
    p = 2

    W = [
        [0, 2, 3],
        [2, 0, 4],
        [3, 4, 0],
    ]

    D = [
        [0, 5, 2],
        [5, 0, 3],
        [2, 3, 0],
    ]

    # F3
    f3_res = solve_hub_arc_F3(n, p, W, D, gurobi_output=True)
    print("\n=== F3 Canonical Formulation Result ===")
    print("Status:", f3_res["status"])
    print("Objective:", f3_res["objective"])
    print("Selected hub arcs:", f3_res["selected_arcs"])
    print("Solve time (s):", round(f3_res["time"], 6))

    # Benders
    benders = HubArcBenders(n=n, p=p, W=W, D=D, verbose=True)
    b_res = benders.solve(time_limit=60)
    print("\n=== Hub-Arc Benders Result ===")
    print("Status:", b_res["status"])
    print("Objective:", b_res["objective"])
    print("Selected arcs:", b_res["selected_arcs"])
    print("Time (s):", round(b_res["time"], 6))
