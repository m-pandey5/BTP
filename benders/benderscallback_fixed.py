"""
Fixed: Benders Decomposition for p-Median Problem using Gurobi lazy callbacks.

Fixes variable naming / lookup issues so callback can reliably read x_hat / y_hat values.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

class BendersDecomposition:
    def __init__(self, N, M, p_open, K, D, distance_matrix, verbose=False):
        self.N = N
        self.M = M
        self.p = p_open
        self.K = K
        self.D = D
        self.dist = distance_matrix
        self.verbose = verbose

        # Precompute facilities at each distance level for each client
        self.facilities_at_distance = {}
        for i in range(self.N):
            self.facilities_at_distance[i] = {}
            for k in range(1, self.K[i] + 1):
                Dk = self.D[i][k]
                self.facilities_at_distance[i][k] = [j for j in range(self.M) if abs(self.dist[i][j] - Dk) < 1e-8]

        # place-holders for master var references (filled when building master)
        self.master_y_vars = []
        self.master_theta_vars = []

    def solve_dual_subproblem(self, i, y_bar):
        model = gp.Model(f"DualSubproblem_{i}")
        model.Params.OutputFlag = 0

        Ki = self.K[i]
        nu = model.addVars(range(1, Ki + 1), lb=0.0, vtype=GRB.CONTINUOUS, name="nu")

        facilities_at_D = self.facilities_at_distance[i]

        # Objective: nu[1]*(1 - sum y_j at D1) + sum_{k>=2} nu[k]*(-sum y_j at Dk)
        obj_expr = nu[1] * (1.0 - sum(y_bar[j] for j in facilities_at_D[1]))
        for k in range(2, Ki + 1):
            facs = facilities_at_D.get(k, [])
            obj_expr += nu[k] * (-sum(y_bar[j] for j in facs))

        model.setObjective(obj_expr, GRB.MAXIMIZE)

        # Constraints
        model.addConstr(nu[1] <= self.D[i][2] - self.D[i][1], name="dual_cons_1")
        for k in range(2, Ki):
            model.addConstr(nu[k] - nu[k+1] <= self.D[i][k+1] - self.D[i][k], 
                        name=f"dual_cons_{k}")
        # Last constraint (if Ki > 1)
        if Ki > 1:
            model.addConstr(nu[Ki] <= 0, name=f"dual_cons_{Ki}")  # or omit if nu[Ki] already lb=0

        model.optimize()
        # ... rest remains the same

        if model.status == GRB.OPTIMAL:
            nu_sol = {k: nu[k].X for k in range(1, Ki + 1)}
            obj = model.ObjVal
            cut = {
                'client': i,
                'D1_i': self.D[i][1],
                'nu1_i': nu_sol[1],
                'nu_k': nu_sol,
                'facilities_at_D1': facilities_at_D[1]
            }
            for k in range(2, Ki + 1):
                cut[f'facilities_at_D{k}'] = facilities_at_D[k]
            return obj, cut
        else:
            return None, None

    def build_master_for_callback(self):
        model = gp.Model("MasterCallback")
        model.Params.OutputFlag = 0

        # Create y_j and theta_i explicitly and keep references
        self.master_y_vars = [model.addVar(vtype=GRB.BINARY, name=f"y_{j}") for j in range(self.M)]
        self.master_theta_vars = [model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"theta_{i}") for i in range(self.N)]

        # objective: minimize sum theta_i
        model.setObjective(gp.quicksum(self.master_theta_vars[i] for i in range(self.N)), GRB.MINIMIZE)

        # p-facility constraint
        model.addConstr(gp.quicksum(self.master_y_vars[j] for j in range(self.M)) == self.p, name="p_facilities")

        model.Params.LazyConstraints = 1
        model.update()
        return model

    def _callback(self, model, where):
        # callback triggered at MIPSOL (integer solution)
        if where == GRB.Callback.MIPSOL:
            # Retrieve solution values using the stored variable *objects* where possible.
            # model.getVarByName should also work since we named them explicitly, but
            # using stored objects avoids name mismatch.
            try:
                # cbGetSolution accepts either a Var object or variable ID retrieved by name
                y_hat = [model.cbGetSolution(self.master_y_vars[j]) for j in range(self.M)]
                theta_hat = [model.cbGetSolution(self.master_theta_vars[i]) for i in range(self.N)]
            except gp.GurobiError as e:
                # fallback to name-based retrieval if needed (should not happen)
                y_hat = [model.cbGetSolution(model.getVarByName(f"y_{j}")) for j in range(self.M)]
                theta_hat = [model.cbGetSolution(model.getVarByName(f"theta_{i}")) for i in range(self.N)]

            tol = 1e-6
            cuts_added = 0
            total_sp_obj = 0.0

            for i in range(self.N):
                sp_obj, cut = self.solve_dual_subproblem(i, y_hat)
                if sp_obj is None:
                    if self.verbose:
                        print(f"[CB] DSP_{i} not optimal - skipping.")
                    continue

                total_sp_obj += sp_obj

                if sp_obj > theta_hat[i] + tol:
                    # build cut expression: D1 + nu1*(1 - sum_{j in D1} y_j) - sum_{k>=2} nu_k sum_{j in Dk} y_j
                    expr = cut['D1_i'] + cut['nu1_i'] * (1.0 - gp.quicksum(model.getVarByName(f"y_{j}") for j in cut['facilities_at_D1']))
                    Ki = self.K[i]
                    for k in range(2, Ki + 1):
                        facs = cut.get(f'facilities_at_D{k}', [])
                        if len(facs) > 0:
                            expr -= cut['nu_k'][k] * gp.quicksum(model.getVarByName(f"y_{j}") for j in facs)

                    theta_var = model.getVarByName(f"theta_{i}")
                    model.cbLazy(theta_var >= expr)
                    cuts_added += 1
                    if self.verbose:
                        print(f"[CB] Added cut for client {i}: SP={sp_obj:.6f} theta_hat={theta_hat[i]:.6f}")

            if self.verbose:
                print(f"[CB] MIPSOL processed. Cuts added: {cuts_added}. Total SP-object sum: {total_sp_obj:.4f}")

    def run_callback_benders(self, time_limit=None):
        master = self.build_master_for_callback()
        if time_limit is not None:
            master.Params.TimeLimit = time_limit

        start = time.time()
        # Pass the instance method as callback: Gurobi accepts a callable (model, where)
        master.optimize(self._callback)
        elapsed = time.time() - start

        # If solve produced a model with variables, use stored references to extract solution.
        try:
            y_sol = [v.X for v in self.master_y_vars]
            theta_sol = [v.X for v in self.master_theta_vars]
            obj = master.ObjVal
        except Exception:
            y_sol = None
            theta_sol = None
            obj = None

        return {
            'y': y_sol,
            'theta': theta_sol,
            'obj': obj,
            'time': elapsed,
            'status': master.status,
            'model': master
        }


if __name__ == "__main__":
    # small test instance
    np.random.seed(42)
    N = 5
    M = 7
    p_open = 2

    dist = np.random.rand(N, M) * 100.0
    K = [3] * N

    D = {}
    for i in range(N):
        sorted_dists = sorted(set(np.round(dist[i, :], 8)))
        D[i] = {k: sorted_dists[min(k - 1, len(sorted_dists) - 1)] for k in range(1, K[i] + 1)}

    b = BendersDecomposition(N=N, M=M, p_open=p_open, K=K, D=D, distance_matrix=dist, verbose=True)

    print("Running Benders via lazy callback (fixed)...")
    res = b.run_callback_benders(time_limit=60)

    print("\n=== Results ===")
    print("Status:", res['status'])
    print("Objective (master):", res['obj'])
    if res['y'] is not None:
        print("Open facilities:", [j for j in range(M) if res['y'][j] > 0.5])
    print("Elapsed time (s):", round(res['time'], 6))
