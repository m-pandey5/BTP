
import gurobipy as gp
from gurobipy import GRB
import numpy as np

class BendersDecomposition:
    def __init__(self, N, M, p, K, D, distance_matrix):
        self.N = N
        self.M = M
        self.p = p
        self.K = K
        self.D = D
        self.dist = distance_matrix
        self.opt_cuts = []   # List of optimality cuts
        self.feas_cuts = []  # List of feasibility cuts

    def solve_master_problem(self):
        model = gp.Model("Master_BP")
        model.Params.OutputFlag = 0

        y = model.addVars(self.M, vtype=GRB.BINARY, name="y")
        theta = model.addVars(self.N, lb=0, vtype=GRB.CONTINUOUS, name="theta")
        model.setObjective(gp.quicksum(theta[i] for i in range(self.N)), GRB.MINIMIZE)
        model.addConstr(gp.quicksum(y[j] for j in range(self.M)) == self.p)

        # Add optimality cuts (from subproblem duals)
        for cut in self.opt_cuts:
            i = cut['client']
            D1_i = cut['D1_i']
            nu1_i = cut['nu1_i']
            nu_k = cut['nu_k']
            expr = D1_i + nu1_i
            expr -= nu1_i * gp.quicksum(y[j] for j in cut['facilities_at_D1'])
            for k in range(2, self.K[i]+1):
                if k in nu_k:
                    expr -= nu_k[k] * gp.quicksum(y[j] for j in cut[f'facilities_at_D{k}'])
            model.addConstr(theta[i] >= expr)

        # Add feasibility cuts
        for feas in self.feas_cuts:
            i, k, faclist = feas['client'], feas['level'], feas['facilities']
            model.addConstr(gp.quicksum(y[j] for j in faclist) >= 1)

        model.optimize()
        if model.status == GRB.OPTIMAL:
            y_sol = [y[j].X for j in range(self.M)]
            theta_sol = [theta[i].X for i in range(self.N)]
            obj = model.ObjVal
            return y_sol, theta_sol, obj
        else:
            return None, None, None
    def solve_dual_subproblem(self, i, y_bar):

    # Check if ANY level has coverage
        has_coverage = False
        for k in range(1, self.K[i]+1):
            faclist = [j for j in range(self.M) if abs(self.dist[i][j] - self.D[i][k]) < 1e-6]
            if sum(y_bar[j] for j in faclist) > 1e-6:
                has_coverage = True
                break
        
        if not has_coverage:
            # Infeasible: add feasibility cut for level 1
            faclist = [j for j in range(self.M) if abs(self.dist[i][j] - self.D[i][1]) < 1e-6]
            return None, None, {'client': i, 'level': 1, 'facilities': faclist}
        
        # Has coverage - solve the dual problem
        model = gp.Model(f"DualSub_{i}")
        model.Params.OutputFlag = 0
        
        # Dual variables
        nu = model.addVars(range(1, self.K[i]+1), lb=0, vtype=GRB.CONTINUOUS, name="nu")
        
        # Build facility sets for each distance level
        fac_D = {}
        for kk in range(1, self.K[i]+1):
            D_kk = self.D[i][kk]
            fac_D[kk] = [j for j in range(self.M) if abs(self.dist[i][j] - D_kk) < 1e-6]
        
        # Dual objective: D[i][1] + nu[1]*(1 - sum y_bar[j]) - sum_k>=2 nu[k]*sum y_bar[j]
        obj_expr = self.D[i][1]
        obj_expr += nu[1] * (1 - sum(y_bar[j] for j in fac_D[1]))
        for kk in range(2, self.K[i]+1):
            obj_expr -= nu[kk] * sum(y_bar[j] for j in fac_D[kk])
        
        model.setObjective(obj_expr, GRB.MAXIMIZE)
        
        # Dual constraints: nu[k] - nu[k+1] <= D[i][k+1] - D[i][k]
        for kk in range(1, self.K[i]):
            model.addConstr(nu[kk] - nu[kk+1] <= self.D[i][kk+1] - self.D[i][kk])
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            nu_sol = {kk: nu[kk].X for kk in range(1, self.K[i]+1)}
            
            # Build the optimality cut
            cut = {
                'client': i,
                'D1_i': self.D[i][1],
                'nu1_i': nu_sol[1],
                'nu_k': nu_sol,
                'facilities_at_D1': fac_D[1]
            }
            for kk in range(2, self.K[i]+1):
                cut[f'facilities_at_D{kk}'] = fac_D[kk]
            
            return obj_expr.getValue(), cut, None
        else:
            # Dual is infeasible (shouldn't happen if primal was feasible)
            faclist = [j for j in range(self.M) if abs(self.dist[i][j] - self.D[i][1]) < 1e-6]
            return None, None, {'client': i, 'level': 1, 'facilities': faclist}


    def solve(self, max_iterations=100, tolerance=1e-6):
        print("Starting Benders Decomposition (with feasibility cuts)...")
        iteration = 0
        LB = -float('inf')
        UB = float('inf')

        while iteration < max_iterations and (UB - LB) > tolerance:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            y_sol, theta_sol, mp_obj = self.solve_master_problem()
            if y_sol is None:
                print("Master infeasible!")
                break
            LB = mp_obj
            print(f"Master Problem Objective (LB): {LB:.4f}")
            print(f"y solution: {[j for j in range(self.M) if y_sol[j] > 0.5]}")
            total_sp_obj = 0
            cuts_added = 0
            feas_added = 0
            for i in range(self.N):
                sp_obj, opt_cut, feas_cut = self.solve_dual_subproblem(i, y_sol)
                if feas_cut is not None:
                    self.feas_cuts.append(feas_cut)
                    print(f"Client {i}: Added FEASIBILITY cut (layer {feas_cut['level']}, facilities {feas_cut['facilities']})")
                    feas_added += 1
                    continue
                if sp_obj is not None:
                    total_sp_obj += sp_obj
                    if opt_cut is not None:
                        # Standard Benders optimality cut
                        if sp_obj > theta_sol[i] + tolerance:
                            self.opt_cuts.append(opt_cut)
                            cuts_added += 1
                            print(f"Client {i}: Added OPTIMALITY cut (SP obj: {sp_obj:.4f}, theta: {theta_sol[i]:.4f})")
            UB = total_sp_obj
            print(f"Total SP Objective (UB): {UB:.4f}")
            print(f"Gap: {UB-LB:.4f}, Cuts added: {cuts_added}, Feas cuts: {feas_added}")
            if (cuts_added + feas_added) == 0:
                print("\nConverged! No cuts added.")
                break

        print("\n=== Final Solution ===")
        print(f"Optimal Value: {LB:.4f}")
        print(f"Iterations: {iteration}")
        print(f"Open facilities: {[j for j in range(self.M) if y_sol[j] > 0.5]}")
        return y_sol, theta_sol, LB

# EXAMPLE USAGE (same as before)
if __name__ == "__main__":
    N = 5
    M = 7
    p = 2
    np.random.seed(42)
    distance_matrix = np.random.rand(N, M) * 100
    K = [3] * N
    D = {}
    for i in range(N):
        sorted_dists = sorted(set(distance_matrix[i, :]))
        D[i] = {k: sorted_dists[min(k-1, len(sorted_dists)-1)] for k in range(1, K[i] + 1)}
    # COMPARE WITH CANONICAL FORM
    def solve_canonical_pmedian(N, M, p, K, D, distance_matrix):
        model = gp.Model("Canonical_p_median")
        model.Params.OutputFlag = 0
        y = model.addVars(M, vtype=GRB.BINARY, name="y")
        z = {}
        for i in range(N):
            for k in range(1, K[i] + 1):
                z[i, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"z_{i}_{k}")
        obj = gp.quicksum(
            D[i][1]
            + gp.quicksum((D[i][k+1] - D[i][k]) * z[i, k] for k in range(1, K[i]))
            for i in range(N)
        )
        model.setObjective(obj, GRB.MINIMIZE)
        model.addConstr(gp.quicksum(y[j] for j in range(M)) == p, name="open_p")
        for i in range(N):
            fac_D1 = [j for j in range(M) if abs(distance_matrix[i][j] - D[i][1]) < 1e-6]
            model.addConstr(
                z[i, 1] + gp.quicksum(y[j] for j in fac_D1) >= 1,
                name=f"cover1_{i}"
            )
        for i in range(N):
            for k in range(2, K[i] + 1):
                fac_Dk = [j for j in range(M) if abs(distance_matrix[i][j] - D[i][k]) < 1e-6]
                model.addConstr(
                    z[i, k] + gp.quicksum(y[j] for j in fac_Dk) >= z[i, k-1],
                    name=f"covk_{i}_{k}"
                )
        model.optimize()
        if model.status == GRB.OPTIMAL:
            y_sol = [y[j].X for j in range(M)]
            fac_open = [j for j in range(M) if y_sol[j] > 0.5]
            return model.objVal, fac_open
        else:
            return None, None

    # Run canonical
    obj_can, fac_can = solve_canonical_pmedian(N, M, p, K, D, distance_matrix)
    print("\nCanonical formulation:")
    print("Objective:", obj_can)
    print("Open facilities:", fac_can)
    print()
    # Run Benders w/ feasibility cuts
    benders = BendersDecomposition(N, M, p, K, D, distance_matrix)
    y_sol, theta_sol, obj_benders = benders.solve(max_iterations=100, tolerance=1e-6)
    fac_benders = [j for j in range(M) if y_sol[j] > 0.5]
    print("Benders decomposition (with feasibility cuts):")
    print("Objective:", obj_benders)
    print("Open facilities:", fac_benders)
    print()
    print("---\nAgreement check:")
    if abs(obj_can - obj_benders) < 1e-4 and set(fac_can) == set(fac_benders):
        print("✓ Both methods match: same objective and open facilities.")
    else:
        print("⚠ Mismatch found!")
        print(f"  Canonical obj: {obj_can}, Facilities: {fac_can}")
        print(f"  Benders obj:   {obj_benders}, Facilities: {fac_benders}")

