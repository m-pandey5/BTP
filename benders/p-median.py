
"""
Benders Decomposition for p-Median Problem
Based on the formulation (F3), (MP), and (SP)
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np

class BendersDecomposition:
    def __init__(self, N, M, p, K, D, distance_matrix):
        """
        Parameters:
        -----------
        N : int - Number of clients
        M : int - Number of facilities
        p : int - Number of facilities to open
        K : array-like - Number of levels for each client (K_i for each i)
        D : dict - Distance thresholds D^k for each client i and level k
        distance_matrix : 2D array - Distance from client i to facility j
        """
        self.N = N
        self.M = M
        self.p = p
        self.K = K  # K[i] is K_i for client i
        self.D = D  # D[i][k] is D_i^k for client i at level k
        self.dist = distance_matrix
        
        # Benders cuts storage
        self.benders_cuts = []
        
    def solve_master_problem(self):
        """
        Solve the Master Problem (MP):
        min sum_{i=1}^N theta_i
        s.t. sum_{j=1}^M y_j = p
             theta_i satisfies BD_i (Benders cuts)
             y_j ∈ {0, 1}
        """
        model = gp.Model("MasterProblem")
        model.Params.OutputFlag = 0
        
        # Variables
        y = model.addVars(self.M, vtype=GRB.BINARY, name="y")
        theta = model.addVars(self.N, lb=0, vtype=GRB.CONTINUOUS, name="theta")
        
        # Objective: minimize sum of theta_i
        model.setObjective(gp.quicksum(theta[i] for i in range(self.N)), GRB.MINIMIZE)
        
        # Constraint: sum y_j = p
        model.addConstr(gp.quicksum(y[j] for j in range(self.M)) == self.p, "p_facilities")
        
        # Add Benders cuts
        for cut in self.benders_cuts:
            i = cut['client']
            D1_i = cut['D1_i']
            nu1_i = cut['nu1_i']
            nu_k = cut['nu_k']
            
            # Benders cut (16):
            # theta_i >= D_i^1 + nu_i^1(1 - sum_{j:d_ij=D_i^1} y_j) 
            #            - sum_{k=2}^{K_i} nu_i^k sum_{j:d_ij=D_i^k} y_j
            
            expr = D1_i + nu1_i
            expr -= nu1_i * gp.quicksum(y[j] for j in cut['facilities_at_D1'])
            for k in range(2, self.K[i] + 1):
                if k in nu_k:
                    expr -= nu_k[k] * gp.quicksum(y[j] for j in cut[f'facilities_at_D{k}'])
            
            model.addConstr(theta[i] >= expr, f"benders_cut_{len(self.benders_cuts)}_{i}")
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            y_sol = [y[j].X for j in range(self.M)]
            theta_sol = [theta[i].X for i in range(self.N)]
            obj = model.ObjVal
            return y_sol, theta_sol, obj
        else:
            return None, None, None
    
    def solve_dual_subproblem(self, i, y_bar):
        """
        Solve the Dual Subproblem (DSP_i) for client i given y_bar:
        
        max D_i^1 + nu_i^1(1 - sum_{j:d_ij=D_i^1} y_bar_j) 
            - sum_{k=2}^{K_i} nu_i^k sum_{j:d_ij=D_i^k} y_bar_j
        s.t. nu_i^k - nu_i^{k+1} <= D_i^{k+1} - D_i^k    k ∈ [K_i - 1]
             nu_i^k >= 0                                  k ∈ [K_i]
        """
        model = gp.Model(f"DualSubproblem_{i}")
        model.Params.OutputFlag = 0
        
        # Variables: nu_i^k for k = 1, ..., K_i
        nu = model.addVars(range(1, self.K[i] + 1), lb=0, vtype=GRB.CONTINUOUS, name="nu")
        
        # Precompute facilities at each distance level
        facilities_at_D = {}
        for k in range(1, self.K[i] + 1):
            D_k = self.D[i][k]
            facilities_at_D[k] = [j for j in range(self.M) if abs(self.dist[i][j] - D_k) < 1e-6]
        
        # Objective
        obj_expr = self.D[i][1]
        obj_expr += nu[1] * (1 - sum(y_bar[j] for j in facilities_at_D[1]))
        for k in range(2, self.K[i] + 1):
            obj_expr -= nu[k] * sum(y_bar[j] for j in facilities_at_D[k])
        
        model.setObjective(obj_expr, GRB.MAXIMIZE)
        
        # Constraints: nu_i^k - nu_i^{k+1} <= D_i^{k+1} - D_i^k
        for k in range(1, self.K[i]):
            model.addConstr(nu[k] - nu[k+1] <= self.D[i][k+1] - self.D[i][k], 
                          f"dual_constraint_{k}")
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            nu_sol = {k: nu[k].X for k in range(1, self.K[i] + 1)}
            obj = model.ObjVal
            
            # Prepare cut information
            cut = {
                'client': i,
                'D1_i': self.D[i][1],
                'nu1_i': nu_sol[1],
                'nu_k': nu_sol,
                'facilities_at_D1': facilities_at_D[1]
            }
            for k in range(2, self.K[i] + 1):
                cut[f'facilities_at_D{k}'] = facilities_at_D[k]
            
            return obj, cut
        else:
            return None, None
    
    def solve(self, max_iterations=100, tolerance=1e-4):
        """
        Main Benders Decomposition algorithm
        """
        print("Starting Benders Decomposition...")
        print(f"N={self.N}, M={self.M}, p={self.p}")
        
        iteration = 0
        LB = -float('inf')  # Lower bound
        UB = float('inf')   # Upper bound
        
        while iteration < max_iterations and (UB - LB) > tolerance:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Step 1: Solve Master Problem
            y_sol, theta_sol, mp_obj = self.solve_master_problem()
            
            if y_sol is None:
                print("Master problem infeasible!")
                break
            
            LB = mp_obj
            print(f"Master Problem Objective (LB): {LB:.4f}")
            print(f"y solution: {[j for j in range(self.M) if y_sol[j] > 0.5]}")
            
            # Step 2: Solve Dual Subproblems for each client
            total_sp_obj = 0
            cuts_added = 0
            
            for i in range(self.N):
                sp_obj, cut = self.solve_dual_subproblem(i, y_sol)
                
                if sp_obj is None:
                    print(f"Subproblem {i} infeasible!")
                    continue
                
                total_sp_obj += sp_obj
                
                # Check if we need to add a cut (optimality check)
                if sp_obj > theta_sol[i] + tolerance:
                    self.benders_cuts.append(cut)
                    cuts_added += 1
                    print(f"Client {i}: Adding cut (SP obj: {sp_obj:.4f}, theta: {theta_sol[i]:.4f})")
            
            # Update upper bound
            UB = total_sp_obj
            print(f"Total Subproblem Objective (UB): {UB:.4f}")
            print(f"Gap: {UB - LB:.4f}, Cuts added: {cuts_added}")
            
            # Check convergence
            if cuts_added == 0:
                print(f"\nConverged! No cuts added.")
                break
        
        print(f"\n=== Final Solution ===")
        print(f"Optimal Value: {LB:.4f}")
        print(f"Iterations: {iteration}")
        print(f"Final Gap: {UB - LB:.4f}")
        print(f"Open facilities: {[j for j in range(self.M) if y_sol[j] > 0.5]}")
        
        return y_sol, theta_sol, LB


# Example usage
if __name__ == "__main__":
    # Small test instance
    N = 5  # clients
    M = 7  # facilities
    p = 2  # facilities to open
    
    # Random distance matrix
    np.random.seed(42)
    distance_matrix = np.random.rand(N, M) * 100
    
    # Define K_i (number of distance levels for each client)
    K = [3, 3, 3, 3, 3]  # Each client has 3 levels
    
    # Define D_i^k (distance thresholds)
    D = {}
    for i in range(N):
        # Sort distances for client i to get levels
        sorted_dists = sorted(set(distance_matrix[i, :]))
        D[i] = {k: sorted_dists[min(k-1, len(sorted_dists)-1)] 
                for k in range(1, K[i] + 1)}
    
    # Create and solve
    benders = BendersDecomposition(N, M, p, K, D, distance_matrix)
    y_sol, theta_sol, optimal_value = benders.solve(max_iterations=50)


