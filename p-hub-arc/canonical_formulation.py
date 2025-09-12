import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_hub_arc_canonical(n, p, W, D):
    """
    Canonical Representation for p-Hub-Arc Problem
    
    n: number of nodes
    p: number of hub arcs to open
    W: flow weight matrix (n x n), W[i][j] is weight between node i and j
    D: distance matrix (n x n), D[k][m] is cost of hub arc (k,m)
    """
    N = range(n)
    H = [(k, m) for k in N for m in N if k != m]  # hub arcs k ≠ m
    
    # Step 1: Create sorted cost vectors D_ij for each OD pair
    D_vectors = {}
    G = {}
    
    for i in N:
        for j in N:
            # Calculate all costs c_{ij,km} = W[i][j] * D[k][m] for this OD pair
            costs = []
            for (k, m) in H:
                cost = W[i][j] * D[k][m]
                costs.append(cost)
            
            # Sort and remove duplicates, add 0 at beginning
            unique_costs = sorted(list(set(costs + [0])))
            D_vectors[(i, j)] = unique_costs
            G[(i, j)] = len(unique_costs)
    
    # Create model
    model = gp.Model("HubArc_Canonical")
    
    # Decision variables
    # z_ijk: cumulative variables
    z = model.addVars([(i, j, k) for i in N for j in N 
                       for k in range(1, G[(i, j)] + 1)],
                      vtype=GRB.CONTINUOUS, lb=0, name="z")
    
    # y_km: hub arc open variables
    y = model.addVars(H, vtype=GRB.BINARY, name="y")
    
    # Objective: minimize sum of (D_ijk - D_ij,k-1) * z_ijk
    obj_expr = gp.LinExpr()
    for i in N:
        for j in N:
            for k in range(2, G[(i, j)] + 1):  # k starts from 2
                D_ijk = D_vectors[(i, j)][k-1]      # k-th element (0-indexed)
                D_ij_k_minus_1 = D_vectors[(i, j)][k-2]  # (k-1)-th element
                diff = D_ijk - D_ij_k_minus_1
                obj_expr += diff * z[i, j, k]
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Constraint 1: exactly p hub arcs open
    model.addConstr(gp.quicksum(y[k, m] for (k, m) in H) == p, 
                   name="pHubArcs")
    
    # Constraint 2: z_ijk + sum of y_k'm' for arcs with cost < D_ijk >= 1
    for i in N:
        for j in N:
            for k in range(1, G[(i, j)] + 1):
                D_ijk = D_vectors[(i, j)][k-1]  # k-th element (0-indexed)
                
                # Find hub arcs with cost < D_ijk
                cheap_arcs = []
                for (kp, mp) in H:
                    c_ij_kpmp = W[i][j] * D[kp][mp]
                    if c_ij_kpmp < D_ijk:
                        cheap_arcs.append((kp, mp))
                
                # Add constraint: z_ijk + sum(y_k'm' for cheap arcs) >= 1
                if cheap_arcs:
                    model.addConstr(
                        z[i, j, k] + gp.quicksum(y[kp, mp] for (kp, mp) in cheap_arcs) >= 1,
                        name=f"canonical[{i},{j},{k}]"
                    )
                else:
                    # If no cheap arcs, z_ijk >= 1
                    model.addConstr(z[i, j, k] >= 1, 
                                   name=f"canonical[{i},{j},{k}]")
    
    # Optimize
    model.optimize()
    
    # Extract solution
    if model.status == GRB.OPTIMAL:
        chosen_arcs = [(k, m) for (k, m) in H if y[k, m].x > 0.5]
        z_values = {(i, j, k): z[i, j, k].x for i in N for j in N 
                   for k in range(1, G[(i, j)] + 1)}
        
        print("Optimal objective value:", model.objVal)
        print("Chosen hub arcs:", chosen_arcs)
        print("Sample z values:", {key: val for key, val in list(z_values.items())[:5]})
        return chosen_arcs, z_values
    else:
        print("No optimal solution found.")
        return None, None

# Example usage
if __name__ == "__main__":
    n = 3
    p = 2
    W = [
        [0, 2, 3],
        [2, 0, 4],
        [3, 4, 0]
    ]
    D = [
        [0, 5, 2],
        [5, 0, 3],
        [2, 3, 0]
    ]
    solve_hub_arc_canonical(n, p, W, D)
