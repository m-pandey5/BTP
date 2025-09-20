import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import random

def solve_hub_arc_classical(n, p, W, D):
    """Classical Formulation for p-Hub-Arc Problem"""
    N = range(n)
    H = [(k, m) for k in N for m in N if k != m]  # hub arcs k ≠ m

    model = gp.Model("HubArc_Classical")
    model.setParam('OutputFlag', 0)  # Suppress output

    # Decision variables
    x = model.addVars([(i, j, k, m) for i in N for j in N for (k, m) in H],
                      vtype=GRB.BINARY, name="x")
    y = model.addVars(H, vtype=GRB.BINARY, name="y")

    # Objective
    model.setObjective(
        gp.quicksum(x[i, j, k, m] * W[i][j] * (D[i][k]+D[k][m] + D[m][j])
                    for i in N for j in N for (k, m) in H),
        GRB.MINIMIZE
    )

    # Constraints
    for i in N:
        for j in N:
            model.addConstr(
                gp.quicksum(x[i, j, k, m] for (k, m) in H) == 1,
                name=f"Assign[{i},{j}]"
            )

    for i in N:
        for j in N:
            for (k, m) in H:
                model.addConstr(
                    x[i, j, k, m] <= y[k, m],
                    name=f"ValidAssign[{i},{j},{k},{m}]"
                )

    model.addConstr(
        gp.quicksum(y[k, m] for (k, m) in H) == p,
        name="pHubArcs"
    )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        chosen_arcs = [(k, m) for (k, m) in H if y[k, m].x > 0.5]
        return model.objVal, chosen_arcs, model.runtime
    else:
        return None, None, model.runtime

def solve_hub_arc_canonical(n, p, W, D):
    """Canonical Representation for p-Hub-Arc Problem"""
    N = range(n)
    H = [(k, m) for k in N for m in N if k != m]  # hub arcs k ≠ m
    
    # Create sorted cost vectors D_ij for each OD pair
    D_vectors = {}
    G = {}
    
    for i in N:
        for j in N:
            costs = []
            for (k, m) in H:
                cost = W[i][j] * (D[i][k]+D[k][m]+D[m][j])
                costs.append(cost)
            
            unique_costs = sorted(list(set(costs + [0])))
            D_vectors[(i, j)] = unique_costs
            G[(i, j)] = len(unique_costs)
    
    model = gp.Model("HubArc_Canonical")
    model.setParam('OutputFlag', 0)  # Suppress output
    
    # Decision variables
    z = model.addVars([(i, j, k) for i in N for j in N 
                       for k in range(2, G[(i, j)] + 1)],
                      vtype=GRB.CONTINUOUS, lb=0, name="z")
    
    y = model.addVars(H, vtype=GRB.BINARY, name="y")
    
    # Objective
    obj_expr = gp.LinExpr()
    for i in N:
        for j in N:
            for k in range(2, G[(i, j)] + 1):  # k starts from 2
                D_ijk = D_vectors[(i, j)][k-1]
                D_ij_k_minus_1 = D_vectors[(i, j)][k-2]
                diff = D_ijk - D_ij_k_minus_1
                obj_expr += diff * z[i, j, k]
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Constraints
    model.addConstr(gp.quicksum(y[k, m] for (k, m) in H) == p, 
                   name="pHubArcs")
    
    for i in N:
        for j in N:
            for k in range(2, G[(i, j)] + 1):
                D_ijk = D_vectors[(i, j)][k-1]
                
                cheap_arcs = []
                for (kp, mp) in H:
                    c_ij_kpmp = W[i][j] * (D[i][kp] + D[kp][mp] + D[mp][j])
                    if c_ij_kpmp < D_ijk:
                        cheap_arcs.append((kp, mp))
                
                if cheap_arcs:
                    model.addConstr(
                        z[i, j, k] + gp.quicksum(y[kp, mp] for (kp, mp) in cheap_arcs) >= 1,
                        name=f"canonical[{i},{j},{k}]"
                    )
                else:
                    model.addConstr(z[i, j, k] >= 1, 
                                   name=f"canonical[{i},{j},{k}]")
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        chosen_arcs = [(k, m) for (k, m) in H if y[k, m].x > 0.5]
        return model.objVal, chosen_arcs, model.runtime
    else:
        return None, None, model.runtime

# def generate_test_case(n, density=0.7):
#     """Generate random test case"""
#     # Generate flow matrix W
#     W = [[0 for _ in range(n)] for _ in range(n)]
#     for i in range(n):
#         for j in range(n):
#             if i != j and random.random() < density:
#                 W[i][j] = random.randint(1, 10)
    
#     # Generate distance matrix D
#     D = [[0 for _ in range(n)] for _ in range(n)]
#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 D[i][j] = random.randint(1, 15)
    
#     return W, D

def generate_large_test_case(n=50, density=0.05):
    W = [[0]*n for _ in range(n)]
    D = [[0]*n for _ in range(n)]
    # Sparse flow matrix W
    for i in range(n):
        for j in range(n):
            if i != j and random.random() < density:
                W[i][j] = random.randint(1, 10)
    # Distance matrix D with random distances 1-50, symmetric for realism
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = 0
            elif j > i:
                val = random.randint(1, 50)
                D[i][j] = val
                D[j][i] = val
    return W, D

# def run_comprehensive_tests():
#     """Run 10 test cases comparing Classical vs Canonical formulations"""
    
#     test_cases = [
#         # Test Case 1: Small instance
#         {
#             'n': 3, 'p': 2,
#             'W': [[0, 2, 3], [2, 0, 4], [3, 4, 0]],
#             'D': [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
#         },
        
#         # Test Case 2: Symmetric instance
#         {
#             'n': 4, 'p': 3,
#             'W': [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]],
#             'D': [[0, 4, 6, 8], [4, 0, 2, 4], [6, 2, 0, 3], [8, 4, 3, 0]]
#         },
        
#         # Test Case 3: High flow variation
#         {
#             'n': 3, 'p': 1,
#             'W': [[0, 10, 1], [1, 0, 10], [10, 1, 0]],
#             'D': [[0, 2, 8], [2, 0, 3], [8, 3, 0]]
#         },
        
#         # Test Case 4: Uniform costs
#         {
#             'n': 4, 'p': 2,
#             'W': [[0, 3, 3, 3], [3, 0, 3, 3], [3, 3, 0, 3], [3, 3, 3, 0]],
#             'D': [[0, 5, 5, 5], [5, 0, 5, 5], [5, 5, 0, 5], [5, 5, 5, 0]]
#         },
        
#         # Test Case 5: Single hub arc
#         {
#             'n': 3, 'p': 1,
#             'W': [[0, 5, 2], [3, 0, 7], [1, 4, 0]],
#             'D': [[0, 1, 9], [8, 0, 2], [3, 6, 0]]
#         }
#     ]
    
#     # Generate 5 more random test cases
#     for i in range(5):
#         n = random.choice([3, 4, 5])
#         p = random.randint(1, min(3, n*(n-1)//2))
#         W, D = generate_test_case(n)
#         test_cases.append({'n': n, 'p': p, 'W': W, 'D': D})
    
#     print("=" * 80)
#     print("COMPREHENSIVE TESTING: Classical vs Canonical Hub Arc Formulations")
#     print("=" * 80)
#     print(f"{'Test':<5} {'n':<3} {'p':<3} {'Classical Obj':<15} {'Canonical Obj':<15} {'Match':<8} {'Classical Time':<15} {'Canonical Time':<15}")
#     print("-" * 80)
    
#     total_tests = 0
#     successful_tests = 0
    
#     for i, test in enumerate(test_cases, 1):
#         n, p, W, D = test['n'], test['p'], test['W'], test['D']
        
#         try:
#             # Solve with Classical formulation
#             start_time = time.time()
#             classical_obj, classical_arcs, classical_time = solve_hub_arc_classical(n, p, W, D)
            
#             # Solve with Canonical formulation
#             start_time = time.time()
#             canonical_obj, canonical_arcs, canonical_time = solve_hub_arc_canonical(n, p, W, D)
            
#             total_tests += 1
            
#             # Check if both found solutions
#             if classical_obj is not None and canonical_obj is not None:
#                 match = "✓" if abs(classical_obj - canonical_obj) < 1e-6 else "✗"
#                 if abs(classical_obj - canonical_obj) < 1e-6:
#                     successful_tests += 1
                
#                 print(f"{i:<5} {n:<3} {p:<3} {classical_obj:<15.3f} {canonical_obj:<15.3f} {match:<8} {classical_time:<15.4f} {canonical_time:<15.4f}")
                
#                 # Print chosen arcs for first few tests
#                 if i <= 3:
#                     print(f"      Classical arcs: {classical_arcs}")
#                     print(f"      Canonical arcs: {canonical_arcs}")
#                     print()
#             else:
#                 print(f"{i:<5} {n:<3} {p:<3} {'INFEASIBLE':<15} {'INFEASIBLE':<15} {'N/A':<8} {classical_time:<15.4f} {canonical_time:<15.4f}")
        
#         except Exception as e:
#             print(f"{i:<5} {n:<3} {p:<3} {'ERROR':<15} {'ERROR':<15} {'✗':<8} {'N/A':<15} {'N/A':<15}")
#             print(f"      Error: {str(e)}")
    
#     print("-" * 80)
#     print(f"SUMMARY: {successful_tests}/{total_tests} tests passed")
#     print(f"Success Rate: {100*successful_tests/total_tests:.1f}%")
#     print("=" * 80)

# if __name__ == "__main__":
#     run_comprehensive_tests()
def run_comprehensive_tests(min_n=3, max_n=30, density=0.2):
    """
    Run test cases for n in range [min_n, max_n] with different p,
    comparing Classical vs Canonical Hub-Arc formulations.
    """
    
    print("=" * 90)
    print("TESTING CLASSICAL VS CANONICAL HUB-ARC FORMULATIONS")
    print("=" * 90)
    print(f"{'Test':<5} {'n':<3} {'p':<3} {'Classical Obj':<15} {'Canonical Obj':<15} {'Match':<8} {'Classical Time':<15} {'Canonical Time':<15}")
    print("-" * 90)
    
    test_id = 0
    total_tests = 0
    successful_tests = 0
    
    for n in range(min_n, max_n+1):
        max_p = min(5, n*(n-1)//2)  # ensure p is feasible
        for p in range(1, max_p+1):
            total_tests += 1
            test_id += 1
            
            # Generate random test case
            W = [[0]*n for _ in range(n)]
            D = [[0]*n for _ in range(n)]
            
            for i in range(n):
                for j in range(n):
                    if i != j and random.random() < density:
                        W[i][j] = random.randint(1, 10)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        D[i][j] = 0
                    elif j > i:
                        val = random.randint(1, 15)
                        D[i][j] = val
                        D[j][i] = val
            
            # Solve Classical
            start = time.time()
            classical_obj, classical_arcs, classical_time = solve_hub_arc_classical(n, p, W, D)
            classical_time = time.time() - start
            
            # Solve Canonical
            start = time.time()
            canonical_obj, canonical_arcs, canonical_time = solve_hub_arc_canonical(n, p, W, D)
            canonical_time = time.time() - start
            
            # Compare
            match = "✓" if classical_obj is not None and canonical_obj is not None and abs(classical_obj - canonical_obj) < 1e-6 else "✗"
            if match == "✓":
                successful_tests += 1
            
            print(f"{test_id:<5} {n:<3} {p:<3} {classical_obj:<15} {canonical_obj:<15} {match:<8} {classical_time:<15.4f} {canonical_time:<15.4f}")
    
    print("-" * 90)
    print(f"SUMMARY: {successful_tests}/{total_tests} tests matched")
    print(f"Success Rate: {100*successful_tests/total_tests:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    run_comprehensive_tests()
# def run_random_n_test_cases(min_n=10, max_n=50, density=0.05):
#     n = random.randint(min_n, max_n)
#     p = max(1, min(10, n // 5))  # example: p is roughly proportional to n

#     print(f"\nRunning test with random n={n}, p={p}, density={density}")

#     W, D = generate_large_test_case(n, density)

#     # Solve canonical only for large problem
#     canonical_obj, canonical_arcs, canonical_time = solve_hub_arc_canonical(n, p, W, D)

#     print(f"Canonical formulation objective: {canonical_obj}")
#     print(f"Solution time: {canonical_time:.2f} seconds")
#     print(f"Number of selected arcs: {len(canonical_arcs)}")


# if __name__ == "__main__":
#     run_random_n_test_cases()