
"""
Verification Script: Benders vs F3 Direct Formulation for p-Median Problem
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time


def solve_F3_formulation(N, M, p, K, D, distance_matrix):
    """
    Solve p-median using direct F3 formulation from the image:

    min sum_i [ D_i^1 + sum_{k=1}^{K_i-1} (D_i^{k+1} - D_i^k) * z_i^k ]
    s.t.
        sum_j y_j = p                                    (11)
        z_i^1 + sum_{j: d_ij=D_i^1} y_j >= 1            (12)
        z_i^k + sum_{j: d_ij=D_i^k} y_j >= z_i^{k-1}    (13)
        z_i^k >= 0                                       (14)
        y_j in {0,1}                                     (15)
    """
    model = gp.Model("F3_pMedian")
    model.Params.OutputFlag = 1

    # Variables
    y = model.addVars(M, vtype=GRB.BINARY, name="y")
    z = {}
    for i in range(N):
        for k in range(1, K[i]+1):
            z[i,k] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"z_{i}_{k}")

    # Precompute facilities at each distance level
    facilities_at_distance = {}
    for i in range(N):
        facilities_at_distance[i] = {}
        for k in range(1, K[i]+1):
            Dk = D[i][k]
            facilities_at_distance[i][k] = [j for j in range(M) 
                                           if abs(distance_matrix[i,j] - Dk) < 1e-8]

    # Objective (10): min sum_i [ D_i^1 + sum_{k=1}^{K_i-1} (D_i^{k+1} - D_i^k) * z_i^k ]
    obj = gp.quicksum(D[i][1] for i in range(N))
    for i in range(N):
        for k in range(1, K[i]):
            obj += (D[i][k+1] - D[i][k]) * z[i,k]

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (11): sum_j y_j = p
    model.addConstr(gp.quicksum(y[j] for j in range(M)) == p, name="p_facilities")

    # Constraint (12): z_i^1 + sum_{j: d_ij=D_i^1} y_j >= 1 for all i
    for i in range(N):
        facs_at_D1 = facilities_at_distance[i][1]
        model.addConstr(z[i,1] + gp.quicksum(y[j] for j in facs_at_D1) >= 1, 
                       name=f"coverage_{i}_k1")

    # Constraint (13): z_i^k + sum_{j: d_ij=D_i^k} y_j >= z_i^{k-1} for k=2,...,K_i
    for i in range(N):
        for k in range(2, K[i]+1):
            facs_at_Dk = facilities_at_distance[i][k]
            model.addConstr(z[i,k] + gp.quicksum(y[j] for j in facs_at_Dk) >= z[i,k-1],
                           name=f"coverage_{i}_k{k}")

    model.update()
    start = time.time()
    model.optimize()
    elapsed = time.time() - start

    y_sol = [y[j].X for j in range(M)] if model.status == GRB.OPTIMAL else None
    z_sol = {(i,k): z[i,k].X for i in range(N) for k in range(1, K[i]+1)} if model.status == GRB.OPTIMAL else None

    return {
        'objective': model.ObjVal if model.status == GRB.OPTIMAL else None,
        'y': y_sol,
        'z': z_sol,
        'time': elapsed,
        'status': model.status
    }


def solve_benders_pmedian(N, M, p, K, D, distance_matrix, verbose=False):
    """
    Benders decomposition for p-median (from previous script)
    """
    from benderscallback_fixed import BendersDecomposition

    bd = BendersDecomposition(
        N=N, M=M, p_open=p, K=K, D=D, 
        distance_matrix=distance_matrix, 
        verbose=verbose
    )

    result = bd.run_callback_benders(time_limit=300)
    return result


def compare_solutions(f3_result, benders_result, tol=1e-4):
    """
    Compare F3 and Benders solutions
    """
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    # Objective values
    f3_obj = f3_result['objective']
    benders_obj = benders_result['obj']

    print(f"F3 Objective:      {f3_obj:.6f}")
    print(f"Benders Objective: {benders_obj:.6f}")
    print(f"Difference:        {abs(f3_obj - benders_obj):.8f}")

    if abs(f3_obj - benders_obj) < tol:
        print("✓ Objectives match within tolerance!")
    else:
        print("✗ Objectives differ significantly!")

    # Facility opening decisions
    f3_open = set([j for j, val in enumerate(f3_result['y']) if val > 0.5])
    benders_open = set([j for j, val in enumerate(benders_result['y']) if val > 0.5])

    print(f"\nF3 Open Facilities:      {sorted(f3_open)}")
    print(f"Benders Open Facilities: {sorted(benders_open)}")

    if f3_open == benders_open:
        print("✓ Facility decisions match exactly!")
    else:
        print(f"✗ Facility decisions differ!")
        print(f"  Only in F3:      {sorted(f3_open - benders_open)}")
        print(f"  Only in Benders: {sorted(benders_open - f3_open)}")

    # Solution times
    print(f"\nF3 Time:      {f3_result['time']:.4f} seconds")
    print(f"Benders Time: {benders_result['time']:.4f} seconds")
    print("="*60)

    return abs(f3_obj - benders_obj) < tol and f3_open == benders_open


if __name__ == "__main__":
    # Test instance
    np.random.seed(42)
    N = 5  # clients
    M = 5 # potential facilities
    p = 3   # facilities to open

    # Generate distance matrix
    dist = np.random.rand(N, M) * 100.0

    # Define K_i (number of distance levels per client)
    K = {i: min(4, M) for i in range(N)}

    # Compute D_i^k (sorted unique distances for each client)
    D = {}
    for i in range(N):
        sorted_dists = sorted(set(np.round(dist[i, :], 8)))
        D[i] = {k: sorted_dists[min(k-1, len(sorted_dists)-1)] 
                for k in range(1, K[i]+1)}

    print("="*60)
    print("P-MEDIAN PROBLEM: F3 vs BENDERS VERIFICATION")
    print("="*60)
    print(f"Clients (N):             {N}")
    print(f"Facilities (M):          {M}")
    print(f"Facilities to open (p):  {p}")
    print("="*60)

    # Solve using F3 formulation
    print("\nSolving with F3 formulation...")
    f3_result = solve_F3_formulation(N, M, p, K, D, dist)

    # Solve using Benders decomposition
    print("\nSolving with Benders decomposition...")
    benders_result = solve_benders_pmedian(N, M, p, K, D, dist, verbose=True)

    # Compare
    match = compare_solutions(f3_result, benders_result)

    if match:
        print("\n✓✓✓ SUCCESS: Both methods produce identical solutions! ✓✓✓")
    else:
        print("\n✗✗✗ WARNING: Solutions differ - investigate further ✗✗✗")