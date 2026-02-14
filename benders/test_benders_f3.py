"""
Test: Benders Decomposition (Polynomial Separation) vs F3 Formulation

Compares the new_model implementation (Algorithms 1, 2, 3 + Phase 2)
with the direct F3 formulation for the p-median problem.

Run: python test_benders_f3.py
"""

import sys
import numpy as np
import time

import os
# Add benders dir for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

import gurobipy as gp
from gurobipy import GRB

# F3 formulation (from p-median.py)
def solve_F3_formulation(N, M, p, K, D, distance_matrix):
    """
    Solve p-median using direct F3 formulation.
    """
    model = gp.Model("F3_pMedian")
    model.Params.OutputFlag = 0

    y = model.addVars(M, vtype=GRB.BINARY, name="y")
    z = {}
    for i in range(N):
        for k in range(1, K[i] + 1):
            z[i, k] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"z_{i}_{k}")

    facilities_at_distance = {}
    for i in range(N):
        facilities_at_distance[i] = {}
        for k in range(1, K[i] + 1):
            Dk = D[i][k]
            facilities_at_distance[i][k] = [
                j
                for j in range(M)
                if abs(distance_matrix[i, j] - Dk) < 1e-8
            ]

    obj = gp.quicksum(D[i][1] for i in range(N))
    for i in range(N):
        for k in range(1, K[i]):
            obj += (D[i][k + 1] - D[i][k]) * z[i, k]

    model.setObjective(obj, GRB.MINIMIZE)
    model.addConstr(gp.quicksum(y[j] for j in range(M)) == p, name="p_facilities")

    for i in range(N):
        facs_at_D1 = facilities_at_distance[i][1]
        model.addConstr(
            z[i, 1] + gp.quicksum(y[j] for j in facs_at_D1) >= 1,
            name=f"coverage_{i}_k1",
        )

    for i in range(N):
        for k in range(2, K[i] + 1):
            facs_at_Dk = facilities_at_distance[i][k]
            model.addConstr(
                z[i, k] + gp.quicksum(y[j] for j in facs_at_Dk) >= z[i, k - 1],
                name=f"coverage_{i}_k{k}",
            )

    model.update()
    start = time.time()
    model.optimize()
    elapsed = time.time() - start

    if model.status == GRB.OPTIMAL:
        y_sol = [y[j].X for j in range(M)]
        return {
            "objective": model.ObjVal,
            "y": y_sol,
            "facilities": sorted([j for j in range(M) if y_sol[j] > 0.5]),
            "time": elapsed,
            "status": "OPTIMAL",
        }
    else:
        return {
            "objective": None,
            "y": None,
            "facilities": None,
            "time": elapsed,
            "status": "FAILED",
        }


def build_instance(N, M, p, seed=None):
    """Build test instance with K and D (all unique distance levels)."""
    if seed is not None:
        np.random.seed(seed)
    dist = np.random.rand(N, M) * 100.0
    K = {}
    D = {}
    for i in range(N):
        unique = sorted(set(np.round(dist[i, :], 8)))
        K[i] = len(unique)
        D[i] = {k: unique[k - 1] for k in range(1, K[i] + 1)}
    return dist, K, D


def run_test(test_id, N, M, p, seed=None):
    """Run single test: F3 vs Benders."""
    from new_model import solve_benders_pmedian

    dist, K, D = build_instance(N, M, p, seed)

    print(f"\n{'='*60}")
    print(f"Test {test_id}: N={N}, M={M}, p={p}")
    print("="*60)

    # F3
    print("Solving F3...", end=" ", flush=True)
    f3_res = solve_F3_formulation(N, M, p, K, D, dist)
    print(f"done. Obj={f3_res['objective']:.4f}, Time={f3_res['time']:.4f}s")

    # Benders (new_model) - try without Phase 1 (callback-only, like benderscallback_fixed)
    print("Solving Benders (dual separation)...", end=" ", flush=True)
    benders_res = solve_benders_pmedian(N, M, p, dist, K=K, D=D, verbose=False, use_phase1=True)
    print(f"done. Obj={benders_res['objective']:.4f}, Time={benders_res['time']:.4f}s")

    # Compare
    diff = abs(f3_res["objective"] - benders_res["objective"]) if (
        f3_res["objective"] is not None and benders_res["objective"] is not None
    ) else None
    match = diff is not None and diff < 1e-4
    print(f"\nF3 objective:      {f3_res['objective']}")
    print(f"Benders objective: {benders_res['objective']}")
    print(f"Difference:        {diff}")
    print(f"Match:             {'YES' if match else 'NO'}")
    print(f"F3 facilities:     {f3_res['facilities']}")
    print(f"Benders facilities:{benders_res['facilities']}")

    return {
        "test_id": test_id,
        "N": N, "M": M, "p": p,
        "f3_obj": f3_res["objective"],
        "benders_obj": benders_res["objective"],
        "f3_time": f3_res["time"],
        "benders_time": benders_res["time"],
        "match": match,
    }


def main():
    print("\n" + "="*70)
    print("BENDERS (Polynomial Separation) vs F3 FORMULATION")
    print("="*70)

    results = []

    # Test 1: Small
    r1 = run_test(1, N=5, M=5, p=2, seed=42)
    results.append(r1)

    # Test 2: Small
    r2 = run_test(2, N=5, M=5, p=3, seed=43)
    results.append(r2)

    # Test 3: Medium
    r3 = run_test(3, N=10, M=10, p=4, seed=44)
    results.append(r3)

    # Test 4: Medium
    r4 = run_test(4, N=10, M=10, p=5, seed=45)
    results.append(r4)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(1 for r in results if r["match"])
    print(f"Passed: {passed}/{len(results)}")
    for r in results:
        status = "OK" if r["match"] else "FAIL"
        print(f"  Test {r['test_id']}: {status}")

    return passed == len(results)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
