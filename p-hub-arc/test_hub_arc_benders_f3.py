"""
Test: Benders Decomposition vs F3 Formulation for p-Hub-Arc

Compares solve_benders_hub_arc (new_model_hub_arc) with solve_hub_arc_F3
from hub_arc_models.

Run: python test_hub_arc_benders_f3.py
"""

import sys
import os
import numpy as np
import time

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from hub_arc_models import solve_hub_arc_F3
from new_model_hub_arc import solve_benders_hub_arc


def build_instance(n, p, seed=None):
    """Build random W (flows) and D (distances) for hub-arc."""
    if seed is not None:
        np.random.seed(seed)
    W = np.random.rand(n, n) * 10
    np.fill_diagonal(W, 0)
    D = np.random.rand(n, n) * 20
    np.fill_diagonal(D, 0)
    return W.tolist(), D.tolist()


def run_test(test_id, n, p, W=None, D=None, seed=None):
    """Run single test: F3 vs Benders."""
    if W is None or D is None:
        W, D = build_instance(n, p, seed)

    print(f"\n{'='*60}")
    print(f"Test {test_id}: n={n}, p={p}")
    print("="*60)

    # F3
    print("Solving F3...", end=" ", flush=True)
    f3_res = solve_hub_arc_F3(n, p, W, D, gurobi_output=False)
    print(f"done. Obj={f3_res['objective']:.4f}, Time={f3_res['time']:.4f}s")

    # Benders
    print("Solving Benders (Phase 1 + Phase 2)...", end=" ", flush=True)
    benders_res = solve_benders_hub_arc(
        n, p, W, D, verbose=False, use_phase1=True
    )
    print(f"done. Obj={benders_res['objective']:.4f}, Time={benders_res['time']:.4f}s")

    # Compare
    diff = None
    if f3_res["objective"] is not None and benders_res["objective"] is not None:
        diff = abs(f3_res["objective"] - benders_res["objective"])
    match = diff is not None and diff < 1e-4

    print(f"\nF3 objective:      {f3_res['objective']}")
    print(f"Benders objective: {benders_res['objective']}")
    print(f"Difference:        {diff}")
    print(f"Match:             {'YES' if match else 'NO'}")
    print(f"F3 arcs:           {sorted(f3_res['selected_arcs'])}")
    print(f"Benders arcs:      {sorted(benders_res['selected_arcs'])}")

    return {
        "test_id": test_id,
        "n": n,
        "p": p,
        "f3_obj": f3_res["objective"],
        "benders_obj": benders_res["objective"],
        "f3_time": f3_res["time"],
        "benders_time": benders_res["time"],
        "match": match,
    }


def main():
    print("\n" + "="*70)
    print("BENDERS vs F3 FORMULATION - p-Hub-Arc")
    print("="*70)

    results = []

    # Test 1: Toy 3-node instance (from hub_arc_models / test_benders)
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    r1 = run_test(1, n=3, p=2, W=W1, D=D1)
    results.append(r1)

    # Test 2: Same 3-node, p=3
    r2 = run_test(2, n=3, p=3, W=W1, D=D1)
    results.append(r2)

    # Test 3: 4-node symmetric
    W3 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D3 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    r3 = run_test(3, n=4, p=3, W=W3, D=D3)
    results.append(r3)

    # Test 4: Random
    r4 = run_test(4, n=5, p=3, seed=42)
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
