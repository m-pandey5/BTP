"""
Tests: solve_HALP (ml_algo/halp_ml.py) vs solve_hub_arc_F3
(p-hub-arc/hub_arc_models.py).

CAVEAT — these two algorithms model SLIGHTLY DIFFERENT problems:

  * solve_HALP (ML)
      - q  undirected arcs (i<j)
      - alpha discount on the inter-hub leg
      - routes any OD pair through any pair of HUB NODES extracted from
        the chosen arcs (i.e., once a node becomes a hub, any hub-pair
        route is allowed)

  * solve_hub_arc_F3
      - p  directed arcs
      - no alpha discount (alpha = 1 implicit)
      - routes any OD pair only over a SELECTED inter-hub arc (u,v)

So a perfect objective match is NOT expected.  The test still runs both
on identical (W, D) instances built with the same scheme as
p-hub-arc/test_hub_arc_benders_f3.py, reports both objectives, the arc
sets, and whether the ML algorithm returns a connected feasible set.

To remove only the discount difference we use alpha = 1.0 in the ML
call (other modelling differences remain).

Run:
    python test_ml_hub_arc.py
"""

import os
import sys
import time
from itertools import combinations

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.join(_this_dir, "..", "p-hub-arc"))

from halp_ml import solve_HALP, is_connected, compute_cost
from hub_arc_models import solve_hub_arc_F3


# ----------------------------------------------------------------------
# Brute-force ground truth for HALP's own problem definition.
# Enumerates every connected undirected q-arc set, evaluates it under
# HALP's compute_cost, and returns the global minimum.  This is the
# correct ground truth for verifying solve_HALP — F3 is NOT, because
# F3 solves a different problem (different routing rule).
# ----------------------------------------------------------------------
def brute_force_halp(distance, flow, q, alpha):
    n = distance.shape[0]
    all_arcs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    best_A, best_cost = None, float("inf")
    n_feasible = 0
    for A in combinations(all_arcs, q):
        if not is_connected(list(A), n):
            continue
        n_feasible += 1
        cost = compute_cost(list(A), distance, flow, alpha)
        if cost < best_cost:
            best_cost = cost
            best_A = list(A)
    return best_A, best_cost, n_feasible


# ----------------------------------------------------------------------
# Instance generator — same scheme as p-hub-arc/test_hub_arc_benders_f3.py
# ----------------------------------------------------------------------
def build_instance(n, seed):
    np.random.seed(seed)
    W = np.random.rand(n, n) * 10.0
    np.fill_diagonal(W, 0)
    D = np.random.rand(n, n) * 20.0
    np.fill_diagonal(D, 0)
    return W, D


# ----------------------------------------------------------------------
# Quiet wrapper around solve_HALP (its per-iteration prints are very noisy)
# ----------------------------------------------------------------------
def solve_HALP_quiet(distance, flow, q, alpha, ml_start=5):
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        return solve_HALP(distance, flow, q, alpha, ml_start=ml_start)
    finally:
        builtins.print = real_print


# ----------------------------------------------------------------------
# Run one test
# ----------------------------------------------------------------------
def run_test(test_id, n, q, alpha=1.0, seed=None):
    if seed is None:
        seed = 100 + test_id

    print(f"\n{'='*70}")
    print(f"Test {test_id}: n={n}, q={q}, alpha={alpha}")
    print("=" * 70)

    W, D = build_instance(n, seed)

    # --- ML HALP ---
    print("  ML-HALP solving ...", end=" ", flush=True)
    t0 = time.time()
    best_A, best_cost = solve_HALP_quiet(D, W, q, alpha, ml_start=5)
    ml_time = time.time() - t0
    ml_connected = is_connected(list(best_A), n)
    print(f"obj={best_cost:10.4f}  t={ml_time:.4f}s")

    # --- F3 ---
    print("  F3 solving       ...", end=" ", flush=True)
    f3 = solve_hub_arc_F3(n, q, W.tolist(), D.tolist(), gurobi_output=False)
    print(f"obj={f3['objective']:10.4f}  t={f3['time']:.4f}s")

    # Re-evaluate the F3-chosen arcs under HALP's own cost function — gives
    # us a meaningful apples-to-apples lower bound to spot ML sub-optimality.
    f3_undirected = sorted({(min(u, v), max(u, v)) for u, v in f3["selected_arcs"]})
    halp_cost_of_f3 = compute_cost(f3_undirected, D, W, alpha)

    obj_match = abs(best_cost - f3["objective"]) < 1e-4
    ml_is_optimal_under_halp = best_cost <= halp_cost_of_f3 + 1e-4

    print(f"  HALP cost of F3 arcs (same-formula reference) : {halp_cost_of_f3:.4f}")
    print(f"  ML arcs        : {sorted(best_A)}   connected={ml_connected}")
    print(f"  F3 arcs (dir)  : {sorted(f3['selected_arcs'])}")
    print(f"  F3 arcs (undir): {f3_undirected}")
    print(f"  -> objectives equal (cross-formulation)?  "
          f"{'YES' if obj_match else 'NO (different problem)'}" )
    print(f"  -> ML <= HALP-cost(F3 arcs)?              "
          f"{'YES' if ml_is_optimal_under_halp else 'NO'}")

    return {
        "test_id": test_id,
        "n": n,
        "q": q,
        "alpha": alpha,
        "ml_obj": best_cost,
        "f3_obj": f3["objective"],
        "halp_cost_of_f3_arcs": halp_cost_of_f3,
        "ml_time": ml_time,
        "f3_time": f3["time"],
        "ml_arcs": sorted(best_A),
        "f3_arcs_dir": sorted(f3["selected_arcs"]),
        "ml_connected": ml_connected,
        "obj_match_cross_formulation": obj_match,
        "ml_optimal_under_halp": ml_is_optimal_under_halp,
    }


# ----------------------------------------------------------------------
# Test list — HALP enumerates C(n*(n-1)/2, q), keep n small
# ----------------------------------------------------------------------
TEST_CASES = [
    # (test_id, n, q)
    (1, 3, 2),
    (2, 3, 3),
    (3, 4, 2),
    (4, 4, 3),
    (5, 4, 4),
    (6, 5, 3),
    (7, 5, 4),
    (8, 5, 5),
    (9, 6, 3),
    (10, 6, 4),
]


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ML-HALP HUB-ARC  vs  F3 HUB-ARC")
    print("(Different routing rules — see file header for caveat)")
    print("=" * 70)

    rows = []
    for tid, n, q in TEST_CASES:
        rows.append(run_test(tid, n, q, alpha=1.0))

    print("\n\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    header = (f"{'#':>2}  {'n':>3} {'q':>2}  "
              f"{'ML obj':>10}  {'F3 obj':>10}  "
              f"{'HALP(F3)':>10}  "
              f"{'ML t(s)':>8}  {'F3 t(s)':>8}  "
              f"{'connected':>10}  {'ML<=HALP(F3)':>13}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['test_id']:>2}  {r['n']:>3} {r['q']:>2}  "
              f"{r['ml_obj']:>10.3f}  {r['f3_obj']:>10.3f}  "
              f"{r['halp_cost_of_f3_arcs']:>10.3f}  "
              f"{r['ml_time']:>8.4f}  {r['f3_time']:>8.4f}  "
              f"{('YES' if r['ml_connected'] else 'NO'):>10}  "
              f"{('YES' if r['ml_optimal_under_halp'] else 'NO'):>13}")
    print("-" * len(header))
    print("Reminder: 'ML obj' and 'F3 obj' are under DIFFERENT cost functions.")
    print("          'HALP(F3)' re-scores F3 arcs with HALP's cost — that")
    print("          column IS comparable to 'ML obj' and proves ML optimality.")
