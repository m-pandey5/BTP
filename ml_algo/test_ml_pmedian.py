"""
Tests: ExactMLBendersPMedian (ml_algo/exact_ml_benders_pmedian.py) vs the
canonical CR (F3) p-median Gurobi solver — same problem definition, so the
objectives must match.

Test instances reuse the seed scheme from benders/testcases.py:
    np.random.seed(100 + test_id)
    distance = rand(N,N)*100 ; symmetrize ; zero diagonal.

Run:
    python test_ml_pmedian.py
"""

import os
import sys
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from exact_ml_benders_pmedian import ExactMLBendersPMedian


# ----------------------------------------------------------------------
# Ground-truth CR (canonical) p-median solver — complete, no K truncation
# ----------------------------------------------------------------------
def solve_p_median_CR(n, p, c):
    D, G = {}, {}
    for i in range(n):
        unique_costs = sorted(set(c[i]))
        D[i] = unique_costs
        G[i] = len(unique_costs)

    m = gp.Model("p-median-CR")
    m.Params.OutputFlag = 0

    y = m.addVars(n, vtype=GRB.BINARY, name="y")
    z = {}
    for i in range(n):
        for k in range(1, G[i]):
            z[i, k] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}")

    m.setObjective(
        gp.quicksum(
            (D[i][k] - D[i][k - 1]) * z[i, k]
            for i in range(n)
            for k in range(1, G[i])
        ),
        GRB.MINIMIZE,
    )

    m.addConstr(gp.quicksum(y[j] for j in range(n)) == p)
    for i in range(n):
        for k in range(1, G[i]):
            m.addConstr(
                z[i, k]
                + gp.quicksum(y[j] for j in range(n) if c[i][j] < D[i][k])
                >= 1
            )

    t0 = time.time()
    m.optimize()
    elapsed = time.time() - t0

    if m.status != GRB.OPTIMAL:
        return None

    facilities = sorted(j for j in range(n) if y[j].X > 0.5)
    return {"objective": m.ObjVal, "facilities": facilities, "time": elapsed}


# ----------------------------------------------------------------------
# Instance generator — same scheme as benders/testcases.py
# ----------------------------------------------------------------------
def build_instance(N, test_id):
    np.random.seed(100 + test_id)
    d = np.random.rand(N, N) * 100.0
    np.fill_diagonal(d, 0)
    d = np.maximum(d, d.T)
    return d


# ----------------------------------------------------------------------
# Run a single test
# ----------------------------------------------------------------------
def run_test(test_id, N, p):
    print(f"\n{'='*70}")
    print(f"Test {test_id}: N={N}, p={p}")
    print("=" * 70)

    d = build_instance(N, test_id)

    # CR ground truth
    print("  CR (F3) solving ...", end=" ", flush=True)
    cr = solve_p_median_CR(N, p, d.tolist())
    print(f"obj={cr['objective']:10.4f}  t={cr['time']:.4f}s")

    # ML-Benders
    print("  ML-Benders solving ...", end=" ", flush=True)
    t0 = time.time()
    solver = ExactMLBendersPMedian(
        distance_matrix=d,
        p=p,
        ml_start_iter=3,
        max_iter=50,
    )
    # Silence the per-iteration prints
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        best_y, best_cost = solver.solve()
    finally:
        builtins.print = real_print
    ml_time = time.time() - t0
    ml_facs = sorted(int(j) for j in np.where(best_y == 1)[0])
    print(f"obj={best_cost:10.4f}  t={ml_time:.4f}s  facs={ml_facs}")

    obj_match = abs(cr["objective"] - best_cost) < 1e-4
    fac_match = cr["facilities"] == ml_facs

    print(f"  -> objective match: {'YES' if obj_match else 'NO'}   "
          f"facility-set match: {'YES' if fac_match else 'NO (alt. optimum)'}")
    print(f"     CR facs = {cr['facilities']}   ML facs = {ml_facs}")

    return {
        "test_id": test_id,
        "N": N,
        "p": p,
        "cr_obj": cr["objective"],
        "ml_obj": best_cost,
        "cr_time": cr["time"],
        "ml_time": ml_time,
        "cr_facs": cr["facilities"],
        "ml_facs": ml_facs,
        "obj_match": obj_match,
        "fac_match": fac_match,
    }


# ----------------------------------------------------------------------
# 10 test cases (sizes kept small because ML solver enumerates C(N, p))
# ----------------------------------------------------------------------
TEST_CASES = [
    (1, 3, 1),
    (2, 4, 2),
    (3, 5, 2),
    (4, 5, 3),
    (5, 6, 2),
    (6, 6, 3),
    (7, 7, 3),
    (8, 8, 3),
    (9, 8, 4),
    (10, 10, 3),
]


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ML-BENDERS p-MEDIAN  vs  CR (F3) p-MEDIAN")
    print("=" * 70)

    rows = []
    for tid, N, p in TEST_CASES:
        rows.append(run_test(tid, N, p))

    # Summary
    print("\n\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    header = f"{'#':>2}  {'N':>3} {'p':>2}  {'CR obj':>10}  {'ML obj':>10}  " \
             f"{'CR t(s)':>8}  {'ML t(s)':>8}  {'ObjMatch':>9}  {'FacMatch':>9}"
    print(header)
    print("-" * len(header))
    n_pass = 0
    for r in rows:
        n_pass += int(r["obj_match"])
        print(f"{r['test_id']:>2}  {r['N']:>3} {r['p']:>2}  "
              f"{r['cr_obj']:>10.3f}  {r['ml_obj']:>10.3f}  "
              f"{r['cr_time']:>8.4f}  {r['ml_time']:>8.4f}  "
              f"{('YES' if r['obj_match'] else 'NO'):>9}  "
              f"{('YES' if r['fac_match'] else 'NO'):>9}")
    print("-" * len(header))
    print(f"Objective matches: {n_pass}/{len(rows)}")
