"""
Test: Benders Decomposition (Polynomial Separation) vs F3 Formulation

Compares the new_model implementation (Algorithms 1, 2, 3 + Phase 2)
with the direct F3 formulation for the p-median problem.

Run: python test_benders_f3.py
      or: python -m pytest benders/test_benders_f3.py -v
"""

import sys
import os
import numpy as np
import time

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

import gurobipy as gp
from gurobipy import GRB

# F3 formulation (from p-median.py)
def solve_F3_formulation(N, M, p, K, D, distance_matrix, time_limit=None):
    """Solve p-median using direct F3 formulation. time_limit: optional seconds."""
    model = gp.Model("F3_pMedian")
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

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
                j for j in range(M) if abs(distance_matrix[i, j] - Dk) < 1e-8
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


def run_test_silent(test_id, N, M, p, seed=None, time_limit=None):
    """Run single test: F3 vs Benders (no print). Returns result dict.
    time_limit: optional seconds for F3 and Benders; use for large N,M.
    """
    from new_model import solve_benders_pmedian

    dist, K, D = build_instance(N, M, p, seed)
    f3_res = solve_F3_formulation(N, M, p, K, D, dist, time_limit=time_limit)
    benders_res = solve_benders_pmedian(
        N, M, p, dist, K=K, D=D, verbose=False, use_phase1=True, time_limit=time_limit
    )
    diff = (
        abs(f3_res["objective"] - benders_res["objective"])
        if (
            f3_res["objective"] is not None and benders_res["objective"] is not None
        )
        else None
    )
    match = diff is not None and diff < 1e-4
    return {
        "test_id": test_id,
        "N": N, "M": M, "p": p,
        "f3_obj": f3_res["objective"],
        "benders_obj": benders_res["objective"],
        "match": match,
    }


# ---------------------------------------------------------------------------
# Unit tests: build_instance
# ---------------------------------------------------------------------------

def test_build_instance_shape():
    """build_instance returns dist of shape (N, M)."""
    dist, K, D = build_instance(5, 7, 2, seed=42)
    assert dist.shape == (5, 7)
    assert len(K) == 5
    assert len(D) == 5


def test_build_instance_K_keys():
    """K[i] is number of unique distances for client i."""
    dist, K, D = build_instance(4, 4, 2, seed=43)
    for i in range(4):
        assert K[i] >= 1 and K[i] <= 4
        assert 1 in D[i] and K[i] in D[i]


def test_build_instance_D_sorted():
    """D[i] values are sorted ascending."""
    dist, K, D = build_instance(6, 5, 2, seed=44)
    for i in range(6):
        vals = [D[i][k] for k in range(1, K[i] + 1)]
        assert vals == sorted(vals)


def test_build_instance_reproducibility():
    """Same seed gives same dist."""
    d1, _, _ = build_instance(5, 5, 2, seed=99)
    d2, _, _ = build_instance(5, 5, 2, seed=99)
    np.testing.assert_array_almost_equal(d1, d2)


def test_build_instance_different_seeds():
    """Different seeds give different dist (with high probability)."""
    d1, _, _ = build_instance(10, 10, 3, seed=1)
    d2, _, _ = build_instance(10, 10, 3, seed=2)
    assert not np.allclose(d1, d2)


# ---------------------------------------------------------------------------
# Unit tests: F3 formulation
# ---------------------------------------------------------------------------

def test_f3_small_5_5_2():
    """F3 solves N=5, M=5, p=2 and returns OPTIMAL."""
    dist, K, D = build_instance(5, 5, 2, seed=42)
    res = solve_F3_formulation(5, 5, 2, K, D, dist)
    assert res["status"] == "OPTIMAL"
    assert res["objective"] is not None
    assert res["objective"] >= 0


def test_f3_small_5_5_3():
    """F3 solves N=5, M=5, p=3."""
    dist, K, D = build_instance(5, 5, 3, seed=43)
    res = solve_F3_formulation(5, 5, 3, K, D, dist)
    assert res["status"] == "OPTIMAL"
    assert len(res["facilities"]) == 3


def test_f3_exactly_p_facilities():
    """F3 solution opens exactly p facilities."""
    dist, K, D = build_instance(6, 8, 4, seed=44)
    res = solve_F3_formulation(6, 8, 4, K, D, dist)
    assert res["status"] == "OPTIMAL"
    assert len(res["facilities"]) == 4


def test_f3_facilities_in_range():
    """F3 facility indices are in [0, M-1]."""
    dist, K, D = build_instance(5, 7, 2, seed=45)
    res = solve_F3_formulation(5, 7, 2, K, D, dist)
    assert res["status"] == "OPTIMAL"
    for j in res["facilities"]:
        assert 0 <= j < 7


def test_f3_p_equals_1():
    """F3 with p=1 (single facility)."""
    dist, K, D = build_instance(4, 5, 1, seed=46)
    res = solve_F3_formulation(4, 5, 1, K, D, dist)
    assert res["status"] == "OPTIMAL"
    assert len(res["facilities"]) == 1


def test_f3_p_equals_M():
    """F3 with p=M (all facilities open)."""
    dist, K, D = build_instance(3, 4, 4, seed=47)
    res = solve_F3_formulation(3, 4, 4, K, D, dist)
    assert res["status"] == "OPTIMAL"
    assert set(res["facilities"]) == {0, 1, 2, 3}


def test_f3_returns_time():
    """F3 result includes non-negative time."""
    dist, K, D = build_instance(5, 5, 2, seed=48)
    res = solve_F3_formulation(5, 5, 2, K, D, dist)
    assert res["time"] >= 0


def test_f3_y_sum_equals_p():
    """F3 y solution sums to p."""
    dist, K, D = build_instance(5, 6, 3, seed=49)
    res = solve_F3_formulation(5, 6, 3, K, D, dist)
    assert res["status"] == "OPTIMAL"
    assert sum(1 for v in res["y"] if v > 0.5) == 3


# ---------------------------------------------------------------------------
# Unit tests: Benders (new_model)
# ---------------------------------------------------------------------------

def test_benders_small_returns_optimal():
    """Benders solve returns OPTIMAL for small instance."""
    from new_model import solve_benders_pmedian
    dist, K, D = build_instance(5, 5, 2, seed=50)
    res = solve_benders_pmedian(5, 5, 2, dist, K=K, D=D, verbose=False, use_phase1=True)
    assert res["status"] == "OPTIMAL"
    assert res["objective"] is not None


def test_benders_facilities_count():
    """Benders opens exactly p facilities."""
    from new_model import solve_benders_pmedian
    dist, K, D = build_instance(5, 6, 3, seed=51)
    res = solve_benders_pmedian(5, 6, 3, dist, K=K, D=D, verbose=False, use_phase1=True)
    assert res["status"] == "OPTIMAL"
    assert len(res["facilities"]) == 3


def test_benders_without_phase1():
    """Benders runs with use_phase1=False."""
    from new_model import solve_benders_pmedian
    dist, K, D = build_instance(5, 5, 2, seed=52)
    res = solve_benders_pmedian(5, 5, 2, dist, K=K, D=D, verbose=False, use_phase1=False)
    assert res["status"] == "OPTIMAL"
    assert res["objective"] is not None


def test_benders_preprocess_internal():
    """Benders with K,D=None uses internal preprocess (same result type)."""
    from new_model import solve_benders_pmedian
    np.random.seed(53)
    dist = np.random.rand(5, 5) * 100.0
    res = solve_benders_pmedian(5, 5, 2, dist, K=None, D=None, verbose=False)
    assert res["status"] in ("OPTIMAL", "FAILED")
    assert "objective" in res and "facilities" in res


# ---------------------------------------------------------------------------
# Integration tests: F3 vs Benders objective match (parametrized)
# ---------------------------------------------------------------------------

def test_benders_vs_f3_match_1():
    r = run_test_silent(1, 5, 5, 2, seed=42)
    assert r["match"], f"F3={r['f3_obj']} Benders={r['benders_obj']}"


def test_benders_vs_f3_match_2():
    r = run_test_silent(2, 5, 5, 3, seed=43)
    assert r["match"]


def test_benders_vs_f3_match_3():
    r = run_test_silent(3, 10, 10, 4, seed=44)
    assert r["match"]


def test_benders_vs_f3_match_4():
    r = run_test_silent(4, 10, 10, 5, seed=45)
    assert r["match"]


def test_benders_vs_f3_match_5():
    r = run_test_silent(5, 6, 6, 2, seed=46)
    assert r["match"]


def test_benders_vs_f3_match_6():
    r = run_test_silent(6, 6, 6, 3, seed=47)
    assert r["match"]


def test_benders_vs_f3_match_7():
    r = run_test_silent(7, 6, 8, 3, seed=48)
    assert r["match"]


def test_benders_vs_f3_match_8():
    r = run_test_silent(8, 7, 7, 4, seed=49)
    assert r["match"]


def test_benders_vs_f3_match_9():
    r = run_test_silent(9, 5, 10, 2, seed=50)
    assert r["match"]


def test_benders_vs_f3_match_10():
    r = run_test_silent(10, 8, 8, 4, seed=51)
    assert r["match"]


def test_benders_vs_f3_match_11():
    r = run_test_silent(11, 4, 4, 2, seed=52)
    assert r["match"]


def test_benders_vs_f3_match_12():
    r = run_test_silent(12, 4, 5, 1, seed=53)
    assert r["match"]


def test_benders_vs_f3_match_13():
    r = run_test_silent(13, 5, 5, 5, seed=54)
    assert r["match"]


def test_benders_vs_f3_match_14():
    r = run_test_silent(14, 9, 9, 3, seed=55)
    assert r["match"]


def test_benders_vs_f3_match_15():
    r = run_test_silent(15, 9, 9, 5, seed=56)
    assert r["match"]


def test_benders_vs_f3_match_16():
    r = run_test_silent(16, 7, 10, 3, seed=57)
    assert r["match"]


def test_benders_vs_f3_match_17():
    r = run_test_silent(17, 8, 12, 4, seed=58)
    assert r["match"]


def test_benders_vs_f3_match_18():
    r = run_test_silent(18, 6, 7, 2, seed=59)
    assert r["match"]


def test_benders_vs_f3_match_19():
    r = run_test_silent(19, 10, 10, 6, seed=60)
    assert r["match"]


def test_benders_vs_f3_match_20():
    r = run_test_silent(20, 5, 7, 2, seed=61)
    assert r["match"]


def test_benders_vs_f3_match_21():
    r = run_test_silent(21, 7, 7, 3, seed=62)
    assert r["match"]


def test_benders_vs_f3_match_22():
    r = run_test_silent(22, 8, 8, 2, seed=63)
    assert r["match"]


def test_benders_vs_f3_match_23():
    r = run_test_silent(23, 6, 9, 4, seed=64)
    assert r["match"]


def test_benders_vs_f3_match_24():
    r = run_test_silent(24, 5, 6, 3, seed=65)
    assert r["match"]


def test_benders_vs_f3_match_25():
    r = run_test_silent(25, 10, 12, 5, seed=66)
    assert r["match"]


# ---------------------------------------------------------------------------
# Large instance tests: N, M from 10 to 50 (with time_limit where needed)
# ---------------------------------------------------------------------------

def test_benders_vs_f3_n10_m10_p4():
    r = run_test_silent(101, 10, 10, 4, seed=70)
    assert r["match"], f"F3={r['f3_obj']} Benders={r['benders_obj']}"


def test_benders_vs_f3_n10_m10_p5():
    r = run_test_silent(102, 10, 10, 5, seed=71)
    assert r["match"]


def test_benders_vs_f3_n12_m12_p4():
    r = run_test_silent(103, 12, 12, 4, seed=72)
    assert r["match"]


def test_benders_vs_f3_n12_m12_p6():
    r = run_test_silent(104, 12, 12, 6, seed=73)
    assert r["match"]


def test_benders_vs_f3_n15_m15_p5():
    r = run_test_silent(105, 15, 15, 5, seed=74, time_limit=120)
    assert r["match"]


def test_benders_vs_f3_n15_m15_p7():
    r = run_test_silent(106, 15, 15, 7, seed=75, time_limit=120)
    assert r["match"]


def test_benders_vs_f3_n20_m20_p6():
    r = run_test_silent(107, 20, 20, 6, seed=76, time_limit=180)
    assert r["match"]


def test_benders_vs_f3_n20_m20_p8():
    r = run_test_silent(108, 20, 20, 8, seed=77, time_limit=180)
    assert r["match"]


def test_benders_vs_f3_n25_m25_p8():
    r = run_test_silent(109, 25, 25, 8, seed=78, time_limit=300)
    assert r["match"]


def test_benders_vs_f3_n25_m25_p10():
    r = run_test_silent(110, 25, 25, 10, seed=79, time_limit=300)
    assert r["match"]


def test_benders_vs_f3_n30_m30_p10():
    r = run_test_silent(111, 30, 30, 10, seed=80, time_limit=300)
    assert r["match"]


def test_benders_vs_f3_n30_m30_p12():
    r = run_test_silent(112, 30, 30, 12, seed=81, time_limit=300)
    assert r["match"]


def test_benders_vs_f3_n35_m35_p12():
    r = run_test_silent(113, 35, 35, 12, seed=82, time_limit=400)
    assert r["match"]


def test_benders_vs_f3_n35_m35_p15():
    r = run_test_silent(114, 35, 35, 15, seed=83, time_limit=400)
    assert r["match"]


def test_benders_vs_f3_n40_m40_p14():
    r = run_test_silent(115, 40, 40, 14, seed=84, time_limit=500)
    assert r["match"]


def test_benders_vs_f3_n40_m40_p18():
    r = run_test_silent(116, 40, 40, 18, seed=85, time_limit=500)
    assert r["match"]


def test_benders_vs_f3_n45_m45_p15():
    r = run_test_silent(117, 45, 45, 15, seed=86, time_limit=600)
    assert r["match"]


def test_benders_vs_f3_n45_m45_p20():
    r = run_test_silent(118, 45, 45, 20, seed=87, time_limit=600)
    assert r["match"]


def test_benders_vs_f3_n50_m50_p18():
    r = run_test_silent(119, 50, 50, 18, seed=88, time_limit=600)
    assert r["match"]


def test_benders_vs_f3_n50_m50_p22():
    r = run_test_silent(120, 50, 50, 22, seed=89, time_limit=600)
    assert r["match"]


def test_benders_vs_f3_n10_m15_p5():
    r = run_test_silent(121, 10, 15, 5, seed=90)
    assert r["match"]


def test_benders_vs_f3_n20_m25_p8():
    r = run_test_silent(122, 20, 25, 8, seed=91, time_limit=240)
    assert r["match"]


def test_benders_vs_f3_n30_m40_p12():
    r = run_test_silent(123, 30, 40, 12, seed=92, time_limit=400)
    assert r["match"]


# ---------------------------------------------------------------------------
# Legacy main (run with python test_benders_f3.py)
# ---------------------------------------------------------------------------

def run_test(test_id, N, M, p, seed=None, time_limit=None):
    """Run single test with print (for script mode)."""
    from new_model import solve_benders_pmedian
    dist, K, D = build_instance(N, M, p, seed)
    tl_str = f", time_limit={time_limit}s" if time_limit else ""
    print(f"\n{'='*60}\nTest {test_id}: N={N}, M={M}, p={p}{tl_str}\n{'='*60}")
    print("Solving F3...", end=" ", flush=True)
    f3_res = solve_F3_formulation(N, M, p, K, D, dist, time_limit=time_limit)
    print(f"done. Obj={f3_res['objective']:.4f}, Time={f3_res['time']:.4f}s")
    print("Solving Benders...", end=" ", flush=True)
    benders_res = solve_benders_pmedian(
        N, M, p, dist, K=K, D=D, verbose=False, use_phase1=True, time_limit=time_limit
    )
    print(f"done. Obj={benders_res['objective']:.4f}, Time={benders_res['time']:.4f}s")
    diff = abs(f3_res["objective"] - benders_res["objective"]) if (
        f3_res["objective"] is not None and benders_res["objective"] is not None
    ) else None
    match = diff is not None and diff < 1e-4
    print(f"Match: {'YES' if match else 'NO'}")
    return {"test_id": test_id, "N": N, "M": M, "p": p, "match": match}


# Test configs for script mode: (test_id, N, M, p, seed, time_limit or None)
SCRIPT_TEST_CONFIGS = [
    (1, 5, 5, 2, 42, None),
    (2, 5, 5, 3, 43, None),
    (3, 10, 10, 4, 44, None),
    (4, 10, 10, 5, 45, None),
    (5, 6, 6, 2, 46, None),
    (6, 6, 6, 3, 47, None),
    (7, 8, 8, 4, 51, None),
    (8, 10, 10, 4, 70, None),
    (9, 10, 10, 5, 71, None),
    (10, 12, 12, 4, 72, None),
    (11, 12, 12, 6, 73, None),
    (12, 15, 15, 5, 74, 120),
    (13, 15, 15, 7, 75, 120),
    (14, 20, 20, 6, 76, 180),
    (15, 20, 20, 8, 77, 180),
    (16, 25, 25, 8, 78, 300),
    (17, 25, 25, 10, 79, 300),
    (18, 30, 30, 10, 80, 300),
    (19, 35, 35, 12, 82, 400),
    (20, 40, 40, 14, 84, 500),
]


def main():
    print("\n" + "="*70)
    print("BENDERS (Polynomial Separation) vs F3 FORMULATION")
    print("="*70)
    print(f"Running {len(SCRIPT_TEST_CONFIGS)} tests (script mode).")
    print("For all pytest tests including n=50, run: python -m pytest benders/test_benders_f3.py -v")
    print("="*70)
    results = []
    for test_id, N, M, p, seed, time_limit in SCRIPT_TEST_CONFIGS:
        r = run_test(test_id, N, M, p, seed=seed, time_limit=time_limit)
        results.append(r)
    passed = sum(1 for r in results if r["match"])
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(results)}")
    return passed == len(results)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
