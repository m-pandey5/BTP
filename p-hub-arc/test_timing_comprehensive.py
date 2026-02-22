"""
Comprehensive Timing Test: F3 vs Benders for p-Hub-Arc

Compares:
  - F3 (direct canonical formulation)
  - Normal Benders (HubArcBenders: master + lazy callback, no Phase 1)
  - New Benders (solve_benders_hub_arc: Phase 1 LP + Phase 2 with warm start)

Run: python test_timing_comprehensive.py
      or: python -m pytest p-hub-arc/test_timing_comprehensive.py -v
"""

import sys
import os
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from hub_arc_models import solve_hub_arc_F3, HubArcBenders
from new_model_hub_arc import solve_benders_hub_arc, preprocess


def build_instance(n: int, p: int, seed: Optional[int] = None) -> Tuple[List, List]:
    """Build random W (flows) and D (distances) for hub-arc."""
    if seed is not None:
        np.random.seed(seed)
    W = np.random.rand(n, n) * 10
    np.fill_diagonal(W, 0)
    D = np.random.rand(n, n) * 20
    np.fill_diagonal(D, 0)
    return W.tolist(), D.tolist()


def run_instance(
    n: int,
    p: int,
    W: List,
    D: List,
    time_limit: Optional[float] = None,
    use_phase1: bool = True,
) -> Dict[str, Any]:
    """
    Run F3, normal Benders (HubArcBenders), and new Benders on one instance.
    Returns dict with objectives, times, status, and match.
    """
    f3_res = solve_hub_arc_F3(n, p, W, D, gurobi_output=False)

    # Normal Benders (HubArcBenders): no Phase 1, callback-only
    b_norm = HubArcBenders(n=n, p=p, W=W, D=D, verbose=False)
    norm_res = b_norm.solve(time_limit=time_limit)
    # HubArcBenders returns master.ObjVal; add constant for OD pairs with K==1
    _, C, _, K, _, _, _ = preprocess(n, W, D)
    if norm_res["objective"] is not None:
        add_const = sum(C[(i, j)][0] for (i, j) in K if i != j and K[(i, j)] == 1)
        norm_res = dict(norm_res, objective=norm_res["objective"] + add_const)

    # New Benders (Phase 1 + Phase 2)
    new_res = solve_benders_hub_arc(
        n, p, W, D, verbose=False, use_phase1=use_phase1, time_limit=time_limit
    )

    ref = f3_res["objective"]
    match_f3 = ref is not None
    diff_norm = diff_new = None
    if ref is not None:
        if norm_res["objective"] is not None:
            diff_norm = abs(ref - norm_res["objective"])
            match_f3 = match_f3 and diff_norm < 1e-4
        if new_res["objective"] is not None:
            diff_new = abs(ref - new_res["objective"])
            match_f3 = match_f3 and diff_new < 1e-4

    return {
        "n": n,
        "p": p,
        "f3_obj": ref,
        "f3_time": f3_res["time"],
        "f3_status": f3_res["status"],
        "norm_obj": norm_res["objective"],
        "norm_time": norm_res["time"],
        "norm_status": norm_res["status"],
        "new_obj": new_res["objective"],
        "new_time": new_res["time"],
        "new_status": new_res["status"],
        "match": match_f3,
        "diff_norm": diff_norm,
        "diff_new": diff_new,
    }


def run_timing_suite(
    instance_specs: List[Dict],
    seeds: List[int],
    time_limit: Optional[float] = 300.0,
    use_phase1: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run full timing suite: each (n, p) × each seed.
    Returns list of results with aggregated stats per (n, p).
    """
    results = []
    idx = 0

    for spec in instance_specs:
        n = spec["n"]
        p = spec["p"]
        fixed = spec.get("fixed", False)
        seed_list = [None] if fixed else seeds

        for seed in seed_list:
            idx += 1
            if verbose:
                label = f"n={n}, p={p}" + (f", seed={seed}" if seed is not None else " (fixed)")
                print(f"  [{idx}] {label} ...", end=" ", flush=True)

            if fixed:
                W, D = spec["W"], spec["D"]
            else:
                W, D = build_instance(n, p, seed)

            try:
                r = run_instance(n, p, W, D, time_limit=time_limit, use_phase1=use_phase1)
                r["seed"] = seed
                r["fixed"] = fixed
                results.append(r)
                if verbose:
                    f3t, nt, nn = r["f3_time"], r["norm_time"], r["new_time"]
                    times = [("F3", f3t), ("Norm", nt), ("New", nn)]
                    fastest = min(times, key=lambda x: x[1] or float("inf"))
                    f3o, no, neo = r.get("f3_obj"), r.get("norm_obj"), r.get("new_obj")
                    f3o_s = f"{f3o:.4f}" if f3o is not None else "N/A"
                    no_s = f"{no:.4f}" if no is not None else "N/A"
                    neo_s = f"{neo:.4f}" if neo is not None else "N/A"
                    match_str = "Match" if r.get("match") else "MISMATCH"
                    print(f"F3={f3t:.3f}s, NormB={nt:.3f}s, NewB={nn:.3f}s (fastest: {fastest[0]})")
                    print(f"      Obj: F3={f3o_s}  NormB={no_s}  NewB={neo_s}  |  {match_str}")
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                results.append({
                    "n": n, "p": p, "seed": seed, "fixed": fixed,
                    "f3_time": None, "norm_time": None, "new_time": None,
                    "match": False, "error": str(e),
                })

    return results


def aggregate_by_instance(results: List[Dict]) -> Dict[Tuple[int, int], Dict]:
    """Aggregate results by (n, p): mean time, std, count, match rate."""
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        key = (r["n"], r["p"])
        by_key[key].append(r)

    agg = {}
    for (n, p), rows in by_key.items():
        f3_t = [x["f3_time"] for x in rows if x.get("f3_time") is not None]
        norm_t = [x["norm_time"] for x in rows if x.get("norm_time") is not None]
        new_t = [x["new_time"] for x in rows if x.get("new_time") is not None]
        matches = sum(1 for x in rows if x.get("match"))
        agg[(n, p)] = {
            "n": n,
            "p": p,
            "count": len(rows),
            "f3_mean": np.mean(f3_t) if f3_t else None,
            "norm_mean": np.mean(norm_t) if norm_t else None,
            "new_mean": np.mean(new_t) if new_t else None,
            "match_count": matches,
            "match_rate": matches / len(rows) if rows else 0,
        }
    return agg


def print_timing_table(results: List[Dict], agg: Dict):
    """Print formatted timing comparison table."""
    print("\n" + "=" * 115)
    print("TIMING COMPARISON TABLE (F3 vs Normal Benders vs New Benders)")
    print("=" * 115)

    header = f"{'n':>4} | {'p':>3} | {'#runs':>6} | {'F3 (s)':>10} | {'Norm Benders (s)':>16} | {'New Benders (s)':>16} | {'Fastest':>12} | Match"
    print(header)
    print("-" * 115)

    for key in sorted(agg.keys(), key=lambda k: (k[0], k[1])):
        a = agg[key]
        n, p = a["n"], a["p"]
        fm = a["f3_mean"]
        nm = a["norm_mean"]
        newm = a["new_mean"]
        count = a["count"]

        f_str = f"{fm:.4f}" if fm is not None else "   N/A   "
        n_str = f"{nm:.4f}" if nm is not None else "   N/A   "
        new_str = f"{newm:.4f}" if newm is not None else "   N/A   "

        times = [("F3", fm), ("NormB", nm), ("NewB", newm)]
        valid = [(name, t) for name, t in times if t is not None and t >= 0]
        fastest = min(valid, key=lambda x: x[1])[0] if valid else "-"
        match_str = f"{a['match_count']}/{count}"
        print(f"{n:>4} | {p:>3} | {count:>6} | {f_str:>10} | {n_str:>16} | {new_str:>16} | {fastest:>12} | {match_str}")


def print_detailed_results(results: List[Dict]):
    """Print per-run details (times, objectives, match)."""
    print("\n" + "=" * 115)
    print("DETAILED PER-RUN RESULTS (times, objectives, match)")
    print("=" * 115)
    for i, r in enumerate(results):
        if "error" in r:
            print(f"  [{i+1}] n={r['n']}, p={r['p']}: ERROR - {r['error']}")
            continue
        status = "Match" if r.get("match") else "MISMATCH"
        f3t, nt, nn = r.get("f3_time"), r.get("norm_time"), r.get("new_time")
        f3o, no, neo = r.get("f3_obj"), r.get("norm_obj"), r.get("new_obj")
        f3s = f"{f3t:.4f}" if f3t is not None else "N/A"
        ns = f"{nt:.4f}" if nt is not None else "N/A"
        nns = f"{nn:.4f}" if nn is not None else "N/A"
        f3o_s = f"{f3o:.4f}" if f3o is not None else "N/A"
        no_s = f"{no:.4f}" if no is not None else "N/A"
        neo_s = f"{neo:.4f}" if neo is not None else "N/A"
        print(f"  [{i+1}] n={r['n']}, p={r['p']}, seed={r.get('seed')}: "
              f"F3={f3s}s, NormB={ns}s, NewB={nns}s")
        print(f"       Obj: F3={f3o_s}  NormB={no_s}  NewB={neo_s}  |  {status}")


# ---------------------------------------------------------------------------
# Unit tests: build_instance
# ---------------------------------------------------------------------------

def test_build_instance_shape():
    """build_instance returns W and D of size n×n."""
    W, D = build_instance(5, 2, seed=42)
    assert len(W) == 5 and len(W[0]) == 5
    assert len(D) == 5 and len(D[0]) == 5


def test_build_instance_diagonal_zero():
    """W and D have zero diagonal."""
    W, D = build_instance(4, 2, seed=43)
    for i in range(4):
        assert W[i][i] == 0
        assert D[i][i] == 0


def test_build_instance_reproducibility():
    """Same seed gives same W and D."""
    W1, D1 = build_instance(5, 2, seed=100)
    W2, D2 = build_instance(5, 2, seed=100)
    assert W1 == W2 and D1 == D2


def test_build_instance_different_seeds():
    """Different seeds give different data."""
    W1, D1 = build_instance(5, 2, seed=1)
    W2, D2 = build_instance(5, 2, seed=2)
    assert W1 != W2 or D1 != D2


def test_build_instance_no_negative():
    """W and D have non-negative entries (after zero diag)."""
    W, D = build_instance(6, 3, seed=44)
    for i in range(6):
        for j in range(6):
            assert W[i][j] >= 0 and D[i][j] >= 0


# ---------------------------------------------------------------------------
# Unit tests: preprocess
# ---------------------------------------------------------------------------

def test_preprocess_returns_all_keys():
    """preprocess returns H, C, L, K, cost_map, arcs_sorted, od_pairs."""
    W, D = build_instance(4, 2, seed=45)
    H, C, L, K, cost_map, arcs_sorted, od_pairs = preprocess(4, W, D)
    assert H is not None and C is not None and L is not None
    assert K is not None and cost_map is not None and arcs_sorted is not None
    assert od_pairs is not None


def test_preprocess_H_size():
    """H has n*(n-1) arcs."""
    W, D = build_instance(5, 2, seed=46)
    H, *_ = preprocess(5, W, D)
    assert len(H) == 5 * 4


def test_preprocess_K_ij_positive():
    """K[(i,j)] >= 1 for i != j."""
    W, D = build_instance(4, 2, seed=47)
    _, _, _, K, *_ = preprocess(4, W, D)
    for (i, j), k in K.items():
        assert i != j
        assert k >= 1


# ---------------------------------------------------------------------------
# Unit tests: F3 (hub_arc_models)
# ---------------------------------------------------------------------------

def test_f3_fixed_3_2_optimal():
    """F3 solves fixed n=3, p=2 instance to optimality."""
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    res = solve_hub_arc_F3(3, 2, W1, D1, gurobi_output=False)
    assert res["status"] == "OPTIMAL"
    assert res["objective"] is not None and res["objective"] >= 0


def test_f3_fixed_4_2_optimal():
    """F3 solves fixed n=4, p=2 instance."""
    W2 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    res = solve_hub_arc_F3(4, 2, W2, D2, gurobi_output=False)
    assert res["status"] == "OPTIMAL"


def test_f3_returns_time():
    """F3 result includes time."""
    W, D = build_instance(4, 2, seed=48)
    res = solve_hub_arc_F3(4, 2, W, D, gurobi_output=False)
    assert "time" in res and res["time"] >= 0


# ---------------------------------------------------------------------------
# Unit tests: run_instance and aggregate
# ---------------------------------------------------------------------------

def test_run_instance_returns_all_keys():
    """run_instance returns n, p, f3_obj, norm_obj, new_obj, match, times, statuses."""
    W, D = build_instance(4, 2, seed=49)
    r = run_instance(4, 2, W, D, time_limit=30.0)
    for key in ("n", "p", "f3_obj", "f3_time", "norm_obj", "new_obj", "match"):
        assert key in r


def test_run_instance_fixed_3_2_match():
    """F3, Norm Benders, New Benders objectives match on fixed n=3, p=2."""
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    r = run_instance(3, 2, W1, D1, time_limit=60.0)
    assert r["match"], f"f3={r['f3_obj']} norm={r['norm_obj']} new={r['new_obj']}"


def test_run_instance_fixed_4_2_match():
    """Objectives match on fixed n=4, p=2."""
    W2 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    r = run_instance(4, 2, W2, D2, time_limit=60.0)
    assert r["match"]


def test_aggregate_by_instance_structure():
    """aggregate_by_instance returns dict keyed by (n, p) with count, means, match_count."""
    W, D = build_instance(4, 2, seed=50)
    results = [run_instance(4, 2, W, D, time_limit=20.0)]
    agg = aggregate_by_instance(results)
    assert (4, 2) in agg
    a = agg[(4, 2)]
    assert "count" in a and "f3_mean" in a and "match_count" in a
    assert a["count"] == 1


# ---------------------------------------------------------------------------
# Integration tests: F3 vs Benders objective match (many seeds/sizes)
# ---------------------------------------------------------------------------

def test_match_n3_p2_seed_42():
    W, D = build_instance(3, 2, seed=42)
    assert run_instance(3, 2, W, D, time_limit=60.0)["match"]


def test_match_n4_p2_seed_43():
    W, D = build_instance(4, 2, seed=43)
    assert run_instance(4, 2, W, D, time_limit=60.0)["match"]


def test_match_n4_p3_seed_44():
    W, D = build_instance(4, 3, seed=44)
    assert run_instance(4, 3, W, D, time_limit=60.0)["match"]


def test_match_n5_p2_seed_45():
    W, D = build_instance(5, 2, seed=45)
    assert run_instance(5, 2, W, D, time_limit=90.0)["match"]


def test_match_n5_p3_seed_46():
    W, D = build_instance(5, 3, seed=46)
    assert run_instance(5, 3, W, D, time_limit=90.0)["match"]


def test_match_n5_p2_seed_47():
    W, D = build_instance(5, 2, seed=47)
    assert run_instance(5, 2, W, D, time_limit=90.0)["match"]


def test_match_n6_p3_seed_48():
    W, D = build_instance(6, 3, seed=48)
    assert run_instance(6, 3, W, D, time_limit=120.0)["match"]


def test_match_n6_p4_seed_49():
    W, D = build_instance(6, 4, seed=49)
    assert run_instance(6, 4, W, D, time_limit=120.0)["match"]


def test_match_n5_p2_seed_50():
    W, D = build_instance(5, 2, seed=50)
    assert run_instance(5, 2, W, D, time_limit=60.0)["match"]


def test_match_n4_p2_seed_51():
    W, D = build_instance(4, 2, seed=51)
    assert run_instance(4, 2, W, D, time_limit=60.0)["match"]


def test_match_n3_p3_fixed():
    """Fixed instance n=3, p=3."""
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    assert run_instance(3, 3, W1, D1, time_limit=60.0)["match"]


def test_match_n4_p3_fixed():
    """Fixed instance n=4, p=3."""
    W2 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    assert run_instance(4, 3, W2, D2, time_limit=60.0)["match"]


def test_new_benders_use_phase1_false():
    """New Benders runs with use_phase1=False and returns consistent result."""
    W, D = build_instance(4, 2, seed=52)
    r = run_instance(4, 2, W, D, use_phase1=False, time_limit=60.0)
    assert "new_obj" in r and "new_status" in r
    assert r["new_obj"] is not None or r["new_status"] != "OPTIMAL"


def test_run_instance_time_limit_short():
    """run_instance respects short time_limit (no crash)."""
    W, D = build_instance(6, 4, seed=53)
    r = run_instance(6, 4, W, D, time_limit=1.0)
    assert "f3_time" in r and "match" in r


def test_f3_objective_non_negative():
    """F3 objective is non-negative for random instance."""
    W, D = build_instance(4, 2, seed=54)
    res = solve_hub_arc_F3(4, 2, W, D, gurobi_output=False)
    assert res["objective"] is None or res["objective"] >= 0


def test_run_instance_n7_p4_seed_55():
    W, D = build_instance(7, 4, seed=55)
    assert run_instance(7, 4, W, D, time_limit=120.0)["match"]


def test_run_instance_n5_p2_seed_56():
    W, D = build_instance(5, 2, seed=56)
    assert run_instance(5, 2, W, D, time_limit=60.0)["match"]


def test_run_instance_n6_p3_seed_57():
    W, D = build_instance(6, 3, seed=57)
    assert run_instance(6, 3, W, D, time_limit=90.0)["match"]


def test_run_instance_n4_p2_seed_58():
    W, D = build_instance(4, 2, seed=58)
    assert run_instance(4, 2, W, D, time_limit=60.0)["match"]


def test_run_instance_n5_p3_seed_59():
    W, D = build_instance(5, 3, seed=59)
    assert run_instance(5, 3, W, D, time_limit=90.0)["match"]


def test_run_instance_n6_p4_seed_60():
    W, D = build_instance(6, 4, seed=60)
    assert run_instance(6, 4, W, D, time_limit=120.0)["match"]


# ---------------------------------------------------------------------------
# Large instance tests: n from 10 to 50 (with time_limit for scalability)
# ---------------------------------------------------------------------------

def test_match_n10_p4_seed_70():
    W, D = build_instance(10, 4, seed=70)
    assert run_instance(10, 4, W, D, time_limit=180.0)["match"]


def test_match_n10_p5_seed_71():
    W, D = build_instance(10, 5, seed=71)
    assert run_instance(10, 5, W, D, time_limit=180.0)["match"]


def test_match_n10_p6_seed_72():
    W, D = build_instance(10, 6, seed=72)
    assert run_instance(10, 6, W, D, time_limit=180.0)["match"]


def test_match_n12_p5_seed_73():
    W, D = build_instance(12, 5, seed=73)
    assert run_instance(12, 5, W, D, time_limit=200.0)["match"]


def test_match_n12_p6_seed_74():
    W, D = build_instance(12, 6, seed=74)
    assert run_instance(12, 6, W, D, time_limit=200.0)["match"]


def test_match_n15_p6_seed_75():
    W, D = build_instance(15, 6, seed=75)
    assert run_instance(15, 6, W, D, time_limit=240.0)["match"]


def test_match_n15_p8_seed_76():
    W, D = build_instance(15, 8, seed=76)
    assert run_instance(15, 8, W, D, time_limit=240.0)["match"]


def test_match_n20_p8_seed_77():
    W, D = build_instance(20, 8, seed=77)
    assert run_instance(20, 8, W, D, time_limit=300.0)["match"]


def test_match_n20_p10_seed_78():
    W, D = build_instance(20, 10, seed=78)
    assert run_instance(20, 10, W, D, time_limit=300.0)["match"]


def test_match_n25_p10_seed_79():
    W, D = build_instance(25, 10, seed=79)
    assert run_instance(25, 10, W, D, time_limit=360.0)["match"]


def test_match_n25_p12_seed_80():
    W, D = build_instance(25, 12, seed=80)
    assert run_instance(25, 12, W, D, time_limit=360.0)["match"]


def test_match_n30_p12_seed_81():
    W, D = build_instance(30, 12, seed=81)
    assert run_instance(30, 12, W, D, time_limit=400.0)["match"]


def test_match_n30_p15_seed_82():
    W, D = build_instance(30, 15, seed=82)
    assert run_instance(30, 15, W, D, time_limit=400.0)["match"]


def test_match_n35_p14_seed_83():
    W, D = build_instance(35, 14, seed=83)
    assert run_instance(35, 14, W, D, time_limit=480.0)["match"]


def test_match_n35_p18_seed_84():
    W, D = build_instance(35, 18, seed=84)
    assert run_instance(35, 18, W, D, time_limit=480.0)["match"]


def test_match_n40_p16_seed_85():
    W, D = build_instance(40, 16, seed=85)
    assert run_instance(40, 16, W, D, time_limit=600.0)["match"]


def test_match_n40_p20_seed_86():
    W, D = build_instance(40, 20, seed=86)
    assert run_instance(40, 20, W, D, time_limit=600.0)["match"]


def test_match_n45_p18_seed_87():
    W, D = build_instance(45, 18, seed=87)
    assert run_instance(45, 18, W, D, time_limit=600.0)["match"]


def test_match_n45_p22_seed_88():
    W, D = build_instance(45, 22, seed=88)
    assert run_instance(45, 22, W, D, time_limit=600.0)["match"]


def test_match_n50_p20_seed_89():
    W, D = build_instance(50, 20, seed=89)
    assert run_instance(50, 20, W, D, time_limit=600.0)["match"]


def test_match_n50_p25_seed_90():
    W, D = build_instance(50, 25, seed=90)
    assert run_instance(50, 25, W, D, time_limit=600.0)["match"]


def test_match_n10_p4_seed_91():
    W, D = build_instance(10, 4, seed=91)
    assert run_instance(10, 4, W, D, time_limit=120.0)["match"]


def test_match_n18_p8_seed_92():
    W, D = build_instance(18, 8, seed=92)
    assert run_instance(18, 8, W, D, time_limit=280.0)["match"]


def test_match_n22_p10_seed_93():
    W, D = build_instance(22, 10, seed=93)
    assert run_instance(22, 10, W, D, time_limit=320.0)["match"]


# Instance specs for script mode: n from 3 to 50 (fixed + random with seeds)
def get_script_instance_specs():
    """Return instance specs for main() script: small fixed + n=5..50 random."""
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    W2 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    return [
        # Fixed (small, deterministic)
        {"n": 3, "p": 2, "fixed": True, "W": W1, "D": D1},
        {"n": 3, "p": 3, "fixed": True, "W": W1, "D": D1},
        {"n": 4, "p": 2, "fixed": True, "W": W2, "D": D2},
        {"n": 4, "p": 3, "fixed": True, "W": W2, "D": D2},
        # Random n=5..8
        {"n": 5, "p": 2, "fixed": False},
        {"n": 5, "p": 3, "fixed": False},
        {"n": 6, "p": 3, "fixed": False},
        {"n": 6, "p": 4, "fixed": False},
        {"n": 7, "p": 4, "fixed": False},
        {"n": 8, "p": 4, "fixed": False},
        {"n": 8, "p": 5, "fixed": False},
        # n=10 to n=50 (script mode: 1 seed each for speed; pytest runs full seeds)
        {"n": 10, "p": 4, "fixed": False},
        {"n": 10, "p": 5, "fixed": False},
        {"n": 10, "p": 6, "fixed": False},
        {"n": 12, "p": 5, "fixed": False},
        {"n": 12, "p": 6, "fixed": False},
        {"n": 15, "p": 6, "fixed": False},
        {"n": 15, "p": 8, "fixed": False},
        {"n": 18, "p": 8, "fixed": False},
        {"n": 20, "p": 8, "fixed": False},
        {"n": 20, "p": 10, "fixed": False},
        {"n": 22, "p": 10, "fixed": False},
        {"n": 25, "p": 10, "fixed": False},
        {"n": 25, "p": 12, "fixed": False},
        {"n": 30, "p": 12, "fixed": False},
        {"n": 30, "p": 15, "fixed": False},
        {"n": 35, "p": 14, "fixed": False},
        {"n": 35, "p": 18, "fixed": False},
        {"n": 40, "p": 16, "fixed": False},
        {"n": 40, "p": 20, "fixed": False},
        {"n": 45, "p": 18, "fixed": False},
        {"n": 45, "p": 22, "fixed": False},
        {"n": 50, "p": 20, "fixed": False},
        {"n": 50, "p": 25, "fixed": False},
    ]


def main():
    print("\n" + "=" * 95)
    print("COMPREHENSIVE TIMING TEST: p-Hub-Arc (F3 vs Normal Benders vs New Benders)")
    print("=" * 95)

    instance_specs = get_script_instance_specs()
    # Script mode: 1 seed per random config for faster run (n=3..50). For 5 seeds, set seeds = [42,43,44,45,46].
    seeds = [42]
    time_limit = 300.0  # allow larger n (up to 50) to complete

    n_fixed = sum(1 for s in instance_specs if s.get("fixed"))
    n_random = len(instance_specs) - n_fixed
    total_runs = n_fixed * 1 + n_random * len(seeds)

    print(f"\nInstance specs: {len(instance_specs)} configurations (n=3..50)")
    print(f"Seeds per random config: {len(seeds)} (or 1 for fixed) -> {total_runs} total runs")
    print(f"Time limit per solve: {time_limit}s")
    print("Full pytest suite (all tests): python -m pytest p-hub-arc/test_timing_comprehensive.py -v")
    print("\nRunning suite...")
    print("-" * 60)

    results = run_timing_suite(
        instance_specs,
        seeds=seeds,
        time_limit=time_limit,
        use_phase1=True,
        verbose=True,
    )

    agg = aggregate_by_instance(results)
    print_timing_table(results, agg)
    print_detailed_results(results)

    # Summary
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    total = len([r for r in results if "error" not in r])
    matches = sum(1 for r in results if r.get("match"))
    f3_wins = norm_wins = new_wins = 0
    for r in results:
        if "error" in r:
            continue
        times = [("F3", r.get("f3_time")), ("NormB", r.get("norm_time")), ("NewB", r.get("new_time"))]
        valid = [(n, t) for n, t in times if t is not None]
        if valid:
            fastest = min(valid, key=lambda x: x[1])[0]
            if fastest == "F3":
                f3_wins += 1
            elif fastest == "NormB":
                norm_wins += 1
            else:
                new_wins += 1

    print(f"Total runs:       {total}")
    print(f"Objectives match: {matches}/{total}")
    print(f"F3 fastest:       {f3_wins} runs")
    print(f"Normal Benders fastest: {norm_wins} runs")
    print(f"New Benders fastest:    {new_wins} runs")

    ok = matches == total
    print("\n" + ("All objectives matched." if ok else "WARNING: Some objectives did not match."))
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
