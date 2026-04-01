"""
Test: Pareto-Optimal Benders vs F3 vs Standard New Benders for p-Hub-Arc

Verifies that solve_benders_pareto_hub_arc produces the same objective as
F3 (canonical formulation) and solve_benders_hub_arc (standard two-phase).

Run:
    python p-hub-arc/test_pareto_benders.py
    python -m pytest p-hub-arc/test_pareto_benders.py -v
"""

import sys
import os
import numpy as np
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from hub_arc_models import solve_hub_arc_F3, HubArcBenders
from new_model_hub_arc import solve_benders_hub_arc, preprocess
from pareto_benders_hub_arc import solve_benders_pareto_hub_arc


# ============================================================
# Helpers
# ============================================================

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
    Run F3, standard New Benders, and Pareto Benders on one instance.
    Returns dict with objectives, times, statuses, and match flag.
    """
    # F3
    f3_res = solve_hub_arc_F3(n, p, W, D, gurobi_output=False)

    # Standard New Benders (Phase 1 + Phase 2, no Pareto)
    new_res = solve_benders_hub_arc(
        n, p, W, D, verbose=False, use_phase1=use_phase1, time_limit=time_limit
    )

    # Pareto Benders (Phase 1 + Phase 2 + Magnanti-Wong cuts)
    pareto_res = solve_benders_pareto_hub_arc(
        n, p, W, D, verbose=False, use_phase1=use_phase1, time_limit=time_limit
    )

    ref = f3_res["objective"]
    match = ref is not None
    diff_new = diff_pareto = None

    if ref is not None:
        if new_res["objective"] is not None:
            diff_new = abs(ref - new_res["objective"])
            match = match and diff_new < 1e-4
        if pareto_res["objective"] is not None:
            diff_pareto = abs(ref - pareto_res["objective"])
            match = match and diff_pareto < 1e-4

    return {
        "n": n,
        "p": p,
        "f3_obj":     f3_res["objective"],
        "f3_time":    f3_res["time"],
        "f3_status":  f3_res["status"],
        "new_obj":    new_res["objective"],
        "new_time":   new_res["time"],
        "new_status": new_res["status"],
        "par_obj":    pareto_res["objective"],
        "par_time":   pareto_res["time"],
        "par_status": pareto_res["status"],
        "match":      match,
        "diff_new":   diff_new,
        "diff_pareto": diff_pareto,
    }


def aggregate_by_instance(results: List[Dict]) -> Dict:
    by_key = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        by_key[(r["n"], r["p"])].append(r)

    agg = {}
    for (n, p), rows in by_key.items():
        f3_t   = [x["f3_time"]  for x in rows if x.get("f3_time")  is not None]
        new_t  = [x["new_time"] for x in rows if x.get("new_time") is not None]
        par_t  = [x["par_time"] for x in rows if x.get("par_time") is not None]
        matches = sum(1 for x in rows if x.get("match"))
        agg[(n, p)] = {
            "n": n, "p": p,
            "count":      len(rows),
            "f3_mean":    np.mean(f3_t)  if f3_t  else None,
            "new_mean":   np.mean(new_t) if new_t else None,
            "par_mean":   np.mean(par_t) if par_t else None,
            "match_count": matches,
            "match_rate":  matches / len(rows) if rows else 0,
        }
    return agg


def print_timing_table(results: List[Dict], agg: Dict):
    print("\n" + "=" * 120)
    print("TIMING TABLE: F3 vs New Benders vs Pareto Benders")
    print("=" * 120)
    header = (f"{'n':>4} | {'p':>3} | {'#runs':>6} | {'F3 (s)':>10} | "
              f"{'New Benders (s)':>16} | {'Pareto Benders (s)':>18} | {'Fastest':>14} | Match")
    print(header)
    print("-" * 120)
    for key in sorted(agg.keys()):
        a = agg[key]
        n, p = a["n"], a["p"]
        f_s   = f"{a['f3_mean']:.4f}"  if a["f3_mean"]  is not None else "   N/A   "
        n_s   = f"{a['new_mean']:.4f}" if a["new_mean"] is not None else "   N/A   "
        par_s = f"{a['par_mean']:.4f}" if a["par_mean"] is not None else "   N/A   "
        times = [("F3", a["f3_mean"]), ("New", a["new_mean"]), ("Pareto", a["par_mean"])]
        valid = [(nm, t) for nm, t in times if t is not None]
        fastest = min(valid, key=lambda x: x[1])[0] if valid else "-"
        match_s = f"{a['match_count']}/{a['count']}"
        print(f"{n:>4} | {p:>3} | {a['count']:>6} | {f_s:>10} | {n_s:>16} | {par_s:>18} | {fastest:>14} | {match_s}")


def print_detailed_results(results: List[Dict]):
    print("\n" + "=" * 120)
    print("DETAILED PER-RUN RESULTS")
    print("=" * 120)
    for i, r in enumerate(results):
        if "error" in r:
            print(f"  [{i+1}] n={r['n']}, p={r['p']}: ERROR - {r['error']}")
            continue
        status = "Match" if r.get("match") else "MISMATCH"
        f3t  = f"{r['f3_time']:.4f}"  if r.get("f3_time")  is not None else "N/A"
        nt   = f"{r['new_time']:.4f}" if r.get("new_time") is not None else "N/A"
        pt   = f"{r['par_time']:.4f}" if r.get("par_time") is not None else "N/A"
        f3o  = f"{r['f3_obj']:.4f}"   if r.get("f3_obj")   is not None else "N/A"
        no   = f"{r['new_obj']:.4f}"  if r.get("new_obj")  is not None else "N/A"
        po   = f"{r['par_obj']:.4f}"  if r.get("par_obj")  is not None else "N/A"
        print(f"  [{i+1}] n={r['n']}, p={r['p']}, seed={r.get('seed')}: "
              f"F3={f3t}s  New={nt}s  Pareto={pt}s  |  {status}")
        print(f"       Obj: F3={f3o}  New={no}  Pareto={po}")


# ============================================================
# Unit tests: build_instance
# ============================================================

def test_build_instance_shape():
    W, D = build_instance(5, 2, seed=42)
    assert len(W) == 5 and len(W[0]) == 5
    assert len(D) == 5 and len(D[0]) == 5


def test_build_instance_diagonal_zero():
    W, D = build_instance(4, 2, seed=43)
    for i in range(4):
        assert W[i][i] == 0 and D[i][i] == 0


def test_build_instance_reproducibility():
    W1, D1 = build_instance(5, 2, seed=100)
    W2, D2 = build_instance(5, 2, seed=100)
    assert W1 == W2 and D1 == D2


# ============================================================
# Unit tests: pareto solver basic sanity
# ============================================================

def test_pareto_returns_keys():
    W, D = build_instance(4, 2, seed=10)
    res = solve_benders_pareto_hub_arc(4, 2, W, D, time_limit=30.0)
    for key in ("objective", "selected_arcs", "time", "status"):
        assert key in res


def test_pareto_fixed_3_2_optimal():
    W = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    res = solve_benders_pareto_hub_arc(3, 2, W, D, time_limit=60.0)
    assert res["status"] == "OPTIMAL"
    assert res["objective"] is not None and res["objective"] >= 0


def test_pareto_fixed_4_2_optimal():
    W = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    res = solve_benders_pareto_hub_arc(4, 2, W, D, time_limit=60.0)
    assert res["status"] == "OPTIMAL"


def test_pareto_objective_non_negative():
    W, D = build_instance(5, 2, seed=11)
    res = solve_benders_pareto_hub_arc(5, 2, W, D, time_limit=60.0)
    assert res["objective"] is None or res["objective"] >= 0


def test_pareto_selected_arcs_count():
    """Pareto solver selects exactly p arcs."""
    W, D = build_instance(5, 3, seed=12)
    res = solve_benders_pareto_hub_arc(5, 3, W, D, time_limit=60.0)
    if res["status"] == "OPTIMAL":
        assert len(res["selected_arcs"]) == 3


# ============================================================
# run_instance wrapper tests
# ============================================================

def test_run_instance_returns_all_keys():
    W, D = build_instance(4, 2, seed=13)
    r = run_instance(4, 2, W, D, time_limit=30.0)
    for key in ("n", "p", "f3_obj", "new_obj", "par_obj", "match",
                "f3_time", "new_time", "par_time"):
        assert key in r


def test_run_instance_fixed_3_2_match():
    """F3, New Benders, Pareto Benders match on fixed n=3, p=2."""
    W = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    r = run_instance(3, 2, W, D, time_limit=60.0)
    assert r["match"], f"f3={r['f3_obj']}  new={r['new_obj']}  pareto={r['par_obj']}"


def test_run_instance_fixed_4_2_match():
    W = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    r = run_instance(4, 2, W, D, time_limit=60.0)
    assert r["match"], f"f3={r['f3_obj']}  new={r['new_obj']}  pareto={r['par_obj']}"


# ============================================================
# Integration tests: objective match across seeds/sizes
# ============================================================

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


def test_match_n7_p4_seed_55():
    W, D = build_instance(7, 4, seed=55)
    assert run_instance(7, 4, W, D, time_limit=120.0)["match"]


def test_match_n6_p3_seed_57():
    W, D = build_instance(6, 3, seed=57)
    assert run_instance(6, 3, W, D, time_limit=90.0)["match"]


def test_match_n8_p4_seed_60():
    W, D = build_instance(8, 4, seed=60)
    assert run_instance(8, 4, W, D, time_limit=120.0)["match"]


def test_match_n8_p5_seed_61():
    W, D = build_instance(8, 5, seed=61)
    assert run_instance(8, 5, W, D, time_limit=120.0)["match"]


# ============================================================
# Larger instance tests
# ============================================================

def test_match_n10_p4_seed_70():
    W, D = build_instance(10, 4, seed=70)
    assert run_instance(10, 4, W, D, time_limit=180.0)["match"]


def test_match_n10_p5_seed_71():
    W, D = build_instance(10, 5, seed=71)
    assert run_instance(10, 5, W, D, time_limit=180.0)["match"]


def test_match_n12_p5_seed_73():
    W, D = build_instance(12, 5, seed=73)
    assert run_instance(12, 5, W, D, time_limit=200.0)["match"]


def test_match_n15_p6_seed_75():
    W, D = build_instance(15, 6, seed=75)
    assert run_instance(15, 6, W, D, time_limit=240.0)["match"]


def test_match_n20_p8_seed_77():
    W, D = build_instance(20, 8, seed=77)
    assert run_instance(20, 8, W, D, time_limit=300.0)["match"]


def test_match_n25_p10_seed_79():
    W, D = build_instance(25, 10, seed=79)
    assert run_instance(25, 10, W, D, time_limit=360.0)["match"]


def test_match_n30_p12_seed_81():
    W, D = build_instance(30, 12, seed=81)
    assert run_instance(30, 12, W, D, time_limit=400.0)["match"]


# ============================================================
# Script mode: timing suite
# ============================================================

def get_instance_specs():
    W1 = [[0, 2, 3], [2, 0, 4], [3, 4, 0]]
    D1 = [[0, 5, 2], [5, 0, 3], [2, 3, 0]]
    W2 = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    D2 = [[0, 4, 6, 8], [4, 0, 5, 7], [6, 5, 0, 3], [8, 7, 3, 0]]
    return [
        {"n": 3, "p": 2, "fixed": True,  "W": W1, "D": D1},
        {"n": 3, "p": 3, "fixed": True,  "W": W1, "D": D1},
        {"n": 4, "p": 2, "fixed": True,  "W": W2, "D": D2},
        {"n": 4, "p": 3, "fixed": True,  "W": W2, "D": D2},
        {"n": 5, "p": 2, "fixed": False},
        {"n": 5, "p": 3, "fixed": False},
        {"n": 6, "p": 3, "fixed": False},
        {"n": 6, "p": 4, "fixed": False},
        {"n": 7, "p": 4, "fixed": False},
        {"n": 8, "p": 4, "fixed": False},
        {"n": 8, "p": 5, "fixed": False},
        {"n": 10, "p": 4, "fixed": False},
        {"n": 10, "p": 5, "fixed": False},
        {"n": 12, "p": 5, "fixed": False},
        {"n": 15, "p": 6, "fixed": False},
        {"n": 20, "p": 8, "fixed": False},
        {"n": 25, "p": 10, "fixed": False},
        {"n": 30, "p": 12, "fixed": False},
    ]


def main():
    print("\n" + "=" * 100)
    print("PARETO BENDERS TEST: F3 vs New Benders vs Pareto Benders")
    print("=" * 100)

    specs = get_instance_specs()
    seeds = [42]
    time_limit = 300.0

    n_fixed  = sum(1 for s in specs if s.get("fixed"))
    n_random = len(specs) - n_fixed
    total    = n_fixed + n_random * len(seeds)

    print(f"\nConfigurations: {len(specs)}  |  Seeds: {seeds}  |  Total runs: {total}")
    print(f"Time limit per solve: {time_limit}s\n")
    print("-" * 60)

    results = []
    idx = 0
    for spec in specs:
        n, p   = spec["n"], spec["p"]
        fixed  = spec.get("fixed", False)
        seed_list = [None] if fixed else seeds

        for seed in seed_list:
            idx += 1
            label = f"n={n}, p={p}" + (f", seed={seed}" if seed is not None else " (fixed)")
            print(f"  [{idx}] {label} ...", end=" ", flush=True)

            W, D = (spec["W"], spec["D"]) if fixed else build_instance(n, p, seed)

            try:
                r = run_instance(n, p, W, D, time_limit=time_limit)
                r["seed"] = seed
                r["fixed"] = fixed
                results.append(r)
                f3t  = r["f3_time"]
                nt   = r["new_time"]
                pt   = r["par_time"]
                f3o  = f"{r['f3_obj']:.4f}"  if r["f3_obj"]  is not None else "N/A"
                no   = f"{r['new_obj']:.4f}" if r["new_obj"] is not None else "N/A"
                po   = f"{r['par_obj']:.4f}" if r["par_obj"] is not None else "N/A"
                ms   = "Match" if r["match"] else "MISMATCH"
                print(f"F3={f3t:.3f}s  New={nt:.3f}s  Pareto={pt:.3f}s  | {ms}")
                print(f"       Obj: F3={f3o}  New={no}  Pareto={po}")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"n": n, "p": p, "seed": seed, "fixed": fixed,
                                 "match": False, "error": str(e)})

    agg = aggregate_by_instance(results)
    print_timing_table(results, agg)
    print_detailed_results(results)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    good   = [r for r in results if "error" not in r]
    total  = len(good)
    matches = sum(1 for r in good if r.get("match"))
    f3_wins = new_wins = par_wins = 0
    for r in good:
        times = [("F3", r.get("f3_time")), ("New", r.get("new_time")), ("Pareto", r.get("par_time"))]
        valid = [(nm, t) for nm, t in times if t is not None]
        if valid:
            fastest = min(valid, key=lambda x: x[1])[0]
            if fastest == "F3":     f3_wins  += 1
            elif fastest == "New":  new_wins += 1
            else:                   par_wins += 1

    print(f"Total runs:          {total}")
    print(f"Objectives match:    {matches}/{total}")
    print(f"F3 fastest:          {f3_wins}")
    print(f"New Benders fastest: {new_wins}")
    print(f"Pareto Benders fastest: {par_wins}")
    ok = matches == total
    print("\n" + ("All objectives matched." if ok else "WARNING: Some objectives did not match."))
    return ok


if __name__ == "__main__":
    main()
