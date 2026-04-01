"""
Tests and timing comparison for McDaniel & Devine two-phase Benders.

Compares four solvers on the same random instances:
  1. F3          — direct canonical MIP formulation (ground truth)
  2. plain_b     — standard Benders (Phase 1 LP warm-start only, cuts discarded)
  3. pareto_b    — Benders + Magnanti-Wong Pareto-optimal cuts
  4. md_b        — McDaniel & Devine (Phase 1 cuts carried into Phase 2)

Run:
    python test_md_benders.py
    python -m pytest p-hub-arc/test_md_benders.py -v
"""

import sys
import os
import time
import numpy as np
from typing import List, Optional, Dict, Any

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from hub_arc_models import solve_hub_arc_F3
from new_model_hub_arc import solve_benders_hub_arc, preprocess
from pareto_benders_hub_arc import solve_benders_pareto_hub_arc
from md_benders_hub_arc import solve_md_benders_hub_arc, _preprocess, _phase1, _allocation_cost
from algo3_hub_arc import phase1_solve_mp, compute_allocation_cost


# ============================================================
# Helpers
# ============================================================

def build_instance(n: int, seed: int):
    np.random.seed(seed)
    W = np.random.rand(n, n) * 10
    np.fill_diagonal(W, 0)
    D = np.random.rand(n, n) * 20
    np.fill_diagonal(D, 0)
    return W.tolist(), D.tolist()


def _obj_match(a, b, tol=1e-4):
    if a is None or b is None:
        return False
    return abs(a - b) < tol


def run_all(n: int, p: int, seed: int, time_limit: Optional[float] = 120.0) -> Dict[str, Any]:
    """Run all four solvers on one instance and return a result dict."""
    W, D = build_instance(n, seed)

    # 1. F3 (ground truth)
    f3 = solve_hub_arc_F3(n, p, W, D, gurobi_output=False)

    # 2. Plain Benders (Phase 1 LP warm-start only)
    pb = solve_benders_hub_arc(n, p, W, D, verbose=False,
                                use_phase1=True, time_limit=time_limit)
    # Add constant for K==1 OD pairs (already done inside; included here for clarity)

    # 3. Pareto Benders (Magnanti-Wong)
    pab = solve_benders_pareto_hub_arc(n, p, W, D, verbose=False,
                                        use_phase1=True, time_limit=time_limit)

    # 4. McDaniel & Devine
    md = solve_md_benders_hub_arc(n, p, W, D, verbose=False,
                                   use_phase1=True, time_limit=time_limit)

    ref = f3["objective"]
    return {
        "n": n, "p": p, "seed": seed,
        # objectives
        "f3_obj":     ref,
        "plain_obj":  pb["objective"],
        "pareto_obj": pab["objective"],
        "md_obj":     md["objective"],
        # times
        "f3_time":     f3["time"],
        "plain_time":  pb["time"],
        "pareto_time": pab["time"],
        "md_time":     md["time"],
        # match vs F3
        "plain_match":  _obj_match(ref, pb["objective"]),
        "pareto_match": _obj_match(ref, pab["objective"]),
        "md_match":     _obj_match(ref, md["objective"]),
        # M&D specific
        "md_phase1_cuts": md.get("phase1_cuts", 0),
    }


def print_row(r: Dict):
    match_str = (
        ("P" if r["plain_match"]  else "X") +
        ("P" if r["pareto_match"] else "X") +
        ("P" if r["md_match"]     else "X")
    )
    print(
        f"n={r['n']:2d} p={r['p']} seed={r['seed']:3d} | "
        f"F3={r['f3_obj']:10.4f} ({r['f3_time']:5.1f}s) | "
        f"Plain={r['plain_time']:5.1f}s | "
        f"Pareto={r['pareto_time']:5.1f}s | "
        f"MD={r['md_time']:5.1f}s cuts={r['md_phase1_cuts']:3d} | "
        f"match=[plain/pareto/md]={match_str}"
    )


# ============================================================
# pytest unit tests
# ============================================================

def test_md_correctness_small():
    """M&D must match F3 on a small instance."""
    r = run_all(n=5, p=4, seed=42, time_limit=60)
    assert r["md_match"], (
        f"M&D obj={r['md_obj']:.6f} != F3 obj={r['f3_obj']:.6f}"
    )


def test_md_correctness_medium():
    """M&D must match F3 on a medium instance."""
    r = run_all(n=7, p=6, seed=7, time_limit=120)
    assert r["md_match"], (
        f"M&D obj={r['md_obj']:.6f} != F3 obj={r['f3_obj']:.6f}"
    )


def test_md_no_phase1():
    """M&D with use_phase1=False (plain B&B) must still match F3."""
    W, D = build_instance(n=5, seed=0)
    md = solve_md_benders_hub_arc(5, 4, W, D, use_phase1=False, time_limit=60)
    f3 = solve_hub_arc_F3(5, 4, W, D, gurobi_output=False)
    assert _obj_match(f3["objective"], md["objective"]), (
        f"M&D (no Phase1) obj={md['objective']:.6f} != F3={f3['objective']:.6f}"
    )


def test_md_cuts_preloaded():
    """Phase 1 must generate at least some cuts for a non-trivial instance."""
    W, D = build_instance(n=6, seed=99)
    md = solve_md_benders_hub_arc(6, 5, W, D, use_phase1=True, time_limit=60)
    assert md["phase1_cuts"] > 0, "Expected Phase 1 to generate at least one cut."


def test_all_match_small():
    """All three Benders variants must match F3 on a small instance."""
    r = run_all(n=5, p=3, seed=123, time_limit=60)
    assert r["plain_match"],  f"Plain mismatch: {r['plain_obj']} vs {r['f3_obj']}"
    assert r["pareto_match"], f"Pareto mismatch: {r['pareto_obj']} vs {r['f3_obj']}"
    assert r["md_match"],     f"MD mismatch: {r['md_obj']} vs {r['f3_obj']}"


# ============================================================
# Timing / comparison suite (run directly)
# ============================================================

INSTANCES = [
    {"n": 5,  "p": 3},
    {"n": 5,  "p": 4},
    {"n": 6,  "p": 4},
    {"n": 7,  "p": 5},
    {"n": 8,  "p": 6},
    {"n": 10, "p": 7},
]
SEEDS = [0, 1, 42]
TIME_LIMIT = 120.0


def run_suite():
    print("\n" + "=" * 110)
    print(f"{'McDaniel & Devine vs Plain Benders vs Pareto Benders vs F3':^110}")
    print("=" * 110)
    print(
        f"{'n':>3} {'p':>2} {'seed':>4} | "
        f"{'F3 obj':>12} {'F3t':>5} | "
        f"{'Plaint':>7} | {'Paretot':>8} | {'MDt':>5} {'cuts':>4} | match"
    )
    print("-" * 110)

    results = []
    for spec in INSTANCES:
        for seed in SEEDS:
            try:
                r = run_all(spec["n"], spec["p"], seed, time_limit=TIME_LIMIT)
                print_row(r)
                results.append(r)
            except Exception as e:
                print(f"n={spec['n']} p={spec['p']} seed={seed} ERROR: {e}")

    print("=" * 110)
    total = len(results)
    ok_plain  = sum(1 for r in results if r["plain_match"])
    ok_pareto = sum(1 for r in results if r["pareto_match"])
    ok_md     = sum(1 for r in results if r["md_match"])
    print(f"Correctness: plain={ok_plain}/{total}  pareto={ok_pareto}/{total}  MD={ok_md}/{total}")

    if results:
        avg_plain  = sum(r["plain_time"]  for r in results) / total
        avg_pareto = sum(r["pareto_time"] for r in results) / total
        avg_md     = sum(r["md_time"]     for r in results) / total
        avg_cuts   = sum(r["md_phase1_cuts"] for r in results) / total
        print(f"Avg time:    plain={avg_plain:.2f}s  pareto={avg_pareto:.2f}s  MD={avg_md:.2f}s")
        print(f"Avg Phase1 cuts preloaded into MD Phase2: {avg_cuts:.1f}")
    print()


def compare_phases_detail(n: int, p: int, seed: int):
    """
    For a single instance, explicitly run Phase 1 and Phase 2 for each method
    and print both the Phase 1 upper bound and the final Phase 2 objective.
    Confirms all three methods agree at every stage with F3.
    """
    W, D = build_instance(n, seed)
    H, C, L, K, cost_map, arcs_sorted, od_pairs = _preprocess(n, W, D)
    const_k1 = sum(C[(i, j)][0] for (i, j) in K if i != j and K[(i, j)] == 1)

    f3 = solve_hub_arc_F3(n, p, W, D, gurobi_output=False)
    ref = f3["objective"]

    print(f"\n{'='*70}")
    print(f"Phase-by-phase comparison  n={n} p={p} seed={seed}")
    print(f"F3 (ground truth) = {ref:.6f}")
    print(f"{'='*70}")

    # ── Plain Benders ────────────────────────────────────────────────────
    y_heur = {a: (1.0 if idx < p else 0.0) for idx, a in enumerate(H)}
    UB_h = sum(
        compute_allocation_cost(i, j, y_heur, cost_map[(i, j)], arcs_sorted[(i, j)])
        for (i, j) in od_pairs
    ) + const_k1

    LB1, _, UB1, _, _, _ = phase1_solve_mp(
        n, p, H, C, L, K, od_pairs, cost_map, arcs_sorted,
        y_heuristic=y_heur, UB_heuristic=UB_h, verbose=False,
    )
    plain_phase2 = solve_benders_hub_arc(n, p, W, D, verbose=False, use_phase1=True)
    print(f"\nPlain Benders:")
    print(f"  Phase 1 LP lower bound (LB)     = {LB1:.6f}")
    print(f"  Phase 1 rounded upper bound (UB) = {UB1:.6f}")
    print(f"  Phase 2 final objective          = {plain_phase2['objective']:.6f}  {'OK' if _obj_match(ref, plain_phase2['objective']) else 'MISMATCH'}")

    # ── Pareto Benders ───────────────────────────────────────────────────
    # Phase 1 LP is identical to plain (same master, same cuts — only Phase 2 differs)
    pareto_phase2 = solve_benders_pareto_hub_arc(n, p, W, D, verbose=False, use_phase1=True)
    print(f"\nPareto Benders (Magnanti-Wong):")
    print(f"  Phase 1 LP lower bound (LB)     = {LB1:.6f}  (same LP as plain)")
    print(f"  Phase 1 rounded upper bound (UB) = {UB1:.6f}  (same LP as plain)")
    print(f"  Phase 2 final objective          = {pareto_phase2['objective']:.6f}  {'OK' if _obj_match(ref, pareto_phase2['objective']) else 'MISMATCH'}")

    # ── McDaniel & Devine ────────────────────────────────────────────────
    md_LB, _, md_UB, md_cuts = _phase1(
        n, p, H, C, L, K, od_pairs, cost_map, arcs_sorted,
        time_budget=None, max_iter=500, verbose=False,
    )
    md_phase2 = solve_md_benders_hub_arc(n, p, W, D, verbose=False, use_phase1=True)
    print(f"\nMcDaniel & Devine:")
    print(f"  Phase 1 LP lower bound (LB)     = {md_LB:.6f}")
    print(f"  Phase 1 rounded upper bound (UB) = {md_UB:.6f}")
    print(f"  Phase 1 cuts accumulated        = {len(md_cuts)}")
    print(f"  Phase 2 final objective          = {md_phase2['objective']:.6f}  {'OK' if _obj_match(ref, md_phase2['objective']) else 'MISMATCH'}")

    # ── Summary ──────────────────────────────────────────────────────────
    all_match = (
        _obj_match(ref, plain_phase2["objective"]) and
        _obj_match(ref, pareto_phase2["objective"]) and
        _obj_match(ref, md_phase2["objective"])
    )
    print(f"\n  All Phase 2 finals match F3: {'YES' if all_match else 'NO'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Detailed phase-by-phase comparison on a few instances
    for (n, p, seed) in [(5, 3, 0), (6, 4, 42), (8, 6, 1)]:
        compare_phases_detail(n, p, seed)

    print()
    # Full timing suite
    run_suite()
