"""
Speed benchmark: New Benders vs ML-assisted Benders (10+ instances).

Run from p-hub-arc:
    python3 benchmark_ml_vs_new_timing.py
"""

import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

from new_model_hub_arc import solve_benders_hub_arc
from ml_benders_hub_arc import solve_ml_benders_hub_arc
from test_timing_comprehensive import build_instance

TIME_LIMIT = 300.0
TOL = 1e-4

# 12 diverse small/medium instances
CASES = [
    (4, 2, None),   # fixed 4x4 below
    (4, 2, 42),
    (4, 3, 43),
    (5, 2, 44),
    (5, 3, 45),
    (6, 3, 46),
    (6, 4, 47),
    (7, 4, 48),
    (8, 4, 49),
    (8, 5, 50),
    (10, 4, 51),
    (10, 5, 52),
]

FIXED_W = [
    [0, 10, 20, 15],
    [12, 0, 18, 25],
    [14, 11, 0, 30],
    [20, 16, 22, 0],
]
FIXED_D = [
    [0, 2, 6, 8],
    [2, 0, 4, 5],
    [6, 4, 0, 3],
    [8, 5, 3, 0],
]


def run_benchmark():
    rows = []
    new_wins = ml_exact_wins = ml_default_wins = 0
    ties = 0

    print("=" * 88)
    print("TIMING BENCHMARK: New Benders vs ML Benders")
    print(f"Time limit per solve: {TIME_LIMIT}s | {len(CASES)} instances")
    print("=" * 88)
    print(
        f"{'Case':<18} {'New(s)':>8} {'ML-ex(s)':>9} {'ML-def(s)':>10} "
        f"{'Winner':<12} {'Match':>6}"
    )
    print("-" * 88)

    for n, p, seed in CASES:
        if seed is None:
            W, D = FIXED_W, FIXED_D
            label = f"n={n},p={p},fix"
        else:
            W, D = build_instance(n, p, seed)
            label = f"n={n},p={p},s={seed}"

        new = solve_benders_hub_arc(
            n, p, W, D, time_limit=TIME_LIMIT, verbose=False, use_phase1=True
        )
        ml_ex = solve_ml_benders_hub_arc(
            n, p, W, D, time_limit=TIME_LIMIT, verbose=False, use_phase1=True,
            arc_model=None, od_model=None,
            ml_pair_fraction=1.0, max_cuts_per_callback=None,
        )
        ml_def = solve_ml_benders_hub_arc(
            n, p, W, D, time_limit=TIME_LIMIT, verbose=False, use_phase1=True,
            arc_model=None, od_model=None,
            ml_pair_fraction=0.25, max_cuts_per_callback=20,
        )

        t_new = new.get("time") or float("inf")
        t_ex = ml_ex.get("time") or float("inf")
        t_def = ml_def.get("time") or float("inf")

        times = {"New": t_new, "ML-exact": t_ex, "ML-default": t_def}
        best = min(times.values())
        winners = [k for k, v in times.items() if abs(v - best) < 1e-6]

        if len(winners) > 1:
            winner = "tie"
            ties += 1
        else:
            winner = winners[0]
            if winner == "New":
                new_wins += 1
            elif winner == "ML-exact":
                ml_exact_wins += 1
            else:
                ml_default_wins += 1

        obj_match = (
            new.get("objective") is not None
            and ml_ex.get("objective") is not None
            and ml_def.get("objective") is not None
            and abs(new["objective"] - ml_ex["objective"]) < TOL
            and abs(new["objective"] - ml_def["objective"]) < TOL
        )

        rows.append({
            "label": label,
            "t_new": t_new,
            "t_ex": t_ex,
            "t_def": t_def,
            "winner": winner,
            "match": obj_match,
            "new_status": new.get("status"),
            "ml_ex_status": ml_ex.get("status"),
            "ml_def_status": ml_def.get("status"),
        })

        print(
            f"{label:<18} {t_new:8.3f} {t_ex:9.3f} {t_def:10.3f} "
            f"{winner:<12} {'YES' if obj_match else 'NO':>6}"
        )

    total_new = sum(r["t_new"] for r in rows)
    total_ex = sum(r["t_ex"] for r in rows)
    total_def = sum(r["t_def"] for r in rows)

    print("-" * 88)
    print(f"{'TOTAL':<18} {total_new:8.3f} {total_ex:9.3f} {total_def:10.3f}")
    print("=" * 88)
    print("WINS (fastest per case):")
    print(f"  New Benders:     {new_wins}/{len(CASES)}")
    print(f"  ML exact:        {ml_exact_wins}/{len(CASES)}")
    print(f"  ML default:      {ml_default_wins}/{len(CASES)}")
    print(f"  Ties:            {ties}/{len(CASES)}")
    print()
    print("TOTAL WALL TIME (sum of all cases):")
    print(f"  New Benders:     {total_new:.3f}s")
    print(f"  ML exact:        {total_ex:.3f}s  ({100*(total_ex/total_new - 1):+.1f}% vs New)")
    print(f"  ML default:      {total_def:.3f}s  ({100*(total_def/total_new - 1):+.1f}% vs New)")
    fastest_total = min(total_new, total_ex, total_def)
    if fastest_total == total_new:
        print("\nOverall fastest (total time): New Benders")
    elif fastest_total == total_def:
        print("\nOverall fastest (total time): ML default")
    else:
        print("\nOverall fastest (total time): ML exact")
    print(f"\nObjectives matched on all cases: {all(r['match'] for r in rows)}")


if __name__ == "__main__":
    run_benchmark()
