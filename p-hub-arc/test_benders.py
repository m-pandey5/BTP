# # """
# # test_hub_arc_models.py

# # Run several small instances of the p-Hub-Arc problem and compare:
# #   - Direct F3 canonical formulation (solve_hub_arc_F3)
# #   - Benders decomposition (HubArcBenders)

# # Usage:
# #   python3 test_hub_arc_models.py
# # """

# # from math import isclose
# # from hub_arc_models import solve_hub_arc_F3, HubArcBenders


# # def run_instance(
# #     name: str,
# #     n: int,
# #     p: int,
# #     W,
# #     D,
# #     tol: float = 1e-5,
# # ):
# #     print("=" * 70)
# #     print(f"TEST INSTANCE: {name}")
# #     print(f"n = {n}, p = {p}")
# #     print("W:")
# #     for row in W:
# #         print("  ", row)
# #     print("D:")
# #     for row in D:
# #         print("  ", row)

# #     # ---- F3 formulation ----
# #     f3_res = solve_hub_arc_F3(n, p, W, D, gurobi_output=False)
# #     print("\n[F3] status:", f3_res["status"])
# #     print("[F3] objective:", f3_res["objective"])
# #     print("[F3] selected arcs:", sorted(f3_res["selected_arcs"]))

# #     # ---- Benders ----
# #     benders = HubArcBenders(n=n, p=p, W=W, D=D, verbose=False)
# #     b_res = benders.solve(time_limit=60)
# #     print("\n[Benders] status:", b_res["status"])
# #     print("[Benders] objective:", b_res["objective"])
# #     print("[Benders] selected arcs:", sorted(b_res["selected_arcs"]))

# #     # ---- Check / compare ----
# #     ok_status = (f3_res["status"] == "OPTIMAL") and (b_res["status"] == 2)
# #     # In Gurobi: 2 == OPTIMAL status
# #     if not ok_status:
# #         print("\nWARNING: One of the models is not optimal, skipping numerical check.")
# #         return

# #     if isclose(f3_res["objective"], b_res["objective"], rel_tol=tol, abs_tol=tol):
# #         print("\nRESULT: ✅ Objectives match within tolerance.")
# #     else:
# #         print("\nRESULT: ❌ Objectives DIFFER!")
# #         print("  |F3 - Benders| =", abs(f3_res["objective"] - b_res["objective"]))

# #     print("=" * 70, "\n")


# # def main():
# #     # ------------------------------------------------------------------
# #     # Test 1: Your original 3-node instance, p = 2
# #     # ------------------------------------------------------------------
# #     n1 = 3
# #     p1 = 2

# #     W1 = [
# #         [0, 2, 3],
# #         [2, 0, 4],
# #         [3, 4, 0],
# #     ]

# #     D1 = [
# #         [0, 5, 2],
# #         [5, 0, 3],
# #         [2, 3, 0],
# #     ]

# #     run_instance("Toy_3_nodes_p2", n1, p1, W1, D1)

# #     # ------------------------------------------------------------------
# #     # Test 2: Same 3-node instance, different p (p = 3)
# #     # ------------------------------------------------------------------
# #     p2 = 3
# #     run_instance("Toy_3_nodes_p3", n1, p2, W1, D1)

# #     # ------------------------------------------------------------------
# #     # Test 3: 3-node instance, p = 1 (edge case: very few hub arcs)
# #     # ------------------------------------------------------------------
# #     p3 = 1
# #     run_instance("Toy_3_nodes_p1", n1, p3, W1, D1)

# #     # ------------------------------------------------------------------
# #     # Test 4: 4-node symmetric flows and distances, p = 3
# #     # ------------------------------------------------------------------
# #     n4 = 4
# #     p4 = 3

# #     # flows: symmetric, more or less increasing with distance
# #     W4 = [
# #         [0, 1, 2, 3],
# #         [1, 0, 1, 2],
# #         [2, 1, 0, 1],
# #         [3, 2, 1, 0],
# #     ]

# #     # distances: simple grid-like symmetric matrix
# #     D4 = [
# #         [0, 4, 6, 8],
# #         [4, 0, 5, 7],
# #         [6, 5, 0, 3],
# #         [8, 7, 3, 0],
# #     ]

# #     run_instance("Symmetric_4_nodes_p3", n4, p4, W4, D4)

# #     # ------------------------------------------------------------------
# #     # Test 5: 4-node, asymmetric flows, p = 4
# #     # ------------------------------------------------------------------
# #     p5 = 4

# #     W5 = [
# #         [0, 1, 3, 2],
# #         [2, 0, 1, 4],
# #         [1, 5, 0, 2],
# #         [4, 1, 3, 0],
# #     ]

# #     D5 = [
# #         [0, 2, 5, 4],
# #         [2, 0, 3, 6],
# #         [5, 3, 0, 1],
# #         [4, 6, 1, 0],
# #     ]

# #     run_instance("AsymmetricFlows_4_nodes_p4", n4, p5, W5, D5)


# # if __name__ == "__main__":
# #     main()
# """
# test_hub_arc_models.py

# Run several small instances of the p-Hub-Arc problem and compare:
#   - Direct F3 canonical formulation (solve_hub_arc_F3)
#   - Benders decomposition (HubArcBenders)

# For each test:
#   - print objectives and selected arcs
#   - print wall-clock time for each solver
#   - print peak Python memory usage (via tracemalloc) for each solver

# Usage:
#   python3 test_hub_arc_models.py
# """

# import time
# import tracemalloc
# from math import isclose

# from hub_arc_models import solve_hub_arc_F3, HubArcBenders


# # ----------------------------------------------------------------------
# # Helper: run a callable with timing + peak memory measurement
# # ----------------------------------------------------------------------
# def run_with_metrics(func, *args, **kwargs):
#     """
#     Run func(*args, **kwargs) while measuring:
#       - wall-clock time (seconds)
#       - peak Python memory usage (bytes, via tracemalloc)

#     Returns: (result, elapsed_time, peak_memory_bytes)
#     """
#     tracemalloc.start()
#     start = time.time()
#     result = func(*args, **kwargs)
#     elapsed = time.time() - start
#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     return result, elapsed, peak


# # ----------------------------------------------------------------------
# # Single test instance runner
# # ----------------------------------------------------------------------
# def run_instance(
#     name: str,
#     n: int,
#     p: int,
#     W,
#     D,
#     tol: float = 1e-5,
# ):
#     print("=" * 80)
#     print(f"TEST INSTANCE: {name}")
#     print(f"n = {n}, p = {p}")
#     print("W:")
#     for row in W:
#         print("  ", row)
#     print("D:")
#     for row in D:
#         print("  ", row)

#     # ---- F3 formulation ----
#     f3_res, f3_time, f3_mem = run_with_metrics(
#         solve_hub_arc_F3, n, p, W, D, False
#     )
#     print("\n[F3] status:", f3_res["status"])
#     print("[F3] objective:", f3_res["objective"])
#     print("[F3] selected arcs:", sorted(f3_res["selected_arcs"]))
#     print(f"[F3] wall-clock time: {f3_time:.6f} s")
#     print(f"[F3] peak Python memory: {f3_mem / (1024 ** 2):.3f} MB")

#     # ---- Benders ----
#     benders = HubArcBenders(n=n, p=p, W=W, D=D, verbose=False)
#     b_res, b_time, b_mem = run_with_metrics(benders.solve, 60)
#     print("\n[Benders] status:", b_res["status"])
#     print("[Benders] objective:", b_res["objective"])
#     print("[Benders] selected arcs:", sorted(b_res["selected_arcs"]))
#     print(f"[Benders] wall-clock time: {b_time:.6f} s")
#     print(f"[Benders] peak Python memory: {b_mem / (1024 ** 2):.3f} MB")

#     # ---- Check / compare ----
#     ok_status = (f3_res["status"] == "OPTIMAL") and (b_res["status"] == 2)
#     # In Gurobi, status 2 == OPTIMAL

#     if not ok_status:
#         print(
#             "\nWARNING: One of the models is not optimal "
#             "(F3 status = {}, Benders status = {}). "
#             "Skipping objective comparison.".format(
#                 f3_res["status"], b_res["status"]
#             )
#         )
#     else:
#         if isclose(f3_res["objective"], b_res["objective"],
#                    rel_tol=tol, abs_tol=tol):
#             print("\nRESULT: ✅ Objectives match within tolerance.")
#         else:
#             print("\nRESULT: ❌ Objectives DIFFER!")
#             print("  |F3 - Benders| =",
#                   abs(f3_res["objective"] - b_res["objective"]))

#     print("=" * 80, "\n")


# # ----------------------------------------------------------------------
# # Main: define several deterministic testcases
# # ----------------------------------------------------------------------
# def main():
#     # ------------------------------------------------------------------
#     # Test 1: Your original 3-node instance, p = 2
#     # ------------------------------------------------------------------
#     n1 = 3
#     p1 = 2

#     W1 = [
#         [0, 2, 3],
#         [2, 0, 4],
#         [3, 4, 0],
#     ]

#     D1 = [
#         [0, 5, 2],
#         [5, 0, 3],
#         [2, 3, 0],
#     ]

#     run_instance("Toy_3_nodes_p2", n1, p1, W1, D1)

#     # ------------------------------------------------------------------
#     # Test 2: Same 3-node instance, different p (p = 3)
#     # ------------------------------------------------------------------
#     p2 = 3
#     run_instance("Toy_3_nodes_p3", n1, p2, W1, D1)

#     # ------------------------------------------------------------------
#     # Test 3: 3-node instance, p = 1 (edge case: very few hub arcs)
#     # ------------------------------------------------------------------
#     p3 = 1
#     run_instance("Toy_3_nodes_p1", n1, p3, W1, D1)

#     # ------------------------------------------------------------------
#     # Test 4: 4-node symmetric flows and distances, p = 3
#     # ------------------------------------------------------------------
#     n4 = 4
#     p4 = 3

#     # flows: symmetric
#     W4 = [
#         [0, 1, 2, 3],
#         [1, 0, 1, 2],
#         [2, 1, 0, 1],
#         [3, 2, 1, 0],
#     ]

#     # distances: simple symmetric matrix
#     D4 = [
#         [0, 4, 6, 8],
#         [4, 0, 5, 7],
#         [6, 5, 0, 3],
#         [8, 7, 3, 0],
#     ]

#     run_instance("Symmetric_4_nodes_p3", n4, p4, W4, D4)

#     # ------------------------------------------------------------------
#     # Test 5: 4-node, asymmetric flows, p = 4
#     # ------------------------------------------------------------------
#     p5 = 4

#     W5 = [
#         [0, 1, 3, 2],
#         [2, 0, 1, 4],
#         [1, 5, 0, 2],
#         [4, 1, 3, 0],
#     ]

#     D5 = [
#         [0, 2, 5, 4],
#         [2, 0, 3, 6],
#         [5, 3, 0, 1],
#         [4, 6, 1, 0],
#     ]

#     run_instance("AsymmetricFlows_4_nodes_p4", n4, p5, W5, D5)


# if __name__ == "__main__":
#     main()
"""
test_hub_arc_bench.py

Benchmark and compare:
  - Direct F3 canonical formulation (solve_hub_arc_F3)
  - Benders decomposition (HubArcBenders)

Metrics per test:
  - Objective values (should match)
  - Wall-clock time
  - Peak Python memory usage (MB)

Requires:
  - hub_arc_models.py in the same directory, containing:
      solve_hub_arc_F3(...)
      HubArcBenders(...)
"""

import time
import tracemalloc
import random
from math import isclose

from hub_arc_models import solve_hub_arc_F3, HubArcBenders


# ----------------------------------------------------------------------
# Helper: generate random W, D for a given n and seed
# ----------------------------------------------------------------------


def generate_random_instance(n: int, seed: int, flow_max: int = 5, dist_max: int = 10):
    """
    Generate a random (W, D) instance for given n and seed.

    - W[i][i] = 0, D[i][i] = 0
    - Off-diagonal entries are positive integers.
    """
    random.seed(seed)
    W = [[0] * n for _ in range(n)]
    D = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            W[i][j] = random.randint(1, flow_max)
            D[i][j] = random.randint(1, dist_max)

    return W, D


# ----------------------------------------------------------------------
# Helper: measure time + peak memory of a solver call
# ----------------------------------------------------------------------


def measure_solver(run_fn):
    """
    run_fn: function with no arguments that runs the solver and returns the result.

    Returns:
      (result, wall_time_sec, peak_memory_mb)
    """
    tracemalloc.start()
    t0 = time.time()
    result = run_fn()
    wall = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    return result, wall, peak_mb


# ----------------------------------------------------------------------
# One benchmark instance
# ----------------------------------------------------------------------


def run_instance(
    name: str,
    category: str,
    n: int,
    p: int,
    seed: int,
    tol: float = 1e-5,
):
    print("=" * 80)
    print(f"TEST: {name}  |  category={category},  n={n},  p={p},  seed={seed}")

    # build W, D
    W, D = generate_random_instance(n, seed)

    # ------------- F3 -------------
    def run_f3():
        return solve_hub_arc_F3(n, p, W, D, gurobi_output=False)

    f3_res, f3_time, f3_mem = measure_solver(run_f3)
    print("\n[F3] status:", f3_res["status"])
    print("[F3] objective:", f3_res["objective"])
    print("[F3] selected arcs:", sorted(f3_res["selected_arcs"]))
    print(f"[F3] wall time (s): {f3_time:.4f}")
    print(f"[F3] peak Python memory (MB): {f3_mem:.3f}")

    # ------------- Benders -------------
    def run_benders():
        b = HubArcBenders(n=n, p=p, W=W, D=D, verbose=False)
        return b.solve(time_limit=600)

    b_res, b_time, b_mem = measure_solver(run_benders)
    print("\n[Benders] status:", b_res["status"])
    print("[Benders] objective:", b_res["objective"])
    print("[Benders] selected arcs:", sorted(b_res["selected_arcs"]))
    print(f"[Benders] wall time (s): {b_time:.4f}")
    print(f"[Benders] peak Python memory (MB): {b_mem:.3f}")

    # ------------- Comparison -------------
    # solve_hub_arc_F3 returns "OPTIMAL" as string,
    # Gurobi's numeric optimal status is 2 for the Benders master model.
    ok_status = (f3_res["status"] == "OPTIMAL") and (b_res["status"] == 2)

    if not ok_status:
        print("\n⚠️  One of the models did not reach OPTIMAL. Skipping objective check.")
    else:
        if isclose(f3_res["objective"], b_res["objective"], rel_tol=tol, abs_tol=tol):
            print("\n✅ Objectives match within tolerance.")
        else:
            diff = abs(f3_res["objective"] - b_res["objective"])
            print("\n❌ Objectives DIFFER!")
            print("   |F3 - Benders| =", diff)

    print("=" * 80, "\n")


# ----------------------------------------------------------------------
# Build and run many test cases (small / medium / large)
# ----------------------------------------------------------------------


def main():
    tests = []

    # ---------------- SMALL INSTANCES ----------------
    # n = 3, 4, 5
    # (ensure p <= n*(n-1))
    tests.extend([
        # n = 3
        ("Small_n3_p1", "small", 3, 1, 1),
        ("Small_n3_p2", "small", 3, 2, 2),
        ("Small_n3_p4", "small", 3, 4, 3),

        # n = 4
        ("Small_n4_p2", "small", 4, 2, 4),
        ("Small_n4_p4", "small", 4, 4, 5),
        ("Small_n4_p6", "small", 4, 6, 6),

        # n = 5
        ("Small_n5_p3", "small", 5, 3, 7),
        ("Small_n5_p6", "small", 5, 6, 8),
        ("Small_n5_p8", "small", 5, 8, 9),
    ])

    # ---------------- MEDIUM INSTANCES ----------------
    # n = 6, 7, 8
    tests.extend([
        # n = 6
        ("Medium_n6_p5", "medium", 6, 5, 10),
        ("Medium_n6_p10", "medium", 6, 10, 11),

        # n = 7
        ("Medium_n7_p7", "medium", 7, 7, 12),
        ("Medium_n7_p12", "medium", 7, 12, 13),

        # n = 8
        ("Medium_n8_p8", "medium", 8, 8, 14),
        ("Medium_n8_p15", "medium", 8, 15, 15),
    ])

    # ---------------- LARGE INSTANCES ----------------
    # n = 9, 10  (still “small” for Gurobi, but larger than before)
    tests.extend([
        # n = 9
        ("Large_n9_p10", "large", 9, 10, 16),
        ("Large_n9_p18", "large", 9, 18, 17),
        ("Large_n9_p25", "large", 9, 25, 18),

        # n = 10
        ("Large_n10_p10", "large", 10, 10, 19),
        ("Large_n10_p20", "large", 10, 20, 20),
        ("Large_n10_p30", "large", 10, 30, 21),
        ("Large_n10_p40", "large", 10, 40, 22),
    ])

    # That’s 9 + 6 + 7 = 22 testcases total.

    for (name, cat, n, p, seed) in tests:
        # Safety check: p cannot exceed number of directed arcs
        max_arcs = n * (n - 1)
        p_eff = min(p, max_arcs)
        if p_eff != p:
            print(f"NOTE: Adjusting p from {p} to {p_eff} for {name} (max arcs = {max_arcs})")
            p = p_eff

        run_instance(name, cat, n, p, seed)


if __name__ == "__main__":
    main()

