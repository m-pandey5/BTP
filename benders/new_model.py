"""
Benders Decomposition for p-Median Problem (Section 3.2 Separation Problem)

Complete implementation using:
- Algorithm 1: Polynomial separation algorithm
- Algorithm 2: Computing k_i
- Algorithm 3: Phase 1 - LP relaxation
- Phase 2: Branch-and-Benders-cut

Reference: Duran-Mateliana, Ales, Dillami - Benders decomposition for p-median.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

from algo1 import separation_algorithm, separation_algorithm_dual
from algo2 import compute_k_i
from algo3 import phase1_solve_mp, rounding_heuristic, compute_allocation_cost


# ============================================================
# Preprocessing: Build S, D, K, facilities_at_distance
# ============================================================

def preprocess(N, M, dist):
    """
    Build S, D, K, and facilities_at_distance from distance matrix.
    """
    S = {}
    D = {}
    K = {}
    facilities_at_distance = {}

    for i in range(N):
        sites_sorted = sorted(range(M), key=lambda j: dist[i][j])
        S[i] = sites_sorted

        unique_dists = sorted(set(round(dist[i][j], 8) for j in range(M)))
        K[i] = len(unique_dists)
        D[i] = {k: unique_dists[k - 1] for k in range(1, K[i] + 1)}

        facilities_at_distance[i] = {}
        for k in range(1, K[i] + 1):
            Dk = D[i][k]
            facilities_at_distance[i][k] = [
                j for j in range(M) if abs(dist[i][j] - Dk) < 1e-8
            ]

    return S, D, K, facilities_at_distance


# ============================================================
# Phase 2: Benders Callback (branch-and-Benders-cut)
# ============================================================

class BendersCutCallback:
    """
    Lazy constraint callback for Phase 2.
    Uses dual-based separation for correct Benders cuts.
    """

    def __init__(self, N, M, p, facilities_at_distance, D, K, y_vars, theta_vars):
        self.N = N
        self.M = M
        self.p = p
        self.facilities_at_distance = facilities_at_distance
        self.D = D
        self.K = K
        self.y_vars = y_vars
        self.theta_vars = theta_vars

    def __call__(self, model, where):
        if where != GRB.Callback.MIPSOL:
            return

        y_bar = [model.cbGetSolution(self.y_vars[j]) for j in range(self.M)]
        theta_bar = [model.cbGetSolution(self.theta_vars[i]) for i in range(self.N)]

        cuts, _ = separation_algorithm_dual(
            y_bar, theta_bar,
            self.facilities_at_distance, self.D, self.K,
            self.N, self.M
        )

        for (i, cut_data) in cuts:
            D1 = cut_data["D1_i"]
            nu = cut_data["nu"]
            fac_D = cut_data["facilities_at_D"]
            expr = D1
            if nu.get(1, 0) != 0 and 1 in fac_D and fac_D[1]:
                expr += nu[1] * (1.0 - gp.quicksum(
                    self.y_vars[j] for j in fac_D[1]
                ))
            for k in range(2, self.K[i] + 1):
                if nu.get(k, 0) != 0 and k in fac_D and fac_D[k]:
                    expr -= nu[k] * gp.quicksum(
                        self.y_vars[j] for j in fac_D[k]
                    )
            model.cbLazy(self.theta_vars[i] >= expr)


# ============================================================
# Full Benders Decomposition (Phase 1 + Phase 2)
# ============================================================

def solve_benders_pmedian(N, M, p, dist, K=None, D=None, time_limit=None, verbose=False, use_phase1=True):
    """
    Solve p-median using Benders decomposition with polynomial separation.

    Parameters
    ----------
    N, M : int
        Number of clients and sites
    p : int
        Number of facilities to open
    dist : array-like
        dist[i][j] = distance from client i to site j
    time_limit : float, optional
        Time limit in seconds
    verbose : bool
        Print progress

    Returns
    -------
    dict with keys: objective, y, facilities, time, status, num_vars, num_constrs
    """
    dist = np.asarray(dist)
    start_time = time.time()

    if K is None or D is None:
        S, D, K, facilities_at_distance = preprocess(N, M, dist)
    else:
        S = {i: sorted(range(M), key=lambda j: dist[i][j]) for i in range(N)}
        facilities_at_distance = {}
        for i in range(N):
            facilities_at_distance[i] = {}
            for k in range(1, K[i] + 1):
                Dk = D[i][k]
                facilities_at_distance[i][k] = [
                    j for j in range(M) if abs(dist[i][j] - Dk) < 1e-8
                ]

    y1 = None
    if use_phase1:
        # Simple heuristic: open first p sites
        y_heuristic = [1.0 if j < p else 0.0 for j in range(M)]
        UB_h = sum(
            compute_allocation_cost(i, y_heuristic, S, D, dist)
            for i in range(N)
        )
        if verbose:
            print("Phase 1: Solving LP relaxation with Benders cuts...")
        LB1, y1, UB1, mp_model, y_vars, theta_vars = phase1_solve_mp(
            N, M, p, S, D, K, dist,
            facilities_at_distance=facilities_at_distance,
            y_heuristic=y_heuristic,
            UB_heuristic=UB_h,
            verbose=verbose
        )
        if verbose:
            print(f"Phase 1 done: LB={LB1:.4f}, UB={UB1:.4f}")

    # Phase 2: Build MIP with lazy constraints (like benderscallback_fixed)
    if verbose:
        print("Phase 2: Branch-and-Benders-cut...")

    model = gp.Model("Master_Phase2")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.LazyConstraints = 1
    model.Params.MIPGap = 1e-9
    if time_limit is not None:
        model.Params.TimeLimit = max(0, time_limit - (time.time() - start_time))

    y_vars = model.addVars(M, vtype=GRB.BINARY, name="y")
    theta_vars = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="theta")

    model.setObjective(gp.quicksum(theta_vars[i] for i in range(N)), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(y_vars[j] for j in range(M)) == p, name="p_facilities")

    # Warm start from Phase 1 best integer solution
    if y1 is not None:
        for j in range(M):
            y_vars[j].Start = 1.0 if y1[j] > 0.5 else 0.0
    callback = BendersCutCallback(
        N, M, p, facilities_at_distance, D, K,
        y_vars, theta_vars
    )
    model.optimize(callback)

    elapsed = time.time() - start_time

    if model.status == GRB.OPTIMAL:
        y_sol = [y_vars[j].X for j in range(M)]
        obj = sum(
            compute_allocation_cost(i, y_sol, S, D, dist)
            for i in range(N)
        )
        facilities = sorted([j for j in range(M) if y_sol[j] > 0.5])
        return {
            "objective": obj,
            "y": y_sol,
            "facilities": facilities,
            "time": elapsed,
            "status": "OPTIMAL",
            "num_vars": model.NumVars,
            "num_constrs": model.NumConstrs,
        }
    else:
        return {
            "objective": None,
            "y": None,
            "facilities": None,
            "time": elapsed,
            "status": "FAILED",
            "num_vars": model.NumVars,
            "num_constrs": model.NumConstrs,
        }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    N = 5
    M = 7
    p = 2

    dist = np.random.rand(N, M) * 100.0

    print("=== Benders Decomposition (Polynomial Separation) ===")
    res = solve_benders_pmedian(N, M, p, dist, verbose=True)

    print("\n=== Result ===")
    print("Status:", res["status"])
    print("Objective:", res["objective"])
    print("Facilities:", res["facilities"])
    print("Time:", res["time"], "s")
