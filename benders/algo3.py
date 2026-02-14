"""
Algorithm 3: Phase 1 - Solving the linear relaxation of the master problem

Implements the two-phase Benders decomposition:
- Phase 1: Solve LP relaxation of MP with Benders cuts
- Phase 2: Add integrity constraints, solve via branch-and-Benders-cut

Reference: Section 3.5.1 from Duran-Mateliana, Ales, Dillami paper.
"""

import gurobipy as gp
from gurobipy import GRB

from algo1 import separation_algorithm_dual


def rounding_heuristic(y_bar, p, M):
    """
    Rounding heuristic: open the p facilities with largest y values.

    Parameters
    ----------
    y_bar : list[float]
        Fractional MP solution
    p : int
        Number of facilities to open
    M : int
        Total number of sites

    Returns
    -------
    y_int : list[float]
        Integer solution (0/1 values)
    """
    indices = sorted(range(M), key=lambda j: y_bar[j], reverse=True)
    y_int = [0.0] * M
    for j in indices[:p]:
        y_int[j] = 1.0
    return y_int


def compute_allocation_cost(i, y_int, S, D, dist):
    """
    Compute allocation cost for client i given integer solution y_int.
    Cost = distance to nearest open facility.
    """
    for j in S[i]:
        if y_int[j] > 0.5:
            return dist[i][j]
    return float("inf")


def phase1_solve_mp(N, M, p, S, D, K, dist, facilities_at_distance=None,
                    y_heuristic=None, UB_heuristic=None, max_iter=200, verbose=False):
    """
    Algorithm 3: Phase 1 - Solve linear relaxation of master problem.

    Parameters
    ----------
    N, M, p : int
        Problem dimensions
    S : dict
        S[i] = sorted facilities by distance for client i
    D, K : dict
        Distance levels
    dist : array
        Distance matrix
    facilities_at_distance : dict
        facilities_at_distance[i][k] = list of sites j with d_ij = D[i][k]
    y_heuristic : list, optional
        Initial heuristic solution (opens p facilities)
    UB_heuristic : float, optional
        Upper bound from heuristic
    max_iter : int
        Maximum iterations (default 200; increase if Phase 1 converges slowly)
    verbose : bool
        Print progress

    Returns
    -------
    LB1 : float
        Lower bound from Phase 1
    y1 : list
        Best feasible integer solution found
    UB1 : float
        Upper bound (objective of y1)
    model : gurobipy.Model
        Master problem with all Phase 1 cuts
    y_vars : list
        y variable references
    theta_vars : list
        theta variable references
    """
    # Build initial master (LP relaxation)
    model = gp.Model("Master_Phase1")
    model.Params.OutputFlag = 1 if verbose else 0

    y_vars = model.addVars(M, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")
    theta_vars = model.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name="theta")

    model.setObjective(gp.quicksum(theta_vars[i] for i in range(N)), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(y_vars[j] for j in range(M)) == p, name="p_facilities")

    LB1 = -float("inf")
    UB_hat = UB_heuristic if UB_heuristic is not None else float("inf")
    y_hat = y_heuristic

    # Initial solution from heuristic
    if y_hat is not None:
        for j in range(M):
            y_vars[j].Start = y_hat[j]

    iter_count = 0
    while iter_count < max_iter:
        model.optimize()
        if model.status != GRB.OPTIMAL:
            if verbose:
                print(f"[Phase1] Master solve failed: status={model.status}")
            break

        y_bar = [y_vars[j].X for j in range(M)]
        theta_bar = [theta_vars[i].X for i in range(N)]
        obj_val = model.ObjVal
        LB1 = max(LB1, obj_val)

        # Separate cuts (dual-based for correctness)
        if facilities_at_distance is None:
            facilities_at_distance = {}
            for i in range(N):
                facilities_at_distance[i] = {}
                for k in range(1, K[i] + 1):
                    Dk = D[i][k]
                    facilities_at_distance[i][k] = [
                        j for j in range(M) if abs(dist[i][j] - Dk) < 1e-8
                    ]
        cuts, UB = separation_algorithm_dual(
            y_bar, theta_bar, facilities_at_distance, D, K, N, M
        )

        if not cuts:
            # No violated cuts - optimal for LP relaxation
            if verbose:
                print(f"[Phase1] No violated cuts. Converged.")
            break

        # Add cuts
        for (i, cut_data) in cuts:
            D1 = cut_data["D1_i"]
            nu = cut_data.get("nu") or cut_data.get("nu_k", {})
            fac_D = cut_data["facilities_at_D"]
            expr = D1
            if nu.get(1, 0) != 0 and 1 in fac_D:
                expr += nu[1] * (1.0 - gp.quicksum(y_vars[j] for j in fac_D[1]))
            for k in range(2, K[i] + 1):
                if nu.get(k, 0) != 0 and k in fac_D and fac_D[k]:
                    expr -= nu[k] * gp.quicksum(y_vars[j] for j in fac_D[k])
            model.addConstr(theta_vars[i] >= expr, name=f"cut_{i}_{iter_count}")

        # Rounding heuristic if fractional
        is_fractional = any(1e-6 < y_bar[j] < 1 - 1e-6 for j in range(M))
        if is_fractional:
            y_round = rounding_heuristic(y_bar, p, M)
            UB_h = sum(
                compute_allocation_cost(i, y_round, S, D, dist)
                for i in range(N)
            )
            if UB_h < UB_hat:
                UB_hat = UB_h
                y_hat = y_round
                if verbose:
                    print(f"[Phase1] Rounding improved UB to {UB_hat:.4f}")

        iter_count += 1
        if verbose:
            print(f"[Phase1] Iter {iter_count}: LB={LB1:.4f}, cuts added={len(cuts)}")

    # Final integer solution
    y1 = y_hat if y_hat is not None else rounding_heuristic(
        [y_vars[j].X for j in range(M)], p, M
    )
    UB1 = sum(
        compute_allocation_cost(i, y1, S, D, dist)
        for i in range(N)
    )
    if UB_heuristic is not None and UB_heuristic < UB1:
        UB1 = UB_heuristic
        y1 = y_heuristic

    return LB1, y1, UB1, model, y_vars, theta_vars
