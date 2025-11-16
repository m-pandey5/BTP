
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_hub_arc_canonical(n, p, W, D):
    """Canonical representation for the p-Hub-Arc Problem.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : int
        Number of hub-arcs to select.
    W : 2D array-like
        OD weights (n x n).
    D : 2D array-like
        Distance matrix (n x n).

    Returns
    -------
    obj_val : float or None
        Optimal objective value (None if infeasible / not optimal).
    chosen_arcs : list of tuple or None
        Selected hub-arcs as (u, v) pairs (None if infeasible / not optimal).
    runtime : float
        Solver runtime in seconds.
    """
    N = range(n)
    hub_pairs = [(u, v) for u in N for v in N if u != v]  # hub arcs u ≠ v

    # Create sorted cost vectors D_ij for each OD pair (include 0)
    D_vectors = {}
    G = {}
    for i in N:
        for j in N:
            if i == j:
                continue  # skip self-loops
            costs = []
            for (u, v) in hub_pairs:
                if u ==v :
                    continue
                cost = W[i][j] * (D[i][u] + D[u][v] + D[v][j])
                costs.append(cost)

            unique_costs = sorted(list(set(costs)))
            D_vectors[(i, j)] = unique_costs
            G[(i, j)] = len(unique_costs)

    model = gp.Model("HubArc_Canonical")
    model.setParam('OutputFlag', 0)  # suppress Gurobi output

    # Decision variables
    # z[i,j,k] for k = 2..G[(i,j)]  (note: k indexes the sorted cost levels; we keep same k naming)
    # z = model.addVars(
    #     ((i, j, k) for i in N for j in N for k in range(1, G[(i, j)])),
    #     vtype=GRB.CONTINUOUS, lb=0.0, name="z"
    # )
    z = {}
    for i in N:
        for j in N:
            if i == j:
                continue
            g = G[(i, j)]
            # if g <= 1 there are no positive levels (no z needed)
            for k in range(1, g):          # k = 1 .. g-1
                z[(i, j, k)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"z_{i}_{j}_{k}")

    # binary selection for hub arcs
    y = model.addVars(hub_pairs, vtype=GRB.BINARY, name="y")

    # # Objective: sum_{i,j} sum_{k=2..G} (D_{i,j}^{(k)} - D_{i,j}^{(k-1)}) * z_{i,j,k}
    # obj_expr = gp.LinExpr()
    # for i in N:
    #     for j in N:
    #         # if G[(i,j)] <= 1 then there are no z-variables for this pair
    #         for k in range(2, G[(i, j)] + 1):
    #             D_ijk = D_vectors[(i, j)][k - 1]
    #             D_ij_k_minus_1 = D_vectors[(i, j)][k - 2]
    #             diff = D_ijk - D_ij_k_minus_1
    #             obj_expr += diff * z[i, j, k]
    # model.setObjective(obj_expr, GRB.MINIMIZE)
        # 3. Objective Function
    # ------------------------------------------------------------------ #
    obj = gp.LinExpr()

    for i in N:
        for j in N:
            if i == j or G[(i, j)] <= 1:
                continue
            # Add base cost: C_{ij}^1
            obj += D_vectors[(i, j)][1]  # first positive cost level (index 1)

            # Add incremental costs: (C^{k+1} - C^k) * z_{ij}^k
            for k in range(1, G[(i, j)]-1):
                diff = D_vectors[(i, j)][k + 1] - D_vectors[(i, j)][k]
                if diff > 0:
                    obj += diff * z[(i, j, k)]

    model.setObjective(obj, GRB.MINIMIZE)
    

    

    # Exactly p hub-arcs must be chosen
    model.addConstr(gp.quicksum(y[u, v] for (u, v) in hub_pairs) == p, name="pHubArcs")

    # Canonical constraints: for each (i,j) and level k
    for i in N:
        for j in N:
            if i == j or G[(i, j)] <= 1:
                continue
            for k in range(1, G[(i, j)]):
                D_ijk = D_vectors[(i, j)][k - 1]

                # find hub arcs cheaper than the k-th level for OD (i,j)
                cheaper_pairs = []
                for (u, v) in hub_pairs:
                    c_ij_uv = W[i][j] * (D[i][u] + D[u][v] + D[v][j])
                    if c_ij_uv < D_ijk:
                        cheaper_pairs.append((u, v))

                if cheaper_pairs:
                    model.addConstr(
                        z[i, j, k] + gp.quicksum(y[u, v] for (u, v) in cheaper_pairs) >= 1,
                        name=f"canonical_{i}_{j}_{k}"
                    )
                else:
                    # no hub-arc produces cost < D_ijk, so z must be at least 1
                    model.addConstr(z[i, j, k] >= 1, name=f"canonical_{i}_{j}_{k}_no_cheaper")

    model.optimize()

    # Use getAttr for solver attributes (safer across Gurobi versions)
    runtime = model.getAttr('Runtime') if model.Status != GRB.INF_OR_UNBD else None

    if model.status == GRB.OPTIMAL:
        obj_val = model.getAttr('ObjVal')
        chosen_arcs = [(u, v) for (u, v) in hub_pairs if y[u, v].X > 0.5]
        return obj_val, chosen_arcs, runtime
    else:
        return None, None, runtime


# Example usage
if __name__ == "__main__":
    n = 3
    p = 2
    W = [
        [0, 2, 3],
        [2, 0, 4],
        [3, 4, 0]
    ]
    D = [
        [0, 5, 2],
        [5, 0, 3],
        [2, 3, 0]
    ]
    obj, arcs, run = solve_hub_arc_canonical(n, p, W, D)
    print("Objective:", obj)
    print("Chosen hub-arcs:", arcs)
    print("Runtime (s):", run)
