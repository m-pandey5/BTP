import gurobipy as gp
from gurobipy import GRB

def solve_hub_arc(n, p, W, D):
    """
    n: number of nodes
    p: number of hub arcs to open
    W: flow weight matrix (n x n), W[i][j] is weight between node i and j
    D: arc cost matrix (n x n), D[k][m] is cost of hub arc (k,m)
    """
    N = range(n)
    H = [(k, m) for k in N for m in N if k != m]  # hub arcs k ≠ m

    model = gp.Model("HubArc")

    # Decision variables
    x = model.addVars([(i, j, k, m) for i in N for j in N for (k, m) in H],
                      vtype=GRB.BINARY, name="x")
    y = model.addVars(H, vtype=GRB.BINARY, name="y")

    # Objective
    model.setObjective(
        gp.quicksum(x[i, j, k, m] * W[i][j] * D[k][m]
                    for i in N for j in N for (k, m) in H),
        GRB.MINIMIZE
    )

    # Constraint 1: each flow assigned to exactly one hub arc
    for i in N:
        for j in N:
            model.addConstr(
                gp.quicksum(x[i, j, k, m] for (k, m) in H) == 1,
                name=f"Assign[{i},{j}]"
            )

    # Constraint 2: can only assign to open hub arc (k ≠ m)
    for i in N:
        for j in N:
            for (k, m) in H:
                model.addConstr(
                    x[i, j, k, m] <= y[k, m],
                    name=f"ValidAssign[{i},{j},{k},{m}]"
                )

    # Constraint 3: exactly p hub arcs open
    model.addConstr(
        gp.quicksum(y[k, m] for (k, m) in H) == p,
        name="pHubArcs"
    )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        chosen_arcs = [(k, m) for (k, m) in H if y[k, m].x > 0.5]
        assignment = {(i, j): [(k, m) for (k, m) in H if x[i, j, k, m].x > 0.5] for i in N for j in N}
        print("Optimal objective value:", model.objVal)
        print("Chosen hub arcs:", chosen_arcs)
        print("Assignments:", assignment)
        return chosen_arcs, assignment
    else:
        print("No optimal solution found.")
        return None, None

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
    solve_hub_arc(n, p, W, D)
