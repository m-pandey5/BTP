import gurobipy as gp
from gurobipy import GRB

def solve_p_median_CR(n, p, c):
    """
    n: number of nodes
    p: number of medians to choose
    c: cost matrix (n x n)
    """
    try:
        # Try to create a model to verify license
        m = gp.Model("license_check")
        print("Gurobi license is active!")
    except gp.GurobiError as e:
        print("Gurobi license error:", e)
        return None
    # Step 1: Construct D_i for each i
    D = {}
    G = {}
    for i in range(n):
        unique_costs = sorted(set(c[i]))
        D[i] = unique_costs
        G[i] = len(unique_costs)

    # Model
    m = gp.Model("p-median-CR")

    # Step 2: Variables
    y = m.addVars(n, vtype=GRB.BINARY, name="y")  # facility open
    z = {}
    for i in range(n):
        for k in range(1, G[i]):  # k starts from 1 since D[i][0]=0 always
            z[i, k] = m.addVar(vtype=GRB.BINARY, name=f"z[{i},{k}]")

    # Step 3: Objective
    m.setObjective(
        gp.quicksum((D[i][k] - D[i][k-1]) * z[i, k]
                    for i in range(n) for k in range(1, G[i])),
        GRB.MINIMIZE
    )

    # Constraint (1): Exactly p facilities open
    m.addConstr(gp.quicksum(y[i] for i in range(n)) == p, name="pFacilities")

    # Constraint (2): Coverage constraints
    for i in range(n):
        for k in range(1, G[i]):  # k>=2 in math notation → index starts at 1 here
            m.addConstr(
                z[i, k] + gp.quicksum(y[j] for j in range(n) if c[i][j] < D[i][k]) >= 1,
                name=f"cover[{i},{k}]"
            )

    # Optimize
    m.optimize()

    # Extract solution
    if m.status == GRB.OPTIMAL:
        chosen_facilities = [i for i in range(n) if y[i].x > 0.5]
        print("Optimal objective value:", m.objVal)
        print("Chosen facilities:", chosen_facilities)
        return chosen_facilities
    else:
        print("No optimal solution found.")
        return None


# Example usage
if __name__ == "__main__":
    n = 5
    p = 2
    c = [
        [0, 2, 3, 1, 4],
        [2, 0, 2, 3, 5],
        [3, 2, 0, 4, 2],
        [1, 3, 4, 0, 3],
        [4, 5, 2, 3, 0]
    ]

    solve_p_median_CR(n, p, c)
