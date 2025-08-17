import gurobipy as gp
from gurobipy import GRB

def solve_p_median(n, p, c):
    """
    n: number of nodes
    p: number of medians to choose
    c: cost matrix (n x n) where c[i][j] = cost of assigning i -> j
    """

    # Create model
    m = gp.Model("p-median")

    # Decision variables
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")   # assignment vars
    y = m.addVars(n, vtype=GRB.BINARY, name="y")      # facility open vars (y_j = x_jj)

    # Objective: minimize sum of assignment costs
    m.setObjective(gp.quicksum(c[i][j] * x[i, j] for i in range(n) for j in range(n)),
                   GRB.MINIMIZE)

    # Constraint (1): each demand assigned to exactly one facility
    for i in range(n):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"assign[{i}]")

    # Constraint (2): valid assignment (can only assign if facility is open)
    for i in range(n):
        for j in range(n):
            m.addConstr(x[i, j] <= y[j], name=f"validAssign[{i},{j}]")

    # Constraint (3): exactly p facilities open
    m.addConstr(gp.quicksum(y[j] for j in range(n)) == p, name="pFacilities")

    # Optimize
    m.optimize()

    # Extract solution
    if m.status == GRB.OPTIMAL:
        chosen_facilities = [j for j in range(n) if y[j].x > 0.5]
        assignment = {i: [j for j in range(n) if x[i, j].x > 0.5][0] for i in range(n)}
        print("Optimal objective value:", m.objVal)
        print("Chosen facilities:", chosen_facilities)
        print("Assignments:", assignment)
        return chosen_facilities, assignment
    else:
        print("No optimal solution found.")
        return None, None


# Example usage
if __name__ == "__main__":
    # Small test (5 nodes, choose 2 facilities)
    n = 5
    p = 2
    c = [
        [0, 2, 3, 1, 4],
        [2, 0, 2, 3, 5],
        [3, 2, 0, 4, 2],
        [1, 3, 4, 0, 3],
        [4, 5, 2, 3, 0]
    ]

    solve_p_median(n, p, c)
