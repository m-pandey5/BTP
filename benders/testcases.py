"""
Comprehensive Benchmark: CR Formulation vs Benders Decomposition for p-Median

Metrics: 
- Time (seconds)
- Memory Usage (MB)
- Peak Memory (MB)
- Objective Value
- Facility Selection
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import psutil
import os


def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class BenchmarkSuite:
    def __init__(self):
        self.results = []

    # ============================================================
    # CR FORMULATION (Direct)
    # ============================================================

    def solve_p_median_CR(self, n, p, cost_matrix, verbose=False):
        """
        CR Formulation (Direct approach):
        min Σᵢ Σₖ (D_i^k - D_i^(k-1)) * z_i^k
        s.t. Σᵢ yᵢ = p
             z_i^k + Σⱼ: c_ij < D_i^k yⱼ >= 1
        """
        mem_start = get_memory_usage()
        time_start = time.time()

        try:
            # Construct distance thresholds
            D = {}
            G = {}
            for i in range(n):
                unique_costs = sorted(set(cost_matrix[i]))
                D[i] = unique_costs
                G[i] = len(unique_costs)

            # Model
            m = gp.Model("p-median-CR")
            m.Params.OutputFlag = 0  # Suppress output

            # Variables
            y = m.addVars(n, vtype=GRB.BINARY, name="y")
            z = {}
            for i in range(n):
                for k in range(1, G[i]):
                    z[i, k] = m.addVar(vtype=GRB.BINARY, name=f"z[{i},{k}]")

            # Objective
            m.setObjective(
                gp.quicksum((D[i][k] - D[i][k-1]) * z[i, k]
                           for i in range(n) for k in range(1, G[i])),
                GRB.MINIMIZE
            )

            # Constraints
            m.addConstr(gp.quicksum(y[i] for i in range(n)) == p, name="pFacilities")
            for i in range(n):
                for k in range(1, G[i]):
                    m.addConstr(
                        z[i, k] + gp.quicksum(y[j] for j in range(n) 
                                             if cost_matrix[i][j] < D[i][k]) >= 1,
                        name=f"cover[{i},{k}]"
                    )

            m.optimize()

            time_elapsed = time.time() - time_start
            mem_end = get_memory_usage()

            if m.status == GRB.OPTIMAL:
                facilities = sorted([i for i in range(n) if y[i].x > 0.5])
                return {
                    'status': 'OPTIMAL',
                    'objective': m.objVal,
                    'facilities': facilities,
                    'time': time_elapsed,
                    'memory_start': mem_start,
                    'memory_end': mem_end,
                    'memory_used': mem_end - mem_start,
                    'num_vars': m.NumVars,
                    'num_constrs': m.NumConstrs
                }
            else:
                return {
                    'status': 'FAILED',
                    'objective': None,
                    'facilities': None,
                    'time': time_elapsed,
                    'memory_start': mem_start,
                    'memory_end': mem_end,
                    'memory_used': mem_end - mem_start,
                    'num_vars': m.NumVars,
                    'num_constrs': m.NumConstrs
                }
        except Exception as e:
            time_elapsed = time.time() - time_start
            mem_end = get_memory_usage()
            return {
                'status': f'ERROR: {str(e)}',
                'objective': None,
                'facilities': None,
                'time': time_elapsed,
                'memory_start': mem_start,
                'memory_end': mem_end,
                'memory_used': mem_end - mem_start,
                'num_vars': 0,
                'num_constrs': 0
            }


    # ============================================================
    # BENDERS DECOMPOSITION (Callback-based)
    # ============================================================

    def solve_p_median_benders(self, n, p, cost_matrix, verbose=False):
        """
        Benders Decomposition via Lazy Callbacks
        """
        mem_start = get_memory_usage()
        time_start = time.time()

        try:
            # Construct distance thresholds
            D = {}
            G = {}
            for i in range(n):
                unique_costs = sorted(set(cost_matrix[i]))
                D[i] = unique_costs
                G[i] = len(unique_costs)

            # Precompute facilities at each distance level
            facilities_at_distance = {}
            for i in range(n):
                facilities_at_distance[i] = {}
                for k in range(1, G[i]):
                    Dk = D[i][k]
                    facilities_at_distance[i][k] = [j for j in range(n) 
                                                   if cost_matrix[i][j] < Dk]

            # Store variable references for callback
            master_y_vars = []
            master_theta_vars = []

            def build_master():
                m = gp.Model("master-benders")
                m.Params.OutputFlag = 0
                m.Params.LazyConstraints = 1

                nonlocal master_y_vars, master_theta_vars
                master_y_vars = [m.addVar(vtype=GRB.BINARY, name=f"y_{i}") 
                                for i in range(n)]
                master_theta_vars = [m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                             name=f"theta_{i}") for i in range(n)]

                m.setObjective(gp.quicksum(master_theta_vars[i] for i in range(n)), 
                              GRB.MINIMIZE)
                m.addConstr(gp.quicksum(master_y_vars[i] for i in range(n)) == p)

                return m

            def solve_subproblem(i, y_bar):
                """Solve subproblem for client i"""
                sp_model = gp.Model(f"subproblem_{i}")
                sp_model.Params.OutputFlag = 0

                z_sp = {}
                for k in range(1, G[i]):
                    z_sp[k] = sp_model.addVar(vtype=GRB.BINARY, name=f"z_{k}")

                obj_expr = 0
                for k in range(1, G[i]):
                    obj_expr += (D[i][k] - D[i][k-1]) * z_sp[k]

                sp_model.setObjective(obj_expr, GRB.MINIMIZE)

                for k in range(1, G[i]):
                    facs = facilities_at_distance[i][k]
                    y_sum = sum(y_bar[j] for j in facs)
                    sp_model.addConstr(z_sp[k] + y_sum >= 1)

                sp_model.optimize()

                if sp_model.status == GRB.OPTIMAL:
                    return sp_model.ObjVal
                else:
                    return None

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    y_hat = [model.cbGetSolution(master_y_vars[j]) for j in range(n)]
                    theta_hat = [model.cbGetSolution(master_theta_vars[i]) 
                               for i in range(n)]

                    tol = 1e-6
                    for i in range(n):
                        sp_obj = solve_subproblem(i, y_hat)
                        if sp_obj is not None and sp_obj > theta_hat[i] + tol:
                            model.cbLazy(master_theta_vars[i] >= sp_obj)

            master = build_master()
            master.optimize(callback)

            time_elapsed = time.time() - time_start
            mem_end = get_memory_usage()

            if master.status == GRB.OPTIMAL:
                facilities = sorted([i for i in range(n) 
                                   if master_y_vars[i].x > 0.5])
                # Calculate actual objective
                obj_val = 0
                y_final = [master_y_vars[i].x for i in range(n)]
                for i in range(n):
                    sp_obj = solve_subproblem(i, y_final)
                    if sp_obj is not None:
                        obj_val += sp_obj

                return {
                    'status': 'OPTIMAL',
                    'objective': obj_val,
                    'facilities': facilities,
                    'time': time_elapsed,
                    'memory_start': mem_start,
                    'memory_end': mem_end,
                    'memory_used': mem_end - mem_start,
                    'num_vars': master.NumVars,
                    'num_constrs': master.NumConstrs
                }
            else:
                return {
                    'status': 'FAILED',
                    'objective': None,
                    'facilities': None,
                    'time': time_elapsed,
                    'memory_start': mem_start,
                    'memory_end': mem_end,
                    'memory_used': mem_end - mem_start,
                    'num_vars': master.NumVars,
                    'num_constrs': master.NumConstrs
                }
        except Exception as e:
            time_elapsed = time.time() - time_start
            mem_end = get_memory_usage()
            return {
                'status': f'ERROR: {str(e)}',
                'objective': None,
                'facilities': None,
                'time': time_elapsed,
                'memory_start': mem_start,
                'memory_end': mem_end,
                'memory_used': mem_end - mem_start,
                'num_vars': 0,
                'num_constrs': 0
            }


    def run_test_case(self, test_id, n, p, cost_matrix, test_name):
        """Run both methods on same test case"""
        print(f"\n{'='*80}")
        print(f"Test {test_id}: {test_name} (n={n}, p={p})")
        print('='*80)

        # CR Formulation
        print("Running CR Formulation...", end=" ", flush=True)
        cr_result = self.solve_p_median_CR(n, p, cost_matrix)
        print(f"✓ Time: {cr_result['time']:.6f}s")

        # Benders
        print("Running Benders Decomposition...", end=" ", flush=True)
        benders_result = self.solve_p_median_benders(n, p, cost_matrix)
        print(f"✓ Time: {benders_result['time']:.6f}s")

        # Compare
        print(f"\nResults Comparison:")
        print(f"  CR Objective:       {cr_result['objective']:.6f}")
        print(f"  Benders Objective:  {benders_result['objective']:.6f}")
        print(f"  Difference:         {abs(cr_result['objective'] - benders_result['objective']):.8f}")
        print(f"  CR Facilities:      {cr_result['facilities']}")
        print(f"  Benders Facilities: {benders_result['facilities']}")

        match = (abs(cr_result['objective'] - benders_result['objective']) < 1e-4 and 
                cr_result['facilities'] == benders_result['facilities'])
        print(f"  Solutions Match:    {'✓ YES' if match else '✗ NO'}")

        print(f"\nPerformance Metrics:")
        print(f"  CR Time:            {cr_result['time']:.6f}s")
        print(f"  Benders Time:       {benders_result['time']:.6f}s")
        print(f"  Speedup:            {cr_result['time']/benders_result['time']:.2f}x")
        print(f"\n  CR Memory:          {cr_result['memory_used']:.2f} MB")
        print(f"  Benders Memory:     {benders_result['memory_used']:.2f} MB")
        print(f"  Memory Difference:  {benders_result['memory_used'] - cr_result['memory_used']:.2f} MB")

        print(f"\n  CR Variables:       {cr_result['num_vars']}")
        print(f"  CR Constraints:     {cr_result['num_constrs']}")
        print(f"  Benders Variables:  {benders_result['num_vars']}")
        print(f"  Benders Constraints:{benders_result['num_constrs']}")

        return {
            'test_id': test_id,
            'test_name': test_name,
            'n': n,
            'p': p,
            'cr_objective': cr_result['objective'],
            'benders_objective': benders_result['objective'],
            'cr_time': cr_result['time'],
            'benders_time': benders_result['time'],
            'cr_memory': cr_result['memory_used'],
            'benders_memory': benders_result['memory_used'],
            'cr_vars': cr_result['num_vars'],
            'cr_constrs': cr_result['num_constrs'],
            'benders_vars': benders_result['num_vars'],
            'benders_constrs': benders_result['num_constrs'],
            'solutions_match': match
        }


    def generate_random_distance_matrix(self, n, seed=None):
        """Generate random symmetric distance matrix"""
        if seed is not None:
            np.random.seed(seed)

        # Generate random distances
        dist = np.random.randint(1, 100, size=(n, n))

        # Make symmetric
        dist = np.maximum(dist, dist.T)

        # Zero diagonal
        np.fill_diagonal(dist, 0)

        return dist


# ============================================================
# 10 TEST CASES
# ============================================================

if __name__ == "__main__":
    suite = BenchmarkSuite()
    all_results = []

    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK: CR vs BENDERS DECOMPOSITION FOR p-MEDIAN")
    print("="*80)
    print(f"Metrics: Time (seconds), Memory (MB), Objectives, Solution Verification")

    # Test Case 1: Small (5 nodes, p=2)
    dist1 = suite.generate_random_distance_matrix(5, seed=42)
    result1 = suite.run_test_case(1, 5, 2, dist1.tolist(), "Small Instance")
    all_results.append(result1)

    # Test Case 2: Small (5 nodes, p=3)
    dist2 = suite.generate_random_distance_matrix(5, seed=43)
    result2 = suite.run_test_case(2, 5, 3, dist2.tolist(), "Small Instance - Higher p")
    all_results.append(result2)

    # Test Case 3: Medium (10 nodes, p=3)
    dist3 = suite.generate_random_distance_matrix(10, seed=44)
    result3 = suite.run_test_case(3, 10, 3, dist3.tolist(), "Medium Instance")
    all_results.append(result3)

    # Test Case 4: Medium (10 nodes, p=5)
    dist4 = suite.generate_random_distance_matrix(10, seed=45)
    result4 = suite.run_test_case(4, 10, 5, dist4.tolist(), "Medium Instance - Higher p")
    all_results.append(result4)

    # Test Case 5: Medium-Large (15 nodes, p=4)
    dist5 = suite.generate_random_distance_matrix(15, seed=46)
    result5 = suite.run_test_case(5, 15, 4, dist5.tolist(), "Medium-Large Instance")
    all_results.append(result5)

    # Test Case 6: Medium-Large (15 nodes, p=6)
    dist6 = suite.generate_random_distance_matrix(15, seed=47)
    result6 = suite.run_test_case(6, 15, 6, dist6.tolist(), "Medium-Large Instance - Higher p")
    all_results.append(result6)

    # Test Case 7: Large (20 nodes, p=5)
    dist7 = suite.generate_random_distance_matrix(20, seed=48)
    result7 = suite.run_test_case(7, 20, 5, dist7.tolist(), "Large Instance")
    all_results.append(result7)

    # Test Case 8: Large (20 nodes, p=8)
    dist8 = suite.generate_random_distance_matrix(20, seed=49)
    result8 = suite.run_test_case(8, 20, 8, dist8.tolist(), "Large Instance - Higher p")
    all_results.append(result8)

    # Test Case 9: Large (25 nodes, p=6)
    dist9 = suite.generate_random_distance_matrix(25, seed=50)
    result9 = suite.run_test_case(9, 25, 6, dist9.tolist(), "Large Instance - 25 nodes")
    all_results.append(result9)

    # Test Case 10: Large (30 nodes, p=7)
    dist10 = suite.generate_random_distance_matrix(30, seed=51)
    result10 = suite.run_test_case(10, 30, 7, dist10.tolist(), "Very Large Instance - 30 nodes")
    all_results.append(result10)


    # ============================================================
    # SUMMARY TABLE
    # ============================================================

    print("\n\n" + "="*80)
    print("SUMMARY TABLE - ALL TEST CASES")
    print("="*80)

    print(f"\n{'Test':<5} {'n':<5} {'p':<5} {'CR Time':<12} {'Benders':<12} {'Speedup':<10} {'CR Mem':<10} {'Benders':<10} {'Match':<8}")
    print("-" * 100)

    for r in all_results:
        speedup = r['cr_time'] / r['benders_time'] if r['benders_time'] > 0 else 0
        match_str = "✓" if r['solutions_match'] else "✗"
        print(f"{r['test_id']:<5} {r['n']:<5} {r['p']:<5} {r['cr_time']:<12.6f} {r['benders_time']:<12.6f} {speedup:<10.2f}x {r['cr_memory']:<10.2f} {r['benders_memory']:<10.2f} {match_str:<8}")

    # Statistics
    cr_times = [r['cr_time'] for r in all_results]
    benders_times = [r['benders_time'] for r in all_results]
    speedups = [r['cr_time'] / r['benders_time'] for r in all_results if r['benders_time'] > 0]

    print("-" * 100)
    print(f"{'AVERAGE':<5} {'':<5} {'':<5} {np.mean(cr_times):<12.6f} {np.mean(benders_times):<12.6f} {np.mean(speedups):<10.2f}x {np.mean([r['cr_memory'] for r in all_results]):<10.2f} {np.mean([r['benders_memory'] for r in all_results]):<10.2f}")

    print(f"\nTotal Tests Passed (solutions match): {sum(1 for r in all_results if r['solutions_match'])}/10")

    # Write CSV
    import csv
    with open('benchmark_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: benchmark_results.csv")