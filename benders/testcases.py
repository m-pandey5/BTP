# """
# Comprehensive Benchmark: CR Formulation vs Benders Decomposition for p-Median

# Metrics: 
# - Time (seconds)
# - Memory Usage (MB)
# - Peak Memory (MB)
# - Objective Value
# - Facility Selection
# """

# import gurobipy as gp
# from gurobipy import GRB
# import numpy as np
# import time
# import psutil
# import os


# def get_memory_usage():
#     """Get current process memory usage in MB"""
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / 1024 / 1024


# class BenchmarkSuite:
#     def __init__(self):
#         self.results = []

#     # ============================================================
#     # CR FORMULATION (Direct)
#     # ============================================================

#     def solve_p_median_CR(self, n, p, cost_matrix, verbose=False):
#         """
#         CR Formulation (Direct approach):
#         min Σᵢ Σₖ (D_i^k - D_i^(k-1)) * z_i^k
#         s.t. Σᵢ yᵢ = p
#              z_i^k + Σⱼ: c_ij < D_i^k yⱼ >= 1
#         """
#         mem_start = get_memory_usage()
#         time_start = time.time()

#         try:
#             # Construct distance thresholds
#             D = {}
#             G = {}
#             for i in range(n):
#                 unique_costs = sorted(set(cost_matrix[i]))
#                 D[i] = unique_costs
#                 G[i] = len(unique_costs)

#             # Model
#             m = gp.Model("p-median-CR")
#             m.Params.OutputFlag = 0  # Suppress output

#             # Variables
#             y = m.addVars(n, vtype=GRB.BINARY, name="y")
#             z = {}
#             for i in range(n):
#                 for k in range(1, G[i]):
#                     z[i, k] = m.addVar(vtype=GRB.BINARY, name=f"z[{i},{k}]")

#             # Objective
#             m.setObjective(
#                 gp.quicksum((D[i][k] - D[i][k-1]) * z[i, k]
#                            for i in range(n) for k in range(1, G[i])),
#                 GRB.MINIMIZE
#             )

#             # Constraints
#             m.addConstr(gp.quicksum(y[i] for i in range(n)) == p, name="pFacilities")
#             for i in range(n):
#                 for k in range(1, G[i]):
#                     m.addConstr(
#                         z[i, k] + gp.quicksum(y[j] for j in range(n) 
#                                              if cost_matrix[i][j] < D[i][k]) >= 1,
#                         name=f"cover[{i},{k}]"
#                     )

#             m.optimize()

#             time_elapsed = time.time() - time_start
#             mem_end = get_memory_usage()

#             if m.status == GRB.OPTIMAL:
#                 facilities = sorted([i for i in range(n) if y[i].x > 0.5])
#                 return {
#                     'status': 'OPTIMAL',
#                     'objective': m.objVal,
#                     'facilities': facilities,
#                     'time': time_elapsed,
#                     'memory_start': mem_start,
#                     'memory_end': mem_end,
#                     'memory_used': mem_end - mem_start,
#                     'num_vars': m.NumVars,
#                     'num_constrs': m.NumConstrs
#                 }
#             else:
#                 return {
#                     'status': 'FAILED',
#                     'objective': None,
#                     'facilities': None,
#                     'time': time_elapsed,
#                     'memory_start': mem_start,
#                     'memory_end': mem_end,
#                     'memory_used': mem_end - mem_start,
#                     'num_vars': m.NumVars,
#                     'num_constrs': m.NumConstrs
#                 }
#         except Exception as e:
#             time_elapsed = time.time() - time_start
#             mem_end = get_memory_usage()
#             return {
#                 'status': f'ERROR: {str(e)}',
#                 'objective': None,
#                 'facilities': None,
#                 'time': time_elapsed,
#                 'memory_start': mem_start,
#                 'memory_end': mem_end,
#                 'memory_used': mem_end - mem_start,
#                 'num_vars': 0,
#                 'num_constrs': 0
#             }


#     # ============================================================
#     # BENDERS DECOMPOSITION (Callback-based)
#     # ============================================================

#     def solve_p_median_benders(self, n, p, cost_matrix, verbose=False):
#         """
#         Benders Decomposition via Lazy Callbacks
#         """
#         mem_start = get_memory_usage()
#         time_start = time.time()

#         try:
#             # Construct distance thresholds
#             D = {}
#             G = {}
#             for i in range(n):
#                 unique_costs = sorted(set(cost_matrix[i]))
#                 D[i] = unique_costs
#                 G[i] = len(unique_costs)

#             # Precompute facilities at each distance level
#             facilities_at_distance = {}
#             for i in range(n):
#                 facilities_at_distance[i] = {}
#                 for k in range(1, G[i]):
#                     Dk = D[i][k]
#                     facilities_at_distance[i][k] = [j for j in range(n) 
#                                                    if cost_matrix[i][j] < Dk]

#             # Store variable references for callback
#             master_y_vars = []
#             master_theta_vars = []

#             def build_master():
#                 m = gp.Model("master-benders")
#                 m.Params.OutputFlag = 0
#                 m.Params.LazyConstraints = 1

#                 nonlocal master_y_vars, master_theta_vars
#                 master_y_vars = [m.addVar(vtype=GRB.BINARY, name=f"y_{i}") 
#                                 for i in range(n)]
#                 master_theta_vars = [m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
#                                              name=f"theta_{i}") for i in range(n)]

#                 m.setObjective(gp.quicksum(master_theta_vars[i] for i in range(n)), 
#                               GRB.MINIMIZE)
#                 m.addConstr(gp.quicksum(master_y_vars[i] for i in range(n)) == p)

#                 return m

#             def solve_subproblem(i, y_bar):
#                 """Solve subproblem for client i"""
#                 sp_model = gp.Model(f"subproblem_{i}")
#                 sp_model.Params.OutputFlag = 0

#                 z_sp = {}
#                 for k in range(1, G[i]):
#                     z_sp[k] = sp_model.addVar(vtype=GRB.BINARY, name=f"z_{k}")

#                 obj_expr = 0
#                 for k in range(1, G[i]):
#                     obj_expr += (D[i][k] - D[i][k-1]) * z_sp[k]

#                 sp_model.setObjective(obj_expr, GRB.MINIMIZE)

#                 for k in range(1, G[i]):
#                     facs = facilities_at_distance[i][k]
#                     y_sum = sum(y_bar[j] for j in facs)
#                     sp_model.addConstr(z_sp[k] + y_sum >= 1)

#                 sp_model.optimize()

#                 if sp_model.status == GRB.OPTIMAL:
#                     return sp_model.ObjVal
#                 else:
#                     return None

#             def callback(model, where):
#                 if where == GRB.Callback.MIPSOL:
#                     y_hat = [model.cbGetSolution(master_y_vars[j]) for j in range(n)]
#                     theta_hat = [model.cbGetSolution(master_theta_vars[i]) 
#                                for i in range(n)]

#                     tol = 1e-6
#                     for i in range(n):
#                         sp_obj = solve_subproblem(i, y_hat)
#                         if sp_obj is not None and sp_obj > theta_hat[i] + tol:
#                             model.cbLazy(master_theta_vars[i] >= sp_obj)

#             master = build_master()
#             master.optimize(callback)

#             time_elapsed = time.time() - time_start
#             mem_end = get_memory_usage()

#             if master.status == GRB.OPTIMAL:
#                 facilities = sorted([i for i in range(n) 
#                                    if master_y_vars[i].x > 0.5])
#                 # Calculate actual objective
#                 obj_val = 0
#                 y_final = [master_y_vars[i].x for i in range(n)]
#                 for i in range(n):
#                     sp_obj = solve_subproblem(i, y_final)
#                     if sp_obj is not None:
#                         obj_val += sp_obj

#                 return {
#                     'status': 'OPTIMAL',
#                     'objective': obj_val,
#                     'facilities': facilities,
#                     'time': time_elapsed,
#                     'memory_start': mem_start,
#                     'memory_end': mem_end,
#                     'memory_used': mem_end - mem_start,
#                     'num_vars': master.NumVars,
#                     'num_constrs': master.NumConstrs
#                 }
#             else:
#                 return {
#                     'status': 'FAILED',
#                     'objective': None,
#                     'facilities': None,
#                     'time': time_elapsed,
#                     'memory_start': mem_start,
#                     'memory_end': mem_end,
#                     'memory_used': mem_end - mem_start,
#                     'num_vars': master.NumVars,
#                     'num_constrs': master.NumConstrs
#                 }
#         except Exception as e:
#             time_elapsed = time.time() - time_start
#             mem_end = get_memory_usage()
#             return {
#                 'status': f'ERROR: {str(e)}',
#                 'objective': None,
#                 'facilities': None,
#                 'time': time_elapsed,
#                 'memory_start': mem_start,
#                 'memory_end': mem_end,
#                 'memory_used': mem_end - mem_start,
#                 'num_vars': 0,
#                 'num_constrs': 0
#             }


#     def run_test_case(self, test_id, n, p, cost_matrix, test_name):
#         """Run both methods on same test case"""
#         print(f"\n{'='*80}")
#         print(f"Test {test_id}: {test_name} (n={n}, p={p})")
#         print('='*80)

#         # CR Formulation
#         print("Running CR Formulation...", end=" ", flush=True)
#         cr_result = self.solve_p_median_CR(n, p, cost_matrix)
#         print(f"✓ Time: {cr_result['time']:.6f}s")

#         # Benders
#         print("Running Benders Decomposition...", end=" ", flush=True)
#         benders_result = self.solve_p_median_benders(n, p, cost_matrix)
#         print(f"✓ Time: {benders_result['time']:.6f}s")

#         # Compare
#         print(f"\nResults Comparison:")
#         print(f"  CR Objective:       {cr_result['objective']:.6f}")
#         print(f"  Benders Objective:  {benders_result['objective']:.6f}")
#         print(f"  Difference:         {abs(cr_result['objective'] - benders_result['objective']):.8f}")
#         print(f"  CR Facilities:      {cr_result['facilities']}")
#         print(f"  Benders Facilities: {benders_result['facilities']}")

#         match = (abs(cr_result['objective'] - benders_result['objective']) < 1e-4 and 
#                 cr_result['facilities'] == benders_result['facilities'])
#         print(f"  Solutions Match:    {'✓ YES' if match else '✗ NO'}")

#         print(f"\nPerformance Metrics:")
#         print(f"  CR Time:            {cr_result['time']:.6f}s")
#         print(f"  Benders Time:       {benders_result['time']:.6f}s")
#         print(f"  Speedup:            {cr_result['time']/benders_result['time']:.2f}x")
#         print(f"\n  CR Memory:          {cr_result['memory_used']:.2f} MB")
#         print(f"  Benders Memory:     {benders_result['memory_used']:.2f} MB")
#         print(f"  Memory Difference:  {benders_result['memory_used'] - cr_result['memory_used']:.2f} MB")

#         print(f"\n  CR Variables:       {cr_result['num_vars']}")
#         print(f"  CR Constraints:     {cr_result['num_constrs']}")
#         print(f"  Benders Variables:  {benders_result['num_vars']}")
#         print(f"  Benders Constraints:{benders_result['num_constrs']}")

#         return {
#             'test_id': test_id,
#             'test_name': test_name,
#             'n': n,
#             'p': p,
#             'cr_objective': cr_result['objective'],
#             'benders_objective': benders_result['objective'],
#             'cr_time': cr_result['time'],
#             'benders_time': benders_result['time'],
#             'cr_memory': cr_result['memory_used'],
#             'benders_memory': benders_result['memory_used'],
#             'cr_vars': cr_result['num_vars'],
#             'cr_constrs': cr_result['num_constrs'],
#             'benders_vars': benders_result['num_vars'],
#             'benders_constrs': benders_result['num_constrs'],
#             'solutions_match': match
#         }


#     def generate_random_distance_matrix(self, n, seed=None):
#         """Generate random symmetric distance matrix"""
#         if seed is not None:
#             np.random.seed(seed)

#         # Generate random distances
#         dist = np.random.randint(1, 100, size=(n, n))

#         # Make symmetric
#         dist = np.maximum(dist, dist.T)

#         # Zero diagonal
#         np.fill_diagonal(dist, 0)

#         return dist


# # ============================================================
# # 10 TEST CASES
# # ============================================================

# if __name__ == "__main__":
#     suite = BenchmarkSuite()
#     all_results = []

#     print("\n" + "="*80)
#     print("COMPREHENSIVE BENCHMARK: CR vs BENDERS DECOMPOSITION FOR p-MEDIAN")
#     print("="*80)
#     print(f"Metrics: Time (seconds), Memory (MB), Objectives, Solution Verification")

#     # Test Case 1: Small (5 nodes, p=2)
#     dist1 = suite.generate_random_distance_matrix(5, seed=42)
#     result1 = suite.run_test_case(1, 5, 2, dist1.tolist(), "Small Instance")
#     all_results.append(result1)

#     # Test Case 2: Small (5 nodes, p=3)
#     dist2 = suite.generate_random_distance_matrix(5, seed=43)
#     result2 = suite.run_test_case(2, 5, 3, dist2.tolist(), "Small Instance - Higher p")
#     all_results.append(result2)

#     # Test Case 3: Medium (10 nodes, p=3)
#     dist3 = suite.generate_random_distance_matrix(10, seed=44)
#     result3 = suite.run_test_case(3, 10, 3, dist3.tolist(), "Medium Instance")
#     all_results.append(result3)

#     # Test Case 4: Medium (10 nodes, p=5)
#     dist4 = suite.generate_random_distance_matrix(10, seed=45)
#     result4 = suite.run_test_case(4, 10, 5, dist4.tolist(), "Medium Instance - Higher p")
#     all_results.append(result4)

#     # Test Case 5: Medium-Large (15 nodes, p=4)
#     dist5 = suite.generate_random_distance_matrix(15, seed=46)
#     result5 = suite.run_test_case(5, 15, 4, dist5.tolist(), "Medium-Large Instance")
#     all_results.append(result5)

#     # Test Case 6: Medium-Large (15 nodes, p=6)
#     dist6 = suite.generate_random_distance_matrix(15, seed=47)
#     result6 = suite.run_test_case(6, 15, 6, dist6.tolist(), "Medium-Large Instance - Higher p")
#     all_results.append(result6)

#     # Test Case 7: Large (20 nodes, p=5)
#     dist7 = suite.generate_random_distance_matrix(20, seed=48)
#     result7 = suite.run_test_case(7, 20, 5, dist7.tolist(), "Large Instance")
#     all_results.append(result7)

#     # Test Case 8: Large (20 nodes, p=8)
#     dist8 = suite.generate_random_distance_matrix(20, seed=49)
#     result8 = suite.run_test_case(8, 20, 8, dist8.tolist(), "Large Instance - Higher p")
#     all_results.append(result8)

#     # Test Case 9: Large (25 nodes, p=6)
#     dist9 = suite.generate_random_distance_matrix(25, seed=50)
#     result9 = suite.run_test_case(9, 25, 6, dist9.tolist(), "Large Instance - 25 nodes")
#     all_results.append(result9)

#     # Test Case 10: Large (30 nodes, p=7)
#     dist10 = suite.generate_random_distance_matrix(30, seed=51)
#     result10 = suite.run_test_case(10, 30, 7, dist10.tolist(), "Very Large Instance - 30 nodes")
#     all_results.append(result10)


#     # ============================================================
#     # SUMMARY TABLE
#     # ============================================================

#     print("\n\n" + "="*80)
#     print("SUMMARY TABLE - ALL TEST CASES")
#     print("="*80)

#     print(f"\n{'Test':<5} {'n':<5} {'p':<5} {'CR Time':<12} {'Benders':<12} {'Speedup':<10} {'CR Mem':<10} {'Benders':<10} {'Match':<8}")
#     print("-" * 100)

#     for r in all_results:
#         speedup = r['cr_time'] / r['benders_time'] if r['benders_time'] > 0 else 0
#         match_str = "✓" if r['solutions_match'] else "✗"
#         print(f"{r['test_id']:<5} {r['n']:<5} {r['p']:<5} {r['cr_time']:<12.6f} {r['benders_time']:<12.6f} {speedup:<10.2f}x {r['cr_memory']:<10.2f} {r['benders_memory']:<10.2f} {match_str:<8}")

#     # Statistics
#     cr_times = [r['cr_time'] for r in all_results]
#     benders_times = [r['benders_time'] for r in all_results]
#     speedups = [r['cr_time'] / r['benders_time'] for r in all_results if r['benders_time'] > 0]

#     print("-" * 100)
#     print(f"{'AVERAGE':<5} {'':<5} {'':<5} {np.mean(cr_times):<12.6f} {np.mean(benders_times):<12.6f} {np.mean(speedups):<10.2f}x {np.mean([r['cr_memory'] for r in all_results]):<10.2f} {np.mean([r['benders_memory'] for r in all_results]):<10.2f}")

#     print(f"\nTotal Tests Passed (solutions match): {sum(1 for r in all_results if r['solutions_match'])}/10")

#     # Write CSV
#     import csv
#     with open('benchmark_results.csv', 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
#         writer.writeheader()
#         writer.writerows(all_results)

#     print(f"\nResults saved to: benchmark_results.csv")
"""
10 Test Cases: F3 Formulation vs Benders Decomposition
Benchmark Suite with N = M (equal clients and facilities)

Metrics:
- Time (seconds)
- Memory (MB)
- Objectives
- Solution Verification
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import psutil
import os
import csv
from datetime import datetime


class BenchmarkF3Benders:
    def __init__(self):
        self.results = []

    def get_memory_usage(self):
        """Get current process memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


    def solve_F3_formulation(self, N, M, p, K, D, distance_matrix):
        """
        Solve p-median using direct F3 formulation.
        min sum_i [ D_i^1 + sum_{k=1}^{K_i-1} (D_i^{k+1} - D_i^k) * z_i^k ]
        """
        mem_start = self.get_memory_usage()
        time_start = time.time()

        try:
            model = gp.Model("F3_pMedian")
            model.Params.OutputFlag = 0

            # Variables
            y = model.addVars(M, vtype=GRB.BINARY, name="y")
            z = {}
            for i in range(N):
                for k in range(1, K[i]+1):
                    z[i,k] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"z_{i}_{k}")

            # Precompute facilities at each distance level
            facilities_at_distance = {}
            for i in range(N):
                facilities_at_distance[i] = {}
                for k in range(1, K[i]+1):
                    Dk = D[i][k]
                    facilities_at_distance[i][k] = [j for j in range(M) 
                                                   if abs(distance_matrix[i,j] - Dk) < 1e-8]

            # Objective
            obj = gp.quicksum(D[i][1] for i in range(N))
            for i in range(N):
                for k in range(1, K[i]):
                    obj += (D[i][k+1] - D[i][k]) * z[i,k]

            model.setObjective(obj, GRB.MINIMIZE)

            # Constraints
            model.addConstr(gp.quicksum(y[j] for j in range(M)) == p, name="p_facilities")

            for i in range(N):
                facs_at_D1 = facilities_at_distance[i][1]
                model.addConstr(z[i,1] + gp.quicksum(y[j] for j in facs_at_D1) >= 1)

            for i in range(N):
                for k in range(2, K[i]+1):
                    facs_at_Dk = facilities_at_distance[i][k]
                    model.addConstr(z[i,k] + gp.quicksum(y[j] for j in facs_at_Dk) >= z[i,k-1])

            model.update()
            model.optimize()

            time_elapsed = time.time() - time_start
            mem_end = self.get_memory_usage()

            if model.status == GRB.OPTIMAL:
                y_sol = [y[j].X for j in range(M)]
                return {
                    'objective': model.ObjVal,
                    'y': y_sol,
                    'facilities': sorted([j for j in range(M) if y_sol[j] > 0.5]),
                    'time': time_elapsed,
                    'memory_used': mem_end - mem_start,
                    'status': 'OPTIMAL',
                    'num_vars': model.NumVars,
                    'num_constrs': model.NumConstrs
                }
            else:
                return {
                    'objective': None,
                    'y': None,
                    'facilities': None,
                    'time': time_elapsed,
                    'memory_used': mem_end - mem_start,
                    'status': 'FAILED',
                    'num_vars': model.NumVars,
                    'num_constrs': model.NumConstrs
                }
        except Exception as e:
            time_elapsed = time.time() - time_start
            mem_end = self.get_memory_usage()
            return {
                'objective': None,
                'y': None,
                'facilities': None,
                'time': time_elapsed,
                'memory_used': mem_end - mem_start,
                'status': f'ERROR: {str(e)}',
                'num_vars': 0,
                'num_constrs': 0
            }


    def solve_benders_pmedian(self, N, M, p, K, D, distance_matrix):
        """
        Benders Decomposition with lazy callbacks for p-median.
        """
        mem_start = self.get_memory_usage()
        time_start = time.time()

        try:
            # Precompute facilities at each distance level
            facilities_at_distance = {}
            for i in range(N):
                facilities_at_distance[i] = {}
                for k in range(1, K[i]+1):
                    Dk = D[i][k]
                    facilities_at_distance[i][k] = [j for j in range(M) 
                                                   if abs(distance_matrix[i,j] - Dk) < 1e-8]

            # Store variable references
            master_y_vars = []
            master_theta_vars = []

            def build_master():
                m = gp.Model("master-benders")
                m.Params.OutputFlag = 0
                m.Params.LazyConstraints = 1

                nonlocal master_y_vars, master_theta_vars
                master_y_vars = [m.addVar(vtype=GRB.BINARY, name=f"y_{i}") 
                                for i in range(M)]
                master_theta_vars = [m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                             name=f"theta_{i}") for i in range(N)]

                m.setObjective(gp.quicksum(master_theta_vars[i] for i in range(N)), 
                              GRB.MINIMIZE)
                m.addConstr(gp.quicksum(master_y_vars[j] for j in range(M)) == p)

                return m

            def solve_subproblem(i, y_bar):
                """Solve subproblem for client i with fixed y_bar"""
                sp_model = gp.Model(f"subproblem_{i}")
                sp_model.Params.OutputFlag = 0

                z_sp = {}
                for k in range(1, K[i]+1):
                    z_sp[k] = sp_model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"z_{k}")

                obj_expr = D[i][1]
                for k in range(1, K[i]):
                    obj_expr += (D[i][k+1] - D[i][k]) * z_sp[k]

                sp_model.setObjective(obj_expr, GRB.MINIMIZE)

                # Coverage constraint at D_i^1
                facs_D1 = facilities_at_distance[i][1]
                y_sum_D1 = sum(y_bar[j] for j in facs_D1)
                sp_model.addConstr(z_sp[1] + y_sum_D1 >= 1)

                # Cascading constraints
                for k in range(2, K[i]+1):
                    facs_Dk = facilities_at_distance[i][k]
                    y_sum_Dk = sum(y_bar[j] for j in facs_Dk)
                    sp_model.addConstr(z_sp[k] + y_sum_Dk >= z_sp[k-1])

                sp_model.optimize()

                if sp_model.status == GRB.OPTIMAL:
                    return sp_model.ObjVal
                else:
                    return None

            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        y_hat = [model.cbGetSolution(master_y_vars[j]) for j in range(M)]
                        theta_hat = [model.cbGetSolution(master_theta_vars[i]) 
                                   for i in range(N)]

                        tol = 1e-6
                        for i in range(N):
                            sp_obj = solve_subproblem(i, y_hat)
                            if sp_obj is not None and sp_obj > theta_hat[i] + tol:
                                model.cbLazy(master_theta_vars[i] >= sp_obj)
                    except:
                        pass

            master = build_master()
            master.optimize(callback)

            time_elapsed = time.time() - time_start
            mem_end = self.get_memory_usage()

            if master.status == GRB.OPTIMAL:
                y_final = [master_y_vars[j].X for j in range(M)]
                # Calculate actual objective
                obj_val = 0
                for i in range(N):
                    sp_obj = solve_subproblem(i, y_final)
                    if sp_obj is not None:
                        obj_val += sp_obj

                return {
                    'objective': obj_val,
                    'y': y_final,
                    'facilities': sorted([j for j in range(M) if y_final[j] > 0.5]),
                    'time': time_elapsed,
                    'memory_used': mem_end - mem_start,
                    'status': 'OPTIMAL',
                    'num_vars': master.NumVars,
                    'num_constrs': master.NumConstrs
                }
            else:
                return {
                    'objective': None,
                    'y': None,
                    'facilities': None,
                    'time': time_elapsed,
                    'memory_used': mem_end - mem_start,
                    'status': 'FAILED',
                    'num_vars': master.NumVars,
                    'num_constrs': master.NumConstrs
                }
        except Exception as e:
            time_elapsed = time.time() - time_start
            mem_end = self.get_memory_usage()
            return {
                'objective': None,
                'y': None,
                'facilities': None,
                'time': time_elapsed,
                'memory_used': mem_end - mem_start,
                'status': f'ERROR: {str(e)}',
                'num_vars': 0,
                'num_constrs': 0
            }


    def run_test_case(self, test_id, N, p):
        """Run complete test case with N=M"""
        M = N  # N equals M

        print(f"\n{'='*80}")
        print(f"TEST {test_id}: N={N}, M={M}, p={p}")
        print('='*80)

        # Generate distance matrix
        np.random.seed(100 + test_id)
        distance_matrix = np.random.rand(N, M) * 100.0
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.maximum(distance_matrix, distance_matrix.T)

        # Compute K and D
        K = {i: min(4, M) for i in range(N)}
        D = {}
        for i in range(N):
            sorted_dists = sorted(set(np.round(distance_matrix[i, :], 8)))
            D[i] = {k: sorted_dists[min(k-1, len(sorted_dists)-1)] 
                   for k in range(1, K[i]+1)}

        # Solve with F3
        print("Solving F3 formulation...", end=" ", flush=True)
        f3_result = self.solve_F3_formulation(N, M, p, K, D, distance_matrix)
        print(f"✓ Time: {f3_result['time']:.6f}s")

        # Solve with Benders
        print("Solving Benders decomposition...", end=" ", flush=True)
        benders_result = self.solve_benders_pmedian(N, M, p, K, D, distance_matrix)
        print(f"✓ Time: {benders_result['time']:.6f}s")

        # Compare
        print(f"\nResults:")
        print(f"  F3 Objective:      {f3_result['objective']:.6f}")
        print(f"  Benders Objective: {benders_result['objective']:.6f}")
        print(f"  Difference:        {abs(f3_result['objective'] - benders_result['objective']):.8f}")
        print(f"  F3 Facilities:     {f3_result['facilities']}")
        print(f"  Benders Facilities:{benders_result['facilities']}")

        match = (abs(f3_result['objective'] - benders_result['objective']) < 1e-4 and 
                f3_result['facilities'] == benders_result['facilities'])
        print(f"  Solutions Match:   {'✓ YES' if match else '✗ NO'}")

        print(f"\nPerformance:")
        print(f"  F3 Time:           {f3_result['time']:.6f}s")
        print(f"  Benders Time:      {benders_result['time']:.6f}s")
        speedup = f3_result['time'] / benders_result['time'] if benders_result['time'] > 0 else 0
        print(f"  Speedup:           {speedup:.2f}x")
        print(f"  F3 Memory:         {f3_result['memory_used']:.2f} MB")
        print(f"  Benders Memory:    {benders_result['memory_used']:.2f} MB")

        print(f"\nModel Complexity:")
        print(f"  F3 Variables:      {f3_result['num_vars']}")
        print(f"  F3 Constraints:    {f3_result['num_constrs']}")
        print(f"  Benders Variables: {benders_result['num_vars']}")
        print(f"  Benders Constrs:   {benders_result['num_constrs']}")

        return {
            'test_id': test_id,
            'N': N,
            'M': M,
            'p': p,
            'p_ratio': p/N,
            'f3_objective': f3_result['objective'],
            'benders_objective': benders_result['objective'],
            'f3_time': f3_result['time'],
            'benders_time': benders_result['time'],
            'speedup': speedup,
            'f3_memory': f3_result['memory_used'],
            'benders_memory': benders_result['memory_used'],
            'f3_vars': f3_result['num_vars'],
            'f3_constrs': f3_result['num_constrs'],
            'benders_vars': benders_result['num_vars'],
            'benders_constrs': benders_result['num_constrs'],
            'solutions_match': match
        }


# ============================================================
# 10 TEST CASES WITH N = M
# ============================================================

if __name__ == "__main__":
    suite = BenchmarkF3Benders()
    all_results = []

    print("\n" + "="*80)
    print("BENCHMARK: F3 FORMULATION vs BENDERS DECOMPOSITION")
    print("Test Cases with N = M (Equal Clients and Facilities)")
    print("="*80)

    # Test Case 1: Small (N=M=5, p=2)
    result1 = suite.run_test_case(1, N=5, p=2)
    all_results.append(result1)

    # Test Case 2: Small (N=M=5, p=3)
    result2 = suite.run_test_case(2, N=5, p=3)
    all_results.append(result2)

    # Test Case 3: Small (N=M=5, p=4)
    result3 = suite.run_test_case(3, N=5, p=4)
    all_results.append(result3)

    # Test Case 4: Medium (N=M=10, p=3)
    result4 = suite.run_test_case(4, N=10, p=3)
    all_results.append(result4)

    # Test Case 5: Medium (N=M=10, p=5)
    result5 = suite.run_test_case(5, N=10, p=5)
    all_results.append(result5)

    # Test Case 6: Medium (N=M=10, p=7)
    result6 = suite.run_test_case(6, N=10, p=7)
    all_results.append(result6)

    # Test Case 7: Medium-Large (N=M=15, p=4)
    result7 = suite.run_test_case(7, N=15, p=4)
    all_results.append(result7)

    # Test Case 8: Medium-Large (N=M=15, p=7)
    result8 = suite.run_test_case(8, N=15, p=7)
    all_results.append(result8)

    # Test Case 9: Large (N=M=20, p=5)
    result9 = suite.run_test_case(9, N=20, p=5)
    all_results.append(result9)

    # Test Case 10: Large (N=M=20, p=10)
    result10 = suite.run_test_case(10, N=20, p=10)
    all_results.append(result10)


    # ============================================================
    # SUMMARY TABLE
    # ============================================================

    print("\n\n" + "="*100)
    print("SUMMARY TABLE - ALL TEST CASES")
    print("="*100)

    print(f"\n{'Test':<5} {'N=M':<5} {'p':<4} {'Ratio':<8} {'F3 Time':<12} {'Benders':<12} {'Speedup':<10} {'F3 Obj':<12} {'Match':<8}")
    print("-" * 100)

    for r in all_results:
        print(f"{r['test_id']:<5} {r['N']:<5} {r['p']:<4} {r['p_ratio']:<8.1%} {r['f3_time']:<12.6f} {r['benders_time']:<12.6f} {r['speedup']:<10.2f}x {r['f3_objective']:<12.2f} {'✓' if r['solutions_match'] else '✗':<8}")

    print("-" * 100)
    print(f"{'AVERAGE':<5} {'':<5} {'':<4} {np.mean([r['p_ratio'] for r in all_results]):<8.1%} {np.mean([r['f3_time'] for r in all_results]):<12.6f} {np.mean([r['benders_time'] for r in all_results]):<12.6f} {np.mean([r['speedup'] for r in all_results]):<10.2f}x")

    print(f"\nTotal Tests Passed (solutions match): {sum(1 for r in all_results if r['solutions_match'])}/10")

    # Save to CSV
    with open('f3_benders_comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: f3_benders_comparison.csv")
    print("="*100)