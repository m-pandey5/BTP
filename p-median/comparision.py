# Enhanced P-Median Formulation Comparison
# Key improvements to your existing code

import gurobipy as gp
from gurobipy import GRB
import time
import random

def solve_p_median_IP_enhanced(n, p, c, verbose=False):
    """
    Enhanced Original IP formulation with detailed statistics
    """
    m = gp.Model("p-median-IP")
    if not verbose:
        m.setParam('OutputFlag', 0)
    
    # Enable more detailed solving statistics
    m.setParam('MIPGap', 1e-9)  # Very tight optimality tolerance
    m.setParam('Threads', 1)    # Single thread for fair comparison
    
    # Decision variables
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    y = m.addVars(n, vtype=GRB.BINARY, name="y")
    
    # Objective
    m.setObjective(gp.quicksum(c[i][j] * x[i, j] for i in range(n) for j in range(n)),
                   GRB.MINIMIZE)
    
    # Constraints
    for i in range(n):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"assign_{i}")
    
    for i in range(n):
        for j in range(n):
            m.addConstr(x[i, j] <= y[j], name=f"valid_{i}_{j}")
    
    m.addConstr(gp.quicksum(y[j] for j in range(n)) == p, name="p_facilities")
    
    start_time = time.time()
    m.optimize()
    solve_time = time.time() - start_time
    
    # Enhanced statistics collection
    stats = {
        'objective': m.objVal if m.status == GRB.OPTIMAL else None,
        'solve_time': solve_time,
        'status': m.status,
        'num_vars': m.NumVars,
        'num_constrs': m.NumConstrs,
        'nodes_explored': m.NodeCount,
        'gap_at_termination': m.MIPGap if m.status == GRB.OPTIMAL else None,
        'root_relaxation': m.ObjBound if hasattr(m, 'ObjBound') else None
    }
    
    return stats, m

def solve_p_median_LP_enhanced(n, p, c, verbose=False):
    """
    Enhanced LP relaxation with integrality analysis
    """
    m = gp.Model("p-median-LP")
    if not verbose:
        m.setParam('OutputFlag', 0)
    
    # Decision variables (continuous)
    x = m.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    y = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")
    
    # Objective
    m.setObjective(gp.quicksum(c[i][j] * x[i, j] for i in range(n) for j in range(n)),
                   GRB.MINIMIZE)
    
    # Constraints
    for i in range(n):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
    
    for i in range(n):
        for j in range(n):
            m.addConstr(x[i, j] <= y[j])
    
    m.addConstr(gp.quicksum(y[j] for j in range(n)) == p)
    
    start_time = time.time()
    m.optimize()
    solve_time = time.time() - start_time
    
    # Detailed integrality analysis
    fractional_vars = 0
    max_fractionality = 0.0
    
    if m.status == GRB.OPTIMAL:
        # Check x variables
        for i in range(n):
            for j in range(n):
                val = x[i, j].x
                frac = min(val, 1-val)  # Distance from nearest integer
                if frac > 1e-6:
                    fractional_vars += 1
                    max_fractionality = max(max_fractionality, frac)
        
        # Check y variables  
        for j in range(n):
            val = y[j].x
            frac = min(val, 1-val)
            if frac > 1e-6:
                fractional_vars += 1
                max_fractionality = max(max_fractionality, frac)
    
    stats = {
        'objective': m.objVal if m.status == GRB.OPTIMAL else None,
        'solve_time': solve_time,
        'status': m.status,
        'fractional_vars': fractional_vars,
        'max_fractionality': max_fractionality,
        'num_vars': m.NumVars,
        'num_constrs': m.NumConstrs
    }
    
    return stats, m

def solve_p_median_CR_IP_enhanced(n, p, c, verbose=False):
    """
    Enhanced Cover-based IP formulation
    """
    # Construct D_i for each i
    D = {}
    G = {}
    for i in range(n):
        unique_costs = sorted(set(c[i]))
        D[i] = unique_costs
        G[i] = len(unique_costs)
    
    m = gp.Model("p-median-CR-IP")
    if not verbose:
        m.setParam('OutputFlag', 0)
    
    m.setParam('MIPGap', 1e-9)
    m.setParam('Threads', 1)
    
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
    m.addConstr(gp.quicksum(y[i] for i in range(n)) == p, name="p_facilities")
    
    for i in range(n):
        for k in range(1, G[i]):
            m.addConstr(
                z[i, k] + gp.quicksum(y[j] for j in range(n) if c[i][j] < D[i][k]) >= 1,
                name=f"cover_{i}_{k}"
            )
    
    start_time = time.time()
    m.optimize()
    solve_time = time.time() - start_time
    
    stats = {
        'objective': m.objVal if m.status == GRB.OPTIMAL else None,
        'solve_time': solve_time,
        'status': m.status,
        'num_vars': m.NumVars,
        'num_constrs': m.NumConstrs,
        'nodes_explored': m.NodeCount,
        'gap_at_termination': m.MIPGap if m.status == GRB.OPTIMAL else None,
        'root_relaxation': m.ObjBound if hasattr(m, 'ObjBound') else None
    }
    
    return stats, m

def solve_p_median_CR_LP_enhanced(n, p, c, verbose=False):
    """
    Enhanced LP relaxation of cover-based formulation
    """
    # Construct D_i for each i
    D = {}
    G = {}
    for i in range(n):
        unique_costs = sorted(set(c[i]))
        D[i] = unique_costs
        G[i] = len(unique_costs)
    
    m = gp.Model("p-median-CR-LP")
    if not verbose:
        m.setParam('OutputFlag', 0)
    
    # Variables (continuous)
    y = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")
    z = {}
    for i in range(n):
        for k in range(1, G[i]):
            z[i, k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"z[{i},{k}]")
    
    # Objective
    m.setObjective(
        gp.quicksum((D[i][k] - D[i][k-1]) * z[i, k]
                    for i in range(n) for k in range(1, G[i])),
        GRB.MINIMIZE
    )
    
    # Constraints
    m.addConstr(gp.quicksum(y[i] for i in range(n)) == p)
    
    for i in range(n):
        for k in range(1, G[i]):
            m.addConstr(
                z[i, k] + gp.quicksum(y[j] for j in range(n) if c[i][j] < D[i][k]) >= 1
            )
    
    start_time = time.time()
    m.optimize()
    solve_time = time.time() - start_time
    
    # Check integrality
    fractional_vars = 0
    max_fractionality = 0.0
    
    if m.status == GRB.OPTIMAL:
        # Check y variables
        for j in range(n):
            val = y[j].x
            frac = min(val, 1-val)
            if frac > 1e-6:
                fractional_vars += 1
                max_fractionality = max(max_fractionality, frac)
        
        # Check z variables
        for i in range(n):
            for k in range(1, G[i]):
                if (i, k) in z:
                    val = z[i, k].x
                    frac = min(val, 1-val)
                    if frac > 1e-6:
                        fractional_vars += 1
                        max_fractionality = max(max_fractionality, frac)
    
    stats = {
        'objective': m.objVal if m.status == GRB.OPTIMAL else None,
        'solve_time': solve_time,
        'status': m.status,
        'fractional_vars': fractional_vars,
        'max_fractionality': max_fractionality,
        'num_vars': m.NumVars,
        'num_constrs': m.NumConstrs
    }
    
    return stats, m

def compare_formulations_comprehensive(n, p, c, verbose=False):
    """
    Comprehensive formulation comparison with statistical rigor
    """
    print(f"\\n{'='*90}")
    print(f"COMPREHENSIVE P-MEDIAN FORMULATION ANALYSIS (n={n}, p={p})")
    print(f"{'='*90}")
    
    # Solve all versions
    print("\\nSolving all formulation variants...")
    
    # Solve with enhanced functions that return model objects too
    orig_ip_stats, orig_ip_model = solve_p_median_IP_enhanced(n, p, c, verbose)
    orig_lp_stats, orig_lp_model = solve_p_median_LP_enhanced(n, p, c, verbose)
    cover_ip_stats, cover_ip_model = solve_p_median_CR_IP_enhanced(n, p, c, verbose)
    cover_lp_stats, cover_lp_model = solve_p_median_CR_LP_enhanced(n, p, c, verbose)
    
    # Validate solutions
    if (orig_ip_stats['objective'] is None or cover_ip_stats['objective'] is None or
        orig_lp_stats['objective'] is None or cover_lp_stats['objective'] is None):
        print("ERROR: Could not solve all problem instances")
        return None
    
    # Check consistency
    ip_diff = abs(orig_ip_stats['objective'] - cover_ip_stats['objective'])
    if ip_diff > 1e-6:
        print(f"WARNING: Different IP optimal values! Difference: {ip_diff:.8f}")
        print(f"Original: {orig_ip_stats['objective']:.8f}")
        print(f"Cover:    {cover_ip_stats['objective']:.8f}")
    
    # Calculate integrality gaps
    orig_gap = ((orig_ip_stats['objective'] - orig_lp_stats['objective']) / 
                orig_ip_stats['objective']) * 100
    cover_gap = ((cover_ip_stats['objective'] - cover_lp_stats['objective']) / 
                 cover_ip_stats['objective']) * 100
    
    # Determine which LP relaxation is tighter (lower bound closer to optimum)
    if abs(orig_gap - cover_gap) < 1e-3:
        tightness_winner = "TIE"
        gap_improvement = 0
    elif orig_gap < cover_gap:
        tightness_winner = "ORIGINAL"
        gap_improvement = cover_gap - orig_gap
    else:
        tightness_winner = "COVER"
        gap_improvement = orig_gap - cover_gap
    
    # Comprehensive results table
    print(f"\\n{'='*130}")
    print(f"{'Metric':<20} {'Original IP':<12} {'Original LP':<12} {'Cover IP':<12} {'Cover LP':<12} {'Analysis':<30}")
    print(f"{'='*130}")
    print(f"{'Objective Value':<20} {orig_ip_stats['objective']:<12.6f} {orig_lp_stats['objective']:<12.6f} "
          f"{cover_ip_stats['objective']:<12.6f} {cover_lp_stats['objective']:<12.6f} {'Both IP same: ' + str(ip_diff < 1e-6):<30}")
    print(f"{'Integrality Gap %':<20} {'N/A':<12} {orig_gap:<12.3f} {'N/A':<12} {cover_gap:<12.3f} {tightness_winner + ' is tighter':<30}")
    print(f"{'Solve Time (s)':<20} {orig_ip_stats['solve_time']:<12.4f} {orig_lp_stats['solve_time']:<12.4f} "
          f"{cover_ip_stats['solve_time']:<12.4f} {cover_lp_stats['solve_time']:<12.4f} {'Time comparison':<30}")
    print(f"{'Variables':<20} {orig_ip_stats['num_vars']:<12} {orig_lp_stats['num_vars']:<12} "
          f"{cover_ip_stats['num_vars']:<12} {cover_lp_stats['num_vars']:<12} {'Model size':<30}")
    print(f"{'Constraints':<20} {orig_ip_stats['num_constrs']:<12} {orig_lp_stats['num_constrs']:<12} "
          f"{cover_ip_stats['num_constrs']:<12} {cover_lp_stats['num_constrs']:<12} {'Model complexity':<30}")
    print(f"{'Fractional Vars':<20} {'N/A':<12} {orig_lp_stats['fractional_vars']:<12} "
          f"{'N/A':<12} {cover_lp_stats['fractional_vars']:<12} {'LP integrality':<30}")
    print(f"{'B&B Nodes':<20} {orig_ip_stats['nodes_explored']:<12} {'N/A':<12} "
          f"{cover_ip_stats['nodes_explored']:<12} {'N/A':<12} {'Search effort':<30}")
    
    # Detailed tightness analysis
    print(f"\\n{'='*80}")
    print("LP RELAXATION TIGHTNESS ANALYSIS")
    print(f"{'='*80}")
    
    print(f"Optimal IP Value:           {orig_ip_stats['objective']:.8f}")
    print(f"Original LP Lower Bound:    {orig_lp_stats['objective']:.8f} (gap: {orig_gap:.4f}%)")
    print(f"Cover-based LP Lower Bound: {cover_lp_stats['objective']:.8f} (gap: {cover_gap:.4f}%)")
    
    if tightness_winner == "ORIGINAL":
        print(f"\\n🏆 ORIGINAL formulation has the TIGHTER LP relaxation")
        print(f"   Advantage: {gap_improvement:.4f} percentage points")
        print(f"   This means better bounds for branch-and-bound algorithms")
    elif tightness_winner == "COVER":
        print(f"\\n🏆 COVER-BASED formulation has the TIGHTER LP relaxation")  
        print(f"   Advantage: {gap_improvement:.4f} percentage points")
        print(f"   This means better bounds for branch-and-bound algorithms")
    else:
        print(f"\\n🟰 Both formulations have equivalent LP relaxation tightness")
        print(f"   Gap difference is negligible ({abs(orig_gap - cover_gap):.6f}%)")
    
    # Solution characteristics
    print(f"\\nLP Solution Fractional Variables:")
    print(f"• Original LP: {orig_lp_stats['fractional_vars']} fractional variables")
    print(f"  Max fractionality: {orig_lp_stats['max_fractionality']:.6f}")
    print(f"• Cover LP: {cover_lp_stats['fractional_vars']} fractional variables")  
    print(f"  Max fractionality: {cover_lp_stats['max_fractionality']:.6f}")
    
    # Performance comparison
    orig_faster = orig_ip_stats['solve_time'] < cover_ip_stats['solve_time']
    time_ratio = (cover_ip_stats['solve_time'] / orig_ip_stats['solve_time'] 
                  if orig_faster else orig_ip_stats['solve_time'] / cover_ip_stats['solve_time'])
    
    print(f"\\nComputational Performance:")
    if orig_faster:
        print(f"• Original IP is {time_ratio:.2f}x faster ({orig_ip_stats['solve_time']:.4f}s vs {cover_ip_stats['solve_time']:.4f}s)")
    else:
        print(f"• Cover IP is {time_ratio:.2f}x faster ({cover_ip_stats['solve_time']:.4f}s vs {orig_ip_stats['solve_time']:.4f}s)")
    
    print(f"• Original required {orig_ip_stats['nodes_explored']} B&B nodes")
    print(f"• Cover required {cover_ip_stats['nodes_explored']} B&B nodes")
    
    # Final recommendation
    print(f"\\n{'='*80}")
    print("RECOMMENDATION FOR FORMULATION CHOICE")
    print(f"{'='*80}")
    
    if tightness_winner == "ORIGINAL":
        print("✅ RECOMMEND: Original Formulation")
        print("   Reasons:")
        print(f"   • Tighter LP relaxation ({gap_improvement:.4f}% better gap)")
        print("   • Provides better bounds for exact algorithms")
        print("   • Simpler model structure")
        print("   • Will likely scale better to larger instances")
    elif tightness_winner == "COVER":
        print("✅ RECOMMEND: Cover-based Formulation") 
        print("   Reasons:")
        print(f"   • Tighter LP relaxation ({gap_improvement:.4f}% better gap)")
        print("   • Provides better bounds for exact algorithms")
        print("   • Better theoretical foundation for approximation algorithms")
        print("   • More sophisticated problem representation")
    else:
        if orig_faster:
            print("✅ RECOMMEND: Original Formulation")
            print("   Reasons:")
            print("   • Equivalent LP relaxation quality")
            print(f"   • Faster solve times ({time_ratio:.2f}x speedup)")
            print("   • Simpler and more intuitive formulation")
        else:
            print("✅ BOTH formulations are viable options")
            print("   • Equivalent LP relaxation quality")
            print("   • Choice depends on specific problem characteristics")
    
    return {
        'tighter_formulation': tightness_winner,
        'gap_improvement': gap_improvement,
        'original_gap': orig_gap,
        'cover_gap': cover_gap,
        'time_advantage': time_ratio,
        'faster_formulation': 'ORIGINAL' if orig_faster else 'COVER',
        'model_size_ratio': cover_ip_stats['num_vars'] / orig_ip_stats['num_vars']
    }

def run_comprehensive_study():
    """
    Run study across multiple problem instances
    """
    print("\\n" + "="*100)
    print("COMPREHENSIVE FORMULATION STUDY - MULTIPLE INSTANCES")
    print("="*100)
    
    # Test cases of varying sizes
    test_cases = [
        (5, 2, "Small"),
        (8, 3, "Small-Med"), 
        (10, 3, "Medium"),
        (12, 4, "Medium"),
        (15, 5, "Large")
    ]
    
    results = []
    
    for n, p, size_label in test_cases:
        print(f"\\n{'='*60}")
        print(f"TEST CASE: {size_label} (n={n}, p={p})")
        print(f"{'='*60}")
        
        # Generate structured test instance
        random.seed(42 + n)  # Reproducible but varied
        c = [[random.randint(1, 15) if i != j else 0 for j in range(n)] for i in range(n)]
        
        # Make symmetric for realism
        for i in range(n):
            for j in range(i+1, n):
                c[j][i] = c[i][j]
        
        result = compare_formulations_comprehensive(n, p, c, verbose=False)
        if result:
            result['n'] = n
            result['p'] = p 
            result['size_label'] = size_label
            results.append(result)
    
    # Overall analysis
    print(f"\\n{'='*100}")
    print("STUDY SUMMARY - WHICH FORMULATION IS BETTER?")
    print(f"{'='*100}")
    
    original_wins = sum(1 for r in results if r['tighter_formulation'] == 'ORIGINAL')
    cover_wins = sum(1 for r in results if r['tighter_formulation'] == 'COVER')  
    ties = sum(1 for r in results if r['tighter_formulation'] == 'TIE')
    
    print(f"LP Relaxation Tightness Results:")
    print(f"• Original formulation wins:  {original_wins}/{len(results)} instances")
    print(f"• Cover-based formulation wins: {cover_wins}/{len(results)} instances")
    print(f"• Equivalent performance:     {ties}/{len(results)} instances")
    
    if original_wins > cover_wins:
        print(f"\\n🏆 OVERALL WINNER: ORIGINAL FORMULATION")
        print(f"   Consistently provides tighter LP bounds")
    elif cover_wins > original_wins:
        print(f"\\n🏆 OVERALL WINNER: COVER-BASED FORMULATION") 
        print(f"   Consistently provides tighter LP bounds")
    else:
        print(f"\\n🟰 NO CLEAR WINNER - Instance dependent")
    
    # Calculate average improvements
    valid_results = [r for r in results if r['gap_improvement'] > 0]
    if valid_results:
        avg_improvement = sum(r['gap_improvement'] for r in valid_results) / len(valid_results)
        print(f"\\nAverage gap improvement: {avg_improvement:.4f} percentage points")
    
    return results

# Example usage - replace your existing code with this
if __name__ == "__main__":
    # Test with your original example
    n = 5
    p = 2
    c = [
        [0, 2, 3, 1, 4],
        [2, 0, 2, 3, 5], 
        [3, 2, 0, 4, 2],
        [1, 3, 4, 0, 3],
        [4, 5, 2, 3, 0]
    ]
    
    # Single instance analysis
    print("ANALYZING YOUR SPECIFIC EXAMPLE:")
    result = compare_formulations_comprehensive(n, p, c, verbose=True)
    
    # Comprehensive study
    print("\\n\\nRUNNING COMPREHENSIVE STUDY...")
    study_results = run_comprehensive_study()