# Mathematical Formulation of p-Hub-Arc Algorithms

This document gives the mathematical description of the **p-Hub-Arc** problem and of **Algorithm 1**, **Algorithm 2**, **Algorithm 3**, and **Phase 1 / Phase 2** as implemented in the codebase.

---

## 1. Problem and Notation

### 1.1 p-Hub-Arc Problem

- **Node set:** \( N = \{0, 1, \ldots, n-1\} \).
- **OD pairs:** \((i,j)\) with \(i, j \in N\), \(i \neq j\).
- **Flow / weight:** \(W_{ij} \geq 0\) for each OD pair \((i,j)\).
- **Distance matrix:** \(D_{ab} \geq 0\) for \(a, b \in N\).
- **Hub arcs:** \(H = \{(u,v) : u, v \in N,\, u \neq v\}\).
- **Parameter:** \(p\) = number of hub arcs to open.

**Arc cost for OD pair \((i,j)\) using hub arc \((u,v)\):**

\[
c_{ij,uv} = W_{ij} \cdot \bigl( D_{iu} + D_{uv} + D_{vj} \bigr).
\]

**Allocation cost for OD \((i,j)\) given a set of open arcs (indicator \(y\)):**

- If at least one open arc has cost \(c_{ij,uv}\), the cost for \((i,j)\) is the **minimum** of \(c_{ij,uv}\) over open arcs \((u,v)\).
- We want to choose exactly \(p\) hub arcs so that the **total allocation cost** over all OD pairs is minimized.

---

## 2. F3 Canonical Formulation (Background for Benders)

For each OD pair \((i,j)\), let the **distinct** arc costs be sorted:

\[
C_{ij}^1 < C_{ij}^2 < \cdots < C_{ij}^{K_{ij}}.
\]

- \(K_{ij}\) = number of distinct cost levels for \((i,j)\).
- \(L_{ij}^k\) = set of arcs \((u,v)\) with \(c_{ij,uv} = C_{ij}^k\).

**Variables:**

- \(y_{uv} \in \{0,1\}\) for each \((u,v) \in H\) (1 if arc is open).
- For each \((i,j)\) with \(K_{ij} > 1\): \(z_{ij}^k \geq 0\) for \(k = 1, \ldots, K_{ij}-1\).

**F3 formulation:**

\[
\min \quad \sum_{(i,j):\, i\neq j} \left[ C_{ij}^1 + \sum_{k=1}^{K_{ij}-1} \bigl( C_{ij}^{k+1} - C_{ij}^k \bigr) z_{ij}^k \right]
\]

subject to:

\[
\sum_{(u,v) \in H} y_{uv} = p,
\]

\[
z_{ij}^1 + \sum_{a \in L_{ij}^1} y_a \geq 1 \quad \forall (i,j):\, K_{ij} > 1,
\]

\[
z_{ij}^k + \sum_{a \in L_{ij}^k} y_a \geq z_{ij}^{k-1} \quad \forall (i,j):\, K_{ij} > 1,\; k = 2,\ldots,K_{ij}-1.
\]

(For \((i,j)\) with \(K_{ij} = 1\), the allocation cost is simply \(C_{ij}^1\) and is added to the objective as a constant.)

Benders decomposition uses the **same** cost levels \(C_{ij}^k\) and arc sets \(L_{ij}^k\).

---

## 3. Benders Reformulation (Master and Subproblem)

**Master variables:**

- \(y_{uv}\) for \((u,v) \in H\) (in Phase 1: continuous in \([0,1]\); in Phase 2: binary).
- \(\theta_{ij} \geq 0\) for each OD pair \((i,j)\) with \(K_{ij} > 1\).

**Master problem (relaxation):**

\[
\min \quad \sum_{(i,j) \in \mathcal{OD}} \theta_{ij}
\]

\[
\text{s.t.} \quad \sum_{(u,v) \in H} y_{uv} = p,
\]

\[
\theta_{ij} \geq \text{[Benders cuts]} \quad \forall (i,j) \in \mathcal{OD},
\]

where \(\mathcal{OD} = \{(i,j) : i \neq j,\; K_{ij} > 1\}\). For \((i,j)\) with \(K_{ij} = 1\), cost \(C_{ij}^1\) is added to the objective after the solve.

**Subproblem for a fixed \(\bar{y}\):** For each \((i,j) \in \mathcal{OD}\), the **primal** subproblem is the F3 constraints involving only \(z_{ij}^k\) and the fixed \(\bar{y}\):

\[
\min \quad C_{ij}^1 + \sum_{k=1}^{K_{ij}-1} \bigl( C_{ij}^{k+1} - C_{ij}^k \bigr) z_{ij}^k
\]

\[
\text{s.t.} \quad z_{ij}^1 + \sum_{a \in L_{ij}^1} \bar{y}_a \geq 1,
\]

\[
z_{ij}^k + \sum_{a \in L_{ij}^k} \bar{y}_a \geq z_{ij}^{k-1}, \quad k = 2,\ldots,K_{ij}-1,
\]

\[
z_{ij}^k \geq 0.
\]

Its **dual** (with dual variables \(\nu_k\) for \(k=1,\ldots,K_{ij}\)) is used in Algorithm 1.

---

## 4. Algorithm 1: Separation Algorithm (Dual-Based Benders Cuts)

**Purpose:** Given a master solution \((\bar{y}, \bar{\theta})\), solve the **dual subproblem** for each \((i,j) \in \mathcal{OD}\) and, if \(\theta_{ij}\) is below the subproblem value, add a Benders cut.

### 4.1 Dual Subproblem for OD Pair \((i,j)\)

**Dual variables:** \(\nu_k \geq 0\) for \(k = 1,\ldots,K_{ij}\).

**Dual problem (maximization):**

\[
\max \quad C_{ij}^1 + \nu_1 \left( 1 - \sum_{a \in L_{ij}^1} \bar{y}_a \right) - \sum_{k=2}^{K_{ij}} \nu_k \left( \sum_{a \in L_{ij}^k} \bar{y}_a \right)
\]

\[
\text{s.t.} \quad \nu_k - \nu_{k+1} \leq C_{ij}^{k} - C_{ij}^{k-1}, \quad k = 1,\ldots,K_{ij}-1,
\]

\[
\nu_k \geq 0, \quad k = 1,\ldots,K_{ij}.
\]

(Indices: \(C_{ij}^k\) in code is the \(k\)-th cost level, so the right-hand side is the increment \(C_{ij}^{k+1} - C_{ij}^k\) when written with 1-based \(k\).)

### 4.2 Benders Cut from Dual Solution

If the dual is solved to optimality and \(\nu^*\) is the optimal dual solution, the **Benders cut** for \((i,j)\) is:

\[
\theta_{ij} \geq C_{ij}^1 + \nu_1^* \left( 1 - \sum_{a \in L_{ij}^1} y_a \right) - \sum_{k=2}^{K_{ij}} \nu_k^* \left( \sum_{a \in L_{ij}^k} y_a \right).
\]

So the cut is linear in \(y\).

### 4.3 Separation (When to Add a Cut)

- Solve the dual for \((i,j)\); let \(V_{ij}(\bar{y})\) be its optimal value.
- **Upper bound contribution:** \(V_{ij}(\bar{y})\) is the allocation cost for \((i,j)\) at \(\bar{y}\); sum over \((i,j)\) gives an upper bound on total cost.
- **Cut violation:** If \(\bar{\theta}_{ij} < V_{ij}(\bar{y}) - \varepsilon\) (with small \(\varepsilon > 0\)), the current \(\bar{\theta}_{ij}\) underestimates the true cost; add the Benders cut above.

**Algorithm 1 (summary):**

1. For each \((i,j) \in \mathcal{OD}\) with \(K_{ij} > 1\):
   - Solve the dual subproblem with \(\bar{y}\).
   - Add \(V_{ij}(\bar{y})\) to the upper bound.
   - If \(\bar{\theta}_{ij} < V_{ij}(\bar{y}) - \varepsilon\), add the Benders cut derived from the optimal \(\nu^*\).
2. Return the list of cuts and the total upper bound.

---

## 5. Algorithm 2: Computing \(k_{ij}\) (for Polynomial Separation)

**Purpose:** Given \(\bar{y}\) and the list of arcs for \((i,j)\) **sorted by cost** \(c_{ij,uv}\) (non-decreasing), compute an index \(k_{ij}\) used in a polynomial separation scheme (e.g., to know how many “cost levels” are “covered” by the current \(\bar{y}\)).

**Inputs:**

- OD pair \((i,j)\).
- \(\bar{y}\): current master solution (arc values).
- **arcs_sorted:** list of arcs \((u,v)\) sorted by \(c_{ij,uv}\) (cheapest first).
- **cost_map:** \((u,v) \mapsto c_{ij,uv}\) for this \((i,j)\).
- \(p\): number of hub arcs to open.

**Computation:**

1. Initialize: \(k_{ij} = 0\), \(r = 0\), \(\text{val} = 0\), \(M = |\text{arcs\_sorted}|\).
2. While \(\text{val} < p - \varepsilon\) and \(r < M\):
   - If \(r+1 < M\) and \(c(\text{arcs\_sorted}[r]) < c(\text{arcs\_sorted}[r+1])\): increment \(k_{ij}\) (new cost level).
   - \(\text{val} \mathrel{+}= \bar{y}(\text{arcs\_sorted}[r])\).
   - \(r \mathrel{+}= 1\).
3. Return \(k_{ij}\).

**Interpretation:** We scan arcs in increasing cost order and accumulate \(\bar{y}\) until the cumulative sum reaches (at least) \(p\). Each time we move to a strictly more expensive arc we increment \(k_{ij}\). So \(k_{ij}\) is the index of the “cost level” at which the cumulative \(\bar{y}\) reaches \(p\), and can be used to build Benders cuts in a polynomial way (e.g., to identify which level’s cut to add).

**Mathematical view:** Let \(a_1, a_2, \ldots, a_M\) be the arcs sorted so that \(c_{ij,a_1} \le c_{ij,a_2} \le \cdots \le c_{ij,a_M}\). Define cumulative sums \(S_0 = 0\), \(S_r = \sum_{t=1}^{r} \bar{y}_{a_t}\). Then \(k_{ij}\) is the number of **strict** cost increases encountered before the smallest \(r\) with \(S_r \geq p\) (if any).

---

## 6. Algorithm 3: Phase 1 — LP Relaxation of the Master

**Purpose:** Solve the **LP relaxation** of the Benders master (i.e., \(y \in [0,1]\)) by iteratively adding Benders cuts from Algorithm 1, and produce a feasible integer solution by rounding plus an upper bound.

### 6.1 Master Problem in Phase 1

**Variables:**

- \(y_a \in [0,1]\) for each \(a \in H\),
- \(\theta_{ij} \geq 0\) for each \((i,j) \in \mathcal{OD}\).

**Objective and constraint:**

\[
\min \quad \sum_{(i,j) \in \mathcal{OD}} \theta_{ij}, \qquad \sum_{a \in H} y_a = p.
\]

Cuts are added dynamically: \(\theta_{ij} \geq C_{ij}^1 + \nu_1^*(1 - \sum_{a \in L_{ij}^1} y_a) - \sum_{k=2}^{K_{ij}} \nu_k^* (\sum_{a \in L_{ij}^k} y_a)\) for each generated \((\nu^*)\).

### 6.2 Rounding Heuristic

Given \(\bar{y}\) (fractional), build an integer \(\hat{y}\):

- Sort arcs by \(\bar{y}_a\) **descending**.
- Set \(\hat{y}_a = 1\) for the top \(p\) arcs, and \(\hat{y}_a = 0\) otherwise.

So \(\hat{y}\) is a feasible “select \(p\) arcs” solution.

### 6.3 Allocation Cost for Integer \(y\)

For a **binary** \(\hat{y}\) and OD \((i,j)\), the allocation cost is the cost of the **cheapest open arc** for \((i,j)\):

\[
\text{cost}_{ij}(\hat{y}) = \min\bigl\{ c_{ij,uv} : (u,v) \in H,\; \hat{y}_{uv} = 1 \bigr\}.
\]

In code, arcs for \((i,j)\) are sorted by cost; we take the first arc \(a\) with \(\hat{y}_a > 0.5\).

### 6.4 Phase 1 Iteration (Algorithm 3)

1. **Initialize:** Lower bound LB1, optional heuristic \((\hat{y}, \widehat{UB})\). Optional warm start for \(y\).
2. **Repeat** (until no violated cuts or max iterations):
   - Solve the current master LP (with existing Benders cuts).
   - Set \((\bar{y}, \bar{\theta})\) = current solution; update LB1 = current master objective.
   - Run **Algorithm 1** (separation) at \((\bar{y}, \bar{\theta})\):
     - Get Benders cuts and an upper bound UB (sum of dual objectives over \((i,j)\)).
   - If no cuts are violated: **break** (converged).
   - Add all violated cuts to the master.
   - If \(\bar{y}\) is fractional:
     - Apply **rounding heuristic** to get \(\hat{y}\).
     - Compute \(\widehat{UB}\) = total allocation cost for \(\hat{y}\) (plus fixed terms for \(K_{ij}=1\)).
     - If \(\widehat{UB}\) improves the best UB so far, store \(\hat{y}\) and \(\widehat{UB}\).
   - Increment iteration count.
3. **Final feasible solution:** Best \(\hat{y}\) from rounding (or round the final \(\bar{y}\) if no improvement). Final UB1 = allocation cost of that \(\hat{y}\) (plus \(C_{ij}^1\) for \(K_{ij}=1\)).

**Outputs:** LB1 (LP bound), \(y_1\) (feasible integer solution), UB1 (its total cost), plus the master model and variables for warm start in Phase 2.

---

## 7. Phase 1 and Phase 2 in `new_model_hub_arc.py`

The same **Benders decomposition** is run in two phases: first solve the LP relaxation (Phase 1), then solve the MIP with binary \(y\) and lazy Benders cuts (Phase 2).

### 7.1 Preprocessing (Before Phase 1)

From \(n, p, W, D\):

1. **\(H\):** all hub arcs \((u,v)\), \(u \neq v\).
2. **For each \((i,j)\), \(i \neq j\):**
   - \(c_{ij,uv} = W_{ij}(D_{iu} + D_{uv} + D_{vj})\) for all \((u,v) \in H\).
   - **Unique cost levels:** \(C_{ij}^1 < \cdots < C_{ij}^{K_{ij}}}\) and sets \(L_{ij}^k = \{ (u,v) : c_{ij,uv} = C_{ij}^k \}\).
   - **cost_map[(i,j)]:** \((u,v) \mapsto c_{ij,uv}\).
   - **arcs_sorted[(i,j)]:** arcs sorted by \(c_{ij,uv}\) (non-decreasing).
3. **\(\mathcal{OD}\):** \(\{(i,j) : i \neq j,\; K_{ij} > 1\}\).

This yields \(C, L, K,\) cost_map, arcs_sorted, \(\mathcal{OD}\).

### 7.2 Phase 1 (Same as Algorithm 3)

- **Master:** \(\min \sum_{(i,j) \in \mathcal{OD}} \theta_{ij}\) s.t. \(\sum_{a \in H} y_a = p\), \(y \in [0,1]\), plus Benders cuts.
- **Separation:** Algorithm 1 (dual subproblem per \((i,j)\), add cut if \(\bar{\theta}_{ij} < V_{ij}(\bar{y}) - \varepsilon\)).
- **Rounding:** Select \(p\) arcs with largest \(\bar{y}_a\); compute allocation cost for this \(\hat{y}\).
- **Output:** LB1, feasible integer solution \(y_1\), UB1. Optionally use \(y_1\) and UB1 as warm start / bound for Phase 2.

### 7.3 Phase 2: Branch-and-Benders-Cut

**Master problem (MIP):**

\[
\min \quad \sum_{(i,j) \in \mathcal{OD}} \theta_{ij}
\]

\[
\text{s.t.} \quad \sum_{(u,v) \in H} y_{uv} = p,
\]

\[
y_{uv} \in \{0,1\}, \quad (u,v) \in H,
\]

\[
\theta_{ij} \geq 0, \quad (i,j) \in \mathcal{OD},
\]

plus **lazy Benders cuts** \(\theta_{ij} \geq \text{cut}_{ij}(y)\) added only when needed.

**Lazy constraint callback (at MIP feasible solution):**

1. At a candidate solution \((\bar{y}, \bar{\theta})\) (e.g. at MIPSOL):
   - Get \(\bar{y}, \bar{\theta}\) from the callback.
2. Run **Algorithm 1** (separation) for \((\bar{y}, \bar{\theta})\):
   - For each \((i,j) \in \mathcal{OD}\), solve the dual subproblem; get \(V_{ij}(\bar{y})\) and optimal \(\nu^*\).
   - If \(\bar{\theta}_{ij} < V_{ij}(\bar{y}) - \varepsilon\), add the lazy constraint:
   \[
   \theta_{ij} \geq C_{ij}^1 + \nu_1^* \left( 1 - \sum_{a \in L_{ij}^1} y_a \right) - \sum_{k=2}^{K_{ij}} \nu_k^* \left( \sum_{a \in L_{ij}^k} y_a \right).
   \]
3. Solver continues with the new cuts.

**Warm start:** If Phase 1 was run, the Phase 2 variables \(y\) can be initialized with the Phase 1 integer solution \(y_1\) (e.g. Start = 1 for arcs with \(y_1(a)=1\)).

**Final objective:** After solving, add to the master objective the constant terms \(C_{ij}^1\) for all \((i,j)\) with \(K_{ij} = 1\) to get the full p-Hub-Arc cost.

---

## 8. Summary Table

| Component | Role |
|-----------|------|
| **Algorithm 1** | Dual-based separation: solve dual subproblem per \((i,j)\), add Benders cut if \(\bar{\theta}_{ij}\) is below dual optimal value. |
| **Algorithm 2** | Compute \(k_{ij}\): scan arcs for \((i,j)\) in cost order, accumulate \(\bar{y}\) until sum \(\geq p\); count cost-level steps for polynomial separation. |
| **Algorithm 3** | Phase 1: LP master (\(y \in [0,1]\)), add Benders cuts via Algorithm 1, rounding heuristic for UB and feasible \(y_1\). |
| **Phase 1 (new_model)** | Preprocessing + Algorithm 3 (LP relaxation + cuts + rounding). |
| **Phase 2 (new_model)** | MIP master (\(y \in \{0,1\}\)) with lazy Benders cuts from Algorithm 1 in a callback. |

---

## 9. Indices and Code Correspondence

- **Cost levels:** \(C_{ij}^k\) in math = `C[(i,j)][k-1]` in code (0-based list).
- **Arc sets:** \(L_{ij}^k\) = `L[(i,j)][k]`.
- **Number of levels:** \(K_{ij}\) = `K[(i,j)]`.
- **OD set:** \(\mathcal{OD}\) = `od_pairs` (only \((i,j)\) with \(K_{ij} > 1\)).
- **Dual constraints:** \(\nu_k - \nu_{k+1} \leq C_{ij}^{k+1} - C_{ij}^k\) implemented as `nu[k] - nu[k+1] <= C[(i,j)][k] - C[(i,j)][k-1]` for \(k=1,\ldots,K_{ij}-1\).

This matches the implementations in `algo1_hub_arc.py`, `algo2_hub_arc.py`, `algo3_hub_arc.py`, and `new_model_hub_arc.py`.
