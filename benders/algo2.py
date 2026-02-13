"""
Algorithm 2: Computing k_i (Section 3.2 Separation Problem)

Given a solution ȳ of the master problem, computes the index k_i for client i
used in the polynomial separation of Benders cuts.

Reference: Definition 1 and Algorithm 2 from Duran-Mateliana, Ales, Dillami paper.
"""


def compute_k_i(i, y_bar, S, dist, p, M):
    """
    Algorithm 2: Compute k_i for client i.

    Parameters
    ----------
    i : int
        Client index
    y_bar : list[float]
        Current master solution (facility opening variables), indexed by site j
    S : list[int]
        S[r] = index of (r+1)-th closest site to client i (0-based: S[0]=closest)
        Precomputed as sorted(range(M), key=lambda j: dist[i][j])
    dist : array-like
        distance_matrix[i][j] = distance from client i to site j
    p : int
        Number of facilities to open
    M : int
        Total number of sites

    Returns
    -------
    k_i : int
        Index used for Benders cut generation (equation 18, 20).
        Counts distinct distance levels crossed while accumulating y until val >= p.
    """
    k_i = 0
    r = 0
    val = 0.0

    # Paper Algorithm 2: while val < p and r < M
    while val < p - 1e-9 and r < M:
        if r + 1 < M:
            d_curr = dist[i][S[r]]
            d_next = dist[i][S[r + 1]]
            if d_curr < d_next:
                k_i += 1

        val += y_bar[S[r]]
        r += 1

    return k_i
