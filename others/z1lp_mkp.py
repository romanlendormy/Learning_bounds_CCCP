"""z1LP bound for the Chance-Constrained Multidimensional Knapsack Problem.

Implements formulation (20) from Ahmed et al. (2017), which is the LP relaxation
of the primal reformulation corresponding to the v1^LD Lagrangian dual.

By Theorem 6 of the paper, v1^LD = z1^LP when P_S = conv(S) and P^i = conv(X^i),
which holds here since S = [0,1]^n and X^i = {x : A^i x <= b^i, 0 <= x <= 1}.

Also includes:
  - OR-Library and Song-format instance loaders
  - Song-Luedtke big-M strengthening (compute_strengthened_M)
  - Standard and strengthened LP relaxations of the big-M MIP (for comparison)

Dependencies: numpy, scipy, pulp
"""

import os
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix


# ---------------------------------------------------------------------------
# Instance representation
# ---------------------------------------------------------------------------

class MKPInst:
    """Chance-constrained MKP instance.

    Attributes
    ----------
    episode : int
    n       : int   number of items
    m       : int   number of constraints
    N       : int   number of scenarios
    epsilon : float violation probability (1-epsilon = required reliability)
    sigma   : float noise parameter used during generation
    c       : (n,)        deterministic profit vector
    p       : (N,)        scenario probabilities (uniform = 1/N)
    A       : (N, m, n)   constraint matrix per scenario
    b       : (N, m)      right-hand side per scenario
    """
    __slots__ = ["episode", "n", "m", "N", "epsilon", "sigma", "c", "p", "A", "b"]


# ---------------------------------------------------------------------------
# Instance loaders
# ---------------------------------------------------------------------------

def parse_orlib(filepath):
    """Parse an OR-Library multi-dimensional knapsack file (Beasley format)."""
    def tokens(path):
        with open(path) as f:
            for line in f:
                for t in line.split():
                    yield t

    gen = tokens(filepath)
    K   = int(next(gen))
    out = []
    basename = os.path.splitext(os.path.basename(filepath))[0]
    for k in range(K):
        n   = int(next(gen)); m = int(next(gen)); opt = float(next(gen))
        profits    = [float(next(gen)) for _ in range(n)]
        weights    = [[float(next(gen)) for _ in range(n)] for _ in range(m)]
        capacities = [float(next(gen)) for _ in range(m)]
        out.append(dict(name=f"{basename}_{k+1}", n=n, m=m, opt=opt,
                        profits=profits, weights=weights, capacities=capacities))
    return out


def load_mkp_instances(filepath, epsilon=0.1, max_episodes=-1):
    """Load CCMKP instances from CSV (Song et al. format).

    Expected CSV row format:
        [ep, scen, m, n, sigma, A_flat(m*n), b_flat(m), c_flat(n)]

    c is deterministic (same across all scenarios of one episode).
    """
    raw = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = list(map(float, line.split(",")))
            ep_id = int(vals[0])
            if max_episodes > 0 and ep_id not in raw and len(raw) >= max_episodes:
                continue
            raw[ep_id].append(vals)

    instances = []
    for ep_id in sorted(raw.keys()):
        rows  = raw[ep_id]
        first = rows[0]
        m = int(first[2]); n = int(first[3]); sigma = first[4]
        N = len(rows)
        expected = 5 + m * n + m + n
        assert len(first) == expected, (
            f"ep {ep_id}: expected {expected} values, got {len(first)} "
            f"(m={m}, n={n})"
        )
        offset_A = 5
        offset_b = offset_A + m * n
        offset_c = offset_b + m
        c_vec = np.array(first[offset_c : offset_c + n])
        p_vec = np.full(N, 1.0 / N)
        A_arr = np.zeros((N, m, n))
        b_arr = np.zeros((N, m))
        for i, row in enumerate(rows):
            A_arr[i] = np.array(row[offset_A : offset_A + m * n]).reshape(m, n)
            b_arr[i] = np.array(row[offset_b : offset_b + m])

        inst = MKPInst()
        inst.episode = ep_id; inst.n = n; inst.m = m; inst.N = N
        inst.epsilon = epsilon; inst.sigma = sigma
        inst.c = c_vec; inst.p = p_vec; inst.A = A_arr; inst.b = b_arr
        instances.append(inst)
    return instances


# ---------------------------------------------------------------------------
# Song-Luedtke big-M strengthening
# ---------------------------------------------------------------------------

def compute_strengthened_M(inst, n_iter=1, verbose=False):
    """Compute per-(scenario, constraint) big-M values via Song-Luedtke.

    For each scenario k and constraint row l, the strengthened M[k,l] is the
    q-th LARGEST of {eta_{i,k,l}}_{i=1..N}, where:
        eta_{i,k,l} = max { A[k,l]^T x - b[k,l] : A[i] x <= b[i], 0 <= x <= 1 }
    and q = ceil((1-eps)*N).

    Parameters
    ----------
    inst   : MKPInst
    n_iter : int   number of strengthening passes (1 is usually sufficient)
    verbose: bool

    Returns
    -------
    M : (N, m) ndarray
    """
    n, m, N = inst.n, inst.m, inst.N
    A, b, eps = inst.A, inst.b, inst.epsilon
    q = int(np.ceil((1.0 - eps) * N))

    M = np.maximum(0.0, np.sum(np.maximum(A, 0.0), axis=2) - b)  # naive init
    bounds_x = [(0.0, 1.0)] * n

    for it in range(n_iter):
        eta = np.zeros((N, m, N))

        for i in range(N):
            if it == 0:
                A_ub_i = A[i]
                b_ub_i = b[i]
            else:
                A_ub_i = np.vstack([A[i], A.reshape(N * m, n)])
                b_ub_i = np.concatenate([b[i], (b + M).reshape(N * m)])

            for k in range(N):
                for l in range(m):
                    res = linprog(
                        -A[k, l],
                        A_ub=A_ub_i, b_ub=b_ub_i,
                        bounds=bounds_x, method="highs",
                        options={"disp": False},
                    )
                    if res.status == 0:
                        eta[k, l, i] = max(0.0, float(-res.fun) - b[k, l])

        M_new = np.maximum(0.0, np.sort(eta, axis=2)[:, :, ::-1][:, :, q - 1])

        if verbose:
            print(f"  iter {it+1}/{n_iter}: mean M  {M.mean():.3f} -> {M_new.mean():.3f}  "
                  f"max M  {M.max():.1f} -> {M_new.max():.1f}")
        M = M_new

    return M


# ---------------------------------------------------------------------------
# z1LP — formulation (20) from Ahmed et al.
# ---------------------------------------------------------------------------

def solve_z1lp_mkp(inst, M=None):
    """Compute the z1_LP upper bound (formulation 20, Ahmed et al.).

    Variables: [x(n), z(N), u(N*n), w(N*n)]

        max  c^T x
        s.t.
            u[i] + w[i]  = x                (N*n equalities)
            A[i] u[i]   <= b[i] z[i]        (N*m inequalities)
            w[i] + z[i]  <= 1 elem-wise     (N*n inequalities)
            sum_i p[i] z[i] >= 1-eps        (1 inequality)
            0<=x<=1,  0<=z<=1,  u,w>=0

    Parameters
    ----------
    inst : MKPInst
    M    : (N, m) ndarray or None
        If provided (from compute_strengthened_M), adds valid cuts
        A[k,l]^T x <= b[k,l] + M[k,l] that tighten the LP bound.

    Returns
    -------
    (objective_value, linprog_status_code)
    """
    n, m, N = inst.n, inst.m, inst.N
    c, p, A, b, eps = inst.c, inst.p, inst.A, inst.b, inst.epsilon

    n_vars = n + N + 2 * N * n
    iu = lambda i: n + N + i * n           # start index of u[i]
    iw = lambda i: n + N + N * n + i * n  # start index of w[i]

    c_obj = np.zeros(n_vars)
    c_obj[:n] = -c

    # Equality: u[i] + w[i] - x = 0
    n_eq = N * n
    Aeq  = lil_matrix((n_eq, n_vars))
    for i in range(N):
        for k in range(n):
            r = i * n + k
            Aeq[r, iu(i) + k] =  1.0
            Aeq[r, iw(i) + k] =  1.0
            Aeq[r, k]          = -1.0

    # Inequalities
    n_cuts = N * m if M is not None else 0
    n_ub   = N * m + N * n + 1 + n_cuts
    Aub    = lil_matrix((n_ub, n_vars))
    bub    = np.zeros(n_ub)

    r = 0
    # A[i] u[i] <= b[i] z[i]
    for i in range(N):
        for j in range(m):
            for kv in range(n):
                Aub[r, iu(i) + kv] = A[i, j, kv]
            Aub[r, n + i] = -b[i, j]
            r += 1

    # w[i] + z[i] <= 1
    for i in range(N):
        for kv in range(n):
            Aub[r, iw(i) + kv] = 1.0
            Aub[r, n + i]       = 1.0
            bub[r] = 1.0
            r += 1

    # sum p[i] z[i] >= 1-eps  <==>  -sum p[i] z[i] <= -(1-eps)
    for i in range(N):
        Aub[r, n + i] = -p[i]
    bub[r] = -(1.0 - eps)
    r += 1

    # Optional strengthening cuts: A[k,l]^T x <= b[k,l] + M[k,l]
    if M is not None:
        for k in range(N):
            for l in range(m):
                for kv in range(n):
                    Aub[r, kv] = A[k, l, kv]
                bub[r] = b[k, l] + M[k, l]
                r += 1

    assert r == n_ub

    bounds = (
        [(0.0, 1.0)] * n
        + [(0.0, 1.0)] * N
        + [(0.0, None)] * (N * n)
        + [(0.0, None)] * (N * n)
    )

    res = linprog(
        c_obj,
        A_ub=csr_matrix(Aub), b_ub=bub,
        A_eq=csr_matrix(Aeq), b_eq=np.zeros(n_eq),
        bounds=bounds, method="highs",
        options={"disp": False},
    )

    if res.status != 0:
        return float("nan"), res.status
    return float(-res.fun), res.status


# ---------------------------------------------------------------------------
# LP relaxations of the big-M MIP (for comparison)
# ---------------------------------------------------------------------------

def solve_lp_ccmkp(inst):
    """LP relaxation of the big-M CCMKP (naive global big-M, via PuLP)."""
    import pulp
    n, m, N = inst.n, inst.m, inst.N
    c, p, A, b, eps = inst.c, inst.p, inst.A, inst.b, inst.epsilon
    M_val = float(np.max(A.sum(axis=2)))

    prob = pulp.LpProblem("CCMKP_LP", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{k}", lowBound=0, upBound=1) for k in range(n)]
    y = [pulp.LpVariable(f"y{i}", lowBound=0, upBound=1) for i in range(N)]

    prob += pulp.lpSum(c[k] * x[k] for k in range(n))
    prob += pulp.lpSum(p[i] * y[i] for i in range(N)) >= 1.0 - eps
    for i in range(N):
        for j in range(m):
            prob += (
                pulp.lpSum(A[i, j, k] * x[k] for k in range(n))
                <= float(b[i, j]) + M_val * (1 - y[i])
            )

    pulp.PULP_CBC_CMD(msg=0).solve(prob)
    obj = pulp.value(prob.objective)
    return float(obj) if obj is not None else float("nan")


def solve_lp_ccmkp_M(inst, M):
    """LP relaxation with per-(scenario, constraint) strengthened big-M."""
    import pulp
    n, m, N = inst.n, inst.m, inst.N
    c, p, A, b, eps = inst.c, inst.p, inst.A, inst.b, inst.epsilon

    prob = pulp.LpProblem("CCMKP_LP_Mstr", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{k}", lowBound=0, upBound=1) for k in range(n)]
    y = [pulp.LpVariable(f"y{i}", lowBound=0, upBound=1) for i in range(N)]

    prob += pulp.lpSum(c[k] * x[k] for k in range(n))
    prob += pulp.lpSum(p[i] * y[i] for i in range(N)) >= 1.0 - eps
    for i in range(N):
        for j in range(m):
            prob += (
                pulp.lpSum(A[i, j, k] * x[k] for k in range(n))
                <= float(b[i, j]) + float(M[i, j]) * (1 - y[i])
            )

    pulp.PULP_CBC_CMD(msg=0).solve(prob)
    obj = pulp.value(prob.objective)
    return float(obj) if obj is not None else float("nan")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python z1lp_mkp.py <path_to_csv_instances> [epsilon]")
        sys.exit(1)

    path    = sys.argv[1]
    epsilon = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    insts   = load_mkp_instances(path, epsilon=epsilon, max_episodes=1)
    inst    = insts[0]
    print(f"Episode {inst.episode}: n={inst.n}, m={inst.m}, N={inst.N}, eps={inst.epsilon}")

    t0 = time.perf_counter()
    val, status = solve_z1lp_mkp(inst)
    t1 = time.perf_counter()
    print(f"z1_LP (no M)  = {val:.4f}   status={status}   t={t1-t0:.3f}s")

    t0 = time.perf_counter()
    M_str = compute_strengthened_M(inst, n_iter=1, verbose=True)
    t_M = time.perf_counter() - t0
    print(f"M strengthening: {t_M:.1f}s")

    t0 = time.perf_counter()
    val_M, status_M = solve_z1lp_mkp(inst, M=M_str)
    t2 = time.perf_counter() - t0
    print(f"z1_LP (with M) = {val_M:.4f}   status={status_M}   t={t2:.3f}s")
    print(f"Improvement from M strengthening: {val - val_M:.4f}")
