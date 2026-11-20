"""test_presolve_box_exact.py — validation de presolve_box_exact.box_min_certified.
Référence EXACTE : énumération de tous les ensembles actifs (3^n, n <= 8) ; v* = min sur les candidats réalisables.
Vérifie : lower <= v* (+1e-9 rel), primal >= v*, primal - lower petit, statut optimal ; compare à l'ancien presolve."""
import itertools, sys
import numpy as np
from presolve_box_exact import box_min_certified, presolve_old

def exact_enum(S, Q, U):
    n = len(S); best = np.inf
    for st in itertools.product((0, 1, 2), repeat=n):
        st = np.array(st); fr = np.where(st == 0)[0]; fx = np.where(st != 0)[0]
        qx = np.where(st == 2, U, 0.0)
        R = Q - qx.sum()
        if len(fr) == 0:
            if abs(R) > 1e-9: continue
            q = qx
        else:
            M = S[np.ix_(fr, fr)]; r = S[np.ix_(fr, fx)] @ qx[fx] if len(fx) else np.zeros(len(fr))
            e = np.linalg.solve(M, np.ones(len(fr))); b = np.linalg.solve(M, r)
            mu = 2 * (R + b.sum()) / e.sum(); q = qx.copy(); q[fr] = mu / 2 * e - b
        if (q < -1e-9).any() or (q > U + 1e-9).any(): continue
        best = min(best, q @ S @ q)
    return best

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worst_low = worst_gap = worst_pr = 0.0; n_old_over = 0; n_cases = 0; bad = 0
    for t in range(3000):
        n = int(rng.integers(2, 9)); U = int(rng.integers(1, 10)); Q = int(rng.integers(1, n * U + 1))
        kind = t % 3
        A = rng.normal(size=(n, n if kind else 2))
        S = A @ A.T + np.diag(rng.uniform(1e-3 if kind == 2 else 0.1, 1.0, n))
        if kind == 1: S *= rng.uniform(1, 1e4)
        v = exact_enum(S, Q, U)
        lo, pr, q, info = box_min_certified(S, Q, U)
        sc = max(1.0, abs(v))
        worst_low = max(worst_low, (lo - v) / sc)       # > 0 = surestimation (interdit)
        worst_gap = max(worst_gap, (pr - lo) / sc); worst_pr = max(worst_pr, (pr - v) / sc)
        if lo > v + 1e-9 * sc or info["status"] != "optimal" or abs(q.sum() - Q) > 1e-7 or (q < -1e-9).any() or (q > U + 1e-9).any():
            bad += 1
        n_old_over += presolve_old(S, Q, U) > v * (1 + 1e-9) + 1e-9
        n_cases += 1
    print(f"{n_cases} cas aléatoires (n 2-8) : échecs={bad}")
    print(f"  max (lower - v*)/v* = {worst_low:.2e}   (doit être <= ~1e-12)")
    print(f"  max (primal - lower)/v* = {worst_gap:.2e} (écart de certificat)")
    print(f"  max (primal - v*)/v* = {worst_pr:.2e} (optimalité du point primal)")
    print(f"  ancien presolve : surestime v* dans {n_old_over}/{n_cases} cas")
    print("STATUT", "OK" if bad == 0 and worst_low <= 1e-9 and worst_gap <= 1e-9 and worst_pr <= 1e-9 else "ECHEC")
