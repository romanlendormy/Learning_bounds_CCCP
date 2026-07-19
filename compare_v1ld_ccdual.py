"""
v1ld_compare.py — Comparaison sous-gradient (v1ld_solver.cpp) vs plan-coupant
exact (V1LD de CCdual/ccdual.cpp) sur une instance MKP chance-contrainte
n=20, m=5, N=100, epsilon=0.1 tirée du dataset de collecte
(data/train_base_s10/instances_n20_m5_N100.txt, inst_id=0).

Objectif : vérifier l'hypothèse que l'écart de qualité rapporté entre
l'implémentation de l'utilisateur et CCdual vient du budget d'optimisation
du dual lagrangien (sous-gradient à pas décroissant, T petit) et non d'une
différence de formulation de la borne v1LD elle-même.

Les deux algorithmes utilisent EXACTEMENT le même oracle ζ_i (0/1 knapsack
multi-dimensionnel exact, précalculé une fois par force brute puisque n=20
est petit) — donc toute différence de valeur finale ne peut venir que de la
qualité d'optimisation du dual, pas du solveur du sous-problème.

Algorithmes :
  A. Sous-gradient, budget "production" (T=200, pas 1/sqrt(t), cold start)
     — reproduit T_root de gecode_bab.hpp (root bound, avant tout warm-start).
  B. Sous-gradient, budget "convergé" (T grand + arrêt anticipé rel_tol serré)
     — même algorithme que A, juste plus d'itérations.
  C. Plan coupant exact de Kelley (reproduit ccdual.cpp::V1LD / lr1) — maître
     LP résolu exactement (OR-Tools GLOP) à chaque itération, aucune limite
     d'itérations (arrêt sur convergence à 1e-5 relatif, comme le code C++).

Dépendances : numpy, ortools (déjà dans $HOME/envs/learning_bounds).

Résultats obtenus le 2026-07-09 (inst_id=0, opt=4889.856) :
  A (T=200)            v1LD*=6102.44   (29s)     — gap vs opt = 24.8%
  B (T=3000)           v1LD*=6036.18   (446s)    — gap vs opt = 23.4% (pas d'early-stop)
  C_best_UB (Kelley)   v1LD*=6759.99   (181s)    — jamais amélioré vs lambda=0 (203 iters)
  C_LB_certif (Kelley) theta=4548.33   (181s)    — encore loin de best_UB, non convergé

Interprétation : le sous-gradient de production laisse un vrai gap (~25%), et
même 15x plus d'itérations ne le referme presque pas (pas ~1pp) — le pas 1/sqrt(t)
sans terme de Polyak est en cause. Mais la reproduction fidèle du plan coupant de
Kelley de CCdual (ccdual.cpp::V1LD) ne s'en sort pas mieux à cette échelle
(n=20, N=100 -> 2000 variables duales lambda) : en 203 itérations elle n'a
jamais dépassé sa propre évaluation initiale à lambda=0. Un plan coupant "nu"
(non stabilisé par bundle/proximal) est connu pour avoir besoin d'à peu près
autant de coupes que de dimensions avant de se stabiliser — probablement pas
praticable ici sans stabilisation. Donc l'écart rapporté par l'auteur n'est
probablement pas qu'une question de "combien d'itérations" : soit son échelle
de benchmark était plus petite, soit CPLEX + ses réglages font mieux qu'un
GLOP naïf. À creuser avec une vraie instance Song et al. (data/knapsack/
datasets-song/ccmk-song-mknap1_4_m5-S100.txt) pour une comparaison directement
face à la table de résultats du papier.
"""

from __future__ import annotations

import sys
import time
import numpy as np
from ortools.linear_solver import pywraplp

INSTANCE_FILE = "/lustre09/project/6062881/romanlen/Learning_bounds_CCCP/data/train_base_s10/instances_n20_m5_N100.txt"
INST_ID = 0


# ─────────────────────────────────────────────────────────────────────────────
# Chargement de l'instance (même format que gnn/train_gnn_v1ld.py:load_instances)
# ─────────────────────────────────────────────────────────────────────────────

def load_instance(path: str, inst_id: int):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(",")]
            if int(vals[0]) != inst_id:
                continue
            m = int(vals[1]); n = int(vals[2]); N = int(vals[3])
            epsilon = vals[4]; opt = vals[5]
            c = np.array(vals[6:6 + n], dtype=np.float64)
            off = 6 + n
            b = np.array(vals[off:off + N * m], dtype=np.float64).reshape(N, m)
            off += N * m
            A = np.array(vals[off:off + N * m * n], dtype=np.float64).reshape(N, m, n)
            return dict(n=n, m=m, N=N, epsilon=epsilon, opt=opt, c=c, A=A, b=b)
    raise ValueError(f"inst_id={inst_id} introuvable dans {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Oracle ζ_i exact — précalcul brute-force des vecteurs binaires faisables
# (n=20 → 2^20 ≈ 1.05M combinaisons, calculé une seule fois, réutilisé par
# les 3 algorithmes). θ_i reste trivial (base non contrainte, cf. mkp_problem.cpp).
# ─────────────────────────────────────────────────────────────────────────────

class ExactOracle:
    def __init__(self, inst: dict):
        self.n = inst["n"]; self.m = inst["m"]; self.N = inst["N"]
        self.c = inst["c"]; self.A = inst["A"]; self.b = inst["b"]
        self.epsilon = inst["epsilon"]
        self.p = np.full(self.N, 1.0 / self.N)

        n = self.n
        t0 = time.time()
        # Toutes les combinaisons binaires (2^n, n) — bit k = item k sélectionné
        idx = np.arange(2 ** n, dtype=np.uint32)
        X = ((idx[:, None] >> np.arange(n, dtype=np.uint32)[None, :]) & 1).astype(np.float32)
        self.feas_X = []  # feas_X[i] : (F_i, n) sous-ensemble faisable pour le scénario i
        for i in range(self.N):
            W = X @ self.A[i].T.astype(np.float32)          # (2^n, m)
            feas = np.all(W <= self.b[i].astype(np.float32) + 1e-6, axis=1)
            self.feas_X.append(X[feas])
        sizes = [f.shape[0] for f in self.feas_X]
        print(f"[oracle] precompute {2**n} combinaisons x {self.N} scenarios : "
              f"{time.time()-t0:.1f}s | |feasible| min/mean/max = "
              f"{min(sizes)}/{np.mean(sizes):.0f}/{max(sizes)}", file=sys.stderr)

    def theta(self, cost: np.ndarray):
        x = (cost > 0.0).astype(np.float64)
        return float(cost @ x), x

    def zeta(self, i: int, cost: np.ndarray):
        Xi = self.feas_X[i]
        scores = Xi @ cost.astype(np.float32)
        j = int(np.argmax(scores))
        x = Xi[j].astype(np.float64)
        return float(cost @ x), x

    def zeta_exact_check(self, i: int, cost: np.ndarray):
        """Vérif indépendante via CP-SAT (utilisé une fois pour valider l'oracle)."""
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{k}") for k in range(self.n)]
        SCALE = 1000
        for j in range(self.m):
            w = [int(round(self.A[i][j][k] * SCALE)) for k in range(self.n)]
            cap = int(np.floor(self.b[i][j] * SCALE))
            model.Add(sum(w[k] * x[k] for k in range(self.n)) <= cap)
        profits = [int(round(cost[k] * SCALE)) for k in range(self.n)]
        model.Maximize(sum(profits[k] * x[k] for k in range(self.n)))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        xval = np.array([solver.Value(x[k]) for k in range(self.n)], dtype=np.float64)
        return float(cost @ xval), xval


# ─────────────────────────────────────────────────────────────────────────────
# A/B — Sous-gradient (reproduit fidèlement solver_generic/v1ld_solver.cpp)
# ─────────────────────────────────────────────────────────────────────────────

def knapsack_min(delta: np.ndarray, p: np.ndarray, eps: float):
    order = np.argsort(delta, kind="stable")
    z = np.zeros_like(delta)
    cumsum = 0.0
    for idx in order:
        z[idx] = 1.0
        cumsum += p[idx]
        if cumsum >= 1.0 - eps - 1e-9:
            break
    return z


def solve_v1ld_subgradient(oracle: ExactOracle, T: int, rel_tol: float = -1.0,
                            tol_window: int = 500, abs_tol: float = -1.0,
                            abs_patience: int = 50, lambda_init=None, label=""):
    N, n, p, c, eps = oracle.N, oracle.n, oracle.p, oracle.c, oracle.epsilon
    lam = np.zeros((N, n)) if lambda_init is None else lambda_init.copy()
    best_upper = 1e18
    best_at_tol_check = 1e18
    best_at_patience_start = 1e18
    stale_iters = 0
    convergence = []
    t0 = time.time()
    iters_run = 0

    for t in range(T):
        Lambda_sum = lam.sum(axis=0)
        theta_vals = np.zeros(N); delta_vals = np.zeros(N)
        x_theta = np.zeros((N, n)); x_zeta = np.zeros((N, n))
        for i in range(N):
            cost_i = c + lam[i] / p[i] - Lambda_sum
            vt, xt = oracle.theta(cost_i)
            vz, xz = oracle.zeta(i, cost_i)
            theta_vals[i] = vt; x_theta[i] = xt
            delta_vals[i] = max(0.0, vt - vz); x_zeta[i] = xz

        z = knapsack_min(delta_vals, p, eps)
        knapsack_val = float(np.sum(p * z * delta_vals))
        L1 = float(np.sum(p * theta_vals)) - knapsack_val
        iters_run = t + 1
        if L1 < best_upper:
            best_upper = L1
        convergence.append(L1)

        x_i_used = np.where(z[:, None] > 0.5, x_zeta, x_theta)
        x_bar = np.sum(p[:, None] * x_i_used, axis=0)

        if abs_tol > 0 and abs_patience > 0:
            if best_at_patience_start - best_upper >= abs_tol:
                best_at_patience_start = best_upper
                stale_iters = 0
            else:
                stale_iters += 1
                if stale_iters >= abs_patience:
                    break
        if rel_tol > 0 and tol_window > 0 and t > 0 and t % tol_window == 0:
            impr = best_at_tol_check - best_upper
            if impr / max(abs(best_upper), 1e-10) < rel_tol:
                break
            best_at_tol_check = best_upper

        step = 1.0 / np.sqrt(t + 1)
        lam += step * (x_bar[None, :] - x_i_used)

    dt = time.time() - t0
    print(f"[subgradient {label}] T_run={iters_run}/{T}  v1LD*={best_upper:.4f}  "
          f"temps={dt:.1f}s", file=sys.stderr)
    return dict(value=best_upper, iters=iters_run, time_s=dt, convergence=convergence)


# ─────────────────────────────────────────────────────────────────────────────
# C — Plan coupant exact de Kelley (reproduit ccdual.cpp :: V1LD / lr1)
# ─────────────────────────────────────────────────────────────────────────────

def lr1_ccdual(oracle: ExactOracle, lam: np.ndarray):
    """
    Reproduit lr1() de ccdual.cpp : p_i = 1/N uniforme, coef = c/N + lam[i],
    sélection gloutonne (coverage 1-epsilon) des scénarios les moins coûteux
    à satisfaire pleinement (ceux-là utilisent zeta ; les autres, theta).
    Retourne (result, xvals) où xvals[i] est le sous-gradient (argmax) utilisé.
    """
    N, n, eps = oracle.N, oracle.n, oracle.epsilon
    c_over_N = oracle.c / N
    u = np.zeros(N); v = np.zeros(N)
    x_u = np.zeros((N, n)); x_v = np.zeros((N, n))
    for i in range(N):
        coef = c_over_N + lam[i]
        u[i], x_u[i] = oracle.theta(coef)
        v[i], x_v[i] = oracle.zeta(i, coef)

    delta = u - v  # >= 0
    order = np.argsort(delta, kind="stable")  # ascendant : moins cher à satisfaire d'abord
    p_unif = np.full(N, 1.0 / N)
    z = np.zeros(N)
    cumsum = 0.0
    for idx in order:
        z[idx] = 1.0
        cumsum += p_unif[idx]
        if cumsum >= 1.0 - eps - 1e-9:
            break

    xvals = np.where(z[:, None] > 0.5, x_v, x_u)
    result = float(np.sum(np.where(z > 0.5, v, u)))
    return result, xvals


def solve_v1ld_kelley(oracle: ExactOracle, lam_bound: float = None,
                       max_outer_iters: int = 250, time_limit_s: float = None,
                       tol: float = 1e-5, log_every: int = 20, label=""):
    """
    Maître LP :  min theta  s.c.  Sigma_i lam[i,j] = 0  forall j,  + coupes
      theta >= result_k + sum_i x_i^k . (lam_i - lam_i^k)
    Variables lam[i,j] bornées dans [-lam_bound, lam_bound] (juste pour que le
    maître soit borné avant la 1ere coupe ; CCdual utilise +-50, ici on prend
    une boite a l'echelle de max|c| car l'echelle de couts de notre instance
    (~100-1000) est tres differente de leurs instances originales).

    Retourne à la fois :
      - theta_master : borne INFÉRIEURE (certificat, style maître de Kelley)
        du vrai optimum du dual v1LD*  (converge vers v1LD* par en dessous).
      - best_result  : le meilleur (min) f(lambda_k) observé sur tous les
        lambda_k visités — c'est une borne SUPÉRIEURE valide à chaque
        itération (dualité faible), directement comparable à best_upper
        du sous-gradient.
    """
    N, n = oracle.N, oracle.n
    if lam_bound is None:
        lam_bound = float(np.max(np.abs(oracle.c)))
    solver = pywraplp.Solver.CreateSolver("GLOP")
    theta_var = solver.NumVar(-solver.infinity(), solver.infinity(), "theta")
    lam_vars = [[solver.NumVar(-lam_bound, lam_bound, f"lam_{i}_{j}")
                 for j in range(n)] for i in range(N)]
    solver.Minimize(theta_var)
    for j in range(n):
        ct = solver.Constraint(0.0, 0.0)
        for i in range(N):
            ct.SetCoefficient(lam_vars[i][j], 1.0)

    lam = np.zeros((N, n))
    theta_master = -1e18
    best_result = 1e18
    t0 = time.time()
    n_iters = 0
    for it in range(max_outer_iters):
        n_iters = it + 1
        result, xvals = lr1_ccdual(oracle, lam)
        best_result = min(best_result, result)
        if it % log_every == 0:
            print(f"[kelley {label}] it={it} theta_master={theta_master:.2f} "
                  f"result={result:.2f} best_result={best_result:.2f} "
                  f"t={time.time()-t0:.1f}s", file=sys.stderr)
        if theta_master >= result * (1 - tol):
            break
        if time_limit_s is not None and time.time() - t0 > time_limit_s:
            print(f"[kelley {label}] time_limit_s={time_limit_s} atteint a l'iter {it}",
                  file=sys.stderr)
            break
        # coupe : theta - sum_i x_i.lam_i >= result - sum_i x_i.lam_i^current
        subtract = float(np.sum(xvals * lam))
        ct = solver.Constraint(result - subtract, solver.infinity())
        ct.SetCoefficient(theta_var, 1.0)
        for i in range(N):
            for j in range(n):
                xij = float(xvals[i, j])
                if xij != 0.0:
                    ct.SetCoefficient(lam_vars[i][j], -xij)
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            print(f"[kelley {label}] maitre LP non-optimal (status={status}) a l'iter {it}",
                  file=sys.stderr)
            break
        theta_master = theta_var.solution_value()
        for i in range(N):
            for j in range(n):
                lam[i, j] = lam_vars[i][j].solution_value()

    dt = time.time() - t0
    print(f"[kelley {label}] TERMINE iters={n_iters}  theta_master(LB)={theta_master:.4f}  "
          f"best_result(UB)={best_result:.4f}  temps={dt:.1f}s", file=sys.stderr)
    return dict(value=theta_master, best_ub=best_result, iters=n_iters, time_s=dt)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    inst = load_instance(INSTANCE_FILE, INST_ID)
    print(f"Instance inst_id={INST_ID} : n={inst['n']} m={inst['m']} N={inst['N']} "
          f"epsilon={inst['epsilon']}  opt(B&B exact)={inst['opt']:.4f}")

    oracle = ExactOracle(inst)

    # Sanity check : oracle brute-force vs CP-SAT sur quelques (i, cost) aleatoires
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(5):
        i = int(rng.integers(0, inst["N"]))
        cost = rng.normal(0, 200, size=inst["n"])
        v_bf, _ = oracle.zeta(i, cost)
        v_cp, _ = oracle.zeta_exact_check(i, cost)
        max_diff = max(max_diff, abs(v_bf - v_cp))
    print(f"[check] écart max oracle brute-force vs CP-SAT sur 5 tirages : {max_diff:.6f}")
    assert max_diff < 1e-3, "oracle brute-force incohérent avec CP-SAT !"

    results = {}
    results["A_subgrad_prod (T=200, cold start)"] = solve_v1ld_subgradient(
        oracle, T=200, label="A(T=200)")
    results["B_subgrad_converge (T=3000, rel_tol=1e-6)"] = solve_v1ld_subgradient(
        oracle, T=3000, rel_tol=1e-6, tol_window=100,
        abs_tol=1e-5, abs_patience=300, label="B(T=3000)")
    kelley = solve_v1ld_kelley(oracle, max_outer_iters=250, time_limit_s=180,
                                log_every=10, label="C")
    results["C_kelley_best_UB (meilleur f(lambda_k) rencontré)"] = dict(
        value=kelley["best_ub"], iters=kelley["iters"], time_s=kelley["time_s"])
    results["C_kelley_LB_certif (theta maitre, non convergé)"] = dict(
        value=kelley["value"], iters=kelley["iters"], time_s=kelley["time_s"])

    print("\n=== Résumé ===")
    print(f"{'Méthode':50s} {'v1LD*':>14s} {'iters':>8s} {'temps(s)':>10s}")
    for name, r in results.items():
        print(f"{name:50s} {r['value']:14.4f} {r['iters']:8d} {r['time_s']:10.1f}")
    print(f"\nopt (B&B exact, référence) = {inst['opt']:.4f}")
    print("Les valeurs 'UB' (A, B, C_best_UB) doivent être >= opt (borne supérieure valide, "
          "dualité faible). La valeur 'LB certif' de Kelley est un CERTIFICAT INFÉRIEUR du "
          "vrai optimum du dual v1LD* — utile seulement si elle a convergé vers 'best_UB' "
          "(ce qui n'arrive pas forcément dans le budget d'itérations alloué ici).")


if __name__ == "__main__":
    main()
