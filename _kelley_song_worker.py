#!/usr/bin/env python3
"""
_kelley_song_worker.py — Processus séparé pour la méthode C (V1LD "SONG",
reproduction du plan coupant de Kelley de external/CCdual/ccdual.cpp).

Isolé dans son propre processus : gnn/_bnb_solver_lib.so (compilé avec
-fopenmp, lié à libgomp.so.1 de CVMFS) et le runtime OpenMP embarqué par
OR-Tools se marchent dessus si les deux sont chargés dans le même process
Python (segfault au chargement, indépendant de l'ordre d'import — testé).
Ce script n'importe que numpy/ortools, jamais torch/gnn, donc jamais
gnn/_bnb_solver_lib.so.

Entrée : un .npz contenant n,m,N,epsilon,opt,c,b,A (une instance).
Sortie : une ligne "RESULT_JSON: {...}" sur stdout (le reste est du log).

ζ_i (knapsack 0/1 multi-dim exact par scénario) est résolu via CP-SAT
(ortools.sat, déjà dans le même package qu'OR-Tools GLOP donc aucune
dépendance native supplémentaire) plutôt que par énumération brute-force
2^n : pour n=10 les deux sont triviaux, mais 2^n devient risqué en mémoire
dès n~20 (jusqu'à N * 2^n combinaisons stockées) — CP-SAT reste exact et
scale correctement quel que soit n.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model


class ExactOracle:
    def __init__(self, inst: dict):
        self.n = inst["n"]; self.m = inst["m"]; self.N = inst["N"]
        self.c = inst["c"]; self.A = inst["A"]; self.b = inst["b"]
        self.epsilon = inst["epsilon"]
        self.p = np.full(self.N, 1.0 / self.N)

    def theta(self, cost: np.ndarray):
        x = (cost > 0.0).astype(np.float64)
        return float(cost @ x), x

    def zeta(self, i: int, cost: np.ndarray):
        """ζ_i(cost) = max cost·x  s.c.  A[i] x <= b[i], x binaire — exact via CP-SAT."""
        SCALE = 1000
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{k}") for k in range(self.n)]
        for j in range(self.m):
            w = [int(round(self.A[i][j][k] * SCALE)) for k in range(self.n)]
            cap = int(np.floor(self.b[i][j] * SCALE))
            model.Add(sum(w[k] * x[k] for k in range(self.n)) <= cap)
        profits = [int(round(cost[k] * SCALE)) for k in range(self.n)]
        model.Maximize(sum(profits[k] * x[k] for k in range(self.n)))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            raise RuntimeError(f"CP-SAT n'a pas trouvé l'optimum (scénario {i}, statut={status})")
        xval = np.array([solver.Value(x[k]) for k in range(self.n)], dtype=np.float64)
        return float(cost @ xval), xval


def lr1_ccdual(oracle: ExactOracle, lam: np.ndarray):
    """Reproduit lr1() de ccdual.cpp : p_i=1/N, coef = c/N + lam[i]."""
    N, n, eps = oracle.N, oracle.n, oracle.epsilon
    c_over_N = oracle.c / N
    u = np.zeros(N); v = np.zeros(N)
    x_u = np.zeros((N, n)); x_v = np.zeros((N, n))
    for i in range(N):
        coef = c_over_N + lam[i]
        u[i], x_u[i] = oracle.theta(coef)
        v[i], x_v[i] = oracle.zeta(i, coef)

    delta = u - v
    order = np.argsort(delta, kind="stable")
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


def solve_v1ld_kelley(oracle: ExactOracle, lam_bound: float = 50.0,
                       max_outer_iters: int = 2000, time_limit_s: float = None,
                       tol: float = 1e-5, log_every: int = 20):
    """Plan coupant exact de Kelley (maître LP OR-Tools GLOP, à la place de
    CPLEX) reproduisant V1LD/lr1 de ccdual.cpp.

    lam_bound=50 par défaut : reproduit fidèlement la borne FIXE (pas mise à
    l'échelle du coût !) de ccdual.cpp:886 (`IloNumVarArray lambda(env, ...,
    -50, 50)`). Une valeur trop large (ex. max|c|, essayé initialement) laisse
    le maître LP sauter vers des coins extrêmes tant que peu de coupes
    contraignent la boîte -> f(lambda) explose (observé : 3280 -> 150k dès
    l'itération 20) et le budget d'itérations est consommé à revenir de ce
    survol au lieu de converger. L'échelle naturelle de lambda est fixée par
    l'accumulation du sous-gradient (pas 1/sqrt(t) sur T itérations ~ O(sqrt(T))
    ~ quelques dizaines), pas par l'échelle du coût — donc indépendante de
    l'instance, d'où le fait que SONG code une constante plutôt qu'un ratio."""
    N, n = oracle.N, oracle.n
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
            print(f"[kelley] it={it} theta_master={theta_master:.2f} "
                  f"result={result:.2f} best_result={best_result:.2f} "
                  f"t={time.time()-t0:.1f}s", file=sys.stderr)
        if theta_master >= result * (1 - tol):
            break
        if time_limit_s is not None and time.time() - t0 > time_limit_s:
            print(f"[kelley] time_limit_s={time_limit_s} atteint a l'iter {it}", file=sys.stderr)
            break
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
            print(f"[kelley] maitre LP non-optimal (status={status}) a l'iter {it}", file=sys.stderr)
            break
        theta_master = theta_var.solution_value()
        for i in range(N):
            for j in range(n):
                lam[i, j] = lam_vars[i][j].solution_value()

    dt = time.time() - t0
    converged = theta_master >= best_result * (1 - 1e-3)
    max_abs_lam = float(np.max(np.abs(lam)))
    at_boundary = max_abs_lam >= lam_bound * 0.99
    print(f"[kelley] TERMINE iters={n_iters}  theta_master(LB)={theta_master:.4f}  "
          f"best_result(UB)={best_result:.4f}  converged={converged}  temps={dt:.1f}s  "
          f"max|lambda|={max_abs_lam:.2f} (borne={lam_bound}){'  <-- COLLE A LA BORNE, agrandir lam_bound' if at_boundary else ''}",
          file=sys.stderr)
    return dict(value=theta_master, best_ub=best_result, iters=n_iters, time_s=dt,
                converged=converged, max_abs_lam=max_abs_lam, at_boundary=at_boundary)


def solve_v1ld_trustregion(oracle: ExactOracle, r_init: float = 20.0, r_min: float = 0.1,
                            r_max: float = 400.0, kappa: float = 0.1,
                            max_outer_iters: int = 3000, time_limit_s: float = None,
                            tol: float = 1e-6, log_every: int = 20):
    """
    Plan coupant de Kelley stabilisé par région de confiance (bundle method,
    style Kiwiel) — PAS une reproduction de ccdual.cpp (qui n'a aucune
    stabilisation, cf. solve_v1ld_kelley : boîte FIXE ±50 autour de 0, jamais
    recentrée). Répond à une question différente : "quel est le meilleur
    v1LD atteignable par ce type de méthode (relaxation lagrangienne + plan
    coupant), sans les limites de convergence d'un Kelley nu ?"

    Diagnostic qui motive cet ajout : avec la boîte fixe de ccdual.cpp
    (correcte, ±50), le maître trouve un `result` qui décroît de façon
    MONOTONE (12516 -> ... -> 6252 sur 560 itérations, cf. session), signe
    que la méthode progresse réellement mais bute sur le coût croissant par
    itération (~0.1s -> ~2s/itération, quasi quadratique) avant de battre
    f(0)=3280. Le certificat LB (theta_master) montait déjà à 2586 — sous
    l'estimation du sous-gradient maison (2926) — suggérant que le vrai
    optimum du dual v1LD est probablement plus serré que le sous-gradient,
    mais un Kelley nu ne peut pas le prouver dans un budget raisonnable.

    Principe (bundle "serious/null step", cf. Kiwiel 1990) : à chaque
    itération, on résout le maître LP habituel (min theta sous TOUTES les
    coupes accumulées, qui restent valables globalement) mais avec lambda
    contraint dans une boîte [centre - r, centre + r] RECENTRÉE sur le
    meilleur point courant (pas fixe autour de 0). Si la baisse réelle de f
    au candidat atteint au moins `kappa` fois la baisse prédite par le
    modèle linéaire par morceaux, on accepte ("serious step" : nouveau
    centre, r agrandi) ; sinon on rejette ("null step" : centre inchangé, r
    réduit) mais la coupe reste ajoutée (elle affine le modèle partout).

    Le certificat LB rapporté (theta_master_global) est recalculé à la fin
    en relâchant la boîte à l'infini sur TOUTES les coupes accumulées — un
    vrai minorant global, indépendant de la région de confiance (qui ne sert
    qu'à guider la recherche du prochain candidat, pas à définir le modèle).
    """
    N, n = oracle.N, oracle.n
    solver = pywraplp.Solver.CreateSolver("GLOP")
    theta_var = solver.NumVar(-solver.infinity(), solver.infinity(), "theta")
    lam_vars = [[solver.NumVar(-solver.infinity(), solver.infinity(), f"lam_{i}_{j}")
                 for j in range(n)] for i in range(N)]
    solver.Minimize(theta_var)
    for j in range(n):
        ct = solver.Constraint(0.0, 0.0)
        for i in range(N):
            ct.SetCoefficient(lam_vars[i][j], 1.0)

    def add_cut(lam_at: np.ndarray, result: float, xvals: np.ndarray) -> None:
        subtract = float(np.sum(xvals * lam_at))
        ct = solver.Constraint(result - subtract, solver.infinity())
        ct.SetCoefficient(theta_var, 1.0)
        for i in range(N):
            for j in range(n):
                xij = float(xvals[i, j])
                if xij != 0.0:
                    ct.SetCoefficient(lam_vars[i][j], -xij)

    center = np.zeros((N, n))
    f_center, xvals_center = lr1_ccdual(oracle, center)
    best_result = f_center
    add_cut(center, f_center, xvals_center)

    r = r_init
    t0 = time.time()
    n_iters = 0
    n_serious = 0
    for it in range(max_outer_iters):
        n_iters = it + 1
        for i in range(N):
            for j in range(n):
                lam_vars[i][j].SetBounds(center[i, j] - r, center[i, j] + r)
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            print(f"[trustregion] maitre LP non-optimal (status={status}) a l'iter {it}", file=sys.stderr)
            break
        theta_hat = theta_var.solution_value()
        lam_cand = np.array([[lam_vars[i][j].solution_value() for j in range(n)] for i in range(N)])

        result, xvals = lr1_ccdual(oracle, lam_cand)
        best_result = min(best_result, result)

        predicted_decrease = f_center - theta_hat
        actual_decrease = f_center - result

        if it % log_every == 0:
            print(f"[trustregion] it={it} r={r:.3f} f_center={f_center:.2f} "
                  f"theta_hat={theta_hat:.2f} result={result:.2f} best_result={best_result:.2f} "
                  f"pred_decr={predicted_decrease:.4f} n_serious={n_serious} t={time.time()-t0:.1f}s",
                  file=sys.stderr)

        add_cut(lam_cand, result, xvals)

        if predicted_decrease > 1e-9 and actual_decrease >= kappa * predicted_decrease:
            center = lam_cand
            f_center = result
            n_serious += 1
            r = min(r * 1.5, r_max)
        else:
            r = max(r * 0.5, r_min)

        if predicted_decrease < tol * max(abs(f_center), 1.0) and r <= r_min * 1.0001:
            print(f"[trustregion] convergence locale (pred_decr~0, r=r_min) a l'iter {it}", file=sys.stderr)
            break
        if time_limit_s is not None and time.time() - t0 > time_limit_s:
            print(f"[trustregion] time_limit_s={time_limit_s} atteint a l'iter {it}", file=sys.stderr)
            break

    lb_local = theta_hat  # dernier modele resolu, valide seulement pres du centre courant

    # Tentative de certificat LB plus global : relache la region de confiance
    # a quelques multiples de r_max (pas +-1e6 : avec seulement quelques
    # centaines de coupes, toutes generees pres de la trajectoire locale, le
    # modele polyedrique peut avoir une direction "non couverte" dans le noyau
    # de la contrainte de non-anticipativite ou aucune coupe n'augmente jamais
    # -> le maitre s'evade vers -l'infini a l'echelle de la boite relachee
    # (observe : -4.1e6 avec une boite +-1e6, clairement degenere, pas un vrai
    # minorant). On borne donc le relachement, ET on verifie que le resultat
    # est un minorant VALIDE (<= best_result) avant de le faire confiance.
    BIG = max(50.0, 10.0 * r_max)
    for i in range(N):
        for j in range(n):
            lam_vars[i][j].SetBounds(-BIG, BIG)
    status = solver.Solve()
    theta_relaxed = theta_var.solution_value() if status == pywraplp.Solver.OPTIMAL else float("-inf")
    if theta_relaxed <= best_result + 1e-6:
        theta_master_global = theta_relaxed
        lb_is_global = True
    else:
        # degenere (theta_relaxed > best_result est impossible pour un vrai
        # minorant) -> on retombe sur le certificat local, honnete mais plus
        # faible.
        theta_master_global = lb_local
        lb_is_global = False

    dt = time.time() - t0
    converged = theta_master_global >= best_result * (1 - 1e-3)
    print(f"[trustregion] TERMINE iters={n_iters} n_serious={n_serious}  "
          f"theta_master({'global' if lb_is_global else 'local, degenere au-dela'})"
          f"(LB)={theta_master_global:.4f}  best_result(UB)={best_result:.4f}  "
          f"converged={converged}  temps={dt:.1f}s", file=sys.stderr)
    return dict(value=theta_master_global, best_ub=best_result, iters=n_iters,
                n_serious=n_serious, time_s=dt, converged=converged, lb_is_global=lb_is_global)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--time_limit", type=float, default=300.0)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--lam_bound", type=float, default=50.0,
                   help="Boite [-lam_bound, lam_bound] pour les lambda du maitre LP "
                        "(defaut 50.0 = valeur fixe de ccdual.cpp:886, PAS mise a "
                        "l'echelle du cout, cf. docstring solve_v1ld_kelley). "
                        "Ignore si --method=trustregion.")
    p.add_argument("--method", choices=["naive", "trustregion"], default="trustregion",
                   help="naive = reproduction fidele de ccdual.cpp (boite fixe, pas de "
                        "stabilisation). trustregion = bundle stabilise (pas une "
                        "reproduction de SONG, mais teste le meilleur v1LD atteignable "
                        "par ce type de methode).")
    p.add_argument("--r_init", type=float, default=20.0)
    p.add_argument("--r_min", type=float, default=0.1)
    p.add_argument("--r_max", type=float, default=400.0)
    p.add_argument("--kappa", type=float, default=0.1)
    args = p.parse_args()

    d = np.load(args.npz)
    inst = dict(n=int(d["n"]), m=int(d["m"]), N=int(d["N"]),
                epsilon=float(d["epsilon"]), c=d["c"], b=d["b"], A=d["A"])

    oracle = ExactOracle(inst)

    # Sanity check : ζ_i(cost) doit être faisable (A[i] x <= b[i]) et ζ_i <= θ_i
    # (θ = relaxation non contrainte de ζ, donc borne supérieure par construction).
    rng = np.random.default_rng(0)
    for _ in range(3):
        i = int(rng.integers(0, inst["N"]))
        cost = rng.normal(0, 200, size=inst["n"])
        v_theta, _ = oracle.theta(cost)
        v_zeta, x_zeta = oracle.zeta(i, cost)
        viol = float(np.max(oracle.A[i] @ x_zeta - oracle.b[i]))
        assert viol < 1e-3, f"ζ_i CP-SAT infaisable (violation={viol:.4f})"
        assert v_zeta <= v_theta + 1e-6, f"ζ_i > θ_i ({v_zeta:.4f} > {v_theta:.4f}) — incohérent"
    print("[check] ζ_i (CP-SAT) : faisabilité + ζ<=θ vérifiées sur 3 tirages aléatoires", file=sys.stderr)

    t0 = time.perf_counter()
    if args.method == "naive":
        kelley = solve_v1ld_kelley(oracle, lam_bound=args.lam_bound, max_outer_iters=args.max_iters,
                                    time_limit_s=args.time_limit, tol=args.tol)
        out = dict(value=kelley["best_ub"], time_s=time.perf_counter() - t0, iters=kelley["iters"],
                   converged=kelley["converged"], max_abs_lam=kelley["max_abs_lam"],
                   at_boundary=kelley["at_boundary"], method="naive")
    else:
        kelley = solve_v1ld_trustregion(oracle, r_init=args.r_init, r_min=args.r_min,
                                         r_max=args.r_max, kappa=args.kappa,
                                         max_outer_iters=args.max_iters,
                                         time_limit_s=args.time_limit, tol=args.tol)
        out = dict(value=kelley["best_ub"], time_s=time.perf_counter() - t0, iters=kelley["iters"],
                   converged=kelley["converged"], n_serious=kelley["n_serious"],
                   lb_certificate=kelley["value"], lb_is_global=kelley["lb_is_global"],
                   method="trustregion")
    print("RESULT_JSON: " + json.dumps(out))


if __name__ == "__main__":
    main()
