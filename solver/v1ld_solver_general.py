"""
Solveur sous-gradient pour v_LD^1 — variante «general» (double décomposition).

Identique à v1ld_solver.py sauf pour la décomposition interne (decomp=True) :

────────────────────────────────────────────────────────────────────────────
Décomposition interne «générale» — non-anticipativité sur les contraintes
────────────────────────────────────────────────────────────────────────────

Pour chaque scénario i, au lieu de dualiser directement A^i x ≥ b^i,
on réécrit le sous-problème ζ_i comme un problème de consensus :

    m copies x_{i,1}, ..., x_{i,m} ∈ {0,1}^n  (une par contrainte j)
    chaque copie ne satisfait que sa contrainte propre : A^i_j x_{i,j} ≥ b^i_j
    contrainte de non-anticipativité interne : x_{i,1} = ... = x_{i,m}

En dualisant le consensus avec ν_i ∈ R^{m×n} (par analogie avec λ pour
le niveau externe), on obtient le coût modifié pour la copie j :

    cost_{i,j}[k] = cost_i[k] + m · ν_i[j,k] − Σ_{j'} ν_i[j',k]

Le sous-problème pour la copie j est alors :
    ζ_{i,j}(ν_i) = min { cost_{i,j}ᵀ x : A^i_j x ≥ b^i_j, x ∈ {0,1}^n }

et la borne interne est :
    ζ̂_i(λ, ν_i) = (1/m) Σ_j ζ_{i,j}(ν_i)     [borne inférieure sur ζ_i]

Comme la contrainte A^i_j x ≥ b^i_j est UNE seule ligne binaire, la
résolution est O(n) (voir _solve_single_covering).

Sous-gradient sur ν_i (même structure que λ) :
    x̄_i^* = (1/m) Σ_j x_{i,j}^*                 (consensus courant)
    g_{ν_i[j]} = (1/m) · (x_{i,j}^* − x̄_i^*)
    ν_i ← ν_i + α_t · g_{ν_i}                   (pas de projection : ν libre)

Différences vs v1ld_solver.py :
    • mu ∈ R^{N×m}    →  mu ∈ R^{N×m×n}   (plus riche, mais non contraint)
    • _solve_single_covering remplace le min non contraint de ζ̃_i
    • La valeur x_zeta[i] passée au sous-gradient λ est x̄_i^* (moyenne des copies)
"""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, Any, Tuple

from .instance_reader import CCSPInstance


# ─────────────────────────────────────────────
# Coût modifié (identique au solver original)
# ─────────────────────────────────────────────

def cost_modified(
    c: np.ndarray,
    lam: np.ndarray,
    Lambda_sum: np.ndarray,
    p: np.ndarray,
    i: int,
) -> np.ndarray:
    return c + lam[i] / p[i] - Lambda_sum


# ─────────────────────────────────────────────
# Sous-problèmes atomiques
# ─────────────────────────────────────────────

def solve_theta_i(cost_i: np.ndarray) -> tuple[float, np.ndarray]:
    """θ_i(λ) = min { cost_iᵀ x : x ∈ {0,1}^n } — O(n)."""
    x = (cost_i < 0).astype(np.float64)
    return float(cost_i @ x), x


def solve_zeta_i_exact(
    cost_i: np.ndarray,
    A_i: np.ndarray,
    b_i: np.ndarray,
) -> tuple[float, np.ndarray]:
    """ζ_i exact par relaxation LP (inchangé)."""
    from scipy.optimize import linprog

    n = cost_i.size
    res = linprog(
        c=cost_i,
        A_ub=-A_i,
        b_ub=-b_i,
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )
    if not res.success:
        return np.inf, np.zeros(n)
    x_i = np.clip(np.round(res.x), 0.0, 1.0)
    return float(cost_i @ x_i), x_i


def _solve_single_covering(
    cost_j: np.ndarray,
    A_row: np.ndarray,
    b_j: float,
) -> tuple[np.ndarray, float]:
    """min cost_jᵀ x  s.c.  Σ_k A_row[k] x_k ≥ b_j,  x ∈ {0,1}^n — O(n).

    A_row ∈ {0,1}^n (une ligne de A^i), b_j ∈ {0,1}.

    Algorithme :
      1. Solution non contrainte : x_k = 1 ssi cost_j[k] < 0.
      2. Si b_j = 0 ou contrainte déjà satisfaite → terminé.
      3. Sinon (b_j=1, aucun item couvrant sélectionné) : forcer l'item
         couvrant le moins coûteux parmi ceux non encore sélectionnés.
    """
    x = (cost_j < 0).astype(np.float64)
    val = float(cost_j @ x)

    if b_j < 0.5 or A_row @ x >= b_j - 1e-9:
        return x, val

    # Contrainte non satisfaite : tous les items couvrants ont cost_j ≥ 0
    # (sinon ils seraient déjà dans x). Forcer le moins coûteux.
    covering_free = np.where((A_row > 0.5) & (x < 0.5))[0]

    if covering_free.size == 0:
        return x, np.inf  # scénario infaisable (rare avec instances bien formées)

    k_star = covering_free[np.argmin(cost_j[covering_free])]
    x = x.copy()
    x[k_star] = 1.0
    return x, val + cost_j[k_star]


def solve_zeta_i_general(
    cost_i: np.ndarray,
    A_i: np.ndarray,
    b_i: np.ndarray,
    nu_i: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """ζ̂_i via décomposition de non-anticipativité interne.

    Paramètres
    ----------
    cost_i : (n,)   coût modifié pour le scénario i (inclut λ)
    A_i    : (m, n) matrice de couverture
    b_i    : (m,)   RHS binaire
    nu_i   : (m, n) multiplicateurs de consensus interne

    Retourne
    --------
    val    : ζ̂_i = (1/m) Σ_j ζ_{i,j}
    x_bar  : (n,)   moyenne (1/m) Σ_j x_{i,j}^*  — pour sous-gradient λ
    x_by_j : (m, n) optimal par copie              — pour sous-gradient ν
    """
    m = A_i.shape[0]
    nu_sum = nu_i.sum(axis=0)  # (n,) — Σ_{j'} ν_i[j']

    x_by_j = np.empty((m, A_i.shape[1]))
    val = 0.0

    for j in range(m):
        # Coût modifié pour la copie j : symétrique au niveau externe
        cost_ij = cost_i + nu_i[j] - nu_sum / m  # (n,)  # La contrainte de n-a ne s'applique qu'aux éléments du même scénario
        x_ij, v_ij = _solve_single_covering(cost_ij, A_i[j], b_i[j]) # résolution d'un set covering de niveau inférieur
        x_by_j[j] = x_ij
        val += v_ij

    x_bar = x_by_j.mean(axis=0)  # (n,)
    return val / m, x_bar, x_by_j


def solve_z_knapsack_min(
    delta: np.ndarray,
    p: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, float]:
    """Knapsack glouton (inchangé)."""
    N = delta.size
    z = np.zeros(N)
    order = np.argsort(delta)
    cumsum = 0.0
    for idx in order:
        z[idx] = 1.0
        cumsum += p[idx]
        if cumsum >= 1.0 - epsilon - 1e-9:
            break
    return z, float((p * delta) @ z)


# ─────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────

def solve_v1LD_general(
    instance: CCSPInstance,
    T: int = 500,
    decomp: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Solveur sous-gradient pour v_LD^1 (variante «general»).

    Paramètres
    ----------
    instance : CCSPInstance
    T        : nombre d'itérations
    decomp   : si True, utilise la décomposition interne par non-anticipativité
               (ν ∈ R^{N×m×n}, non contraint) plutôt que le LP exact.
    verbose  : trace toutes les 50 itérations
    """
    n, m, N = instance.n, instance.m, instance.N
    c, p, epsilon = instance.c, instance.p, instance.epsilon
    A, b = instance.A, instance.b

    lam = np.zeros((N, n))
    mu  = np.zeros((N, m, n)) if decomp else None  # ν ∈ R^{N×m×n}, libre

    best_lower = -np.inf
    best_lam   = lam.copy()
    best_nu    = mu.copy() if decomp else None
    convergence: list[float] = []

    t0 = time.perf_counter()

    for t in range(T):
        step = 1.0 / np.sqrt(t + 1)

        # ── (A) θ_i et ζ̂_i (ou ζ_i exact) pour chaque scénario ──────────
        theta_vals = np.empty(N)
        zeta_vals  = np.empty(N)
        x_theta    = np.empty((N, n))
        x_zeta     = np.empty((N, n))   # x̄_i^* = moyenne des copies (ou x_ζ exact)
        x_by_j_all = np.empty((N, m, n)) if decomp else None  # copies par contrainte
        Lambda_sum = lam.sum(axis=0)

        for i in range(N):
            cost_i = cost_modified(c, lam, Lambda_sum, p, i) #c_i + lam[i] / p[i] - Lambda_sum
            theta_vals[i], x_theta[i] = solve_theta_i(cost_i)

            if decomp:
                zeta_vals[i], x_zeta[i], x_by_j_all[i] = solve_zeta_i_general(
                    cost_i, A[i], b[i], mu[i]
                )
            else:
                zeta_vals[i], x_zeta[i] = solve_zeta_i_exact(cost_i, A[i], b[i])

        # ── (B) Knapsack sur z ───────────────────────────────────────────
        delta = np.maximum(0.0, zeta_vals - theta_vals)
        z_star, knapsack_val = solve_z_knapsack_min(delta, p, epsilon)

        # ── (C) Borne L1 ─────────────────────────────────────────────────
        L1 = float(p @ theta_vals) + knapsack_val
        convergence.append(L1)

        if L1 > best_lower:
            best_lower = L1
            best_lam   = lam.copy()
            if decomp:
                best_nu = mu.copy()

        # ── (D) Sous-gradient sur λ ─────────────────────────────────────
        # x_zeta[i] = x̄_i^* (moyenne des copies) si decomp, sinon x_ζ_i exact
        x_star = np.where(z_star[:, np.newaxis] > 0.5, x_zeta, x_theta)
        x_bar  = p @ x_star
        g_lam  = x_star - x_bar[np.newaxis, :]

        # ── (E) Update λ ────────────────────────────────────────────────
        lam = lam + step * g_lam

        # ── (F) Sous-gradient sur ν et update (mode decomp uniquement) ──
        if decomp:
            for i in range(N):
                if z_star[i] > 0.5:
                    # x_by_j_all[i] : (m, n) — copies optima pour le scénario i
                    x_bar_i = x_by_j_all[i].mean(axis=0)            # (n,)
                    g_nu_i  = (x_by_j_all[i] - x_bar_i) / m        # (m, n)
                    mu[i]   = mu[i] + step * g_nu_i                 # pas de projection

        if verbose and (t % 50 == 0 or t == T - 1):
            grad_norm = np.linalg.norm(g_lam)
            nu_str = f" ||ν||={np.linalg.norm(mu):.3f}" if decomp else ""
            print(
                f"  iter {t:4d} | L1={L1:10.4f}  best={best_lower:10.4f}"
                f"  Σpθ={p @ theta_vals:.4f}  knap={knapsack_val:.4f}"
                f"  ||g_λ||={grad_norm:.4f}{nu_str}"
            )

    out: Dict[str, Any] = {
        "lambda_star": best_lam.tolist(),
        "v1LD_star":   best_lower,
        "convergence": convergence,
        "solve_time_s": time.perf_counter() - t0,
        "decomp": decomp,
    }
    if decomp:
        out["nu_star"] = best_nu.tolist()  # (N, m, n)
    return out


# ─────────────────────────────────────────────
# Borne v_C(M)  — Ahmed 2016, §3.1
# ─────────────────────────────────────────────

def compute_vc_m(
    instance: CCSPInstance,
    time_limit: int = 300,
) -> Dict[str, Any]:
    """v_C(M=1) — borne de relaxation des contraintes d'intégralité sur z.

    Ahmed 2016, §3.1 : v_LD^1 = v_C(M) sous conditions de régularité.

    Deux calculs sont effectués et retournés :

    (A) Définition stricte Ahmed — «v_CM_exact» :
        x ∈ {0,1}^n  (intégralité NON relaxée),  z ∈ [0,1]^N  (relaxée)
        → Programme mixte-entier (MILP) résolu par CBC/PuLP.
        Remarque : pour A, b ∈ {0,1} et x ∈ {0,1}^n, z est automatiquement
        entier à l'optimum (A^i_j x ∈ {0,1,...} force z_i ∈ {0,1}).
        Dans ce cas, v_CM_exact = v_MIP  (aucun gain par rapport au MIP).

    (B) Relaxation LP complète — «v_CM_lp» :
        x ∈ [0,1]^n  ET  z ∈ [0,1]^N
        → LP résolu par HiGHS/scipy.
        Ceci correspond à v_C(M) quand S = polytope LP (pas de contraintes
        d'intégralité sur x). C'est la borne la plus informative en pratique.

    Hiérarchie :  v_CM_lp ≤ v_CM_exact = v_MIP ≤ v*
    (pour A, b ∈ {0,1}, l'égalité v_CM_exact = v_MIP est toujours vérifiée.)

    Paramètres
    ----------
    instance   : CCSPInstance
    time_limit : limite de temps CBC (secondes) pour le MILP

    Retour
    ------
    dict avec :
      "v_CM_exact"  : float — borne stricte Ahmed (= v_MIP pour instances binaires)
      "v_CM_lp"     : float — relaxation LP complète (toujours ≤ v_CM_exact)
      "x_lp"        : np.ndarray (n,) — solution x de la LP
      "z_lp"        : np.ndarray (N,) — solution z de la LP
      "x_exact"     : np.ndarray (n,) — solution x du MILP
      "z_exact"     : np.ndarray (N,) — solution z du MILP
      "z_lp_frac"   : list[(i, val)] — composantes z fractionnaires (LP)
      "status_exact": str — statut du solveur MILP
    """
    from scipy.optimize import linprog

    n, m, N       = instance.n, instance.m, instance.N
    c, p, epsilon = instance.c, instance.p, instance.epsilon
    A, b          = instance.A, instance.b

    # ── (B) LP complète : x ∈ [0,1]^n, z ∈ [0,1]^N ──────────────────────────
    # Variables [x (n), z (N)].  Inégalités : A_ub y ≤ b_ub.
    nv = n + N
    c_lp = np.concatenate([c, np.zeros(N)])
    rows, rhs = [], []

    row = np.zeros(nv); row[n:] = -p          # chance : -Σ p_i z_i ≤ -(1-ε)
    rows.append(row); rhs.append(-(1.0 - epsilon))

    for i in range(N):
        for j in range(m):
            if b[i, j] < 0.5:
                continue                        # contrainte triviale (b=0)
            row = np.zeros(nv)
            row[:n]    = -A[i, j, :]            # couverture : -A^i_j x + b^i_j z_i ≤ 0
            row[n + i] =  b[i, j]
            rows.append(row); rhs.append(0.0)

    res_lp = linprog(
        c_lp, A_ub=np.array(rows), b_ub=np.array(rhs),
        bounds=[(0.0, 1.0)] * nv, method="highs",
    )
    if not res_lp.success:
        v_cm_lp = float("nan")
        x_lp = z_lp = np.full(n, float("nan"))
    else:
        v_cm_lp = float(res_lp.fun)
        x_lp    = res_lp.x[:n]
        z_lp    = res_lp.x[n:]

    z_lp_frac = [(i, float(z_lp[i])) for i in range(N)
                 if 1e-6 < z_lp[i] < 1 - 1e-6] if res_lp.success else []

    # ── (A) MILP : x ∈ {0,1}^n, z ∈ [0,1]^N ─────────────────────────────────
    try:
        import pulp
        prob = pulp.LpProblem(f"vCM_ep{instance.episode}", pulp.LpMinimize)

        xv = [pulp.LpVariable(f"x{k}", cat="Binary")              for k in range(n)]
        zv = [pulp.LpVariable(f"z{i}", lowBound=0.0, upBound=1.0) for i in range(N)]

        prob += pulp.lpSum(c[k] * xv[k] for k in range(n))
        prob += pulp.lpSum(p[i] * zv[i] for i in range(N)) >= 1.0 - epsilon

        for i in range(N):
            for j in range(m):
                if b[i, j] < 0.5:
                    continue
                prob += (
                    pulp.lpSum(A[i, j, k] * xv[k] for k in range(n))
                    >= b[i, j] * zv[i]
                )

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=0, gapAbs=0)
        prob.solve(solver)

        obj_val    = pulp.value(prob.objective)
        status_str = pulp.LpStatus[prob.status]
        x_exact    = np.array([pulp.value(xv[k]) or 0.0 for k in range(n)])
        z_exact    = np.array([pulp.value(zv[i]) or 0.0 for i in range(N)])
        v_cm_exact = float(obj_val) if obj_val is not None else float("nan")

    except ImportError:
        v_cm_exact = float("nan")
        status_str = "PuLP non disponible"
        x_exact    = z_exact = np.full(n, float("nan"))

    return {
        "v_CM_exact":   v_cm_exact,
        "v_CM_lp":      v_cm_lp,
        "x_lp":         x_lp,
        "z_lp":         z_lp,
        "x_exact":      x_exact,
        "z_exact":      z_exact,
        "z_lp_frac":    z_lp_frac,
        "status_exact": status_str,
    }
