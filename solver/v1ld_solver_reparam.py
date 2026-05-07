"""
Solveur sous-gradient pour v_LD^1 (Ahmed et al. 2016, Section 3.1).
Formulation : min cᵀx s.t. P(A^i x ≥ b^i) ≥ 1 − ε, x ∈ {0,1}^n.

────────────────────────────────────────────────────────────────────────────
Dualité de non-anticipativité (forme symétrique de Rockafellar-Wets)
────────────────────────────────────────────────────────────────────────────

On introduit N copies x^i de x et on dualise les contraintes
        x^j − Σ_i p_i x^i = 0     ∀ j ∈ {1,...,N}.

Reparamétrisation : on travaille avec λ ∈ R^{N×n} sous la contrainte
        Σ_i p_i λ_i = 0     (vecteur n-dim).
        En pratique donne les même résultats et est un eu plus rapide que la formulation explicite

Sous-problèmes par scénario (chacun a son propre coût modifié c + λ_i) :
    θ_i(λ_i) = min { (c+λ_i)ᵀx : x ∈ {0,1}^n }       (sous-problème libre)
    ζ_i(λ_i) = min { (c+λ_i)ᵀx : x ∈ X^i }            (sous-problème contraint)

Formule de la borne (eq. (12) du papier, avec h = 0) :
    L1(λ) = Σ_i p_i θ_i(λ_i) + min_{z ∈ Z} Σ_i p_i (ζ_i(λ_i) − θ_i(λ_i)) z_i
    v_LD^1 = max_λ L1(λ)        (borne INFÉRIEURE sur v*)

Sous-gradient (par scénario j) :
    g_j = x_j* − x̄*       où x̄* = Σ_i p_i x_i*
    x_i* = arg de ζ_i si z_i* = 1, sinon arg de θ_i

Update :  λ_j ← λ_j + α_t · g_j   puis projection sur {Σ_i p_i λ_i = 0}.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, Any

from .instance_reader import CCSPInstance


# ─────────────────────────────────────────────
# Sous-problèmes atomiques
# ─────────────────────────────────────────────

def solve_theta_i(c_plus_lam_i: np.ndarray) -> tuple[float, np.ndarray]:
    """θ_i(λ_i) = min_{x ∈ {0,1}^n} (c+λ_i)ᵀx — O(n).

    Solution explicite : x_k = 1 ssi (c+λ_i)_k < 0.
    """
    x = (c_plus_lam_i < 0).astype(np.float64)
    return float(c_plus_lam_i @ x), x


def solve_zeta_i_exact(
    c_plus_lam_i: np.ndarray,
    A_i: np.ndarray,
    b_i: np.ndarray,
) -> tuple[float, np.ndarray]:
    """ζ_i(λ_i) = min { (c+λ_i)ᵀx : A^i x ≥ b^i, x ∈ {0,1}^n }.

    Résolu par relaxation LP (gap = 0 garanti pour set covering binaire
    avec A, b ∈ {0,1} — Proposition 5 d'Ahmed).

    Retourne (+∞, 0) si le scénario est infaisable.
    """
    from scipy.optimize import linprog

    n = c_plus_lam_i.size
    res = linprog(
        c=c_plus_lam_i,
        A_ub=-A_i,
        b_ub=-b_i,
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )

    if not res.success:
        return np.inf, np.zeros(n)

    # Pour set covering binaire l'optimum LP est entier ; on arrondit pour
    # éliminer les bruits numériques.
    x_i = np.clip(np.round(res.x), 0.0, 1.0)
    return float(c_plus_lam_i @ x_i), x_i


def solve_z_knapsack_min(
    delta: np.ndarray,
    p: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, float]:
    """Knapsack : min Σ_i p_i δ_i z_i  s.c.  Σ_i p_i z_i ≥ 1−ε,  z ∈ {0,1}^N.

    Algorithme glouton : on prend les z_i = 1 par δ_i croissant jusqu'à
    saturer la masse 1−ε. O(N log N).
    """
    N = delta.size
    z = np.zeros(N)
    order = np.argsort(delta)  # croissant
    cumsum = 0.0
    for idx in order:
        z[idx] = 1.0
        cumsum += p[idx]
        if cumsum >= 1.0 - epsilon:
            break
    return z, float((p * delta) @ z)


# ─────────────────────────────────────────────
# Projection sur {Σ_i p_i λ_i = 0}
# ─────────────────────────────────────────────

def project_lambda(lam: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Projection orthogonale (sens L2 standard) sur {λ : Σ_i p_i λ_i = 0}.

    On retire la moyenne pondérée à chaque λ_i :
        λ_i ← λ_i − Σ_j p_j λ_j
    """
    weighted_mean = p @ lam  # shape (n,)
    return lam - weighted_mean[np.newaxis, :]


# ─────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────

def solve_v1LD(
    instance: CCSPInstance,
    T: int = 500,
    verbose: bool = False,
) -> Dict[str, Any]:
    n, m, N = instance.n, instance.m, instance.N
    c, p, epsilon = instance.c, instance.p, instance.epsilon
    A, b = instance.A, instance.b

    # λ ∈ R^{N×n}, initialisé à 0 (qui satisfait Σ p_i λ_i = 0)
    lam = np.zeros((N, n))

    best_lower = -np.inf
    best_lam = lam.copy()
    convergence: list[float] = []

    t0 = time.perf_counter()

    for t in range(T):
        step = 1.0 / np.sqrt(t + 1)

        # ── (A) θ_i et ζ_i pour chaque scénario ──────────────────────────
        theta_vals = np.empty(N)
        zeta_vals = np.empty(N)
        x_theta = np.empty((N, n))
        x_zeta = np.empty((N, n))

        for i in range(N):
            cost_i = c + lam[i]
            theta_vals[i], x_theta[i] = solve_theta_i(cost_i)
            zeta_vals[i], x_zeta[i] = solve_zeta_i_exact(cost_i, A[i], b[i])

        # ── (B) Knapsack sur z ───────────────────────────────────────────
        delta = np.maximum(0.0, zeta_vals - theta_vals)
        z_star, knapsack_val = solve_z_knapsack_min(delta, p, epsilon)

        # ── (C) Borne L1(λ) ──────────────────────────────────────────────
        # h = 0 dans la formulation symétrique → pas de terme à soustraire
        L1 = float(p @ theta_vals) + knapsack_val
        convergence.append(L1)

        if L1 > best_lower:
            best_lower = L1
            best_lam = lam.copy()

        # ── (D) Sous-gradient ────────────────────────────────────────────
        # x_i* = arg de ζ_i si z_i* = 1, sinon arg de θ_i
        x_star = np.where(z_star[:, np.newaxis] > 0.5, x_zeta, x_theta)
        x_bar = p @ x_star  # shape (n,) — moyenne pondérée
        g = x_star - x_bar[np.newaxis, :]  # shape (N, n) — un sous-gradient par λ_i

        # ── (E) Update + projection ──────────────────────────────────────
        # La projection sert à la robustesse à l'initialisation et la stabilité numérique
        lam = lam + step * g
        lam = project_lambda(lam, p)

        if verbose and (t % 50 == 0 or t == T - 1):
            grad_norm = np.linalg.norm(g)
            print(
                f"  iter {t:4d} | L1={L1:10.4f}  best={best_lower:10.4f}"
                f"  Σpθ={p @ theta_vals:.4f}  knap={knapsack_val:.4f}"
                f"  ||g||={grad_norm:.4f}"
            )

    return {
        "lambda_star": best_lam.tolist(),
        "v1LD_star": best_lower,
        "convergence": convergence,
        "solve_time_s": time.perf_counter() - t0,
    }