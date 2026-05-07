"""
Solveur sous-gradient pour v_LD^1 (Ahmed et al. 2016, Section 3.1).
Formulation : min cᵀx s.t. P(A^i x ≥ b^i) ≥ 1 − ε, x ∈ {0,1}^n.

────────────────────────────────────────────────────────────────────────────
Version notation Ahmed : λ et H_i explicites (sans reparamétrisation)
────────────────────────────────────────────────────────────────────────────

On introduit N copies x^i de x et on dualise les contraintes
        Σ_i p_i H_i x^i = h     (forme symétrique de Rockafellar-Wets)

avec, pour la non-anticipativité x^j − Σ_i p_i x^i = 0 ∀j,
        H_i ∈ R^{Nn × n}  défini par blocs n × n :
            [H_i]_j = ((1-p_i)/p_i) · I_n   si j = i
                    = − I_n                  si j ≠ i
        h = 0 ∈ R^{Nn}

Le multiplicateur dual λ ∈ R^{Nn} se décompose en N blocs λ_j ∈ R^n,
représentés en pratique par une matrice de shape (N, n).

Sous-problèmes par scénario (eq. (10)–(11) du papier) :
    θ_i(λ) = min { cᵀx + λᵀH_i x : x ∈ {0,1}^n }
    ζ_i(λ) = min { cᵀx + λᵀH_i x : x ∈ X^i }

avec λᵀH_i x = ((1-p_i)/p_i) · λ_iᵀx − Σ_{j≠i} λ_jᵀx
            = (λ_i/p_i − Λ)ᵀ x        où Λ = Σ_j λ_j

donc le coût modifié pour le scénario i est : c + λ_i/p_i − Λ

Formule de la borne (eq. (12), avec h = 0) :
    L1(λ) = Σ_i p_i θ_i(λ) + min_{z ∈ Z} Σ_i p_i (ζ_i(λ) − θ_i(λ)) z_i
    v_LD^1 = max_λ L1(λ)        (borne INFÉRIEURE sur v*)

Sous-gradient (par bloc j) :
    g_j = p_j (x_j* − x̄*)       où x̄* = Σ_i p_i x_i*
    x_i* = arg ζ_i si z_i* = 1, sinon arg θ_i

Update :  λ_j ← λ_j + α_t · g_j

Note sur le noyau de H : la direction λ_i = p_i v (∀i, pour un v ∈ R^n) est
inerte — elle ne change ni le lagrangien ni les sous-problèmes. Le sous-
gradient ci-dessus n'a pas de composante dans cette direction
(Σ_i g_i = Σ_i p_i x_i* − Σ_i p_i · x̄* = x̄* − x̄* = 0), donc λ ne dérive
pas dans le noyau au cours des itérations. Pas besoin de projection
explicite.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, Any

from .instance_reader import CCSPInstance


# ─────────────────────────────────────────────
# Coût modifié par scénario : c + H_iᵀ λ
# ─────────────────────────────────────────────

def cost_modified(c: np.ndarray, lam: np.ndarray, p: np.ndarray, i: int) -> np.ndarray:
    """Calcule c + H_iᵀ λ pour le scénario i.

    Dérivation :
        H_iᵀ λ = ((1-p_i)/p_i) λ_i − Σ_{j≠i} λ_j
               = λ_i/p_i − Σ_j λ_j
               = λ_i/p_i − Λ            où Λ = Σ_j λ_j
    """
    Lambda_sum = lam.sum(axis=0)         # Λ = Σ_j λ_j, shape (n,)
    return c + lam[i] / p[i] - Lambda_sum


# ─────────────────────────────────────────────
# Sous-problèmes atomiques
# ─────────────────────────────────────────────

def solve_theta_i(cost_i: np.ndarray) -> tuple[float, np.ndarray]:
    """θ_i(λ) = min { cost_iᵀ x : x ∈ {0,1}^n } — O(n).

    Solution explicite : x_k = 1 ssi (cost_i)_k < 0.
    """
    x = (cost_i < 0).astype(np.float64)
    return float(cost_i @ x), x


def solve_zeta_i_exact(
    cost_i: np.ndarray,
    A_i: np.ndarray,
    b_i: np.ndarray,
) -> tuple[float, np.ndarray]:
    """ζ_i(λ) = min { cost_iᵀ x : A^i x ≥ b^i, x ∈ {0,1}^n }.

    Résolu par relaxation LP (gap = 0 garanti pour set covering binaire
    avec A, b ∈ {0,1} — Proposition 5 d'Ahmed).

    Retourne (+∞, 0) si le scénario est infaisable.
    """
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
    order = np.argsort(delta)
    cumsum = 0.0
    for idx in order:
        z[idx] = 1.0
        cumsum += p[idx]
        if cumsum >= 1.0 - epsilon:
            break
    return z, float((p * delta) @ z)


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

    # λ ∈ R^{Nn} représenté comme matrice (N, n)
    lam = np.zeros((N, n))

    best_lower = -np.inf
    best_lam = lam.copy()
    convergence: list[float] = []

    t0 = time.perf_counter()

    for t in range(T):
        step = 1.0 / np.sqrt(t + 1)

        # ── (A) θ_i et ζ_i pour chaque scénario ──────────────────────────
        # Coût modifié c + H_iᵀ λ (calculé via la formule λ_i/p_i − Λ)
        theta_vals = np.empty(N)
        zeta_vals = np.empty(N)
        x_theta = np.empty((N, n))
        x_zeta = np.empty((N, n))

        for i in range(N):
            cost_i = cost_modified(c, lam, p, i)
            theta_vals[i], x_theta[i] = solve_theta_i(cost_i)
            zeta_vals[i], x_zeta[i] = solve_zeta_i_exact(cost_i, A[i], b[i])

        # ── (B) Knapsack sur z ───────────────────────────────────────────
        delta = np.maximum(0.0, zeta_vals - theta_vals)
        z_star, knapsack_val = solve_z_knapsack_min(delta, p, epsilon)

        # ── (C) Borne L1(λ) − λᵀ h ──────────────────────────────────────
        # h = 0 dans la forme symétrique → terme − λᵀh = 0
        L1 = float(p @ theta_vals) + knapsack_val
        convergence.append(L1)

        if L1 > best_lower:
            best_lower = L1
            best_lam = lam.copy()

        # ── (D) Sous-gradient g = Σ_i p_i H_i x_i*  (bloc par bloc) ─────
        # x_i* = arg ζ_i si z_i* = 1, sinon arg θ_i
        x_star = np.where(z_star[:, np.newaxis] > 0.5, x_zeta, x_theta)
        x_bar = p @ x_star                                    # x̄* = Σ_i p_i x_i*
        g = p[:, np.newaxis] * (x_star - x_bar[np.newaxis, :])  # g_j = p_j (x_j* − x̄*)

        # ── (E) Update λ ────────────────────────────────────────────────
        lam = lam + step * g

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