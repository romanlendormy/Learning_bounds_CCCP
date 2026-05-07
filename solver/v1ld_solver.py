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
    ζ_i(λ) = min { cᵀx + λᵀH_i x : A^i x ≥ b^i, x ∈ {0,1}^n }

avec λᵀH_i x = ((1-p_i)/p_i) · λ_iᵀx − Σ_{j≠i} λ_jᵀx
            = (λ_i/p_i − Λ)ᵀ x        où Λ = Σ_j λ_j

donc le coût modifié pour le scénario i est : c + λ_i/p_i − Λ

Formule de la borne (eq. (12), avec h = 0) :
    L1(λ) = Σ_i p_i θ_i(λ) + min_{z ∈ Z} Σ_i p_i (ζ_i(λ) − θ_i(λ)) z_i
    v_LD^1 = max_λ L1(λ)        (borne INFÉRIEURE sur v*)

────────────────────────────────────────────────────────────────────────────
Décomposition interne optionnelle (decomp=True)
────────────────────────────────────────────────────────────────────────────

ζ_i implique m contraintes A^i_j x ≥ b^i_j (j = 1,...,m). Au lieu de les
résoudre exactement par LP, on peut les dualiser avec μ_i ∈ R^m_+ :

    ζ̃_i(λ, μ_i) = μ_iᵀ b^i + min_x { (c + H_iᵀ λ − A_iᵀ μ_i)ᵀ x : x ∈ {0,1}^n }

Le min sur x est trivial (signe par signe), donc le coût passe de O(LP) à
O(n·m). Comme μ_i ≥ 0, on a ζ̃_i ≤ ζ_i, donc la borne L̃_1 reste valide
mais plus faible que L_1 exact.

On effectue alors une montée duale **jointe** sur (λ, μ) :
    Sous-gradient sur λ : g_λ_j = p_j (x_j* − x̄*)        (comme avant)
    Sous-gradient sur μ_i : g_μ_i = b^i − A^i x_i*        (violations)
    Update : μ_i ← max(0, μ_i + α_t · g_μ_i)              (projection ≥ 0)

Le δ_i utilisé dans le knapsack est clampé : δ_i = max(0, ζ̃_i − θ_i),
parce que ζ̃_i peut être < θ_i pour certains μ_i (alors que ζ_i ≥ θ_i
toujours).
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

def cost_modified(c: np.ndarray, lam: np.ndarray, Lambda_sum : np.ndarray, p: np.ndarray, i: int) -> np.ndarray:
    """Calcule c + H_iᵀ λ pour le scénario i.

    Dérivation :
        H_iᵀ λ = ((1-p_i)/p_i) λ_i − Σ_{j≠i} λ_j
               = λ_i/p_i − Σ_j λ_j
               = λ_i/p_i − Λ            où Λ = Σ_j λ_j
    """
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


def solve_zeta_i_decomp(
    cost_i: np.ndarray,
    A_i: np.ndarray,
    b_i: np.ndarray,
    mu_i: np.ndarray,
) -> tuple[float, np.ndarray]:
    """ζ̃_i(λ, μ_i) : relaxation lagrangienne des contraintes A^i x ≥ b^i.

    ζ̃_i = μ_iᵀ b^i + min_x { (cost_i − A_iᵀ μ_i)ᵀ x : x ∈ {0,1}^n }

    Coût O(n·m) (produit matriciel), pas de LP. Pour μ_i ≥ 0 courant,
    on a ζ̃_i ≤ ζ_i (relaxation), donc la borne L̃_1 reste ≤ v*.
    """
    cost_with_mu = cost_i - A_i.T @ mu_i  # shape (n,)
    x_i = (cost_with_mu < 0).astype(np.float64)
    val = float(mu_i @ b_i) + float(cost_with_mu @ x_i)
    return val, x_i


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
    decomp: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Solveur sous-gradient pour v_LD^1.

    Paramètres
    ----------
    instance : CCSPInstance
        Instance du problème.
    T : int
        Nombre d'itérations de montée duale.
    decomp : bool
        Si True, utilise la décomposition interne ζ̃_i avec μ_i ∈ R^m_+
        au lieu de résoudre ζ_i exactement par LP.
        - decomp=False (défaut) : ζ_i exact, borne plus serrée, plus lent.
        - decomp=True            : ζ̃_i lagrangien, borne plus faible mais
                                   coût O(n·m) au lieu de O(LP).
    verbose : bool
        Affiche la trace toutes les 50 itérations.
    """
    n, m, N = instance.n, instance.m, instance.N
    c, p, epsilon = instance.c, instance.p, instance.epsilon
    A, b = instance.A, instance.b

    # λ ∈ R^{Nn} représenté comme matrice (N, n)
    lam = np.zeros((N, n))

    # μ ∈ R^{N×m}_+ uniquement utilisé en mode décomposition
    mu = np.zeros((N, m)) if decomp else None

    best_lower = -np.inf
    best_lam = lam.copy()
    best_mu = mu.copy() if decomp else None
    convergence: list[float] = []

    t0 = time.perf_counter()

    for t in range(T):
        step = 1.0 / np.sqrt(t + 1)

        # ── (A) θ_i et ζ_i (ou ζ̃_i) pour chaque scénario ───────────────
        theta_vals = np.empty(N)
        zeta_vals = np.empty(N)
        x_theta = np.empty((N, n))
        x_zeta = np.empty((N, n))
        Lambda_sum = lam.sum(axis=0)  # shape (n,)

        for i in range(N):
            cost_i = cost_modified(c, lam, Lambda_sum, p, i)
            theta_vals[i], x_theta[i] = solve_theta_i(cost_i)

            if decomp:
                zeta_vals[i], x_zeta[i] = solve_zeta_i_decomp(
                    cost_i, A[i], b[i], mu[i]
                )
            else:
                zeta_vals[i], x_zeta[i] = solve_zeta_i_exact(
                    cost_i, A[i], b[i]
                )

        # ── (B) Knapsack sur z ───────────────────────────────────────────
        # Clamp à 0 nécessaire en mode decomp (ζ̃_i peut être < θ_i)
        delta = np.maximum(0.0, zeta_vals - theta_vals)
        z_star, knapsack_val = solve_z_knapsack_min(delta, p, epsilon)

        # ── (C) Borne L1(λ, μ) ──────────────────────────────────────────
        L1 = float(p @ theta_vals) + knapsack_val
        convergence.append(L1)

        if L1 > best_lower:
            best_lower = L1
            best_lam = lam.copy()
            if decomp:
                best_mu = mu.copy()

        # ── (D) Sous-gradient sur λ ─────────────────────────────────────
        # x_i* = arg ζ_i si z_i* = 1, sinon arg θ_i
        # @@@@ avant on pondérait par p, et ça convergait plus vite, mais pas correct théoriquement
        x_star = np.where(z_star[:, np.newaxis] > 0.5, x_zeta, x_theta)
        x_bar = p @ x_star
        g_lam = (x_star - x_bar[np.newaxis, :])

        # ── (E) Update λ ────────────────────────────────────────────────
        lam = lam + step * g_lam

        # ── (F) Sous-gradient sur μ et update (mode decomp uniquement) ──
        if decomp:
            # g_μ_i = b^i − A^i x_i*  (violation des contraintes)
            # On utilise x_zeta[i] (l'arg de ζ̃_i) pour la mise à jour de μ_i,
            # car μ_i n'agit que sur le sous-problème scénario i.
            # Pour les scénarios avec z_i* = 0, ζ̃_i n'apparaît pas dans la
            # borne, mais on continue à descendre pour préparer le terrain.
            # @@@@  avant on avait g_μ_i = b^i − A^i x_θ_i (arg de θ_i) pour tous les scénarios, ça convergeait plus vite
            for i in range(N):
                if z_star[i] > 0.5:                         # z_i* = 1
                    violation = b[i] - A[i] @ x_zeta[i]     # shape (m,)
                    mu[i] = np.maximum(0.0, mu[i] + step * p[i] * violation)
                # sinon : μ_i inchangé (sous-gradient nul sur cette coordonnée)

        if verbose and (t % 50 == 0 or t == T - 1):
            grad_norm = np.linalg.norm(g_lam)
            mu_str = f" ||μ||={np.linalg.norm(mu):.3f}" if decomp else ""
            print(
                f"  iter {t:4d} | L1={L1:10.4f}  best={best_lower:10.4f}"
                f"  Σpθ={p @ theta_vals:.4f}  knap={knapsack_val:.4f}"
                f"  ||g_λ||={grad_norm:.4f}{mu_str}"
            )

    out: Dict[str, Any] = {
        "lambda_star": best_lam.tolist(),
        "v1LD_star": best_lower,
        "convergence": convergence,
        "solve_time_s": time.perf_counter() - t0,
        "decomp": decomp,
    }
    if decomp:
        out["mu_star"] = best_mu.tolist()
    return out