"""
presolve_box_exact.py — prototype d'un remplaçant EXACT et CERTIFIÉ de
presolve_min_variance_box (solver_bab/ccpo/ccpo_sld.hpp).

Problème (relaxation « boîte » d'un scénario) :
    v* = min q'Σq   s.c.  Σ_i q_i = Q,  0 <= q_i <= U          (Σ symétrique définie positive)

Défaut de l'existant : ensemble actif sans relâchement (une variable fixée à 0/U ne revient jamais),
plusieurs fixations par itération ; le point final peut même violer Σq = Q. La valeur renvoyée peut
donc SURESTIMER v* -> scénarios satisfaisables écartés à tort (cf. diag_val_nodes/MANIFEST.md).

Méthode :
 1. Ensemble actif PRIMAL standard (Nocedal & Wright, alg. 16.3) : on part d'un point réalisable,
    on ajoute UNE contrainte bloquante à la fois, on relâche la contrainte de multiplicateur le plus
    négatif quand le pas est complet. Converge en un nombre fini d'itérations vers l'optimum exact
    (problème strictement convexe).
 2. Certificat DUAL : pour tout μ ∈ R et α, β >= 0, la fonction duale
        d(μ,α,β) = μQ - β'U·1 - ¼ w'Σ⁻¹w,   w = μ1 + α - β
    est <= v* (dualité faible). On l'évalue avec les multiplicateurs KKT du point final (α, β tronqués
    à >= 0). C'est la valeur renvoyée : elle ne peut PAS surestimer v*, même si l'ensemble actif s'est
    arrêté trop tôt ou si l'arithmétique flottante perd des chiffres ; à l'optimum elle égale v*.
    w'Σ⁻¹w via Cholesky de Σ (une seule factorisation par scénario).

API : box_min_certified(Sigma, Q, U) -> (lower, primal, q, info)
   lower  : borne inférieure certifiée (à utiliser pour le presolve : écarter ssi lower > B)
   primal : q'Σq au point réalisable final (majorant de v*), gap = primal - lower >= 0
"""
from __future__ import annotations

import numpy as np


def _eq_qp_free(S, free, fixed_val, fixed_idx, R):
    """min q_F'S_FF q_F + 2 q_F'S_FX q_X  s.c. 1'q_F = R  ->  (q_F, μ) ; ∇ = 2Sq = μ sur F."""
    M = S[np.ix_(free, free)]
    r = S[np.ix_(free, fixed_idx)] @ fixed_val if len(fixed_idx) else np.zeros(len(free))
    Lc = np.linalg.cholesky(M)
    e = np.linalg.solve(Lc.T, np.linalg.solve(Lc, np.ones(len(free))))
    b = np.linalg.solve(Lc.T, np.linalg.solve(Lc, r))
    mu = 2.0 * (R + b.sum()) / e.sum()
    return mu / 2.0 * e - b


def box_min_certified(S, Q, U, max_iter=None):
    S = np.asarray(S, dtype=np.float64)
    S = 0.5 * (S + S.T)
    n = S.shape[0]
    if Q > n * U + 1e-12 or Q < -1e-12:
        return np.inf, np.inf, None, {"status": "box_infeasible"}
    max_iter = max_iter or 50 * (n + 1)
    # point de départ réalisable
    q = np.full(n, Q / n)
    # état : 0 = libre, 1 = à 0, 2 = à U
    state = np.zeros(n, dtype=np.int8)
    if Q / n > U:  # impossible (exclu ci-dessus) -- garde-fou
        state[:] = 2
    it = 0
    status = "max_iter"
    for it in range(max_iter):
        free = np.where(state == 0)[0]
        fx = np.where(state != 0)[0]
        if len(free) == 0:
            target = q.copy()
        else:
            qF = _eq_qp_free(S, free, q[fx], fx, Q - q[fx].sum())
            target = q.copy()
            target[free] = qF
        p = target - q
        if np.max(np.abs(p)) <= 1e-12 * max(1.0, U):
            # pas nul : vérifier les multiplicateurs des contraintes actives
            g = 2.0 * S @ q
            mu = g[free].mean() if len(free) else _mu_all_fixed(g, state)
            mult = np.where(state == 1, g - mu, np.where(state == 2, mu - g, 0.0))
            j = int(np.argmin(mult))
            if len(fx) == 0 or mult[j] >= -1e-10 * max(1.0, np.abs(g).max()):
                status = "optimal"
                break
            state[j] = 0  # relâcher UNE contrainte (la plus violée)
            continue
        # pas vers target : plus grand t in (0,1] restant dans [0,U]
        t, block = 1.0, -1
        for i in np.where(state == 0)[0]:
            if p[i] < 0 and q[i] + p[i] < 0:
                ti = -q[i] / p[i]
                if ti < t:
                    t, block = ti, i
            elif p[i] > 0 and q[i] + p[i] > U:
                ti = (U - q[i]) / p[i]
                if ti < t:
                    t, block = ti, i
        q = q + t * p
        if block >= 0:
            if p[block] < 0:
                q[block] = 0.0; state[block] = 1
            else:
                q[block] = U; state[block] = 2
        q = np.clip(q, 0.0, U)
    primal = float(q @ S @ q)
    lower = dual_value(S, Q, U, q, state)
    return lower, primal, q, {"status": status, "iter": it + 1}


def _mu_all_fixed(g, state):
    # tous fixés : μ dans l'intervalle [max g_i (à U), min g_i (à 0)] ; milieu
    lo = g[state == 2].max() if (state == 2).any() else -np.inf
    hi = g[state == 1].min() if (state == 1).any() else np.inf
    if np.isfinite(lo) and np.isfinite(hi):
        return 0.5 * (lo + hi)
    return lo if np.isfinite(lo) else hi


def dual_value(S, Q, U, q, state=None):
    """Borne inférieure certifiée : maximise d(μ,α,β) en μ (forme fermée) avec α, β tirés du
    gradient au point q (α_i = (g_i - μ)_+ hors de U, β_i = (μ - g_i)_+ à U). Dualité faible :
    valeur <= v* pour tout μ et α,β >= 0, quelle que soit la qualité de q."""
    n = S.shape[0]
    g = 2.0 * S @ q
    Lc = np.linalg.cholesky(S)

    def sinv(v):
        return np.linalg.solve(Lc.T, np.linalg.solve(Lc, v))

    def d(mu):
        # α, β >= 0 choisis au point q (signe du gradient réduit), puis valeur duale exacte
        atU = st == 2
        at0 = st == 1
        alpha = np.where(at0, np.maximum(g - mu, 0.0), 0.0)
        beta = np.where(atU, np.maximum(mu - g, 0.0), 0.0)
        w = mu * np.ones(n) + alpha - beta
        return mu * Q - U * beta.sum() - 0.25 * w @ sinv(w)

    st = state if state is not None else np.where(q <= 1e-12, 1, np.where(q >= U - 1e-12, 2, 0)).astype(np.int8)
    free = st == 0
    # μ KKT : g_i = μ sur les libres ; sans libre, tout μ de [max_{à U} g, min_{à 0} g] donne α,β >= 0
    # et w = g exactement, donc d = q'Σq (certificat sans écart).
    mu0 = float(g[free].mean()) if free.any() else float(_mu_all_fixed(g, st))
    best = d(mu0)
    # petite recherche 1-D sur μ (d concave en μ à α,β fixés par règle) -- ne peut qu'améliorer la borne
    for scale in (1e-6, 1e-4, 1e-2):
        for s in (-1, 1):
            best = max(best, d(mu0 * (1 + s * scale)))
    return float(best)


def presolve_infeasible_exact(Sigma_dense, Q, U, B, rel_tol=1e-9):
    """Remplaçant de presolve_infeasible_scenarios : scénario écarté ssi borne certifiée > B."""
    out = np.zeros(len(Sigma_dense), dtype=bool)
    for j, S in enumerate(Sigma_dense):
        lower, *_ = box_min_certified(S, Q, U)
        out[j] = lower > B * (1.0 + rel_tol)
    return out


def presolve_old(S, Q, U):
    """Port EXACT de presolve_min_variance_box (C++) -- pour comparaison."""
    n = S.shape[0]
    fixed = np.zeros(n, int); q = np.zeros(n)
    for _ in range(n + 2):
        fr = np.where(fixed == 0)[0]; fx = np.where(fixed != 0)[0]
        if len(fr) == 0:
            break
        R = Q - q[fx].sum()
        M = S[np.ix_(fr, fr)]
        r = S[np.ix_(fr, fx)] @ q[fx] if len(fx) else np.zeros(len(fr))
        e = np.linalg.solve(M, np.ones(len(fr))); b = np.linalg.solve(M, r)
        mu = 2 * (R + b.sum()) / e.sum()
        viol = False
        for a, i in enumerate(fr):
            qa = mu / 2 * e[a] - b[a]
            if qa < -1e-9:
                fixed[i] = 1; q[i] = 0; viol = True
            elif qa > U + 1e-9:
                fixed[i] = 2; q[i] = U; viol = True
            else:
                q[i] = qa
        if not viol:
            break
    return float(q @ S @ q)
