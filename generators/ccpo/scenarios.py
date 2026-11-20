"""
scenarios.py — Perturbation de l'instance mère en N scénarios de covariance,
étages (a)(b)(c)(d).

On perturbe les **facteurs** (beta, psi, sigma_f), jamais Sigma directement : toute
matrice de la forme `sigma_f**2 * outer(beta, beta) + diag(psi)` avec `psi > 0` est PSD
par construction, ce qu'une perturbation entrée par entrée ne garantirait pas.

Étages :
    (a) bruit d'estimation sur les chargements :
            beta_j = beta * (1 + delta),   delta ~ N(0, noise_beta**2)
    (b) bruit log-normal sur les variances idiosyncratiques (positivité garantie) :
            psi_j  = psi * exp(eta),       eta   ~ N(0, noise_psi**2)
    (c) régime de volatilité de marché — COMMUN à tous les actifs du scénario j :
            sigma_f_j = sigma_f * exp(zeta_j),   zeta_j ~ N(0, noise_market**2)  (scalaire)
        C'est ce qui corrèle les scénarios entre eux : sans lui, ils sont trop
        interchangeables et `∑ z_j ≥ (1-eps)N` devient triviale.
    (d) chocs idiosyncratiques discrets :
            psi_j *= where(shock_j, kappa, 1.0),   shock_j ~ Bernoulli(p_shock)
        Analogue du Bernoulli du protocole MKP (item absent -> poids nuls). Sans lui,
        `eps` ne fait que découper une queue gaussienne et l'optimum varie à peine.

`psi_j > 0` reste automatique (produit de `psi > 0` par des facteurs strictement
positifs : `exp(eta)`, et `kappa`/`1.0` tous deux > 0), donc aucun rejet-retirage n'est
nécessaire.

Les étages (c) et (d) sont désactivables via `include_market_regime`/`include_shocks`
(défaut : activés) — utile pour le test de monotonicité de dispersion
(`test_dispersion_increases_with_each_stage`) qui vérifie que chaque étage
ajoute effectivement de la dispersion, pas pour un usage en production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generators.ccpo.instance import MotherInstance

NOISE_BETA_DEFAULT = 0.10
NOISE_PSI_DEFAULT = 0.10
NOISE_MARKET_DEFAULT = 0.15
P_SHOCK_DEFAULT = 0.05
KAPPA_DEFAULT = 5.0
# Coefficients par défaut, sous la forme attendue par writers.write_episode(noise_params=...) (écrits dans meta.txt).
NOISE_PARAMS_DEFAULT = {"noise_beta": NOISE_BETA_DEFAULT, "noise_psi": NOISE_PSI_DEFAULT,
                        "noise_market": NOISE_MARKET_DEFAULT, "p_shock": P_SHOCK_DEFAULT, "kappa": KAPPA_DEFAULT}


@dataclass
class ScenarioSet:
    """N scénarios de covariance dérivés d'une instance mère."""

    N: int
    beta: np.ndarray  # (N, n)
    psi: np.ndarray  # (N, n)
    sigma_f: np.ndarray  # (N,) -- varie si include_market_regime (étage (c))
    Sigma: np.ndarray  # (N, n, n)


def generate_scenarios(
    rng: np.random.Generator,
    mother: MotherInstance,
    N: int,
    noise_beta: float = NOISE_BETA_DEFAULT,
    noise_psi: float = NOISE_PSI_DEFAULT,
    noise_market: float = NOISE_MARKET_DEFAULT,
    p_shock: float = P_SHOCK_DEFAULT,
    kappa: float = KAPPA_DEFAULT,
    include_market_regime: bool = True,
    include_shocks: bool = True,
) -> ScenarioSet:
    """
    Génère N scénarios de covariance par perturbation des facteurs de `mother`,
    étages (a)(b)(c)(d) (cf. docstring du module).

    `include_market_regime`/`include_shocks` permettent de désactiver (c)/(d)
    individuellement — réservé aux tests d'ablation, pas à un usage en production.
    """
    n = mother.n
    delta = rng.normal(0.0, noise_beta, size=(N, n))
    beta_j = mother.beta[None, :] * (1.0 + delta)

    eta = rng.normal(0.0, noise_psi, size=(N, n))
    psi_j = mother.psi[None, :] * np.exp(eta)

    if include_shocks:
        shock = rng.random((N, n)) < p_shock
        psi_j = psi_j * np.where(shock, kappa, 1.0)

    if include_market_regime:
        zeta = rng.normal(0.0, noise_market, size=N)
        sigma_f_j = mother.sigma_f * np.exp(zeta)
    else:
        sigma_f_j = np.full(N, mother.sigma_f, dtype=np.float64)

    Sigma_j = np.empty((N, n, n), dtype=np.float64)
    for j in range(N):
        Sigma_j[j] = sigma_f_j[j] ** 2 * np.outer(beta_j[j], beta_j[j]) + np.diag(psi_j[j])

    return ScenarioSet(N=N, beta=beta_j, psi=psi_j, sigma_f=sigma_f_j, Sigma=Sigma_j)
