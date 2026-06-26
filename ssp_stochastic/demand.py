"""
demand.py — Module C : génération de demande stochastique AR(1) log-normal.

Chaîne déterministe (seul eps[t,k] ~ N(0,1) est tiré) :
    eta[1,k]  = eps[1,k]
    eta[t,k]  = rho * eta[t-1,k] + sqrt(1-rho²) * eps[t,k]   # AR(1) stationnaire
    d̃[t,k]   = d_bar[t] * exp(sigma * eta[t,k] - 0.5*sigma²)  # correction de biais
    d[t,k]    = max(0, ceil(d̃[t,k]))                           # entier ≥ 0

Le nominal d_bar est un profil bimodal (deux gaussiennes) sur {1..T}.
"""

from __future__ import annotations
import math
import numpy as np


def make_bimodal_nominal(
    T: int,
    peak1_pos: float = 0.35,
    peak2_pos: float = 0.75,
    peak1_amp: float = 6.0,
    peak2_amp: float = 5.0,
    sigma_g: float = 0.12,
    base: float = 1.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Profil bimodal d_bar ∈ R^T, entiers ≥ 1.

    Si rng fourni, jitter léger sur les amplitudes et positions pour varier entre épisodes.
    """
    t = np.arange(1, T + 1) / T  # positions normalisées dans [1/T, 1]

    if rng is not None:
        p1 = peak1_pos + rng.uniform(-0.05, 0.05)
        p2 = peak2_pos + rng.uniform(-0.05, 0.05)
        a1 = peak1_amp * rng.uniform(0.8, 1.2)
        a2 = peak2_amp * rng.uniform(0.8, 1.2)
    else:
        p1, p2, a1, a2 = peak1_pos, peak2_pos, peak1_amp, peak2_amp

    g1 = a1 * np.exp(-0.5 * ((t - p1) / sigma_g) ** 2)
    g2 = a2 * np.exp(-0.5 * ((t - p2) / sigma_g) ** 2)
    raw = base + g1 + g2
    return np.maximum(1, np.round(raw)).astype(int)


def generate_demand_scenarios(
    d_bar: np.ndarray,
    rho: float,
    sigma_dem: float,
    N: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Génère N scénarios de demande entière ≥ 0.

    Paramètres
    ----------
    d_bar     : vecteur nominal (T,) int ou float
    rho       : autocorrélation AR(1) dans [0, 1)
    sigma_dem : écart-type log-normal
    N         : nombre de scénarios
    rng       : np.random.Generator (numpy Generator, jamais random global)

    Retourne
    --------
    d : (T, N) int32, d[t, k] = demande période t scénario k
    """
    T = len(d_bar)
    sqrt_one_minus_rho2 = math.sqrt(1.0 - rho * rho)

    # eps[t, k] ~ N(0, 1) i.i.d., forme (T, N)
    eps = rng.standard_normal((T, N))

    # AR(1) stationnaire
    eta = np.empty((T, N), dtype=float)
    eta[0] = eps[0]
    for t in range(1, T):
        eta[t] = rho * eta[t - 1] + sqrt_one_minus_rho2 * eps[t]

    # Log-normal bias-corrigé : E[exp(sigma*eta - 0.5*sigma²)] = 1
    log_factor = sigma_dem * eta - 0.5 * sigma_dem ** 2
    d_tilde = d_bar[:, None] * np.exp(log_factor)  # (T, N)

    # Arrondi vers le haut, plancher à 0
    d = np.maximum(0, np.ceil(d_tilde)).astype(np.int32)
    return d
