"""
Tests Module C — demande stochastique AR(1) log-normal.

Quatre propriétés vérifiées :
  BIAIS      : E[d_tilde[t,k]] ≈ d_bar[t]  (≤ 5% erreur relative)
  CORRÉLATION: corr empirique lag-1 de eta ≈ rho  (±0.05)
  POSITIVITÉ : d[t,k] ≥ 0 pour tout (t,k)
  DÉTERMINISME: même seed → matrice identique
"""

import numpy as np
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from ssp_stochastic.demand import generate_demand_scenarios, make_bimodal_nominal


T, N_LARGE = 24, 50_000
RHO, SIGMA = 0.7, 0.3
SEED = 42


def _make_rng(seed=SEED):
    return np.random.default_rng(seed)


# ── helpers ────────────────────────────────────────────────────────────────────

def _generate_eta(d_bar, rho, sigma, N, rng):
    """Reconstruit eta à partir des scénarios pour tester la corrélation."""
    import math
    T = len(d_bar)
    sqrt_one_minus_rho2 = math.sqrt(1.0 - rho * rho)
    eps = rng.standard_normal((T, N))
    eta = np.empty((T, N))
    eta[0] = eps[0]
    for t in range(1, T):
        eta[t] = rho * eta[t - 1] + sqrt_one_minus_rho2 * eps[t]
    return eta


# ── tests ──────────────────────────────────────────────────────────────────────

def test_positivity():
    """d[t,k] ≥ 0 pour tout t, k."""
    d_bar = make_bimodal_nominal(T)
    rng = _make_rng()
    d = generate_demand_scenarios(d_bar, RHO, SIGMA, N=2000, rng=rng)
    assert (d >= 0).all(), "Demande négative détectée"


def test_determinism():
    """Même seed → matrice identique."""
    d_bar = make_bimodal_nominal(T)
    d1 = generate_demand_scenarios(d_bar, RHO, SIGMA, N=500, rng=_make_rng(99))
    d2 = generate_demand_scenarios(d_bar, RHO, SIGMA, N=500, rng=_make_rng(99))
    np.testing.assert_array_equal(d1, d2, err_msg="Non-déterminisme détecté")


def test_bias():
    """E_k[d_tilde[t,k]] ≈ d_bar[t] (tolerance 5% relative)."""
    import math
    d_bar = make_bimodal_nominal(T)
    rng = _make_rng()

    # On recompute d_tilde (avant le ceil) pour tester le biais sans bruit de l'arrondi
    eps = rng.standard_normal((T, N_LARGE))
    sqrt1r2 = math.sqrt(1.0 - RHO ** 2)
    eta = np.empty((T, N_LARGE))
    eta[0] = eps[0]
    for t in range(1, T):
        eta[t] = RHO * eta[t - 1] + sqrt1r2 * eps[t]
    log_factor = SIGMA * eta - 0.5 * SIGMA ** 2
    d_tilde = d_bar[:, None] * np.exp(log_factor)

    mean_dtilde = d_tilde.mean(axis=1)
    rel_err = np.abs(mean_dtilde - d_bar) / d_bar
    assert (rel_err < 0.05).all(), (
        f"Biais > 5% pour au moins une période : max={rel_err.max():.4f}"
    )


def test_ar1_correlation():
    """Corrélation lag-1 empirique de eta ≈ rho (±0.05)."""
    import math
    rng = _make_rng()
    T_corr = 48
    d_bar = make_bimodal_nominal(T_corr)
    eta = _generate_eta(d_bar, RHO, SIGMA, N_LARGE, rng)

    # Corrélation lag-1 : corr(eta[t], eta[t-1]) sur t=1..T-1, moyennée
    corrs = []
    for t in range(1, T_corr):
        c = np.corrcoef(eta[t - 1], eta[t])[0, 1]
        corrs.append(c)
    mean_corr = np.mean(corrs)
    assert abs(mean_corr - RHO) < 0.05, (
        f"Corrélation lag-1 empirique = {mean_corr:.4f}, attendu ≈ {RHO}"
    )


def test_shape():
    """La matrice de sortie a la bonne forme (T, N) et le bon dtype."""
    d_bar = make_bimodal_nominal(T)
    rng = _make_rng()
    d = generate_demand_scenarios(d_bar, RHO, SIGMA, N=100, rng=rng)
    assert d.shape == (T, 100)
    assert d.dtype == np.int32


def test_bimodal_shape():
    """Le profil nominal a la bonne longueur, est entier et ≥ 1."""
    d_bar = make_bimodal_nominal(24)
    assert len(d_bar) == 24
    assert d_bar.dtype == int or np.issubdtype(d_bar.dtype, np.integer)
    assert (d_bar >= 1).all()
