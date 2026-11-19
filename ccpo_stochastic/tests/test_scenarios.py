"""
Tests de la perturbation en scénarios, étages (a)(b)(c)(d).

Deux tests centraux :
- `test_psd_over_many_scenarios` : 1000 scénarios tirés (tous étages), aucun échec PSD
  (check 1).
- `test_dispersion_increases_with_each_stage` : la dispersion de `q'Sigma_j q`
  (portefeuille équipondéré) doit augmenter strictement en ajoutant (c) puis (d). C'est
  le test qui justifie l'existence de (c)/(d) — cf. docstring de scenarios.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccpo_stochastic.instance import draw_mother_instance
from ccpo_stochastic.scenarios import KAPPA_DEFAULT, generate_scenarios


def _mother(n: int = 20, seed: int = 0):
    return draw_mother_instance(np.random.default_rng(seed), n=n)


def test_determinism_same_seed_same_scenarios():
    mother = _mother()
    a = generate_scenarios(np.random.default_rng(1), mother, N=50)
    b = generate_scenarios(np.random.default_rng(1), mother, N=50)
    np.testing.assert_array_equal(a.Sigma, b.Sigma)
    np.testing.assert_array_equal(a.psi, b.psi)
    np.testing.assert_array_equal(a.beta, b.beta)


def test_shapes():
    n, N = 20, 50
    mother = _mother(n=n)
    sc = generate_scenarios(np.random.default_rng(2), mother, N=N)
    assert sc.beta.shape == (N, n)
    assert sc.psi.shape == (N, n)
    assert sc.sigma_f.shape == (N,)
    assert sc.Sigma.shape == (N, n, n)


def test_psi_positive_for_all_scenarios():
    """psi_j > 0 doit être automatique (produit de psi>0 par exp(eta)>0) -- check 2."""
    mother = _mother()
    sc = generate_scenarios(np.random.default_rng(3), mother, N=1000)
    assert np.all(sc.psi > 0)


def test_symmetric_for_all_scenarios():
    mother = _mother(n=15)
    sc = generate_scenarios(np.random.default_rng(4), mother, N=200)
    for j in range(sc.N):
        np.testing.assert_array_almost_equal(sc.Sigma[j], sc.Sigma[j].T)


def test_psd_over_many_scenarios():
    """
    1000 scénarios tirés (étages (a)(b)(c)(d), coefficients par défaut), aucun échec PSD.
    """
    mother = _mother(n=20, seed=0)
    sc = generate_scenarios(np.random.default_rng(123), mother, N=1000)

    min_eigvals = np.array([np.linalg.eigvalsh(sc.Sigma[j]).min() for j in range(sc.N)])
    n_failures = int((min_eigvals <= -1e-12).sum())
    assert n_failures == 0, f"{n_failures}/1000 scénarios non PSD (min eigval={min_eigvals.min()})"


def test_sigma_f_constant_when_market_regime_disabled():
    mother = _mother()
    sc = generate_scenarios(np.random.default_rng(5), mother, N=100, include_market_regime=False)
    np.testing.assert_array_equal(sc.sigma_f, np.full(100, mother.sigma_f))


def test_sigma_f_varies_when_market_regime_enabled():
    mother = _mother()
    sc = generate_scenarios(np.random.default_rng(5), mother, N=100, include_market_regime=True)
    assert not np.allclose(sc.sigma_f, mother.sigma_f)
    assert sc.sigma_f.std() > 0


def test_shock_inflates_some_psi_entries():
    """
    Avec p_shock > 0, une fraction des entrées psi_j doit être multipliée par kappa —
    sinon (d) est un no-op silencieux.
    """
    mother = _mother(n=50)
    sc_no_shock = generate_scenarios(
        np.random.default_rng(6), mother, N=200, include_shocks=False
    )
    sc_shock = generate_scenarios(
        np.random.default_rng(6), mother, N=200, include_shocks=True
    )
    # même graine, même bruit (a)(b) consommé avant le tirage du choc : psi_j ne peut
    # différer entre les deux que par le facteur kappa/1.0 de l'étage (d).
    ratio = sc_shock.psi / sc_no_shock.psi
    assert np.any(np.isclose(ratio, KAPPA_DEFAULT))
    assert np.all(np.isclose(ratio, 1.0) | np.isclose(ratio, KAPPA_DEFAULT))


def test_dispersion_increases_with_each_stage():
    """
    La dispersion de `q'Sigma_j q` (portefeuille équipondéré) doit
    strictement augmenter en ajoutant (c) puis (d). Sans (c), les scénarios sont trop
    interchangeables (∑z_j ≥ (1-eps)N triviale) ; sans (d), eps ne fait que découper une
    queue gaussienne (cf. scenarios.py).
    """
    n = 20
    mother = _mother(n=n, seed=0)
    q = np.ones(n) / n  # portefeuille équipondéré

    def portfolio_var_std(include_market: bool, include_shocks: bool, seed: int) -> float:
        sc = generate_scenarios(
            np.random.default_rng(seed),
            mother,
            N=5000,
            include_market_regime=include_market,
            include_shocks=include_shocks,
        )
        port_var = np.einsum("i,nij,j->n", q, sc.Sigma, q)
        return float(port_var.std())

    std_ab = portfolio_var_std(False, False, seed=10)
    std_abc = portfolio_var_std(True, False, seed=11)
    std_abcd = portfolio_var_std(True, True, seed=12)

    assert std_ab < std_abc, f"std_ab={std_ab} >= std_abc={std_abc} : (c) n'ajoute pas de dispersion"
    assert std_abc < std_abcd, f"std_abc={std_abc} >= std_abcd={std_abcd} : (d) n'ajoute pas de dispersion"
