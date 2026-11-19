"""
Tests de la discrétisation entière et des checks bloquants 1-5.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccpo_stochastic.instance import draw_mother_instance
from ccpo_stochastic.scenarios import generate_scenarios
from ccpo_stochastic.discretize import (
    check_structural_feasibility,
    discretize_instance,
)


def _setup(n=20, N=50, seed=0):
    mother = draw_mother_instance(np.random.default_rng(seed), n=n)
    sc = generate_scenarios(np.random.default_rng(seed + 1), mother, N=N)
    return mother, sc


# ---------------------------------------------------------------------------
# Check 4 -- faisabilité structurelle
# ---------------------------------------------------------------------------


def test_structural_feasibility_ok():
    check_structural_feasibility(Q=100, k=10, L=5, U=20)  # 50 <= 100 <= 200


def test_structural_feasibility_raises_when_kL_exceeds_Q():
    with pytest.raises(ValueError):
        check_structural_feasibility(Q=20, k=10, L=5, U=20)  # k*L=50 > Q=20


def test_structural_feasibility_raises_when_Q_exceeds_kU():
    with pytest.raises(ValueError):
        check_structural_feasibility(Q=300, k=10, L=5, U=20)  # Q=300 > k*U=200


def test_discretize_rejects_infeasible_QkLU():
    mother, sc = _setup()
    with pytest.raises(ValueError):
        discretize_instance(mother, sc, Q=20, k=10, L=5, U=20)


# ---------------------------------------------------------------------------
# Discrétisation nominale (petite grille de départ)
# ---------------------------------------------------------------------------


def test_discretize_success_default_grid():
    mother, sc = _setup(n=20, N=50)
    di = discretize_instance(mother, sc, Q=100, k=10, L=5, U=20)
    assert di.c_int.shape == (20,)
    assert di.c_int.dtype == np.int64
    assert di.Sigma_int.shape == (50, 20, 20)
    assert di.Sigma_int.dtype == np.int64
    assert di.psi_int.shape == (50, 20)
    assert isinstance(di.B, int)
    assert di.B > 0


def test_c_int_matches_floor_S_c_times_c():
    mother, sc = _setup()
    di = discretize_instance(mother, sc, Q=100, k=10, L=5, U=20)
    expected = np.floor(di.S_c * mother.c).astype(np.int64)
    np.testing.assert_array_equal(di.c_int, expected)


def test_B_uses_mother_sigma_bar_not_scenario_sigma_bar():
    """B doit être calculé sur Sigma_bar de la MERE, commun à tous les scénarios."""
    mother, sc = _setup()
    di = discretize_instance(mother, sc, Q=100, k=10, L=5, U=20, gamma=1.2)
    expected_B = int(np.floor(di.S * 1.2 * mother.Sigma_bar)) * 100**2
    assert di.B == expected_B


def test_psi_int_positive_and_matches_definition():
    mother, sc = _setup(n=20, N=200)
    di = discretize_instance(mother, sc, Q=100, k=10, L=5, U=20)
    assert np.all(di.psi_int > 0)

    diag_int = np.diagonal(di.Sigma_int, axis1=1, axis2=2)
    rank1_diag_int = np.floor(di.S * (sc.sigma_f[:, None] ** 2) * (sc.beta**2)).astype(np.int64)
    np.testing.assert_array_equal(di.psi_int, diag_int - rank1_diag_int)


def test_symmetric_and_diagonal_strictly_positive():
    mother, sc = _setup(n=15, N=100)
    di = discretize_instance(mother, sc, Q=100, k=10, L=5, U=15)
    np.testing.assert_array_equal(di.Sigma_int, np.transpose(di.Sigma_int, (0, 2, 1)))
    diag_int = np.diagonal(di.Sigma_int, axis1=1, axis2=2)
    assert np.all(diag_int > 0)


def test_determinism():
    mother, sc = _setup()
    a = discretize_instance(mother, sc, Q=100, k=10, L=5, U=20)
    b = discretize_instance(mother, sc, Q=100, k=10, L=5, U=20)
    np.testing.assert_array_equal(a.Sigma_int, b.Sigma_int)
    np.testing.assert_array_equal(a.psi_int, b.psi_int)
    np.testing.assert_array_equal(a.c_int, b.c_int)
    assert a.B == b.B


# ---------------------------------------------------------------------------
# Check 5 -- non-débordement
# ---------------------------------------------------------------------------


def test_overflow_raises_with_S_too_large_for_Q():
    mother, sc = _setup(n=20, N=20)
    # Sigma_int.max() ~ S * max|Sigma| ; avec S=1e12 et Q=100, on dépasse 2**30 de loin.
    with pytest.raises(ValueError):
        discretize_instance(mother, sc, Q=100, k=10, L=5, U=20, S=10**12)


def test_no_overflow_at_recommended_grid():
    """Grille cible (Q=100, S=1e6) : ne doit PAS déborder."""
    mother, sc = _setup(n=20, N=50)
    di = discretize_instance(mother, sc, Q=100, k=10, L=5, U=20, S=10**6)
    assert int(np.abs(di.Sigma_int).max()) * di.Q**2 < 2**30


# ---------------------------------------------------------------------------
# Cohérence rang-1 (test seulement, pas un check bloquant de discretize.py)
# ---------------------------------------------------------------------------


def test_rank1_consistency_within_rounding_tolerance():
    """
    Sigma_int[j] - diag(psi_int[j]) doit être de rang 1 à la tolérance d'arrondi près :
    deuxième valeur propre petite devant la première.
    """
    mother, sc = _setup(n=20, N=20)
    di = discretize_instance(mother, sc, Q=100, k=10, L=5, U=20)

    for j in range(di.N):
        rank1_part = di.Sigma_int[j] - np.diag(di.psi_int[j])
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(rank1_part.astype(np.float64))))[::-1]
        # tolérance dérivée de l'arrondi (S) plutôt que de la théorie exacte : chaque
        # entrée porte une erreur d'arrondi indépendante d'ordre 1, donc la 2e valeur
        # propre reste petite devant la 1re mais pas nulle.
        assert eigvals[1] < 0.05 * eigvals[0], (
            f"scénario {j}: lambda2/lambda1={eigvals[1] / eigvals[0]:.4f} trop grand"
        )
