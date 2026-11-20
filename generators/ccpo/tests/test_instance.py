"""
Tests du tirage de l'instance mère.

Le test central est `test_target_statistics_over_many_draws` : sur 1000 tirages à n=20,
(rho_bar, sigma_bar, Sigma_bar) doivent tomber dans les plages cibles (check 6).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from generators.ccpo.instance import (
    MotherInstance,
    draw_mother_instance,
    _induce_rank_correlation,
    _truncated_lognormal,
)
from generators.ccpo.seeding import episode_rngs


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Corrélation de rang, sans dépendance à scipy : Pearson sur les rangs."""
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


# ---------------------------------------------------------------------------
# Déterminisme
# ---------------------------------------------------------------------------


def test_determinism_same_seed_same_arrays():
    inst_a = draw_mother_instance(_rng(42), n=20)
    inst_b = draw_mother_instance(_rng(42), n=20)
    np.testing.assert_array_equal(inst_a.sigma, inst_b.sigma)
    np.testing.assert_array_equal(inst_a.h, inst_b.h)
    np.testing.assert_array_equal(inst_a.c, inst_b.c)
    np.testing.assert_array_equal(inst_a.Sigma, inst_b.Sigma)


def test_determinism_different_seed_differs():
    inst_a = draw_mother_instance(_rng(1), n=20)
    inst_b = draw_mother_instance(_rng(2), n=20)
    assert not np.array_equal(inst_a.sigma, inst_b.sigma)


def test_episode_rngs_determinism_and_independence():
    """Même (seed_master, ep_id, n_episodes) -> même instance mère, indépendamment de N."""
    rng_nom_a, _ = episode_rngs(seed_master=123, ep_id=0, n_episodes=10)
    rng_nom_b, _ = episode_rngs(seed_master=123, ep_id=0, n_episodes=10)
    inst_a = draw_mother_instance(rng_nom_a, n=20)
    inst_b = draw_mother_instance(rng_nom_b, n=20)
    np.testing.assert_array_equal(inst_a.Sigma, inst_b.Sigma)

    # changer n_episodes ne doit pas changer le flux "nominal" de l'épisode 0
    # (spawn hiérarchique : le sous-flux nominal ne dépend que de ep_seq, pas de N)
    rng_nom_c, _ = episode_rngs(seed_master=123, ep_id=0, n_episodes=999)
    inst_c = draw_mother_instance(rng_nom_c, n=20)
    np.testing.assert_array_equal(inst_a.Sigma, inst_c.Sigma)


def test_episode_rngs_rejects_out_of_range_ep_id():
    with pytest.raises(ValueError):
        episode_rngs(seed_master=1, ep_id=5, n_episodes=5)


# ---------------------------------------------------------------------------
# Forme / positivité (check 1-3 partiels, à l'échelle de l'instance mère)
# ---------------------------------------------------------------------------


def test_shapes_and_bounds():
    n = 20
    inst = draw_mother_instance(_rng(7), n=n)
    assert inst.sigma.shape == (n,)
    assert inst.h.shape == (n,)
    assert inst.c.shape == (n,)
    assert inst.beta.shape == (n,)
    assert inst.psi.shape == (n,)
    assert inst.Sigma.shape == (n, n)

    assert np.all((inst.sigma >= 0.036) & (inst.sigma <= 0.069))
    assert np.all((inst.h > 0) & (inst.h < 1))
    assert np.all((inst.c >= 1.0e-4) & (inst.c <= 1.2e-2))
    assert np.all(inst.psi > 0)


def test_symmetric_and_psd():
    inst = draw_mother_instance(_rng(11), n=30)
    np.testing.assert_array_almost_equal(inst.Sigma, inst.Sigma.T)
    eigvals = np.linalg.eigvalsh(inst.Sigma)
    assert eigvals.min() > -1e-12


def test_psi_rejects_when_h_forced_to_one():
    """psi_i <= 0 doit lever une exception explicite (check 2, pas un warning silencieux)."""
    rng = _rng(0)
    with pytest.raises(RuntimeError):
        draw_mother_instance(rng, n=5, h_alpha=1e6, h_beta=1e-6)


# ---------------------------------------------------------------------------
# Check 6 — statistiques cibles de l'instance mère
# ---------------------------------------------------------------------------


def test_target_statistics_over_many_draws():
    """
    Sur 1000 tirages à n=20, les moyennes agrégées de rho_bar,
    sigma_bar et Sigma_bar doivent tomber dans les plages du check 6.
    """
    n_draws = 1000
    rho_bars = np.empty(n_draws)
    sigma_bars = np.empty(n_draws)
    Sigma_bars = np.empty(n_draws)
    c_bars = np.empty(n_draws)

    seed_seq = np.random.SeedSequence(2026)
    for i, s in enumerate(seed_seq.spawn(n_draws)):
        inst = draw_mother_instance(np.random.default_rng(s), n=20)
        rho_bars[i] = inst.rho_bar
        sigma_bars[i] = inst.sigma_bar
        Sigma_bars[i] = inst.Sigma_bar
        c_bars[i] = inst.c_bar

    rho_bar = rho_bars.mean()
    sigma_bar = sigma_bars.mean()
    Sigma_bar = Sigma_bars.mean()

    assert 0.45 <= rho_bar <= 0.60, f"rho_bar={rho_bar} hors [0.45, 0.60]"
    assert 0.042 <= sigma_bar <= 0.050, f"sigma_bar={sigma_bar} hors [0.042, 0.050]"
    assert 9e-4 <= Sigma_bar <= 1.4e-3, f"Sigma_bar={Sigma_bar} hors [9e-4, 1.4e-3]"

    # c_bar (cible ~3.5e-3) n'a pas de plage fixée : reporté à titre informatif seulement.
    print(
        f"[stats check 6] rho_bar={rho_bar:.4f} sigma_bar={sigma_bar:.5f} "
        f"Sigma_bar={Sigma_bar:.6f} c_bar={c_bars.mean():.5f}"
    )


# ---------------------------------------------------------------------------
# capm_alignment — question ouverte, testée pour son comportement, pas pour une valeur
# ---------------------------------------------------------------------------


def test_capm_alignment_zero_is_noop():
    rng_a = _rng(5)
    rng_b = _rng(5)
    x, y = _truncated_lognormal(rng_a, 100, 2.5e-3, 0.85, 1e-4, 1.2e-2), rng_b.beta(5.0, 4.4, size=100)
    x_out, y_out = _induce_rank_correlation(_rng(0), x, y, 0.0)
    np.testing.assert_array_equal(x, x_out)
    np.testing.assert_array_equal(y, y_out)


def test_capm_alignment_increases_rank_correlation():
    """Un rho plus grand doit induire une corrélation de rang (Spearman) plus forte."""
    rng = np.random.default_rng(3)
    c = rng.lognormal(mean=math.log(2.5e-3), sigma=0.85, size=500)
    h = rng.beta(5.0, 4.4, size=500)

    c0, h0 = _induce_rank_correlation(np.random.default_rng(1), c.copy(), h.copy(), 0.0)
    c8, h8 = _induce_rank_correlation(np.random.default_rng(1), c.copy(), h.copy(), 0.8)

    assert abs(_spearman(c8, h8)) > abs(_spearman(c0, h0))


def test_capm_alignment_rejects_out_of_range():
    with pytest.raises(ValueError):
        draw_mother_instance(_rng(0), n=5, capm_alignment=1.0)
    with pytest.raises(ValueError):
        draw_mother_instance(_rng(0), n=5, capm_alignment=-1.5)
