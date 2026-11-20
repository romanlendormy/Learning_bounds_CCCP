"""
Tests de l'écriture texte : arborescence, manifest, gzip.

Couvre le déterminisme (y compris `--gzip`, dont l'en-tête contient un mtime par
défaut), le round-trip complet et le parsing
croisé (readers.py est écrit indépendamment de writers.py, ne réutilise que le format).
"""

from __future__ import annotations

import numpy as np
import pytest

from generators.ccpo.instance import draw_mother_instance
from generators.ccpo.scenarios import generate_scenarios
from generators.ccpo.discretize import discretize_instance
from generators.ccpo.writers import write_episode
from generators.ccpo.readers import read_instance_txt, read_scenarios_txt
from generators.ccpo.txt_format import epsilon_to_fraction, saa_threshold

NOISE_PARAMS = {
    "noise_beta": 0.10,
    "noise_psi": 0.10,
    "noise_market": 0.15,
    "p_shock": 0.05,
    "kappa": 5.0,
}


def _build_discretized(n=10, N=20, seed=0, Q=100, k=5, L=5, U=20):
    mother = draw_mother_instance(np.random.default_rng(seed), n=n)
    sc = generate_scenarios(np.random.default_rng(seed + 1), mother, N=N)
    di = discretize_instance(mother, sc, Q=Q, k=k, L=L, U=U)
    return mother, di


# ---------------------------------------------------------------------------
# epsilon <-> fraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "epsilon,expected",
    [(0.1, (1, 10)), (0.05, (1, 20)), (0.2, (1, 5))],
)
def test_epsilon_to_fraction(epsilon, expected):
    assert epsilon_to_fraction(epsilon) == expected


def test_epsilon_to_fraction_rejects_out_of_range():
    with pytest.raises(ValueError):
        epsilon_to_fraction(0.0)
    with pytest.raises(ValueError):
        epsilon_to_fraction(1.0)


def test_saa_threshold():
    eps_num, eps_den = epsilon_to_fraction(0.1)
    assert saa_threshold(eps_num, eps_den, N=50) == 45  # ceil(0.9*50)=45
    eps_num, eps_den = epsilon_to_fraction(0.05)
    assert saa_threshold(eps_num, eps_den, N=50) == 48  # ceil(0.95*50)=47.5 -> 48


# ---------------------------------------------------------------------------
# Déterminisme (avec et sans --gzip)
# ---------------------------------------------------------------------------


def test_write_episode_is_byte_identical_across_runs(tmp_path):
    mother, di = _build_discretized()

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    for d in (dir_a, dir_b):
        write_episode(
            d, ep_id=0, mother=mother, discretized=di, epsilon=0.1,
            seed_master=42, n_episodes=1, noise_params=NOISE_PARAMS,
        )

    for fname in ("instance.txt", "scenarios.txt", "meta.txt"):
        bytes_a = (dir_a / "ep_000000" / fname).read_bytes()
        bytes_b = (dir_b / "ep_000000" / fname).read_bytes()
        assert bytes_a == bytes_b, f"{fname} diffère entre deux runs à seed identique"


def test_write_episode_gzip_is_byte_identical_across_runs(tmp_path):
    """Test du piège identifié : sans mtime=0, deux .gz à contenu identique diffèrent."""
    mother, di = _build_discretized()

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    for d in (dir_a, dir_b):
        write_episode(
            d, ep_id=0, mother=mother, discretized=di, epsilon=0.1,
            seed_master=42, n_episodes=1, noise_params=NOISE_PARAMS,
            gzip_scenarios=True,
        )

    gz_a = (dir_a / "ep_000000" / "scenarios.txt.gz").read_bytes()
    gz_b = (dir_b / "ep_000000" / "scenarios.txt.gz").read_bytes()
    assert gz_a == gz_b, "scenarios.txt.gz diffère entre deux runs -- mtime non forcé à 0 ?"


# ---------------------------------------------------------------------------
# Round-trip complet
# ---------------------------------------------------------------------------


def test_roundtrip_instance_txt(tmp_path):
    mother, di = _build_discretized()
    ep_dir = write_episode(
        tmp_path, ep_id=0, mother=mother, discretized=di, epsilon=0.1,
        seed_master=1, n_episodes=1, noise_params=NOISE_PARAMS,
    )
    parsed = read_instance_txt(ep_dir / "instance.txt")
    assert parsed["n"] == di.n
    assert parsed["N"] == di.N
    assert parsed["k"] == di.k
    assert parsed["Q"] == di.Q
    assert parsed["L"] == di.L
    assert parsed["U"] == di.U
    assert (parsed["epsilon_num"], parsed["epsilon_den"]) == epsilon_to_fraction(0.1)
    assert parsed["B"] == di.B
    np.testing.assert_array_equal(parsed["c_int"], di.c_int)


def test_roundtrip_scenarios_txt(tmp_path):
    mother, di = _build_discretized(n=12, N=30)
    ep_dir = write_episode(
        tmp_path, ep_id=0, mother=mother, discretized=di, epsilon=0.1,
        seed_master=1, n_episodes=1, noise_params=NOISE_PARAMS,
    )
    psi_int, Sigma_int = read_scenarios_txt(ep_dir / "scenarios.txt", n=di.n, N=di.N)
    np.testing.assert_array_equal(psi_int, di.psi_int)
    np.testing.assert_array_equal(Sigma_int, di.Sigma_int)


def test_roundtrip_scenarios_txt_gzip(tmp_path):
    mother, di = _build_discretized(n=8, N=15)
    ep_dir = write_episode(
        tmp_path, ep_id=0, mother=mother, discretized=di, epsilon=0.1,
        seed_master=1, n_episodes=1, noise_params=NOISE_PARAMS,
        gzip_scenarios=True,
    )
    psi_int, Sigma_int = read_scenarios_txt(ep_dir / "scenarios.txt.gz", n=di.n, N=di.N)
    np.testing.assert_array_equal(psi_int, di.psi_int)
    np.testing.assert_array_equal(Sigma_int, di.Sigma_int)


def test_roundtrip_rejects_truncated_scenarios_file(tmp_path):
    mother, di = _build_discretized(n=8, N=15)
    ep_dir = write_episode(
        tmp_path, ep_id=0, mother=mother, discretized=di, epsilon=0.1,
        seed_master=1, n_episodes=1, noise_params=NOISE_PARAMS,
    )
    path = ep_dir / "scenarios.txt"
    truncated = "\n".join(path.read_text().splitlines()[:-3]) + "\n"
    path.write_text(truncated)
    with pytest.raises((ValueError, IndexError)):
        read_scenarios_txt(path, n=di.n, N=di.N)


# ---------------------------------------------------------------------------
# manifest.txt
# ---------------------------------------------------------------------------


def test_manifest_has_one_line_per_episode(tmp_path):
    for ep_id in range(3):
        mother, di = _build_discretized(seed=ep_id)
        write_episode(
            tmp_path, ep_id=ep_id, mother=mother, discretized=di, epsilon=0.1,
            seed_master=100 + ep_id, n_episodes=3, noise_params=NOISE_PARAMS,
        )
    manifest_lines = (tmp_path / "manifest.txt").read_text().strip("\n").split("\n")
    assert len(manifest_lines) == 3
    for ep_id, line in enumerate(manifest_lines):
        fields = line.split()
        assert int(fields[0]) == ep_id
        assert int(fields[-1]) == 100 + ep_id  # seed_master en dernière colonne
