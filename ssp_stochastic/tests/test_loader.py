"""
Tests writers HDF5 + loader — round-trip écrire→lire identique.
"""

import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import numpy as np
import pytest

from ssp_stochastic.automaton import AutomatonParams, build_shift_dfa, ALPHABET
from ssp_stochastic.cost import generate_cost
from ssp_stochastic.demand import generate_demand_scenarios, make_bimodal_nominal
from ssp_stochastic.calibrate import calibrate_E
from ssp_stochastic.writers import open_dataset, write_episode, read_episode, list_episodes
from ssp_stochastic.loader import (
    load_episode, load_episode_from_file, build_bipartite_graph, regenerate_oos,
)

T = 10; N = 15; K_OOS = 30; EPS = 0.1

PARAMS = AutomatonParams(L_min=4, L_max=7, m=3, n_P=1, meal_len=1,
                          a_min=1, b_max=5, win_start=2, win_end=6)


def _build_episode(ep_id=0):
    rng = np.random.default_rng(ep_id * 100 + 1)
    dfa = build_shift_dfa(PARAMS)
    d_bar = make_bimodal_nominal(T)
    demand = generate_demand_scenarios(d_bar, rho=0.5, sigma_dem=0.2, N=N, rng=rng)
    salary, cost_params = generate_cost(T, np.random.default_rng(ep_id * 100 + 3))
    E, calib_meta = calibrate_E(demand, EPS, tightness_factor=1.1, dfa=dfa, T=T)
    ep_seed = ep_id * 100 + 99   # graine pour OOS
    meta = {**cost_params, **calib_meta, "ep_id": ep_id, "T": T, "N": N,
            "K_oos": K_OOS, "seed_episode": ep_seed}
    return dfa, salary, demand, d_bar, E, meta, ep_seed


def _write(h5_path, ep_id=0):
    dfa, salary, demand, d_bar, E, meta, _ = _build_episode(ep_id)
    with open_dataset(h5_path, mode="w") as h5f:
        write_episode(h5f, ep_id, dfa, salary, demand, d_bar,
                      E, T, N, EPS, 0.5, 0.2, meta)
    return dfa, salary, demand, d_bar, E, meta


# ── DFA round-trip ─────────────────────────────────────────────────────────────

def test_dfa_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = pathlib.Path(f.name)
    try:
        dfa, *_ = _write(h5_path)
        with open_dataset(h5_path) as h5f:
            data = read_episode(h5f, 0)
        dfa2 = data["dfa"]
        assert dfa.num_states == dfa2.num_states
        assert dfa.initial    == dfa2.initial
        assert dfa.accepting  == dfa2.accepting
        assert dfa.delta      == dfa2.delta
    finally:
        h5_path.unlink(missing_ok=True)


# ── salary round-trip ──────────────────────────────────────────────────────────

def test_salary_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = pathlib.Path(f.name)
    try:
        _, salary, *_ = _write(h5_path)
        with open_dataset(h5_path) as h5f:
            data = read_episode(h5f, 0)
        np.testing.assert_allclose(salary, data["salary"], rtol=1e-6)
    finally:
        h5_path.unlink(missing_ok=True)


# ── demand round-trip ─────────────────────────────────────────────────────────

def test_demand_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = pathlib.Path(f.name)
    try:
        _, _, demand, d_bar, *_ = _write(h5_path)
        with open_dataset(h5_path) as h5f:
            data = read_episode(h5f, 0)
        np.testing.assert_array_equal(demand, data["demand"])
        np.testing.assert_array_equal(d_bar,  data["d_bar"])
        assert abs(data["rho"] - 0.5) < 1e-6
        assert abs(data["sigma_dem"] - 0.2) < 1e-6
    finally:
        h5_path.unlink(missing_ok=True)


# ── demand_oos non stocké ─────────────────────────────────────────────────────

def test_demand_oos_not_stored():
    """demand_oos ne doit pas être présent dans le HDF5."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = pathlib.Path(f.name)
    try:
        _write(h5_path)
        import h5py
        with h5py.File(h5_path, "r") as h5f:
            assert "demand_oos" not in h5f["ep_00000"]
    finally:
        h5_path.unlink(missing_ok=True)


# ── regenerate_oos ────────────────────────────────────────────────────────────

def test_regenerate_oos_deterministic():
    """Deux appels regenerate_oos avec les mêmes params donnent le même résultat."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = pathlib.Path(f.name)
    try:
        _write(h5_path)
        inst = load_episode_from_file(h5_path, ep_id=0)
        oos1 = regenerate_oos(inst, K=50)
        oos2 = regenerate_oos(inst, K=50)
        np.testing.assert_array_equal(oos1, oos2)
        assert oos1.shape == (T, 50)
    finally:
        h5_path.unlink(missing_ok=True)


def test_regenerate_oos_matches_original():
    """OOS régénéré = OOS original produit par generate_episode."""
    from ssp_stochastic.demand import generate_demand_scenarios
    ep_seed = 99   # même que _build_episode(ep_id=0)
    seq = np.random.SeedSequence(ep_seed)
    _, rng_oos, _ = (np.random.default_rng(s) for s in seq.spawn(3))
    d_bar = make_bimodal_nominal(T)
    expected_oos = generate_demand_scenarios(d_bar, rho=0.5, sigma_dem=0.2,
                                             N=K_OOS, rng=rng_oos)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = pathlib.Path(f.name)
    try:
        _write(h5_path)
        inst = load_episode_from_file(h5_path, ep_id=0)
        oos = regenerate_oos(inst, K=K_OOS)
        np.testing.assert_array_equal(oos, expected_oos)
    finally:
        h5_path.unlink(missing_ok=True)


# ── épisode complet round-trip ────────────────────────────────────────────────

def test_episode_roundtrip():
    dfa, salary, demand, d_bar, E, meta, _ = _build_episode(ep_id=7)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = pathlib.Path(f.name)
    try:
        with open_dataset(h5_path, mode="w") as h5f:
            write_episode(h5f, 7, dfa, salary, demand, d_bar,
                          E, T, N, EPS, 0.5, 0.2, meta)
        inst = load_episode_from_file(h5_path, ep_id=7)

        assert inst.T == T
        assert inst.N == N
        assert inst.E == E
        assert abs(inst.epsilon - EPS) < 1e-6
        assert inst.ep_id == 7
        assert inst.dfa.num_states == dfa.num_states
        assert inst.dfa.accepting  == dfa.accepting
        assert inst.dfa.delta      == dfa.delta
        np.testing.assert_array_equal(inst.demand, demand)
        np.testing.assert_allclose(inst.salary, salary, rtol=1e-6)
    finally:
        h5_path.unlink(missing_ok=True)


# ── list_episodes ─────────────────────────────────────────────────────────────

def test_list_episodes():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = pathlib.Path(f.name)
    try:
        dfa, salary, demand, d_bar, E, meta, _ = _build_episode()
        with open_dataset(h5_path, mode="w") as h5f:
            for ep_id in [0, 3, 7]:
                write_episode(h5f, ep_id, dfa, salary, demand, d_bar,
                              E, T, N, EPS, 0.5, 0.2, meta)
        with open_dataset(h5_path) as h5f:
            assert list_episodes(h5f) == [0, 3, 7]
    finally:
        h5_path.unlink(missing_ok=True)


# ── build_bipartite_graph ─────────────────────────────────────────────────────

def test_bipartite_graph_shapes():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = pathlib.Path(f.name)
    try:
        _, _, _, _, E, *_ = _write(h5_path)
        inst = load_episode_from_file(h5_path, ep_id=0)
        g = build_bipartite_graph(inst)

        n_sigma = 4
        n_var = E * T * n_sigma + N
        n_con = E + T * N + 1

        assert g["node_features_var"].shape     == (n_var, 4)
        assert g["node_features_con"].shape     == (n_con, 4)
        assert g["edge_index_var2con"].shape[0] == 2
        assert g["edge_attr"].shape[1]          == 1
        assert g["var_type"].shape              == (n_var,)
        assert g["edge_index_var2con"].shape[1] == g["edge_attr"].shape[0]
    finally:
        h5_path.unlink(missing_ok=True)


def test_bipartite_graph_no_oob():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = pathlib.Path(f.name)
    try:
        _write(h5_path)
        inst = load_episode_from_file(h5_path, ep_id=0)
        g    = build_bipartite_graph(inst)
        n_var = g["node_features_var"].shape[0]
        n_con = g["node_features_con"].shape[0]
        ei    = g["edge_index_var2con"]
        assert (ei[0] >= 0).all() and (ei[0] < n_var).all()
        assert (ei[1] >= 0).all() and (ei[1] < n_con).all()
    finally:
        h5_path.unlink(missing_ok=True)
