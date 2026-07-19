"""
loader.py — Chargement d'un épisode depuis dataset.h5 + graphe bipartite pour GNN.

Interface principale
--------------------
    with open_dataset("dataset.h5") as h5f:
        instance = load_episode(h5f, ep_id=0)
        graph    = build_bipartite_graph(instance)

    # ou raccourci (ouvre/ferme le fichier à chaque appel)
    instance = load_episode_from_file("dataset.h5", ep_id=0)

Classe Instance
---------------
Regroupe l'automate, les coûts, la demande in-sample, la demande OOS,
et tous les paramètres scalaires de l'instance.

build_bipartite_graph
---------------------
Construit une représentation graphe bipartite variable↔contrainte inspirée de
Gasse et al. (2019), adapté au SSP chance-constrained.

Nœuds variables  : x[e,t,σ] ∈ {0,1}  (|Σ|=4 activités, E*T*|Σ| variables)
                   z[k] ∈ {0,1}       (N variables de couverture de scénario)
Nœuds contraintes: C_auto[e]          (regular constraint par employé)
                   C_cov[t,k]         (y[t] ≥ d[t,k]*z[k])
                   C_quantile         (Σ_k z[k] ≥ ceil((1-ε)*N))

Arêtes (pondérées) :
  x[e,t,σ] — C_auto[e]   : poids WORK_IND(σ)
  x[e,t,W] — C_cov[t,k]  : poids 1/E (contribution normalisée à y[t])
  z[k]     — C_cov[t,k]  : poids -d[t,k]
  z[k]     — C_quantile  : poids 1
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .automaton import DFA
from .demand import generate_demand_scenarios
from .writers import open_dataset, read_episode


# ── dataclass Instance ────────────────────────────────────────────────────────

@dataclass
class Instance:
    """Représentation complète d'un épisode SSP chance-constrained."""
    ep_id: int
    T: int
    E: int
    N: int
    epsilon: float
    quantile: int                   # ceil((1-ε)*N)

    dfa: DFA                        # automate des quarts légaux
    salary: np.ndarray              # (T,) float64 — coûts salariaux
    demand: np.ndarray              # (T, N) int32 — scénarios in-sample
    d_bar: np.ndarray               # (T,) int32 — demande nominale
    rho: float
    sigma_dem: float

    meta: dict = field(default_factory=dict)
    # demand_oos n'est pas stocké ; utiliser regenerate_oos(inst, K)


# ── chargement d'un épisode ───────────────────────────────────────────────────

def load_episode(h5f, ep_id: int) -> Instance:
    """
    Charge un épisode depuis un fichier HDF5 déjà ouvert.

    Paramètres
    ----------
    h5f   : fichier h5py ouvert (via open_dataset ou h5py.File)
    ep_id : identifiant de l'épisode

    Retourne
    --------
    Instance complète
    """
    data = read_episode(h5f, ep_id)
    return Instance(
        ep_id=data["ep_id"],
        T=data["T"],
        E=data["E"],
        N=data["N"],
        epsilon=data["epsilon"],
        quantile=data["quantile"],
        dfa=data["dfa"],
        salary=data["salary"],
        demand=data["demand"],
        d_bar=data["d_bar"],
        rho=data["rho"],
        sigma_dem=data["sigma_dem"],
        meta=data["meta"],
    )


def load_episode_from_file(dataset_path: str | Path, ep_id: int) -> Instance:
    """
    Raccourci : ouvre dataset.h5, charge l'épisode, ferme le fichier.

    À utiliser pour charger un épisode isolément ; préférer open_dataset +
    load_episode en boucle pour charger plusieurs épisodes (évite les ouvertures
    répétées).
    """
    with open_dataset(dataset_path, mode="r") as h5f:
        return load_episode(h5f, ep_id)


# ── régénération OOS ─────────────────────────────────────────────────────────

def regenerate_oos(inst: Instance, K: int) -> np.ndarray:
    """
    Régénère K scénarios out-of-sample pour une instance.

    Utilise la même chaîne de dérivation que le générateur :
        SeedSequence(seed_episode).spawn(3)[1]  →  rng_oos

    Paramètres
    ----------
    inst : Instance chargée via load_episode / load_episode_from_file
    K    : nombre de scénarios OOS souhaités

    Retourne
    --------
    demand_oos : (T, K) int32
    """
    seed_ep = inst.meta.get("seed_episode")
    if seed_ep is None:
        raise ValueError(
            "seed_episode absent de inst.meta — épisode généré sans seed stocké."
        )
    seq = np.random.SeedSequence(int(seed_ep))
    # spawn(3) produit [rng_dem, rng_oos, rng_cost] — même ordre que generate_episode
    _rng_dem, rng_oos, _rng_cost = (np.random.default_rng(s) for s in seq.spawn(3))
    return generate_demand_scenarios(inst.d_bar, inst.rho, inst.sigma_dem, K, rng_oos)


# ── graphe bipartite pour GNN ─────────────────────────────────────────────────

def build_bipartite_graph(inst: Instance) -> dict[str, Any]:
    """
    Construit une représentation graphe bipartite variable↔contrainte.

    Structure inspirée de Gasse et al. 2019 (Learning to Branch in Mixed
    Integer Programming).  Adapté au SSP chance-constrained SAA.

    Variables (n_var = E*T*|Σ| positions x[e,t,σ] + N variables z[k])
    ------------------------------------------------------------------
    x[e,t,σ] : indicatrice binaire "employé e fait activité σ à la période t"
    Index : x[e,t,σ] = e * T * |Σ| + t * |Σ| + σ_idx
    Index z[k] = E*T*|Σ| + k

    Contraintes (n_con)
    -------------------
    C_auto[e]    : 1 par employé (regular/automate) — index 0..E-1
    C_cov[t,k]   : T*N contraintes y[t] ≥ d[t,k]*z[k] — index E..E+T*N-1
    C_quantile   : 1 contrainte globale — index E+T*N

    Arêtes
    ------
    x[e,t,σ] — C_auto[e]   : poids WORK_IND(σ)
    x[e,t,W] — C_cov[t,k]  : poids 1/E (contribution normalisée à y[t])
    z[k]     — C_cov[t,k]  : poids -d[t,k]
    z[k]     — C_quantile  : poids 1

    Retourne
    --------
    dict avec arrays numpy prêts pour un GNN PyG-style.
    """
    from .automaton import ALPHABET as _ALF, WORK_IND

    E, T, N = inst.E, inst.T, inst.N
    n_sigma = len(_ALF)
    sigma_idx = {sym: i for i, sym in enumerate(sorted(_ALF))}

    # ── index des nœuds variables ──────────────────────────────────────────────
    def x_idx(e: int, t: int, s_i: int) -> int:
        return e * T * n_sigma + t * n_sigma + s_i

    def z_idx(k: int) -> int:
        return E * T * n_sigma + k

    n_x = E * T * n_sigma
    n_z = N
    n_var = n_x + n_z

    # ── index des nœuds contraintes ───────────────────────────────────────────
    def c_auto_idx(e: int) -> int:
        return e

    def c_cov_idx(t: int, k: int) -> int:
        return E + t * N + k

    c_quantile_idx = E + T * N

    n_con = E + T * N + 1

    # ── features des nœuds (variables) ────────────────────────────────────────
    # Features x[e,t,σ] : [WORK_IND(σ), s[t]/s_max, e/E, t/T]
    s_max = float(inst.salary.max()) if inst.salary.max() > 0 else 1.0
    var_feats = np.zeros((n_var, 4), dtype=np.float32)
    for e in range(E):
        for t in range(T):
            for si, sym in enumerate(sorted(_ALF)):
                idx = x_idx(e, t, si)
                var_feats[idx, 0] = float(WORK_IND[sym])
                var_feats[idx, 1] = float(inst.salary[t]) / s_max
                var_feats[idx, 2] = float(e) / max(E - 1, 1)
                var_feats[idx, 3] = float(t) / max(T - 1, 1)
    # Features z[k] : [0, 0, 0, k/N]
    for k in range(N):
        var_feats[z_idx(k), 3] = float(k) / max(N - 1, 1)

    # ── features des nœuds (contraintes) ──────────────────────────────────────
    # Features C_auto[e]   : [type=0, e/E, 0, 0]
    # Features C_cov[t,k]  : [type=1, t/T, k/N, d_mean/s_max]
    # Features C_quantile  : [type=2, quantile/N, 0, 0]
    con_feats = np.zeros((n_con, 4), dtype=np.float32)
    for e in range(E):
        con_feats[c_auto_idx(e), 0] = 0.0
        con_feats[c_auto_idx(e), 1] = float(e) / max(E - 1, 1)

    d_mean_t = inst.demand.mean(axis=1)   # (T,)
    for t in range(T):
        for k in range(N):
            ci = c_cov_idx(t, k)
            con_feats[ci, 0] = 1.0
            con_feats[ci, 1] = float(t) / max(T - 1, 1)
            con_feats[ci, 2] = float(k) / max(N - 1, 1)
            con_feats[ci, 3] = float(d_mean_t[t]) / s_max

    con_feats[c_quantile_idx, 0] = 2.0
    con_feats[c_quantile_idx, 1] = float(inst.quantile) / N

    # ── arêtes ────────────────────────────────────────────────────────────────
    edges_src: list[int] = []
    edges_dst: list[int] = []
    edges_w: list[float] = []

    # x[e,t,σ] — C_auto[e]
    for e in range(E):
        for t in range(T):
            for si, sym in enumerate(sorted(_ALF)):
                edges_src.append(x_idx(e, t, si))
                edges_dst.append(c_auto_idx(e))
                edges_w.append(float(WORK_IND[sym]))

    # x[e,t,W] — C_cov[t,k]
    wi = sigma_idx['W']
    for e in range(E):
        for t in range(T):
            for k in range(N):
                edges_src.append(x_idx(e, t, wi))
                edges_dst.append(c_cov_idx(t, k))
                edges_w.append(1.0 / E)

    # z[k] — C_cov[t,k]
    for k in range(N):
        for t in range(T):
            edges_src.append(z_idx(k))
            edges_dst.append(c_cov_idx(t, k))
            edges_w.append(-float(inst.demand[t, k]))

    # z[k] — C_quantile
    for k in range(N):
        edges_src.append(z_idx(k))
        edges_dst.append(c_quantile_idx)
        edges_w.append(1.0)

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
    edge_attr  = np.array(edges_w, dtype=np.float32).reshape(-1, 1)

    var_type = np.concatenate([
        np.zeros(n_x, dtype=np.int32),
        np.ones(n_z, dtype=np.int32),
    ])

    return {
        "node_features_var": var_feats,
        "node_features_con": con_feats,
        "edge_index_var2con": edge_index,
        "edge_attr": edge_attr,
        "var_type": var_type,
        "meta": {
            "n_var": n_var,
            "n_con": n_con,
            "n_edges": len(edges_src),
            "E": E, "T": T, "N": N,
            "automaton_note": (
                "inst.dfa contient le graphe état/transition de l'automate. "
                "Pour l'intégration GNN : créer des nœuds 'transition' (q,sym,q') "
                "reliés à x[e,t,σ] pour chaque (e,t,σ)."
            ),
        },
    }
