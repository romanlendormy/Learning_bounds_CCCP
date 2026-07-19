"""
writers.py — Lecture/écriture HDF5 du dataset SSP chance-constrained.

Format : dataset.h5
  /ep_XXXXX/                 groupe par épisode (zero-padded sur 5 chiffres)
    Attributs HDF5 :         ep_id, E, T, N, K_oos, epsilon, quantile, rho, sigma_dem,
                             aut_num_states, aut_initial
                             + champs scalaires de meta (préfixés "meta_")
    Datasets :
      salary         (T,)        float64  — coûts salariaux par période
      demand         (T, N)      int32    — scénarios de demande in-sample
      demand_oos     (T, K_oos)  int32    — scénarios out-of-sample
      d_bar          (T,)        int32    — demande nominale bimodale
      aut_accepting  (n_acc,)    int32    — états acceptants du DFA
      aut_trans      (n_trans,3) int32    — transitions [src, sym_int, dst]
                                           encodage sym : O=0, P=1, R=2, W=3

Utilisation typique
-------------------
    from ssp_stochastic.writers import open_dataset, write_episode, read_episode

    # écriture
    with open_dataset("dataset.h5", mode="a") as h5f:
        write_episode(h5f, ep_id=0, dfa=dfa, salary=s, ...)

    # lecture
    with open_dataset("dataset.h5", mode="r") as h5f:
        data = read_episode(h5f, ep_id=0)
"""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .automaton import DFA, ALPHABET

# ── encodage alphabet ─────────────────────────────────────────────────────────

_ALPHA_SORTED = sorted(ALPHABET)          # ['O', 'P', 'R', 'W']
SYM_TO_INT: dict[str, int] = {s: i for i, s in enumerate(_ALPHA_SORTED)}
INT_TO_SYM: dict[int, str] = {i: s for i, s in enumerate(_ALPHA_SORTED)}


def _ep_key(ep_id: int) -> str:
    return f"ep_{ep_id:05d}"


# ── context manager ───────────────────────────────────────────────────────────

@contextmanager
def open_dataset(path: str | Path, mode: str = "r"):
    """Ouvre dataset.h5 en tant que context manager.

    mode : 'r' lecture seule, 'a' lecture+écriture (crée si absent), 'w' écrase.
    """
    with h5py.File(path, mode) as h5f:
        yield h5f


# ── écriture ──────────────────────────────────────────────────────────────────

def write_episode(
    h5f: h5py.File,
    ep_id: int,
    dfa: DFA,
    salary: np.ndarray,
    demand: np.ndarray,
    d_bar: np.ndarray,
    E: int,
    T: int,
    N: int,
    epsilon: float,
    rho: float,
    sigma_dem: float,
    meta: dict,
) -> None:
    """Écrit un épisode dans le groupe /ep_XXXXX/ du fichier HDF5 ouvert.

    demand_oos n'est PAS stocké : il est régénérable depuis seed_episode
    via loader.regenerate_oos(inst, K).
    """
    key = _ep_key(ep_id)
    if key in h5f:
        del h5f[key]
    grp = h5f.create_group(key)

    # ── arrays ─────────────────────────────────────────────────────────────────
    grp.create_dataset("salary", data=salary.astype(np.float64), compression="gzip")
    grp.create_dataset("demand", data=demand.astype(np.int32),   compression="gzip")
    grp.create_dataset("d_bar",  data=d_bar.astype(np.int32),    compression="gzip")

    # ── DFA ────────────────────────────────────────────────────────────────────
    accepting_arr = np.array(sorted(dfa.accepting), dtype=np.int32)
    grp.create_dataset("aut_accepting", data=accepting_arr)

    trans_sorted = sorted(dfa.delta.items())   # ordre déterministe
    if trans_sorted:
        trans_arr = np.array(
            [[src, SYM_TO_INT[sym], dst] for (src, sym), dst in trans_sorted],
            dtype=np.int32,
        )
    else:
        trans_arr = np.empty((0, 3), dtype=np.int32)
    grp.create_dataset("aut_trans", data=trans_arr)

    # ── attributs scalaires ────────────────────────────────────────────────────
    grp.attrs["ep_id"]          = ep_id
    grp.attrs["E"]              = E
    grp.attrs["T"]              = T
    grp.attrs["N"]              = N
    grp.attrs["epsilon"]        = float(epsilon)
    grp.attrs["quantile"]       = int(np.ceil((1.0 - epsilon) * N))
    grp.attrs["rho"]            = float(rho)
    grp.attrs["sigma_dem"]      = float(sigma_dem)
    grp.attrs["aut_num_states"] = int(dfa.num_states)
    grp.attrs["aut_initial"]    = int(dfa.initial)

    # Meta : uniquement les scalaires (int/float/str/bool)
    for k, v in meta.items():
        if isinstance(v, (bool, int, float, str, np.integer, np.floating)):
            grp.attrs[f"meta_{k}"] = v


# ── lecture ───────────────────────────────────────────────────────────────────

def read_episode(h5f: h5py.File, ep_id: int) -> dict:
    """Lit un épisode depuis un fichier HDF5 ouvert.

    Retourne un dict avec toutes les données brutes (arrays numpy + DFA + scalaires).
    """
    grp = h5f[_ep_key(ep_id)]

    # Arrays
    salary = grp["salary"][:]
    demand = grp["demand"][:]
    d_bar  = grp["d_bar"][:]

    # DFA
    accepting = frozenset(int(x) for x in grp["aut_accepting"][:])
    trans_arr = grp["aut_trans"][:]
    delta = {(int(r[0]), INT_TO_SYM[int(r[1])]): int(r[2]) for r in trans_arr}
    dfa = DFA(
        num_states=int(grp.attrs["aut_num_states"]),
        delta=delta,
        initial=int(grp.attrs["aut_initial"]),
        accepting=accepting,
    )

    # Meta (attributs préfixés "meta_")
    meta: dict = {}
    for k, v in grp.attrs.items():
        if k.startswith("meta_"):
            val = v.item() if isinstance(v, np.generic) else v
            meta[k[5:]] = val

    attrs = grp.attrs
    return {
        "ep_id":     int(attrs["ep_id"]),
        "E":         int(attrs["E"]),
        "T":         int(attrs["T"]),
        "N":         int(attrs["N"]),
        "epsilon":   float(attrs["epsilon"]),
        "quantile":  int(attrs["quantile"]),
        "rho":       float(attrs["rho"]),
        "sigma_dem": float(attrs["sigma_dem"]),
        "dfa":       dfa,
        "salary":    salary,
        "demand":    demand,
        "d_bar":     d_bar,
        "meta":      meta,
    }


def list_episodes(h5f: h5py.File) -> list[int]:
    """Retourne la liste triée des ep_id présents dans le fichier HDF5."""
    ids = []
    for key in h5f.keys():
        if key.startswith("ep_"):
            try:
                ids.append(int(key[3:]))
            except ValueError:
                pass
    return sorted(ids)
