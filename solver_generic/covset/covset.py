"""
solver_generic/covset/covset.py — Loader pour les instances CCSP (covering set).

Format CSV (identique à solver_cpp2/instance_reader.hpp) :
    [episode, scenario, m, n, pA, pB, A_flat(m*n), b_flat(m)]
    c ∈ R^n non inclus → ones(n) par défaut.

A et b sont binaires : A[i][j][k] ∈ {0,1}, b[i][j] ∈ {0,1}.
"""

from __future__ import annotations
from collections import defaultdict
import numpy as np


class CCSPInstance:
    """Instance chance-contrainte de covering set."""
    __slots__ = ["episode", "n", "m", "N", "epsilon", "pA", "pB", "c", "p", "A", "b"]
    # A : (N, m, n) float64 binaire
    # b : (N, m)    float64 binaire
    # c : (n,)      déterministe (ones par défaut)
    # p : (N,)      probabilités uniformes


def load_covset_instances(
    filepath: str,
    epsilon: float = 0.1,
    max_episodes: int = -1,
    c_override: np.ndarray | None = None,
) -> list[CCSPInstance]:
    """Charge les instances CCSP depuis le CSV.

    Paramètres
    ----------
    filepath     : chemin vers le fichier CSV
    epsilon      : probabilité de violation admissible
    max_episodes : -1 = tout charger
    c_override   : vecteur c déterministe à utiliser (None → ones(n))
    """
    raw: dict[int, list] = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = list(map(float, line.split(",")))
            ep_id = int(vals[0])
            if max_episodes > 0 and ep_id not in raw and len(raw) >= max_episodes:
                continue
            raw[ep_id].append(vals)

    instances = []
    for ep_id in sorted(raw.keys()):
        rows  = raw[ep_id]
        first = rows[0]
        m = int(first[2]); n = int(first[3])
        pA = first[4];     pB = first[5]
        N  = len(rows)

        expected = 6 + m * n + m
        assert len(first) == expected, (
            f"ep {ep_id}: {len(first)} valeurs reçues, {expected} attendues "
            f"(m={m}, n={n})"
        )

        c_vec = c_override if c_override is not None else np.ones(n)
        p_vec = np.full(N, 1.0 / N)

        A_arr = np.zeros((N, m, n))
        b_arr = np.zeros((N, m))
        for i, row in enumerate(rows):
            A_arr[i] = np.array(row[6 : 6 + m * n]).reshape(m, n)
            b_arr[i] = np.array(row[6 + m * n : 6 + m * n + m])

        inst = CCSPInstance()
        inst.episode = ep_id
        inst.n = n; inst.m = m; inst.N = N
        inst.epsilon = epsilon; inst.pA = pA; inst.pB = pB
        inst.c = c_vec.astype(np.float64)
        inst.p = p_vec
        inst.A = A_arr
        inst.b = b_arr
        instances.append(inst)

    return instances


def covset_to_cpp_v1ld(inst: CCSPInstance):
    """Convertit en v1ld_generic_cpp.CovsetProblem (v1LD, DP exact)."""
    import v1ld_generic_cpp as cpp
    return cpp.CovsetProblem(
        inst.c.astype(np.float64),
        inst.p.astype(np.float64),
        inst.A.astype(np.float64),
        inst.b.astype(np.float64),
        inst.epsilon,
    )


def covset_to_cpp_v1cld(inst: CCSPInstance):
    """Convertit en v1ld_generic_cpp.CovsetCLDProblem (v1CLD, glouton 1-contrainte)."""
    import v1ld_generic_cpp as cpp
    return cpp.CovsetCLDProblem(
        inst.c.astype(np.float64),
        inst.p.astype(np.float64),
        inst.A.astype(np.float64),
        inst.b.astype(np.float64),
        inst.epsilon,
    )
