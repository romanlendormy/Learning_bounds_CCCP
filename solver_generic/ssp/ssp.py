"""
solver_generic/ssp.py — Loader pour les instances CCSSP (Pesant 2004).

La classe SSPProblem est implémentée en C++ (v1ld_generic_cpp.SSPProblem).
Ce fichier ne contient que la structure SSPInstance et le loader CSV.
"""

from __future__ import annotations
from collections import defaultdict
import numpy as np


class SSPInstance:
    """Instance CCSSP chargée depuis le format CSV de Pesant (2004)."""
    __slots__ = ["episode", "n", "m", "k", "Q", "N", "epsilon", "p",
                 "profit", "automata"]
    # automata : list[list[dict]]
    #   automata[i]    = liste des k automates du scénario i
    #   automata[i][j] = {"delta": (Q, m) int32, "accepting": (Q,) bool}


def load_ssp_instances(filepath: str,
                       epsilon: float = 0.1,
                       max_episodes: int = -1) -> list[SSPInstance]:
    """
    Charge les instances CCSSP depuis le format CSV de create_database_ssp_pesant.py.

    Format d'une ligne :
        episode, scenario, n, m, k, Q,
        profit[n*m],
        [delta_j[Q*m], acc_j[Q]]  pour j = 0..k-1

    Retourne une liste d'SSPInstance (une par épisode).
    """
    raw: dict[int, list] = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals  = list(map(float, line.split(",")))
            ep_id = int(vals[0])
            if max_episodes > 0 and ep_id not in raw and len(raw) >= max_episodes:
                continue
            raw[ep_id].append(vals)

    instances = []
    for ep_id in sorted(raw.keys()):
        rows  = raw[ep_id]
        first = rows[0]
        n, m, k, Q = int(first[2]), int(first[3]), int(first[4]), int(first[5])
        N = len(rows)

        expected = 6 + n * m + k * Q * (m + 1)
        assert len(first) == expected, (
            f"ep {ep_id} : {len(first)} valeurs reçues, {expected} attendues"
            f"  (n={n}, m={m}, k={k}, Q={Q})"
        )

        profit = np.array(first[6 : 6 + n * m], dtype=np.float64).reshape(n, m)

        automata: list[list[dict]] = []
        for row in rows:
            scen_auts = []
            offset    = 6 + n * m
            for j in range(k):
                delta     = np.array(row[offset : offset + Q * m],
                                     dtype=np.int32).reshape(Q, m)
                offset   += Q * m
                accepting = np.array(row[offset : offset + Q], dtype=bool)
                offset   += Q
                scen_auts.append({"delta": delta, "accepting": accepting})
            automata.append(scen_auts)

        inst          = SSPInstance()
        inst.episode  = ep_id
        inst.n        = n;  inst.m = m;  inst.k = k;  inst.Q = Q
        inst.N        = N;  inst.epsilon = epsilon
        inst.p        = np.full(N, 1.0 / N)
        inst.profit   = profit
        inst.automata = automata
        instances.append(inst)

    return instances


def ssp_to_cpp_v1cld(inst: SSPInstance):
    """
    Convertit un SSPInstance en v1ld_generic_cpp.SSPCLDProblem (C++).

    Décomposition CLD : résolution séparée par automate (n_constraints = k).

    Exemple :
        import v1ld_generic_cpp as cpp
        from solver_generic.ssp import load_ssp_instances, ssp_to_cpp_v1cld

        insts = load_ssp_instances('...')
        prob  = ssp_to_cpp_v1cld(insts[0])
        res   = cpp.solve_v1cld(prob, T=500)
    """
    import v1ld_generic_cpp as cpp
    delta_arr = np.array(
        [[a["delta"]     for a in scen] for scen in inst.automata], dtype=np.int32)
    acc_arr   = np.array(
        [[a["accepting"] for a in scen] for scen in inst.automata], dtype=np.uint8)
    return cpp.SSPCLDProblem(inst.profit, inst.p, inst.epsilon, delta_arr, acc_arr)


def ssp_to_cpp(inst: SSPInstance):
    """
    Convertit un SSPInstance en v1ld_generic_cpp.SSPProblem (C++).

    Exemple :
        import v1ld_generic_cpp as cpp
        from solver_generic.ssp import load_ssp_instances, ssp_to_cpp

        insts = load_ssp_instances('...')
        prob  = ssp_to_cpp(insts[0])
        res   = cpp.solve_v1ld(prob, T=500)
    """
    import v1ld_generic_cpp as cpp
    delta_arr = np.array(
        [[a["delta"]     for a in scen] for scen in inst.automata], dtype=np.int32)
    acc_arr   = np.array(
        [[a["accepting"] for a in scen] for scen in inst.automata], dtype=np.uint8)
    return cpp.SSPProblem(inst.profit, inst.p, inst.epsilon, delta_arr, acc_arr)
