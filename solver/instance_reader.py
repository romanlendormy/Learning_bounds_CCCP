"""
Loader Python pour les instances CCSP (set covering stochastique).

Format CSV :
  [episode, scenario, m, n, pA, pB, A_flat(m*n), b_flat(m)]

Note : le vecteur c (objectif) n'est pas dans le fichier.
       Par défaut c = ones(n) — profit uniforme par item.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CCSPScenario:
    index: int
    A: np.ndarray  # (m, n), binaire {0,1}
    b: np.ndarray  # (m,),  binaire {0,1}


@dataclass
class CCSPInstance:
    episode: int
    n: int           # nombre de variables (items)
    m: int           # nombre de contraintes (cibles)
    N: int           # nombre de scénarios
    epsilon: float   # tolérance de violation (chance constraint)
    c: np.ndarray    # (n,) — objectif déterministe
    p: np.ndarray    # (N,) — probabilités (uniformes = 1/N)
    A: np.ndarray    # (N, m, n) — matrices de couverture
    b: np.ndarray    # (N, m)   — vecteurs RHS
    pA: float = 0.0  # densité (info seulement)
    pB: float = 0.0  # taux d'activation (info seulement)

    def __repr__(self) -> str:
        return (
            f"CCSPInstance(ep={self.episode}, n={self.n}, m={self.m}, "
            f"N={self.N}, ε={self.epsilon})"
        )


def load_instances(
    filepath: str,
    epsilon: float = 0.1,
    max_episodes: int = -1,
    c: Optional[np.ndarray] = None,
) -> List[CCSPInstance]:
    """Charge les instances CCSP depuis un fichier CSV.

    Parameters
    ----------
    filepath     : chemin vers le fichier .txt
    epsilon      : tolérance de violation pour la contrainte de chance
    max_episodes : limiter le nombre d'épisodes lus (-1 = tous)
    c            : vecteur objectif (n,) ; si None → ones(n) par défaut
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

    instances: List[CCSPInstance] = []
    for ep_id in sorted(raw.keys()):
        rows = raw[ep_id]
        first = rows[0]
        m, n = int(first[2]), int(first[3])
        pA, pB = first[4], first[5]
        N = len(rows)

        expected = 6 + m * n + m
        if len(first) != expected:
            raise ValueError(
                f"Épisode {ep_id} : longueur ligne incorrecte "
                f"(attendu {expected}, reçu {len(first)})"
            )

        A_arr = np.zeros((N, m, n))
        b_arr = np.zeros((N, m))

        for i, row in enumerate(rows):
            offset = 6
            A_arr[i] = np.array(row[offset : offset + m * n]).reshape(m, n)
            b_arr[i] = np.array(row[offset + m * n : offset + m * n + m])

        c_vec = c if c is not None else np.ones(n)

        instances.append(
            CCSPInstance(
                episode=ep_id,
                n=n, m=m, N=N,
                epsilon=epsilon,
                c=c_vec.copy(),
                p=np.full(N, 1.0 / N),
                A=A_arr,
                b=b_arr,
                pA=pA, pB=pB,
            )
        )

    return instances
