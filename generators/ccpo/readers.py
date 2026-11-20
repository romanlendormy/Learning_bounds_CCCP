"""
readers.py — Lecteur indépendant du format texte (cf. writers.py).

Écrit sans réutiliser les structures internes de `writers.py` (au-delà du format texte
lui-même) : sert de "lecteur indépendant" pour le test de parsing croisé (tests/test_writers.py).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


def read_instance_txt(path: Path) -> dict:
    text = Path(path).read_text()
    lines = text.strip("\n").split("\n")
    n, N, k, Q, L, U, eps_num, eps_den = (int(x) for x in lines[0].split())
    B = int(lines[1])
    c_int = np.array([int(x) for x in lines[2].split()], dtype=np.int64)
    return {
        "n": n,
        "N": N,
        "k": k,
        "Q": Q,
        "L": L,
        "U": U,
        "epsilon_num": eps_num,
        "epsilon_den": eps_den,
        "B": B,
        "c_int": c_int,
    }


def read_scenarios_txt(path: Path, n: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruit (psi_int, Sigma_int) à partir de scenarios.txt (ou .txt.gz)."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            text = f.read()
    else:
        text = path.read_text()

    lines = text.strip("\n").split("\n")
    n_triplets = n * (n + 1) // 2

    psi_int = np.empty((N, n), dtype=np.int64)
    Sigma_int = np.zeros((N, n, n), dtype=np.int64)

    idx = 0
    for j in range(N):
        scen_id = int(lines[idx])
        idx += 1
        if scen_id != j + 1:
            raise ValueError(
                f"scenarios.txt tronqué ou corrompu : bloc {j + 1} attendu, "
                f"indice {scen_id} lu à la position {idx - 1}"
            )
        psi_int[j] = [int(x) for x in lines[idx].split()]
        idx += 1
        for _ in range(n_triplets):
            r, c, v = (int(x) for x in lines[idx].split())
            idx += 1
            Sigma_int[j, r - 1, c - 1] = v
            Sigma_int[j, c - 1, r - 1] = v

    if idx != len(lines):
        raise ValueError(
            f"scenarios.txt : {len(lines) - idx} ligne(s) résiduelle(s) non consommée(s) "
            "-- N ou n incohérent avec le fichier."
        )

    return psi_int, Sigma_int
