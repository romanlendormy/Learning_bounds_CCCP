"""
ccpo_bruteforce.py — Énumération exhaustive des affectations des variables libres d'un nœud CCPO.

Sert à prouver qu'un sous-arbre est (in)faisable (find_infeasible_nodes.py,
collect/build_extra_feasible_nodes.py) sur les petits nœuds.
"""
from __future__ import annotations

import numpy as np


def gen_free(nf, Q_res, k_res, Lo, Up, cap):
    """Toutes les affectations des variables libres (Σ = Q_res, ≤ k_res actives, valeurs {0}∪[Lo,Up])."""
    out = []

    def rec(i, rem, act, cur):
        if len(out) > cap:
            return
        if i == nf:
            if rem == 0:
                out.append(tuple(cur))
            return
        # élagage : le reste doit pouvoir être atteint
        left = nf - i
        if rem > min(left, k_res - act) * Up:
            return
        rec(i + 1, rem, act, cur + [0])
        if act < k_res:
            for v in range(Lo, min(Up, rem) + 1):
                rec(i + 1, rem - v, act + 1, cur + [v])
    rec(0, Q_res, 0, [])
    if len(out) > cap:
        return None
    return np.array(out, dtype=np.float64).reshape(-1, nf)
