"""
Solveur MIP exact pour le CCSP (set covering stochastique).

Utilisé uniquement pour la validation du gap de dualité :
  gap = v1LD_star − v_MIP  (devrait tendre vers 0 pour set covering)

Formulation :
  max  cᵀ x
  s.t. Σ_i p_i z_i ≥ 1 − ε                          [chance constraint]
       Σ_k A^i_{jk} x_k ≥ b^i_j · z_i  ∀ i, j       [couverture conditionnelle]
       x ∈ {0,1}^n,  z ∈ {0,1}^N

La contrainte de couverture "Σ A^i_{jk} x_k ≥ b^i_j · z_i" est correcte :
  - z_i=1 → la contrainte j du scénario i doit être satisfaite ✓
  - z_i=0 → trivial (RHS = 0) ✓
  - b^i_j ∈ {0,1} : quand b^i_j=0, la contrainte est triviale et n'est pas ajoutée.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .instance_reader import CCSPInstance


def solve_mip_exact(instance: CCSPInstance, time_limit: int = 300) -> Tuple[float, str]:
    """Résoud le CCSP par MIP exact via PuLP/CBC.

    Parameters
    ----------
    instance   : instance CCSP
    time_limit : limite de temps CBC en secondes

    Returns
    -------
    (v_star, status_str)
      v_star     : valeur optimale (ou meilleure borne inférieure si time_limit atteint)
      status_str : statut PuLP ("Optimal", "Infeasible", etc.)
    """
    try:
        import pulp
    except ImportError as e:
        raise ImportError("PuLP requis : pip install pulp") from e

    n, m, N = instance.n, instance.m, instance.N
    c, p, epsilon = instance.c, instance.p, instance.epsilon
    A, b = instance.A, instance.b

    prob = pulp.LpProblem(f"CCSP_ep{instance.episode}", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x{k}", cat="Binary") for k in range(n)]
    z = [pulp.LpVariable(f"z{i}", cat="Binary") for i in range(N)]

    # Objectif
    prob += pulp.lpSum(c[k] * x[k] for k in range(n))

    # Chance constraint : Σ p_i z_i ≥ 1 − ε
    prob += pulp.lpSum(p[i] * z[i] for i in range(N)) >= 1.0 - epsilon

    # Couverture conditionnelle par scénario et contrainte
    for i in range(N):
        for j in range(m):
            if b[i, j] <= 0:
                # Contrainte triviale (RHS=0) : toujours satisfaite
                continue
            # Σ_k A^i_{jk} x_k ≥ b^i_j · z_i
            prob += (
                pulp.lpSum(A[i, j, k] * x[k] for k in range(n))
                >= b[i, j] * z[i]
            )

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    obj_val = pulp.value(prob.objective)
    status  = pulp.LpStatus[prob.status]

    return (float(obj_val) if obj_val is not None else float("nan"), status)
