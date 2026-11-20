"""
ccpo_dataset.py — Parsing des traces B&B collectées (collect/ccpo/collect_ccpo_samples.py) et
reconstruction du sous-problème résiduel à chaque nœud échantillonné.

Format des fichiers (écrits par solver_bab/ccpo/collect_gecode_ccpo.hpp) :

  instances_*.txt (une ligne par instance) :
    instance_id,n,N,k,Q,L,U,B,min_feas,opt,
    c_int[0..n-1],
    infeasible_scenario[0..N-1] (0/1),
    (i,k,val) x (n(n+1)/2) par scénario, N fois -- triangle sup de Sigma_int

  nodes_*.txt (une ligne par nœud échantillonné) :
    instance_id,depth,curr_cost,q_mask[0..n-1]   (-1 = libre, sinon valeur fixée)

reconstruct_residual_instance() DOIT rester en synchronisation exacte avec le
calcul fait par CCPOSpaceSLD::more() (gecode_bab_ccpo.hpp), correctif du
terme croisé fixé×libre inclus -- sinon les exemples
d'entraînement seraient incohérents avec ce que le solveur évalue réellement
en C++ à l'inférence. Toute modification de more() doit être reportée ici.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ccpo_graph import CCPOResidualInstance


@dataclass
class CCPOInstanceRecord:
    instance_id: int
    n: int
    N: int
    k: int
    Q: int
    L: int
    U: int
    B: float
    min_feas: int
    opt: float
    c_int: np.ndarray            # (n,)
    infeasible: np.ndarray       # (N,) bool
    Sigma_dense: np.ndarray      # (N, n, n)


def load_instances_ccpo(path: str) -> dict[int, CCPOInstanceRecord]:
    out: dict[int, CCPOInstanceRecord] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts or parts[0] == "":
                continue
            idx = 0
            instance_id = int(parts[idx]); idx += 1
            n = int(parts[idx]); idx += 1
            N = int(parts[idx]); idx += 1
            k = int(parts[idx]); idx += 1
            Q = int(parts[idx]); idx += 1
            L = int(parts[idx]); idx += 1
            U = int(parts[idx]); idx += 1
            B = float(parts[idx]); idx += 1
            min_feas = int(parts[idx]); idx += 1
            opt = float(parts[idx]); idx += 1

            c_int = np.array([int(x) for x in parts[idx:idx + n]], dtype=np.float64)
            idx += n

            infeasible = np.array([int(x) for x in parts[idx:idx + N]], dtype=bool)
            idx += N

            Sigma_dense = np.zeros((N, n, n), dtype=np.float64)
            n_triplets = n * (n + 1) // 2
            for j in range(N):
                for _ in range(n_triplets):
                    i = int(parts[idx]); kk = int(parts[idx + 1]); val = float(parts[idx + 2])
                    idx += 3
                    Sigma_dense[j, i, kk] = val
                    Sigma_dense[j, kk, i] = val

            out[instance_id] = CCPOInstanceRecord(
                instance_id=instance_id, n=n, N=N, k=k, Q=Q, L=L, U=U, B=B,
                min_feas=min_feas, opt=opt, c_int=c_int, infeasible=infeasible,
                Sigma_dense=Sigma_dense,
            )
    return out


@dataclass
class CCPONodeRecord:
    instance_id: int
    depth: int
    curr_cost: float
    q_mask: np.ndarray  # (n,) int -- -1 = libre, sinon valeur fixée


def load_nodes_ccpo(path: str) -> list[CCPONodeRecord]:
    out = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts or parts[0] == "":
                continue
            instance_id = int(parts[0])
            depth = int(parts[1])
            curr_cost = float(parts[2])
            q_mask = np.array([int(x) for x in parts[3:]], dtype=np.int64)
            out.append(CCPONodeRecord(instance_id, depth, curr_cost, q_mask))
    return out


def sample_nodes(nodes: list[CCPONodeRecord], rate: float = 0.10,
                  rng: np.random.Generator | None = None) -> list[CCPONodeRecord]:
    """Racine (depth==0) toujours gardée + `rate` des autres, tirage uniforme --
    même politique que collect/mkp/collect_mkp_samples.py::_sample_lines (MKP)."""
    if rng is None:
        rng = np.random.default_rng(0)
    roots = [r for r in nodes if r.depth == 0]
    others = [r for r in nodes if r.depth != 0]
    n_keep = round(len(others) * rate)
    idx = rng.choice(len(others), size=min(n_keep, len(others)), replace=False) if others else []
    sampled = roots + [others[i] for i in idx]
    return sampled


def reconstruct_residual_instance(
    inst: CCPOInstanceRecord, node: CCPONodeRecord,
    c_scale: float, sigma_scale: float,
) -> CCPOResidualInstance | None:
    """Reconstruit le sous-problème résiduel EXACTEMENT comme
    CCPOSpaceSLD::more() (gecode_bab_ccpo.hpp), correctif du terme croisé
    inclus. Retourne None si le nœud est une feuille (aucune variable libre)
    -- rien à prédire, pas un exemple d'entraînement utile.

      C_ff       = Σ_{i,i' fixés actifs} Sigma[i][i']*v_i*v_i'   (constante)
      B_res[j]   = B - C_ff
      d_full[i]  = Σ_{i' fixé actif, i'!=i} v_i'*Sigma_j[i'][i]  (tout i, pour R_ext)
      Q_res      = Q - Σ v_i (fixés actifs)
      k_res      = k - #{i fixé actif}
    """
    n = inst.n
    q_mask = node.q_mask
    fixed_active = (q_mask > 0)
    free = (q_mask < 0)  # -1 = libre ; 0 = fixé exclu (compte comme "fixé", pas actif)
    free_vars = np.where(free)[0]
    n_free = len(free_vars)
    if n_free == 0:
        return None

    v_fixed = np.where(fixed_active, q_mask, 0).astype(np.float64)  # (n,) 0 hors support fixé actif

    Q_res = inst.Q - int(v_fixed.sum())
    k_res = inst.k - int(fixed_active.sum())

    cost_free = inst.c_int[free_vars].copy()
    cost_fixed = float((inst.c_int * v_fixed).sum())

    N = inst.N
    Sigma_free = np.zeros((N, n_free, n_free), dtype=np.float64)
    B_res = np.zeros(N, dtype=np.float64)
    R_ext = np.zeros((N, n_free), dtype=np.float64)

    for j in range(N):
        if inst.infeasible[j]:
            continue
        Sig = inst.Sigma_dense[j]
        C_ff = float(v_fixed @ Sig @ v_fixed)
        B_res[j] = inst.B - C_ff
        Sigma_free[j] = Sig[np.ix_(free_vars, free_vars)]
        # d_full[i] = sum_{i' fixe actif} v_i' * Sig[i'][i], restreint aux libres.
        d_full = Sig @ v_fixed  # (n,) -- inclut le terme i=i' (v_fixed[i]=0 pour i libre, sans effet)
        R_ext[j] = d_full[free_vars]

    return CCPOResidualInstance(
        n_free=n_free, N=N, k_res=k_res, Q_res=Q_res, L=inst.L, U=inst.U,
        cost_free=cost_free, Sigma_free=Sigma_free, B_res=B_res, R_ext=R_ext,
        infeasible=inst.infeasible,
        depth=node.depth, n_total=n,
        frac_fixed_active=float(fixed_active.sum()) / max(1, n),
        cost_fixed=cost_fixed,
        c_scale=c_scale, sigma_scale=sigma_scale, epsilon=1.0 - inst.min_feas / inst.N,
        B_orig=float(inst.B),
    )
