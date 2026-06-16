"""
train_gnn_v1ld.py — Chargement et entraînement GNN pour la borne V1LD sur CC-MKP.

Format des fichiers d'entrée (générés par bab_mkp_gecode_collect v1ld, commit c2fdc2f) :

  instances_n{n}.txt — une ligne par instance résolue à optimalité :
    inst_id, m, n, N, epsilon, opt,
    c[n],
    b[i][j] pour i in N, j in m     (N*m valeurs, row-major),
    A[i][j][k] pour i in N, j in m, k in n  (N*m*n valeurs)

  nodes_n{n}.txt — une ligne par nœud B&B collecté :
    inst_id, depth, curr_cost, violated_count,
    fix_ones[n]  (0 = non-fixé-à-1, 1 = fixé-à-1)
    b_res[N*m]   (capacités résiduelles pour tous N scénarios, row-major)

  Total champs : 4 + n + N*m  (ex. n=5, m=2, N=100 → 209)

  violated_count : nombre de scénarios i tels que b_res[i][j] < 0 pour au moins un j
                   (capacité violée par les items fixés à 1 → z_i forcé à 1).
  fix_ones[k] ∈ {0,1}: 0 = non fixé à 1 (actif), 1 = fixé à 1.

Structure du graphe produit (un graphe par nœud B&B) :
  - Scénarios libres : free_z = {i : b_res[i][j] >= 0 pour tout j}
    N_z = |free_z| ; n_fixed_z1 = N - N_z  (scénarios forcés à z=1)
  - Nœuds   : N_z * n_g paires (scénario libre i_idx, item actif k_idx)
              indexé par i_idx * n_g + k_idx
  - Features : 5m + 3 :
      [c_k, A_ijk(m), A_ijk/b_res(m), c_k/max(A_ijk,1)(m), dantzig_rank(m), b_res/b(m), k/n_g, i/N_z]
  - Arêtes   : item_varies (poids 1/n_g), scenario_varies (poids 1/N_z)
  - Contexte : [depth, frac_fixés, 0, tightest, curr_cost, epsilon, n_fixed_z1/N]
  - Target   : minimiser L1(λ) la borne V1LD
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib, sys
_pyg_data_mod = importlib.import_module("torch_geometric.data")
pyg_data = _pyg_data_mod
DataLoader = importlib.import_module("torch_geometric.loader").DataLoader
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import global_mean_pool


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des instances
# ─────────────────────────────────────────────────────────────────────────────

def load_instances(instances_file: str | Path) -> dict[int, dict]:
    """
    Lit instances_n{n}.txt et retourne un dict {inst_id: {n,m,N,c,b,A,opt}}.
    b  : ndarray (N, m)
    A  : ndarray (N, m, n)
    c  : ndarray (n,)
    """
    instances: dict[int, dict] = {}
    with open(instances_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(",")]

            inst_id = int(vals[0])
            m       = int(vals[1])
            n       = int(vals[2])
            N       = int(vals[3])
            epsilon = vals[4]
            opt     = vals[5]

            c = np.array(vals[6 : 6 + n], dtype=np.float32)

            off = 6 + n
            b = np.array(vals[off : off + N * m], dtype=np.float32).reshape(N, m)

            off += N * m
            A = np.array(vals[off : off + N * m * n], dtype=np.float32).reshape(N, m, n)

            instances[inst_id] = {
                "n": n, "m": m, "N": N,
                "epsilon": epsilon, "opt": opt,
                "c": c, "b": b, "A": A,
            }
    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Construction du graphe pour un nœud B&B
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(inst: dict, node_rec: dict) -> Optional[pyg_data.Data]:
    """
    Construit un graphe torch_geometric à partir d'une instance et d'un état
    de nœud B&B V1LD.

    Doit accepter un seul format de node_rec :

    Format B — nouveau (nodes_n10+ dans train_v1ld/, 5+n+N champs) :
      x_mask[k] ∈ {0,1,2} : 0=fixé_0, 1=fixé_1, 2=libre
      z_mask[i] ∈ {0,1,2} : 0=fixé_0, 1=fixé_1, 2=libre
      b_res[N, m]          : recalculé depuis instances + x_mask
      → free_z = indices où z_mask == 2
      → n_fixed_z1 = count(z_mask == 1)

    Nœuds : N_z × n_g paires (scénario libre, item actif).
    Features (5m + 3) : [c_k, A_ijk(m), A_ijk/b_res(m), c/A(m),
                          dantzig_rank(m), b_res/b(m), k/n_g, i/N_z]
    Arêtes : item_varies (poids 1/n_g), scenario_varies (poids 1/N_z).

    Retourne None si n_g == 0 ou N_z == 0.
    """
    n, m, N  = inst["n"], inst["m"], inst["N"]
    c, b, A  = inst["c"], inst["b"], inst["A"]
    x_mask   = node_rec["x_mask"]   # (n,)  int8
    b_res    = node_rec["b_res"]    # (N, m)  float32

    z_mask = node_rec.get("z_mask", None)   # présent seulement en format B

    if z_mask is not None:
        # ── Format B : x_mask ternaire, z_mask explicite ──────────────────────
        active_x   = np.where(x_mask == 2)[0]   # items libres (ni fixé_0 ni fixé_1)
        free_z     = np.where(z_mask == 2)[0]
        n_fixed_z1 = int((z_mask == 1).sum())
    else:
        # ── Format A : x_mask binaire, free_z déduit de b_res ─────────────────
        active_x   = np.where(x_mask == 0)[0]   # non-fixé-à-1
        free_z     = np.where(np.all(b_res >= -1e-9, axis=1))[0]
        n_fixed_z1 = N - len(free_z)

    n_g = len(active_x)
    N_z = len(free_z)
    if n_g == 0 or N_z == 0:
        return None

    # ── Sous-matrices : scénarios libres × items actifs ───────────────────────
    b_res_z = b_res[free_z, :]                       # (N_z, m)
    A_g     = A[np.ix_(free_z, range(m), active_x)]  # (N_z, m, n_g)
    c_g     = c[active_x]                            # (n_g,)
    b_z     = b[free_z, :]                           # (N_z, m)

    # ── Features des nœuds (N_z * n_g, 5m + 3) ───────────────────────────────
    c_2d = np.broadcast_to(c_g[None, :], (N_z, n_g))   # (N_z, n_g)

    A_T    = A_g.transpose(0, 2, 1).copy()              # (N_z, n_g, m)
    bres_T = b_res_z[:, np.newaxis, :]                  # (N_z, 1, m)

    f_c    = c_2d[:, :, None]                                                  # (N_z, n_g, 1)
    f_A    = A_T                                                               # (N_z, n_g, m)
    f_frac = A_T / np.maximum(np.broadcast_to(bres_T, (N_z, n_g, m)), 1e-9)   # (N_z, n_g, m)
    f_eff  = c_2d[:, :, None] / np.maximum(A_T, 1.0)                          # (N_z, n_g, m)

    eff_dantzig  = c_g[None, None, :] / np.maximum(A_g, 1e-9)                 # (N_z, m, n_g)
    dantzig_rank = (
        np.argsort(np.argsort(-eff_dantzig, axis=2), axis=2)
        / max(n_g - 1, 1)
    ).astype(np.float32)                                                        # (N_z, m, n_g)
    f_rank = dantzig_rank.transpose(0, 2, 1).copy()                            # (N_z, n_g, m)

    cap_frac = b_res_z / np.maximum(b_z, 1e-9)                                # (N_z, m)
    f_cap    = np.broadcast_to(cap_frac[:, None, :], (N_z, n_g, m)).copy()    # (N_z, n_g, m)

    k_norm = np.broadcast_to(
        np.arange(n_g)[None, :, None] / n_g, (N_z, n_g, 1)).copy()            # (N_z, n_g, 1)
    i_norm = np.broadcast_to(
        np.arange(N_z)[:, None, None] / N_z, (N_z, n_g, 1)).copy()            # (N_z, n_g, 1)

    X = np.concatenate(
        [f_c, f_A, f_frac, f_eff, f_rank, f_cap, k_norm, i_norm], axis=-1
    ).reshape(-1, 5 * m + 3).astype(np.float32)                                # (N_z*n_g, 5m+3)

    # ── Arêtes ────────────────────────────────────────────────────────────────
    def flat(i_idx, k_idx):
        return i_idx * n_g + k_idx

    src_list, dst_list, w_list = [], [], []

    # 1. Item varie (même scénario i_idx) : poids 1/n_g
    ki, kj = np.triu_indices(n_g, k=1)
    for i in range(N_z):
        a  = flat(i, ki);  b_ = flat(i, kj)
        src_list.append(a);  dst_list.append(b_)
        src_list.append(b_); dst_list.append(a)
        w = np.full(len(a), 1.0 / n_g, dtype=np.float32)
        w_list.append(w); w_list.append(w)

    # 2. Scénario varie (même item k_idx) : poids 1/N_z
    ii, ij = np.triu_indices(N_z, k=1)
    for k in range(n_g):
        a  = flat(ii, k);  b_ = flat(ij, k)
        src_list.append(a);  dst_list.append(b_)
        src_list.append(b_); dst_list.append(a)
        w = np.full(len(a), 1.0 / N_z, dtype=np.float32)
        w_list.append(w); w_list.append(w)

    edge_index  = torch.tensor(
        np.stack([np.concatenate(src_list), np.concatenate(dst_list)]),
        dtype=torch.long)
    edge_weight = torch.tensor(np.concatenate(w_list), dtype=torch.float32)

    # ── Contexte (features au niveau du graphe) ───────────────────────────────
    depth      = float(node_rec["depth"])
    frac_fixed = 1.0 - n_g / n
    curr_cost  = node_rec["curr_cost"]
    tightest   = float(np.min(cap_frac))

    context = torch.tensor(
        [depth, frac_fixed, tightest, curr_cost,
         float(inst["epsilon"]), n_fixed_z1 / N],
        dtype=torch.float32)

    return pyg_data.Data(
        x           = torch.from_numpy(X),
        edge_index  = edge_index,
        edge_weight = edge_weight,
        context     = context,
        opt         = torch.tensor(float(inst["opt"]), dtype=torch.float32),
        n_g         = n_g,
        N           = N,
        N_z         = N_z,
        m           = m,
        n_fixed_z1  = n_fixed_z1,
        epsilon     = torch.tensor(float(inst["epsilon"]), dtype=torch.float32),
        # Matrices pour le calcul de L1(λ) — scénarios libres seulement
        A_g     = torch.from_numpy(A_g.copy()),       # (N_z, m, n_g)
        b_res_g = torch.from_numpy(b_res_z.copy()),   # (N_z, m)
        c_g     = torch.from_numpy(c_g.copy()),       # (n_g,)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chargement du dataset complet
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    instances_file: str | Path,
    nodes_file: str | Path,
    max_graphs: Optional[int] = None,
) -> list[pyg_data.Data]:
    """
    Charge tous les états de nœuds B&B comme graphes PyG.

    Détecte automatiquement le format :
      Format A (ancien, 4+n+N*m champs) :
        inst_id, depth, curr_cost, violated_count, fix_ones[n], b_res[N*m]
      Format B (5+n+N champs, bab_mkp_gecode_collect) :
        inst_id, depth, curr_cost, ub_node, incumbent, x_mask[n]{0/1/2}, z_mask[N]{0/1/2}
      Format C (3+n+N champs, bab_mkp_gecode_base_collect) :
        inst_id, depth, curr_cost, x_mask[n]{0/1/2}, z_mask[N]{0/1/2}
    """
    instances = load_instances(instances_file)
    graphs: list[pyg_data.Data] = []
    skipped = 0

    with open(nodes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            vals = [float(x) for x in line.split(",")]
            inst_id   = int(vals[0])
            depth     = int(vals[1])
            curr_cost = vals[2]

            if inst_id not in instances:
                skipped += 1
                continue

            inst = instances[inst_id]
            n, m, N = inst["n"], inst["m"], inst["N"]

            n_fmt_A = 4 + n + N * m
            n_fmt_B = 5 + n + N
            n_fmt_C = 3 + n + N

            if len(vals) == n_fmt_C:
                # ── Format C : sans ub_node/incumbent (collect_gecode_base) ──────
                x_mask = np.array(vals[3 : 3 + n], dtype=np.int8)
                z_mask = np.array(vals[3 + n : 3 + n + N], dtype=np.int8)
                fixed_to_1 = np.where(x_mask == 1)[0]
                b_res = inst["b"].copy()
                if len(fixed_to_1):
                    b_res -= inst["A"][:, :, fixed_to_1].sum(axis=2)
                node_rec = {
                    "depth":     depth,
                    "curr_cost": curr_cost,
                    "x_mask":    x_mask,
                    "z_mask":    z_mask,
                    "b_res":     b_res,
                }
            elif len(vals) >= n_fmt_B and len(vals) < n_fmt_A:
                # ── Format B : x_mask ternaire + z_mask, b_res à reconstruire ──
                x_mask = np.array(vals[5 : 5 + n], dtype=np.int8)
                z_mask = np.array(vals[5 + n : 5 + n + N], dtype=np.int8)
                fixed_to_1 = np.where(x_mask == 1)[0]
                b_res = inst["b"].copy()
                if len(fixed_to_1):
                    b_res -= inst["A"][:, :, fixed_to_1].sum(axis=2)
                node_rec = {
                    "depth":     depth,
                    "curr_cost": curr_cost,
                    "x_mask":    x_mask,
                    "z_mask":    z_mask,
                    "b_res":     b_res,
                }
            elif len(vals) >= n_fmt_A:
                # ── Format A : fix_ones binaire + b_res explicite ─────────────
                x_mask = np.array(vals[4 : 4 + n], dtype=np.int8)
                b_res  = np.array(vals[4 + n : 4 + n + N * m],
                                  dtype=np.float32).reshape(N, m)
                node_rec = {
                    "depth":     depth,
                    "curr_cost": curr_cost,
                    "x_mask":    x_mask,
                    "b_res":     b_res,
                }
            else:
                skipped += 1
                continue

            g = build_graph(inst, node_rec)
            if g is not None:
                g.inst_id = torch.tensor(inst_id, dtype=torch.long)
                graphs.append(g)
            else:
                skipped += 1

            if max_graphs and len(graphs) >= max_graphs:
                break

    print(f"[load_dataset] {len(graphs)} graphes chargés, {skipped} ignorés")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Modèle GNN
# ─────────────────────────────────────────────────────────────────────────────

class GNN(nn.Module):
    """
    Prédit λ[i,k] ∈ R par nœud (scénario libre i, item actif k).
    Le GNN retourne les valeurs brutes (pas de softplus) car λ ∈ R dans V1LD.

    Architecture :
      embed  : X(5m+3) → 64 → ReLU → 64 → ReLU
      conv1  : GatedGraphConv(64, 2 it.) + LayerNorm
      conv2  : GatedGraphConv(64, 2 it.) + LayerNorm + skip(h1)
      conv3  : GatedGraphConv(64, 2 it.) + LayerNorm + skip(h2)
      proj   : 64 → 256 → ReLU → 128          → h_v
      pool   : nœud → scénario [mean] → graphe [mean]  (hiérarchique sur N_z)
      head   : [h_v | g | context] → 64 → ReLU → 64 → ReLU → 1  → λ ∈ R

    Paramètres :
      m (int) : nombre de contraintes (détermine NODE_DIM = 5m+3)
      hidden  : dimension des couches cachées (défaut 64)
    """
    CONTEXT_DIM = 6  # [depth, frac_fixed, tightest, curr_cost, epsilon, n_fixed_z1/N]

    def __init__(self, m: int = 2, hidden: int = 64):
        super().__init__()
        self.m      = m
        self.hidden = hidden
        node_dim    = 5 * m + 3

        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.conv1 = GatedGraphConv(hidden, num_layers=2)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = GatedGraphConv(hidden, num_layers=2)
        self.norm2 = nn.LayerNorm(hidden)
        self.conv3 = GatedGraphConv(hidden, num_layers=2)
        self.norm3 = nn.LayerNorm(hidden)
        self.proj  = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.ReLU(),
            nn.Linear(4 * hidden, 2 * hidden),
        )
        head_in = 2 * hidden + 2 * hidden + self.CONTEXT_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: pyg_data.Data) -> torch.Tensor:
        x, ei, ew = data.x, data.edge_index, data.edge_weight
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        n_v      = x.size(0)
        dev      = x.device
        n_graphs = int(batch.max().item()) + 1

        # ── Message passing ───────────────────────────────────────────────────
        h   = self.embed(x)
        h1  = self.norm1(self.conv1(h,  ei, ew))
        h2  = self.norm2(self.conv2(h1, ei, ew)) + h1
        h3  = self.norm3(self.conv3(h2, ei, ew)) + h2
        h_v = self.proj(h3)                                        # (V, 2h)

        # ── Pooling hiérarchique : nœud → scénario → graphe ──────────────────
        # Nœuds ordonnés i_idx*n_g + k_idx ; n_g consécutifs = un scénario.
        n_g = data.n_g if isinstance(data.n_g, int) else int(data.n_g.item())

        counts   = torch.bincount(batch, minlength=n_graphs)
        node_off = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        node_off[1:] = counts[:-1].cumsum(0)
        local_idx = torch.arange(n_v, dtype=torch.long, device=dev) - node_off[batch]

        n_scen_per = counts // n_g
        scen_off   = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        scen_off[1:] = n_scen_per[:-1].cumsum(0)
        scen_batch = scen_off[batch] + local_idx // n_g            # (V,)

        h_scen = global_mean_pool(h_v, scen_batch)                # (Σ N_z, 2h)

        scen_graph_batch = torch.repeat_interleave(
            torch.arange(n_graphs, device=dev), n_scen_per)
        g_pool  = global_mean_pool(h_scen, scen_graph_batch)      # (B, 2h)
        g_exp   = g_pool[batch]                                    # (V, 2h)

        ctx = data.context
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        ctx_exp = ctx[batch]                                       # (V, ctx_dim)

        cat = torch.cat([h_v, g_exp, ctx_exp], dim=-1)
        lam = self.head(cat).squeeze(-1)                          # (V,)  λ ∈ R
        return lam


# ─────────────────────────────────────────────────────────────────────────────
# Solveur knapsack multi-dimensionnel exact (B&B)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_zeta_multi_bb(
    cost: np.ndarray,   # (n_g,) coûts lagrangiens
    A_i:  np.ndarray,   # (m, n_g) matrice de contraintes du scénario i
    b_i:  np.ndarray,   # (m,) capacités résiduelles
) -> tuple[float, np.ndarray]:
    """
    Knapsack 0/1 multi-dimensionnel exact par Branch & Bound.
    Port de mkp_problem_bnb.cpp.

    max cost^T x  s.t.  A_i x ≤ b_i,  x ∈ {0,1}^n_g
    Retourne (valeur_optimale, x* ∈ {0,1}^n_g).
    """
    n_g = len(cost)
    m   = len(b_i)
    x_out    = np.zeros(n_g, dtype=np.float32)
    val_free = 0.0

    active = []
    for k in range(n_g):
        if cost[k] <= 0.0:
            continue
        if np.all(A_i[:, k] < 1e-9):
            x_out[k] = 1.0
            val_free += cost[k]
        else:
            active.append(k)

    if not active:
        return val_free, x_out

    na = len(active)

    def surrogate(k):
        w = sum(A_i[j, k] / b_i[j] for j in range(m) if b_i[j] > 1e-12)
        return cost[k] / w if w > 1e-12 else 1e18

    active.sort(key=surrogate, reverse=True)

    p     = np.array([cost[k] for k in active], dtype=np.float64)
    A_act = np.array([[A_i[j, k] for k in active] for j in range(m)],
                     dtype=np.float64)

    sort_j = []
    for j in range(m):
        eff = [p[ki] / A_act[j, ki] if A_act[j, ki] > 1e-12 else 1e18
               for ki in range(na)]
        sort_j.append(sorted(range(na), key=lambda x, e=eff: -e[x]))

    best   = [val_free]
    best_x = [np.zeros(na, dtype=bool)]
    cur_x  = np.zeros(na, dtype=bool)

    def lp_bound_j(j, idx, rem_j, cur_val):
        v, r = cur_val, rem_j
        for ki in sort_j[j]:
            if ki < idx:
                continue
            a = A_act[j, ki]
            if a < 1e-12:
                v += p[ki]
            elif a <= r + 1e-12:
                v += p[ki]; r -= a
            else:
                v += p[ki] * r / a; break
        return v

    def upper_bound(idx, rem_b, cur_val):
        ub = 1e30
        for j in range(m):
            ub = min(ub, lp_bound_j(j, idx, rem_b[j], cur_val))
        return ub

    def bb(idx, rem_b, cur_val):
        if upper_bound(idx, rem_b, cur_val) <= best[0] + 1e-9:
            return
        if idx == na:
            if cur_val > best[0] + 1e-9:
                best[0] = cur_val
                best_x[0] = cur_x.copy()
            return
        feasible = all(A_act[j, idx] <= rem_b[j] + 1e-12 for j in range(m))
        if feasible:
            for j in range(m): rem_b[j] -= A_act[j, idx]
            cur_x[idx] = True
            bb(idx + 1, rem_b, cur_val + p[idx])
            cur_x[idx] = False
            for j in range(m): rem_b[j] += A_act[j, idx]
        bb(idx + 1, rem_b, cur_val)

    bb(0, b_i.copy().astype(np.float64), val_free)

    for ki, k in enumerate(active):
        if best_x[0][ki]:
            x_out[k] = 1.0

    return best[0], x_out


# ─────────────────────────────────────────────────────────────────────────────
# Borne L1(λ) — implémentation différentiable de V1LD
# ─────────────────────────────────────────────────────────────────────────────

def lagrangian_v1ld(lam: torch.Tensor, data: pyg_data.Data) -> torch.Tensor:
    """
    Calcule L1(λ) sur les scénarios libres, différentiable en λ.

    lam     : (N_z*n_g,) ou (N_z, n_g) — sortie brute du GNN
    data    : graphe avec A_g (N_z, m, n_g), b_res_g (N_z, m), c_g (n_g,),
              N (total scénarios), N_z (scénarios libres), n_fixed_z1

    Algorithme (théorème de l'enveloppe) :
      1. cost_i[k] = c[k] + λ_ik / p_i − Λ_sum[k]   (p_i = 1/N)
      2. θ_i = Σ max(0, cost_i[k])
      3. ζ_i = knapsack B&B multi-dimensionnel (detach)
      4. Δ_i = max(0, θ_i − ζ_i)
      5. required_prob = max(0, (1-ε) - n_fixed_z1/N)
      6. y* = sélection gloutonne des N_z scénarios libres (detach)
      7. L1 ≈ p Σ_i θ_i(λ) − p Σ_i y*_i Δ_i(λ)   linéaire en λ
    """
    N_z        = data.N_z
    n_g        = data.n_g
    N          = data.N
    n_fixed_z1 = data.n_fixed_z1
    dev        = lam.device

    lam_2d  = lam.view(N_z, n_g)

    A_g     = data.A_g.to(dev)      # (N_z, m, n_g)
    b_res_g = data.b_res_g.to(dev)  # (N_z, m)
    c_g     = data.c_g.to(dev)      # (n_g,)
    epsilon = data.epsilon.item()

    p_unif = 1.0 / N   # probabilité de chaque scénario (sur N total)

    # (A) Λ_sum = Σ_i λ_i   (sur N_z scénarios libres)
    Lambda_sum = lam_2d.sum(dim=0)                              # (n_g,)

    # (B) cost_i[k] = c[k] + λ_ik/p_i − Λ_sum[k]
    cost = c_g.unsqueeze(0) + lam_2d / p_unif - Lambda_sum.unsqueeze(0)  # (N_z, n_g)

    # ── Sous-problèmes (hors graphe de calcul) ────────────────────────────────
    cost_np = cost.detach().cpu().numpy()
    A_np    = A_g.cpu().numpy()
    bres_np = b_res_g.cpu().numpy()

    x_theta_np = (cost_np > 0).astype(np.float32)
    x_zeta_np  = np.zeros_like(x_theta_np)

    for i in range(N_z):
        _, xz = _solve_zeta_multi_bb(cost_np[i], A_np[i], bres_np[i])
        x_zeta_np[i] = xz

    # (C) Δ_i et y* greedy
    theta_np = np.sum(np.maximum(cost_np, 0.0), axis=1)
    zeta_np  = np.sum(cost_np * x_zeta_np, axis=1)
    Delta_np = np.maximum(0.0, theta_np - zeta_np)

    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_z1 * p_unif)
    order   = np.argsort(Delta_np)
    y_star  = np.zeros(N_z, dtype=np.float32)
    cumprob = 0.0
    for idx in order:
        if required_prob <= 0.0:
            break
        y_star[idx] = 1.0
        cumprob += p_unif
        if cumprob >= required_prob - 1e-9:
            break

    # (D) L1 linéaire en λ à x*, y* fixés
    x_theta_t = torch.tensor(x_theta_np, device=dev)
    x_zeta_t  = torch.tensor(x_zeta_np,  device=dev)
    y_star_t  = torch.tensor(y_star,     device=dev)

    theta_lam = (cost * x_theta_t).sum(dim=1)
    delta_lam = (cost * (x_theta_t - x_zeta_t)).sum(dim=1)

    L1_theta = (p_unif * theta_lam).sum()
    val_lam  = (p_unif * y_star_t * delta_lam).sum()
    return L1_theta - val_lam


# ─────────────────────────────────────────────────────────────────────────────
# Vérification par sous-gradient Python pur — port de v1ld_solver.cpp
# ─────────────────────────────────────────────────────────────────────────────

def verify_v1ld(
    data: pyg_data.Data,
    T: int = 200,
    precision: float | None = None,
    patience:  int | None   = None,
    lam_init:  np.ndarray | None = None,
    return_n_iters: bool = False,
) -> float | tuple[float, int]:
    """
    Reproduit l'algorithme v1ld_solver.cpp en Python pur (pas 1/√(t+1))
    et retourne la meilleure L1 vue.

    lam_init : (N_z, n_g) float64 ; None = cold start λ=0.
    """
    N_z        = data.N_z
    n_g        = data.n_g
    N          = data.N
    n_fixed_z1 = data.n_fixed_z1
    epsilon    = data.epsilon.item()
    A_np       = data.A_g.numpy()      # (N_z, m, n_g)
    b_np       = data.b_res_g.numpy()  # (N_z, m)
    c_np       = data.c_g.numpy()      # (n_g,)

    p_unif        = 1.0 / N
    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_z1 * p_unif)

    lam = (lam_init.copy() if lam_init is not None
           else np.zeros((N_z, n_g), dtype=np.float64))

    best       = np.inf
    no_improve = 0
    n_iters    = T

    for t in range(T):
        Lsum = lam.sum(axis=0)
        cost = c_np + lam / p_unif - Lsum               # (N_z, n_g)

        x_theta    = (cost > 0).astype(np.float64)
        x_zeta     = np.zeros_like(x_theta)
        theta_vals = np.sum(np.maximum(cost, 0), axis=1)

        for i in range(N_z):
            _, xz = _solve_zeta_multi_bb(
                cost[i].astype(np.float32),
                A_np[i].astype(np.float32),
                b_np[i].astype(np.float32))
            x_zeta[i] = xz.astype(np.float64)

        zeta_vals = np.sum(cost * x_zeta, axis=1)
        Delta     = np.maximum(0.0, theta_vals - zeta_vals)

        order   = np.argsort(Delta)
        y_star  = np.zeros(N_z)
        cumprob = 0.0
        for idx in order:
            if required_prob <= 0.0:
                break
            y_star[idx] = 1.0
            cumprob += p_unif
            if cumprob >= required_prob - 1e-9:
                break

        L1 = p_unif * theta_vals.sum() - p_unif * (y_star * Delta).sum()

        if L1 < best - (precision or 0.0):
            best = L1; no_improve = 0
        else:
            no_improve += 1
        if patience is not None and precision is not None and no_improve >= patience:
            n_iters = t + 1
            break

        x_bar = np.zeros(n_g)
        for i in range(N_z):
            xi = x_zeta[i] if y_star[i] > 0.5 else x_theta[i]
            x_bar += p_unif * xi

        step = 1.0 / np.sqrt(t + 1)

        for i in range(N_z):
            xi = x_zeta[i] if y_star[i] > 0.5 else x_theta[i]
            lam[i] += step * (x_bar - xi)

    if return_n_iters:
        return best, n_iters
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Overfit sur un graphe — validation de l'architecture
# ─────────────────────────────────────────────────────────────────────────────

def overfit_one_graph(data: pyg_data.Data, n_epochs: int = 1000, lr: float = 1e-3):
    """Entraîne le GNN sur un seul graphe avec la vraie borne V1LD comme loss."""
    device = torch.device("cpu")
    data   = data.to(device)

    model     = GNN(m=data.m).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    opt_val = data.opt.item()
    with torch.no_grad():
        lam0    = torch.zeros(data.N_z * data.n_g)
        l1_init = lagrangian_v1ld(lam0, data).item()

    print(f"\n{'='*60}")
    print(f"Overfit — n_g={data.n_g}  N_z={data.N_z}  N={data.N}  m={data.m}")
    print(f"  opt={opt_val:.2f}   L1(λ=0)={l1_init:.2f}")
    print(f"  graphe : {data.x.shape[0]} nœuds, {data.edge_index.shape[1]} arêtes")
    print(f"{'='*60}")
    print(f"{'epoch':>6}  {'L1(λ_GNN)':>12}  {'gap_vs_opt%':>12}  {'gap_vs_L1_0%':>13}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lam  = model(data)
        loss = lagrangian_v1ld(lam, data)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            b     = loss.item()
            g_opt = (b - opt_val) / opt_val * 100 if opt_val > 0 else float("nan")
            g_l0  = (b - l1_init) / abs(l1_init) * 100 if l1_init != 0 else float("nan")
            print(f"{epoch:6d}  {b:12.4f}  {g_opt:+11.2f}%  {g_l0:+12.2f}%")

    print(f"{'='*60}")
    print(f"Borne finale : {loss.item():.4f}  (opt={opt_val:.2f}, L1_0={l1_init:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train_v1LD")
    n_val    = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    mode     = sys.argv[3] if len(sys.argv) > 3 else "inspect"

    inst_file  = data_dir / f"instances_n{n_val}.txt"
    nodes_file = data_dir / f"nodes_n{n_val}.txt"

    graphs = load_dataset(inst_file, nodes_file, max_graphs=5)

    for i, g in enumerate(graphs):
        print(
            f"  graph {i:3d} | nodes={g.x.shape[0]:6d} | "
            f"edges={g.edge_index.shape[1]:8d} | "
            f"n_g={g.n_g} N_z={g.N_z}/{g.N} m={g.m} | "
            f"depth={int(g.context[0].item())} | opt={g.opt.item():.2f}"
        )

    if mode in ("verify", "overfit") and graphs:
        g0 = graphs[0]
        lam0    = torch.zeros(g0.N_z * g0.n_g)
        l1_init = lagrangian_v1ld(lam0, g0).item()
        print(f"\n── Vérification V1LD (T=100 itérations subgradient) ──")
        print(f"  L1(λ=0)          : {l1_init:.4f}")
        py_best = verify_v1ld(g0, T=100)
        print(f"  subgradient T=100 : {py_best:.4f}")
        print(f"  opt               : {g0.opt.item():.4f}")
        print(f"  amélioration      : {(l1_init - py_best) / l1_init * 100:.1f}%")

    if mode == "overfit" and graphs:
        overfit_one_graph(graphs[0])
