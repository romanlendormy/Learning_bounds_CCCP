"""
train_gnn_v1cld.py — Chargement et entraînement GNN pour la borne V1CLD sur CC-MKP.

Format des fichiers d'entrée (générés par bab_mkp_gecode_collect) :

  instances_n{n}.txt — une ligne par instance résolue à optimalité :
    inst_id, m, n, N, epsilon, opt,
    c[n],
    b[i][j]   pour i in N, j in m     (N*m valeurs),
    A[i][j][k] pour i in N, j in m, k in n  (N*m*n valeurs)

  nodes_n{n}.txt — une ligne par nœud B&B collecté :
    inst_id, depth, curr_cost,
    x_mask[n]   (0=fixé_0, 1=fixé_1, 2=libre)
    z_mask[N]   (0=fixé_0, 1=fixé_1, 2=libre)

Structure du graphe produit (un graphe par nœud B&B) :
  - Nœuds   : un par triplet (i_idx, j, k_idx) où
                i_idx ∈ [0, N)  — TOUS les scénarios (comme dans le solveur C++)
                j     ∈ [0, m)
                k_idx ∈ items libres (x_mask==2)
    → N*m*n_g nœuds ; indexé par i_idx*(m*n_g) + j*n_g + k_idx
  - Features : [c_k, A_ijk, A_ijk/b_res_ij, c_k/max(A_ijk,1), dantzig_rank, b_res_ij/b_ij, k/n_g, j/m, i/N_z]
  - Arêtes   : entre nœuds qui ne diffèrent que sur une dimension
      • item varie    → poids 1/n_g
      • contrainte    → poids 1/m
      • scénario      → poids 1/N
  - Contexte : [depth, frac_fixés, contrainte_serrée, curr_cost, epsilon, n_fixed_1_z/N]
  - Target   : implicite — minimiser L1(λ) la borne V1CLD
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importer uniquement le sous-module nécessaire pour éviter le conflit
# torch_geometric.datasets → transformers → tensorflow → NumPy 1.x
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
    de nœud B&B.

    Symétrie x / z :
      - items   : uniquement les x libres (x_mask == 2), indices unfixed_x
      - scénarios : uniquement les z libres (z_mask == 2), indices free_z

    Nœuds : N_z * m * n_g triplets (scénario_libre, contrainte, item_libre).
    Features (9) : [c_k, A_ijk, A_ijk/b_res, c_k/max(A_ijk,1), dantzig_rank, b_res/b, k/n_g, j/m, i/N_z]
    Indices normalisés ∈ [0,1] pour la généralisation cross-instance.
    Contexte (6) : [depth, frac_fixed, tightest, curr_cost, epsilon, n_fixed_1_z/N]

    Retourne None si n_g == 0 ou N_z == 0.
    """
    n, m, N     = inst["n"], inst["m"], inst["N"]
    c, b, A     = inst["c"], inst["b"], inst["A"]
    x_mask      = node_rec["x_mask"]   # (n,)  int8
    z_mask      = node_rec["z_mask"]   # (N,)  int8

    # ── Items libres ──────────────────────────────────────────────────────────
    unfixed_x = np.where(x_mask == 2)[0]
    n_g = len(unfixed_x)
    if n_g == 0:
        return None

    # ── Scénarios libres ──────────────────────────────────────────────────────
    free_z      = np.where(z_mask == 2)[0]   # indices des z libres dans [0,N)
    N_z         = len(free_z)
    n_fixed_1_z = int((z_mask == 1).sum())   # scénarios déjà satisfaits
    if N_z == 0:
        return None

    # ── Capacités résiduelles (scénarios libres seulement) ───────────────────
    fixed_1_x = np.where(x_mask == 1)[0]
    b_res = b[free_z, :].copy()              # (N_z, m)
    if len(fixed_1_x):
        b_res -= A[np.ix_(free_z, range(m), fixed_1_x)].sum(axis=2)

    # ── Sous-matrices : scénarios libres × items libres ───────────────────────
    A_g = A[np.ix_(free_z, range(m), unfixed_x)]   # (N_z, m, n_g)
    c_g = c[unfixed_x]                              # (n_g,)

    # ── Features des nœuds (N_z * m * n_g, 7) ────────────────────────────────
    c_brd    = np.broadcast_to(c_g,               (N_z, m, n_g))
    bres_brd = np.broadcast_to(b_res[:, :, None], (N_z, m, n_g))

    feat_profit     = c_brd.astype(np.float32)
    feat_weight     = A_g.astype(np.float32)
    feat_frac_cap   = (A_g / np.maximum(bres_brd, 1e-9)).astype(np.float32)
    feat_efficiency = (c_brd / np.maximum(A_g, 1.0)).astype(np.float32)

    # Rang Dantzig normalisé ∈ [0,1] : position de k dans le classement
    # décroissant d'efficacité c_k/A_ijk au sein du knapsack (i,j).
    # 0.0 = item le plus profitable (greedy le sélectionne en premier),
    # 1.0 = item le moins profitable. Complémentaire à feat_efficiency
    # (absolu) : encode l'ordre de sélection de ζ, pas juste la valeur.
    eff_dantzig  = c_brd / np.maximum(A_g, 1e-9)
    dantzig_rank = (np.argsort(np.argsort(-eff_dantzig, axis=2), axis=2)
                    / max(n_g - 1, 1)).astype(np.float32)                # (N_z, m, n_g)

    # Fraction de capacité résiduelle ∈ [0,1] : b_res_ij / b_ij.
    # Constante sur k, varie sur (i,j). Indique la sévérité de la contrainte :
    # 1.0 = aucun item fixé dans ce knapsack, ~0 = contrainte presque saturée.
    b_orig       = b[free_z, :]                                          # (N_z, m)
    cap_frac     = (b_res / np.maximum(b_orig, 1e-9))                   # (N_z, m)
    cap_frac_brd = np.broadcast_to(cap_frac[:, :, None],
                                   (N_z, m, n_g)).astype(np.float32)    # (N_z, m, n_g)

    k_idx = np.broadcast_to(np.arange(n_g) [None, None, :] / n_g,  (N_z, m, n_g)).astype(np.float32)
    j_idx = np.broadcast_to(np.arange(m)   [None, :, None] / m,   (N_z, m, n_g)).astype(np.float32)
    i_idx = np.broadcast_to(np.arange(N_z) [:, None, None] / N_z, (N_z, m, n_g)).astype(np.float32)

    X = np.stack([
        feat_profit, feat_weight, feat_frac_cap, feat_efficiency,
        dantzig_rank, cap_frac_brd, k_idx, j_idx, i_idx,
    ], axis=-1).reshape(-1, 9).astype(np.float32)   # (N_z*m*n_g, 9)

    # ── Arêtes ────────────────────────────────────────────────────────────────
    def flat(i, j, k):
        return i * (m * n_g) + j * n_g + k

    src_list, dst_list, w_list = [], [], []

    # 1. Item varie (même i, j) : poids 1/n_g
    ki, kj = np.triu_indices(n_g, k=1)
    for i in range(N_z):
        for j in range(m):
            a = flat(i, j, ki); b_ = flat(i, j, kj)
            src_list.append(a); dst_list.append(b_)
            src_list.append(b_); dst_list.append(a)
            w = np.full(len(a), 1.0 / n_g, dtype=np.float32)
            w_list.append(w); w_list.append(w)

    # 2. Contrainte varie (même i, k) : poids 1/m
    ji, jj = np.triu_indices(m, k=1)
    for i in range(N_z):
        for k in range(n_g):
            a = flat(i, ji, k); b_ = flat(i, jj, k)
            src_list.append(a); dst_list.append(b_)
            src_list.append(b_); dst_list.append(a)
            w = np.full(len(a), 1.0 / m, dtype=np.float32)
            w_list.append(w); w_list.append(w)

    # 3. Scénario varie (même j, k) : poids 1/N_z
    ii, ij = np.triu_indices(N_z, k=1)
    for j in range(m):
        for k in range(n_g):
            a = flat(ii, j, k); b_ = flat(ij, j, k)
            src_list.append(a); dst_list.append(b_)
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
    b_orig_free = b[free_z, :]
    tightest   = float(np.min(b_res / np.maximum(b_orig_free, 1e-9)))
    curr_cost  = node_rec["curr_cost"]

    context = torch.tensor(
        [depth, frac_fixed, tightest, curr_cost,
         float(inst["epsilon"]), n_fixed_1_z / N],
        dtype=torch.float32)

    return pyg_data.Data(
        x           = torch.from_numpy(X),
        edge_index  = edge_index,
        edge_weight = edge_weight,
        context     = context,
        opt         = torch.tensor(float(inst["opt"]), dtype=torch.float32),
        n_g         = n_g,
        N_z         = N_z,    # scénarios libres dans ce nœud
        N           = N,      # N original (pour les probabilités p_i = 1/N)
        m           = m,
        n_fixed_1_z = n_fixed_1_z,
        epsilon     = torch.tensor(float(inst["epsilon"]), dtype=torch.float32),
        # Matrices pour le calcul de L1(λ)
        A_g     = torch.from_numpy(A_g.copy()),      # (N_z, m, n_g)
        b_res_g = torch.from_numpy(b_res.copy()),    # (N_z, m)
        c_g     = torch.from_numpy(c_g.copy()),      # (n_g,)
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

    Args:
        instances_file : chemin vers instances_n{n}.txt
        nodes_file     : chemin vers nodes_n{n}.txt
        max_graphs     : limite optionnelle (utile pour debug)

    Returns:
        Liste de pyg_data.Data, un par nœud B&B non dégénéré.
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
            inst_id  = int(vals[0])
            depth    = int(vals[1])
            curr_cost = vals[2]

            if inst_id not in instances:
                skipped += 1
                continue

            inst = instances[inst_id]
            n, N = inst["n"], inst["N"]

            if len(vals) < 3 + n + N:
                skipped += 1
                continue

            x_mask = np.array(vals[3       : 3 + n    ], dtype=np.int8)
            z_mask = np.array(vals[3 + n   : 3 + n + N], dtype=np.int8)

            node_rec = {
                "depth":     depth,
                "curr_cost": curr_cost,
                "x_mask":    x_mask,
                "z_mask":    z_mask,
            }

            g = build_graph(inst, node_rec)
            if g is not None:
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
    Prédit λ[i,j,k] ∈ R par nœud (scénario i, contrainte j, item libre k).
    Le GNN retourne les valeurs brutes (pas de softplus) car λ ∈ R dans V1CLD.

    Architecture :
      embed  : X(9) → 64 → ReLU → 64 → ReLU
      conv1  : GatedGraphConv(64, 2 it.) + LayerNorm
      conv2  : GatedGraphConv(64, 2 it.) + LayerNorm + skip(h1)
      conv3  : GatedGraphConv(64, 2 it.) + LayerNorm + skip(h2)
      proj   : 64 → 256 → ReLU → 128      → h_v
      pool   : nœud → knapsack(i,j) [mean] → graphe [mean]  (hiérarchique)
      head   : [h_v | g | context] → 64 → ReLU → 64 → ReLU → 1  → λ ∈ R
    """
    NODE_DIM    = 9   # [c, A, A/b_res, c/A, dantzig_rank, b_res/b, k/n_g, j/m, i/N_z]
    CONTEXT_DIM = 6   # [depth, frac_fixed, tightest, curr_cost, epsilon, n_fixed_1_z/N]

    def __init__(self, node_dim: int = 9, context_dim: int = 6):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(node_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),       nn.ReLU(),
        )
        self.conv1  = GatedGraphConv(64, num_layers=2)
        self.norm1  = nn.LayerNorm(64)
        self.conv2  = GatedGraphConv(64, num_layers=2)
        self.norm2  = nn.LayerNorm(64)
        self.conv3  = GatedGraphConv(64, num_layers=2)
        self.norm3  = nn.LayerNorm(64)
        self.proj   = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 128 + context_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),                      nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data: pyg_data.Data) -> torch.Tensor:
        x, ei, ew = data.x, data.edge_index, data.edge_weight
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        n_v = x.size(0)
        dev = x.device

        # ── Message passing ───────────────────────────────────────────────────
        h  = self.embed(x)                           # (V, 64)
        h1 = self.norm1(self.conv1(h,  ei, ew))      # (V, 64)
        h2 = self.norm2(self.conv2(h1, ei, ew)) + h1 # (V, 64) + skip
        h3 = self.norm3(self.conv3(h2, ei, ew)) + h2 # (V, 64) + skip
        h_v = self.proj(h3)                           # (V, 128)

        # ── Pooling hiérarchique : nœud → knapsack (i,j) → graphe ────────────
        # Les nœuds sont ordonnés i*(m*n_g) + j*n_g + k, donc n_g nœuds
        # consécutifs forment un groupe (i,j). Fonctionne en mode single-graph
        # et en mode batché (n_g constant sur le batch).
        n_g = data.n_g if isinstance(data.n_g, int) else int(data.n_g.item())
        n_graphs = int(batch.max().item()) + 1

        # Indice local de chaque nœud dans son graphe (vectorisé)
        counts   = torch.bincount(batch, minlength=n_graphs)       # (B,)
        node_off = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        node_off[1:] = counts[:-1].cumsum(0)
        local_idx = torch.arange(n_v, dtype=torch.long, device=dev) - node_off[batch]

        # Indice global du groupe (i,j) pour chaque nœud
        n_ij_per = counts // n_g                                    # (B,)
        ij_off   = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        ij_off[1:] = n_ij_per[:-1].cumsum(0)
        ij_batch = ij_off[batch] + local_idx // n_g                 # (V,)

        # Niveau 1 : moyenne par knapsack (i,j)
        h_ij = global_mean_pool(h_v, ij_batch)                     # (Σ N_z*m, 128)

        # Niveau 2 : moyenne des knapsacks par graphe
        ij_graph_batch = torch.repeat_interleave(
            torch.arange(n_graphs, device=dev), n_ij_per)           # (Σ N_z*m,)
        g     = global_mean_pool(h_ij, ij_graph_batch)             # (B, 128)
        g_exp = g[batch]                                            # (V, 128)

        ctx = data.context
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        ctx_exp = ctx[batch]                                        # (V, context_dim)

        cat = torch.cat([h_v, g_exp, ctx_exp], dim=-1)
        lam = self.head(cat).squeeze(-1)                            # (V,)  — λ ∈ R
        return lam


# ─────────────────────────────────────────────────────────────────────────────
# Solveur knapsack 0/1 exact (B&B, Python) — port de solve_zeta_i_exact_bb
# ─────────────────────────────────────────────────────────────────────────────

def _solve_zeta_bb(cost: np.ndarray, a_row: np.ndarray, b_j: float):
    """
    0/1 knapsack exact par B&B avec borne de Dantzig.
    Même algorithme que solve_zeta_i_exact_bb dans mkp_subproblems.hpp.
    Retourne (valeur optimale, x* ∈ {0,1}^n).
    """
    n = len(cost)
    x = np.zeros(n, dtype=np.float32)
    val_free = 0.0

    items = []
    for k in range(n):
        if a_row[k] < 1e-9:
            if cost[k] > 0.0:
                x[k] = 1.0
                val_free += cost[k]
        elif cost[k] > 0.0:
            items.append(k)

    if not items:
        return val_free, x

    items.sort(key=lambda k: -cost[k] / a_row[k])   # ratio décroissant
    p = np.array([cost[k] for k in items], dtype=np.float64)
    w = np.array([a_row[k] for k in items], dtype=np.float64)
    ni = len(items)

    def dantzig(idx, cap, cur):
        v, r = cur, cap
        for i in range(idx, ni):
            if r < 1e-12: break
            t = min(1.0, r / w[i])
            v += t * p[i]; r -= t * w[i]
        return v

    best = [0.0]
    best_sel = [np.zeros(ni, dtype=bool)]
    cur_sel  = np.zeros(ni, dtype=bool)

    def bb(idx, cap, cur):
        if dantzig(idx, cap, cur) <= best[0] + 1e-9: return
        if idx == ni:
            if cur > best[0] + 1e-9:
                best[0] = cur
                best_sel[0] = cur_sel.copy()
            return
        if w[idx] <= cap + 1e-9:
            cur_sel[idx] = True
            bb(idx + 1, cap - w[idx], cur + p[idx])
            cur_sel[idx] = False
        bb(idx + 1, cap, cur)

    bb(0, float(b_j), 0.0)

    for i, k in enumerate(items):
        if best_sel[0][i]:
            x[k] = 1.0
    return val_free + best[0], x


# ─────────────────────────────────────────────────────────────────────────────
# Borne L1(λ) — implémentation exacte de v1cld_solver.cpp
# ─────────────────────────────────────────────────────────────────────────────

def lagrangian_v1cld(lam: torch.Tensor, data: pyg_data.Data) -> torch.Tensor:
    """
    Calcule L1(λ) sur les scénarios libres, différentiable en λ.

    lam     : (N_z*m*n_g,) ou (N_z, m, n_g) — sortie brute du GNN
    data    : graphe avec A_g (N_z, m, n_g), b_res_g (N_z, m), c_g (n_g,)

    Probabilités : p_i = 1/N (N original), p_ij = 1/(N*m).
    Seuil de sélection : Σ p_i y_i ≥ (1−ε) − n_fixed_1_z/N
      (ajusté des scénarios déjà satisfaits ; pour depth=0 → seuil = 1−ε).

    Algorithme (théorème de l'enveloppe) :
      1. Résoudre θ et ζ → x*_θ, x*_ζ  [detach]
      2. Sélection gloutonne y*           [detach]
      3. L1 comme fonction linéaire de λ à x*, y* fixés → autograd
    """
    N_z = data.N_z
    N   = data.N      # N original pour p_i = 1/N
    m   = data.m
    n_g = data.n_g
    n_fixed_1_z = data.n_fixed_1_z
    dev = lam.device

    lam_3d = lam.view(N_z, m, n_g)                        # (N_z, m, n_g)

    A_g     = data.A_g.to(dev)      # (N_z, m, n_g)
    b_res_g = data.b_res_g.to(dev)  # (N_z, m)
    c_g     = data.c_g.to(dev)      # (n_g,)
    epsilon = data.epsilon.item()

    p_unif = 1.0 / N    # probabilité originale par scénario
    inv_m  = 1.0 / m
    p_ij   = p_unif * inv_m

    # ── (A) Λ_sum = Σ_{i free, j} λ[i,j] ────────────────────────────────────
    Lambda_sum = lam_3d.sum(dim=(0, 1))                    # (n_g,)

    # ── (B) cost_{i,j}[k] = c[k] + λ[i,j,k] / p_ij − Λ_sum[k] ─────────────
    c_brd = c_g.unsqueeze(0).unsqueeze(0)                  # (1, 1, n_g)
    L_brd = Lambda_sum.unsqueeze(0).unsqueeze(0)           # (1, 1, n_g)
    cost  = c_brd + lam_3d / p_ij - L_brd                 # (N_z, m, n_g)

    # ── Sous-problèmes (hors graphe) ──────────────────────────────────────────
    cost_np = cost.detach().cpu().numpy()
    A_np    = A_g.cpu().numpy()
    bres_np = b_res_g.cpu().numpy()

    x_theta_np = (cost_np > 0).astype(np.float32)
    x_zeta_np  = np.zeros_like(x_theta_np)
    for i in range(N_z):
        for j in range(m):
            _, x_z = _solve_zeta_bb(cost_np[i, j], A_np[i, j], bres_np[i, j])
            x_zeta_np[i, j] = x_z

    # ── (C) Δ_i, sélection gloutonne y* ──────────────────────────────────────
    theta_np = np.sum(np.maximum(cost_np, 0.0), axis=2)   # (N_z, m)
    zeta_np  = np.sum(cost_np * x_zeta_np,      axis=2)   # (N_z, m)
    Delta_np = np.maximum(0.0, theta_np - zeta_np).mean(axis=1)  # (N_z,)

    # Seuil corrigé des scénarios déjà satisfaits (z_mask==1)
    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_1_z * p_unif)
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

    # ── (D) L1 linéaire en λ à x*, y* fixés ──────────────────────────────────
    x_theta_t = torch.tensor(x_theta_np, device=dev)
    x_zeta_t  = torch.tensor(x_zeta_np,  device=dev)
    y_star_t  = torch.tensor(y_star,     device=dev)      # (N_z,)

    theta_lam = (cost * x_theta_t).sum(dim=2)                       # (N_z, m)
    delta_lam = (cost * (x_theta_t - x_zeta_t)).sum(dim=2)         # (N_z, m)

    L1_theta = p_ij * theta_lam.sum()
    val_lam  = (p_unif * inv_m * y_star_t.unsqueeze(1) * delta_lam).sum()
    return L1_theta - val_lam


# ─────────────────────────────────────────────────────────────────────────────
# Overfit sur un graphe — validation de l'architecture
# ─────────────────────────────────────────────────────────────────────────────

def verify_v1cld(
    data: pyg_data.Data,
    T: int = 500,
    precision: float | None = None,
    patience: int | None = None,
    lam_init: np.ndarray | None = None,
    return_n_iters: bool = False,
) -> float | tuple[float, int]:
    """
    Reproduit l'algorithme de v1cld_solver.cpp en Python pur
    (subgradient 1/√(t+1)) et retourne la meilleure L1 vue.

    Args:
        lam_init     : warm start (N_z, m, n_g) float64 ; None = cold start λ=0.
        return_n_iters : si True, retourne (best_L1, n_iters_effectuées).

    Arrêt anticipé : si precision et patience sont fournis, s'arrête dès que
    la meilleure L1 n'a pas amélioré de plus de `precision` sur `patience`
    itérations consécutives.
    """
    N_z = data.N_z
    N   = data.N
    m   = data.m
    n_g = data.n_g
    n_fixed_1_z = data.n_fixed_1_z
    epsilon   = data.epsilon.item()
    A_np      = data.A_g.numpy()     # (N_z, m, n_g)
    b_np      = data.b_res_g.numpy() # (N_z, m)
    c_np      = data.c_g.numpy()     # (n_g,)

    p_unif = 1.0 / N
    inv_m  = 1.0 / m
    p_ij   = p_unif * inv_m
    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_1_z * p_unif)

    lam = lam_init.copy() if lam_init is not None else np.zeros((N_z, m, n_g), dtype=np.float64)
    best = np.inf
    no_improve = 0
    n_iters = T

    for t in range(T):
        # (A) Λ_sum
        Lsum = lam.sum(axis=(0, 1))                            # (n_g,)

        # (B) cost, θ, ζ
        cost = c_np + lam / p_ij - Lsum                       # (N_z, m, n_g)

        x_theta = np.zeros((N_z, m, n_g), dtype=np.float64)
        x_zeta  = np.zeros((N_z, m, n_g), dtype=np.float64)
        for i in range(N_z):
            for j in range(m):
                x_theta[i, j] = (cost[i, j] > 0).astype(np.float64)
                _, xz = _solve_zeta_bb(cost[i, j].astype(np.float32),
                                       A_np[i, j].astype(np.float32),
                                       float(b_np[i, j]))
                x_zeta[i, j] = xz

        theta_vals = np.sum(np.maximum(cost, 0.0), axis=2)    # (N_z, m)
        delta_ij   = np.maximum(0.0, theta_vals - np.sum(cost * x_zeta, axis=2))
        Delta      = delta_ij.mean(axis=1)                     # (N_z,)

        # (C) sélection gloutonne y* avec seuil corrigé
        order  = np.argsort(Delta)
        y_star = np.zeros(N_z); cumprob = 0.0
        for idx in order:
            if required_prob <= 0.0:
                break
            y_star[idx] = 1.0; cumprob += p_unif
            if cumprob >= required_prob - 1e-9: break

        val = np.sum(p_unif * y_star * Delta)
        L1  = p_ij * theta_vals.sum() - val
        if L1 < best - (precision or 0.0):
            best = L1
            no_improve = 0
        else:
            no_improve += 1
        if patience is not None and precision is not None and no_improve >= patience:
            n_iters = t + 1
            break

        # (E) x̄
        active = y_star > 0.5
        x_bar  = np.zeros(n_g)
        for i in range(N_z):
            for j in range(m):
                x_bar += p_ij * (x_zeta[i, j] if active[i] else x_theta[i, j])

        # (F) pas 1/√(t+1) — l'indice t est global même en warm start
        step = 1.0 / np.sqrt(t + 1)

        # (G) mise à jour λ
        for i in range(N_z):
            for j in range(m):
                xij = x_zeta[i, j] if active[i] else x_theta[i, j]
                lam[i, j] += step * (x_bar - xij)

    if return_n_iters:
        return best, n_iters
    return best


def overfit_one_graph(data: pyg_data.Data, n_epochs: int = 1000, lr: float = 1e-3):
    """
    Entraîne le GNN sur un seul graphe avec la vraie borne V1CLD comme loss.
    """
    device = torch.device("cpu")
    data   = data.to(device)

    model     = GNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    opt_val = data.opt.item()
    # L1(λ=0) comme référence de départ
    with torch.no_grad():
        lam0    = torch.zeros(data.N_z * data.m * data.n_g)
        l1_init = lagrangian_v1cld(lam0, data).item()

    print(f"\n{'='*60}")
    print(f"Overfit — n_g={data.n_g}  N={data.N}  m={data.m}")
    print(f"  opt={opt_val:.2f}   L1(λ=0)={l1_init:.2f}")
    print(f"  graphe : {data.x.shape[0]} nœuds, {data.edge_index.shape[1]} arêtes")
    print(f"{'='*60}")
    print(f"{'epoch':>6}  {'L1(λ_GNN)':>12}  {'gap_vs_opt%':>12}  {'gap_vs_L1_0%':>13}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lam  = model(data)
        loss = lagrangian_v1cld(lam, data)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            b     = loss.item()
            g_opt = (b - opt_val) / opt_val * 100
            g_l0  = (b - l1_init) / l1_init * 100
            print(f"{epoch:6d}  {b:12.4f}  {g_opt:+11.2f}%  {g_l0:+12.2f}%")

    print(f"{'='*60}")
    print(f"Borne finale : {loss.item():.4f}  (opt={opt_val:.2f}, L1_0={l1_init:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train_v1cld")
    n_val    = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    mode     = sys.argv[3] if len(sys.argv) > 3 else "inspect"  # "inspect" ou "overfit"

    inst_file  = data_dir / f"instances_n{n_val}.txt"
    nodes_file = data_dir / f"nodes_n{n_val}.txt"

    graphs = load_dataset(inst_file, nodes_file, max_graphs=5)

    for i, g in enumerate(graphs):
        n_nodes = g.x.shape[0]
        n_edges = g.edge_index.shape[1]
        print(
            f"  graph {i:3d} | nodes={n_nodes:6d} | edges={n_edges:8d} | "
            f"n_g={g.n_g} N={g.N} m={g.m} | "
            f"depth={int(g.context[0].item())} | opt={g.opt.item():.2f}"
        )

    if mode in ("verify", "overfit") and graphs:
        g0 = graphs[0]
        # L1(λ=0) — référence de départ du subgradient
        lam0 = torch.zeros(g0.N_z * g0.m * g0.n_g)
        l1_init = lagrangian_v1cld(lam0, g0).item()
        print(f"\n── Vérification V1CLD (T=200 itérations subgradient) ──")
        print(f"  L1(λ=0)            : {l1_init:.4f}  ← vraie borne Lagrangienne au départ")
        py_best = verify_v1cld(g0, T=200)
        print(f"  subgradient T=200  : {py_best:.4f}  ← meilleure L1 après 200 itérations")
        print(f"  opt                : {g0.opt.item():.4f}  ← optimum (borne inférieure)")
        print(f"  amélioration vs L1(0) : {(l1_init - py_best) / l1_init * 100:.1f}%")

    if mode == "overfit" and graphs:
        overfit_one_graph(graphs[0])
