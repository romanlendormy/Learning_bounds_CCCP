"""
train_gnn_bipartite.py — GNN bipartite pour la borne V1LD sur CC-MKP.

Architecture bipartite HeteroData :
  - Nœuds-item  : (n_g, 2)       [c_k, k/n_g]
  - Nœuds-scén  : (N_z, 2m+1)    [b_res_i, b_res_i/b_i, i/N_z]
  - Arêtes bidirectionnelles item→scen et scen→item avec 4m edge features
  - 3 aller-retours de message-passing bipartite (GeneralConv)

Réutilise load_raw_nodes et lagrangian_v1ld depuis gnn.train_gnn_v1ld.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from torch_geometric.data import HeteroData
from torch_geometric.nn import GeneralConv

# Réutilisation des fonctions de chargement et du calcul de borne
from gnn.train_gnn_v1ld import load_raw_nodes   # noqa: F401 (réexporté)


# ─────────────────────────────────────────────────────────────────────────────
# Construction du graphe bipartite
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_bipartite(inst: dict, node_rec: dict) -> Optional[HeteroData]:
    """
    Construit un HeteroData PyG bipartite item ↔ scénario.

    Nœuds-item  (n_g, 2)     : [c_k, k/n_g]
    Nœuds-scén  (N_z, 2m+1)  : [b_res_i[0..m-1], b_res_i/max(b_i,1e-9)[0..m-1], i/N_z]

    Arêtes item→scen et scen→item — ENTIÈREMENT vectorisé, zéro boucle Python.
    Edge features (E, 4m) :
        A_ijk                   shape (E, m)
        A_ijk / max(b_res, ε)   shape (E, m)
        c_k  / max(A_ijk, 1.0)  shape (E, m)
        Dantzig_rank            shape (E, m)

    Attributs supplémentaires pour compatibilité avec lagrangian_v1ld :
        A_g, b_res_g, c_g, N, N_z, n_g, n_fixed_z1, epsilon, context, opt

    Retourne None si n_g == 0 ou N_z == 0.
    """
    n, m, N  = inst["n"], inst["m"], inst["N"]
    c, b, A  = inst["c"], inst["b"], inst["A"]

    x_mask = node_rec["x_mask"]   # (n,)  int8
    b_res  = node_rec["b_res"]    # (N, m)  float32
    z_mask = node_rec.get("z_mask", None)

    if z_mask is not None:
        # Format B/C : x_mask ternaire, z_mask explicite
        active_x   = np.where(x_mask == 2)[0]
        free_z     = np.where(z_mask == 2)[0]
        n_fixed_z1 = int((z_mask == 1).sum())
    else:
        # Format A : x_mask binaire, free_z déduit de b_res
        active_x   = np.where(x_mask == 0)[0]
        free_z     = np.where(np.all(b_res >= -1e-9, axis=1))[0]
        n_fixed_z1 = N - len(free_z)

    n_g = len(active_x)
    N_z = len(free_z)
    if n_g == 0 or N_z == 0:
        return None

    # ── Sous-matrices : scénarios libres × items actifs ───────────────────────
    b_res_z = b_res[free_z, :]                          # (N_z, m)
    A_g     = A[np.ix_(free_z, range(m), active_x)]     # (N_z, m, n_g)
    c_g     = c[active_x]                               # (n_g,)
    b_z     = b[free_z, :]                              # (N_z, m)

    # ── Nœuds-item (n_g, 1+2m) ──────────────────────────────────────────────
    # A_g : (N_z, m, n_g) → mean/std sur les scénarios → (m, n_g) → (n_g, m)
    A_mean = A_g.mean(axis=0).T.astype(np.float32)      # (n_g, m)
    A_std  = A_g.std(axis=0).T.astype(np.float32)       # (n_g, m)
    x_item = np.concatenate(
        [c_g[:, None], A_mean, A_std], axis=1
    ).astype(np.float32)                                  # (n_g, 1+2m)

    # ── Nœuds-scén (N_z, 2m) ─────────────────────────────────────────────────
    b_ratio = b_res_z / np.maximum(b_z, 1e-9)           # (N_z, m)
    x_scen  = np.concatenate(
        [b_res_z, b_ratio], axis=1
    ).astype(np.float32)                                  # (N_z, 2m)

    # ── Arêtes — vectorisé (zéro boucle Python) ───────────────────────────────
    k_idx = np.arange(n_g, dtype=np.int64)
    i_idx = np.arange(N_z, dtype=np.int64)
    kk, ii = np.meshgrid(k_idx, i_idx, indexing='ij')   # (n_g, N_z)
    kk = kk.ravel()                                      # (n_g*N_z,)
    ii = ii.ravel()                                      # (n_g*N_z,)
    E = n_g * N_z

    # Edge feature 1 : A_ijk  shape (E, m)
    feat_A = A_g[ii, :, kk]                              # (E, m)

    # Edge feature 2 : A_ijk / max(b_res_i, eps)  shape (E, m)
    bres_i = b_res_z[ii, :]                              # (E, m)
    feat_A_over_bres = feat_A / np.maximum(bres_i, 1e-9)

    # Edge feature 3 : c_k / max(A_ijk, 1.0)  shape (E, m)
    feat_c_over_A = c_g[kk, None] / np.maximum(feat_A, 1.0)  # (E, m)

    # Edge feature 4 : Dantzig rank normalisé  shape (E, m)
    # eff[i, j, k] = c_k / max(A_ijk, 1e-9) — (N_z, m, n_g)
    eff = c_g[None, None, :] / np.maximum(A_g, 1e-9)    # (N_z, m, n_g)
    # argsort sur axis=2 → rang de chaque item par (scen, ressource)
    dantzig_rank = (
        np.argsort(np.argsort(-eff, axis=2), axis=2)
        / max(n_g - 1, 1)
    ).astype(np.float32)                                  # (N_z, m, n_g)
    feat_rank = dantzig_rank[ii, :, kk]                  # (E, m)

    # Concaténation : (E, 4m)
    edge_feat = np.concatenate(
        [feat_A, feat_A_over_bres, feat_c_over_A, feat_rank], axis=1
    ).astype(np.float32)

    # ── Contexte graphe ───────────────────────────────────────────────────────
    depth      = float(node_rec["depth"])
    frac_fixed = 1.0 - n_g / n
    curr_cost  = float(node_rec["curr_cost"])
    cap_frac   = b_res_z / np.maximum(b_z, 1e-9)
    tightest   = float(np.min(cap_frac))
    epsilon    = float(inst["epsilon"])

    context = torch.tensor(
        [depth, frac_fixed, tightest, curr_cost, epsilon, n_fixed_z1 / N],
        dtype=torch.float32,
    )

    # ── Construction HeteroData ───────────────────────────────────────────────
    data = HeteroData()

    data['item'].x = torch.from_numpy(x_item)          # (n_g, 2)
    data['scen'].x = torch.from_numpy(x_scen)          # (N_z, 2m+1)

    ei_i2s = torch.from_numpy(np.stack([kk, ii]).astype(np.int64))  # (2, E)
    ei_s2i = torch.from_numpy(np.stack([ii, kk]).astype(np.int64))  # (2, E)
    ea     = torch.from_numpy(edge_feat)                             # (E, 4m)

    data['item', 'to', 'scen'].edge_index = ei_i2s
    data['item', 'to', 'scen'].edge_attr  = ea
    data['scen', 'to', 'item'].edge_index = ei_s2i
    data['scen', 'to', 'item'].edge_attr  = ea   # mêmes features

    # Attributs de compatibilité avec lagrangian_v1ld
    data.context     = context
    data.opt         = torch.tensor(float(inst["opt"]), dtype=torch.float32)
    data.n_g         = n_g
    data.N_z         = N_z
    data.N           = N
    data.n_fixed_z1  = n_fixed_z1
    data.epsilon     = torch.tensor(epsilon, dtype=torch.float32)
    data.m           = m
    # Matrices nécessaires pour lagrangian_v1ld
    data.A_g         = torch.from_numpy(A_g.copy())       # (N_z, m, n_g)
    data.b_res_g     = torch.from_numpy(b_res_z.copy())   # (N_z, m)
    data.c_g         = torch.from_numpy(c_g.copy())        # (n_g,)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Dataset lazy bipartite
# ─────────────────────────────────────────────────────────────────────────────

class LazyBipartiteDataset(torch.utils.data.Dataset):
    """
    Dataset PyTorch qui construit les HeteroData bipartites à la volée.

    Interface identique à LazyGraphDataset (gnn.train_gnn_v1ld) :
      inst_ids(), filter_by_inst_ids(), subset(), sample()

    Garde en RAM uniquement les enregistrements bruts.
    """

    def __init__(self, instances: dict, raw_records: list):
        self.instances   = instances
        self.raw_records = raw_records

    def __len__(self) -> int:
        return len(self.raw_records)

    def __getitem__(self, idx: int) -> HeteroData:
        rec  = self.raw_records[idx]
        inst = self.instances[rec["inst_id"]]

        if rec["b_res_explicit"] is not None:
            b_res = rec["b_res_explicit"]
        else:
            fixed_to_1 = np.where(rec["x_mask"] == 1)[0]
            b_res = inst["b"].copy()
            if len(fixed_to_1):
                b_res -= inst["A"][:, :, fixed_to_1].sum(axis=2)

        node_rec: dict = {
            "depth":     rec["depth"],
            "curr_cost": rec["curr_cost"],
            "x_mask":    rec["x_mask"],
            "b_res":     b_res,
        }
        if rec["z_mask"] is not None:
            node_rec["z_mask"] = rec["z_mask"]

        g = build_graph_bipartite(inst, node_rec)
        if g is None:
            raise ValueError(
                f"build_graph_bipartite a retourné None pour inst_id={rec['inst_id']} "
                f"depth={rec['depth']} — ce record aurait dû être filtré au chargement."
            )
        g.inst_id = torch.tensor(rec["inst_id"], dtype=torch.long)
        return g

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def inst_ids(self) -> list[int]:
        return [rec["inst_id"] for rec in self.raw_records]

    def filter_by_inst_ids(self, ids_set: set) -> "LazyBipartiteDataset":
        filtered = [rec for rec in self.raw_records if rec["inst_id"] in ids_set]
        return LazyBipartiteDataset(self.instances, filtered)

    def subset(self, start: int, stop: int) -> "LazyBipartiteDataset":
        return LazyBipartiteDataset(self.instances, self.raw_records[start:stop])

    def sample(self, n: int, rng: np.random.Generator) -> "LazyBipartiteDataset":
        n = min(n, len(self))
        idx = rng.choice(len(self), n, replace=False)
        return LazyBipartiteDataset(self.instances, [self.raw_records[i] for i in idx])


# ─────────────────────────────────────────────────────────────────────────────
# Modèle GNN bipartite
# ─────────────────────────────────────────────────────────────────────────────

class GNNBipartite(nn.Module):
    """
    GNN bipartite item ↔ scénario pour prédire λ[i, k] ∈ R.

    3 aller-retours item→scen puis scen→item avec GeneralConv (supporte
    edge features via in_edge_channels — équivalent de SAGEConv+edge_dim
    disponible dans PyG ≥ 2.8).

    Entrées attendues sur HeteroData (batch_size=1) :
      data['item'].x              (n_g, 1+2m)   [c_k, A_mean(m), A_std(m)]
      data['scen'].x              (N_z, 2m)     [b_res_i(m), b_res_i/b_i(m)]
      data['item','to','scen'].edge_index  (2, E)
      data['item','to','scen'].edge_attr   (E, 4m)
      data['scen','to','item'].edge_index  (2, E)
      data['scen','to','item'].edge_attr   (E, 4m)
      data.context                (6,)
      data.n_g, data.N_z          int

    Sortie : (n_g * N_z,)  — même ordre que build_graph_bipartite
             (kk, ii = meshgrid(arange(n_g), arange(N_z), indexing='ij'))
    """
    CONTEXT_DIM = 6

    def __init__(self, m: int, hidden: int = 64):
        super().__init__()
        self.m      = m
        self.hidden = hidden
        edge_dim    = 4 * m

        self.embed_item = nn.Linear(1 + 2 * m, hidden)
        self.embed_scen = nn.Linear(2 * m, hidden)

        # 3 couches aller-retour — aggr='max' pour capturer les extremes
        self.convs_i2s = nn.ModuleList([
            GeneralConv(
                in_channels=(hidden, hidden),
                out_channels=hidden,
                in_edge_channels=edge_dim,
                aggr='max',
            )
            for _ in range(3)
        ])
        self.convs_s2i = nn.ModuleList([
            GeneralConv(
                in_channels=(hidden, hidden),
                out_channels=hidden,
                in_edge_channels=edge_dim,
                aggr='max',
            )
            for _ in range(3)
        ])
        self.norms_item = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.norms_scen = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])

        self.proj_item = nn.Linear(hidden, 2 * hidden)
        self.proj_scen = nn.Linear(hidden, 2 * hidden)

        # head voit [h_k | h_i | ea_ki | ctx] — edge features directes pour la paire (i,k)
        head_in = 4 * hidden + edge_dim + self.CONTEXT_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        # ── Embeddings initiaux ───────────────────────────────────────────────
        h_item = torch.relu(self.embed_item(data['item'].x))   # (n_g, hidden)
        h_scen = torch.relu(self.embed_scen(data['scen'].x))   # (N_z, hidden)

        ei_i2s = data['item', 'to', 'scen'].edge_index    # (2, E)
        ei_s2i = data['scen', 'to', 'item'].edge_index    # (2, E)
        ea     = data['item', 'to', 'scen'].edge_attr      # (E, 4m)

        # ── 3 aller-retours bipartites ────────────────────────────────────────
        for r in range(3):
            # item → scen : messages des items vers les scénarios
            h_scen_new = self.convs_i2s[r](
                (h_item, h_scen), ei_i2s, edge_attr=ea
            )
            h_scen_new = self.norms_scen[r](h_scen_new)
            h_scen = h_scen_new + (h_scen if r > 0 else 0)

            # scen → item : messages des scénarios vers les items
            h_item_new = self.convs_s2i[r](
                (h_scen, h_item), ei_s2i, edge_attr=ea
            )
            h_item_new = self.norms_item[r](h_item_new)
            h_item = h_item_new + (h_item if r > 0 else 0)

        # ── Projection finale ─────────────────────────────────────────────────
        h_item = self.proj_item(h_item)   # (n_g, 2*hidden)
        h_scen = self.proj_scen(h_scen)   # (N_z, 2*hidden)

        # ── Broadcast sur toutes les paires (k, i) ────────────────────────────
        # Ordre identique à build_graph_bipartite : meshgrid(arange(n_g), arange(N_z), indexing='ij')
        n_g = data.n_g if isinstance(data.n_g, int) else int(data.n_g.item())
        N_z = data.N_z if isinstance(data.N_z, int) else int(data.N_z.item())
        dev = h_item.device

        k_idx = torch.arange(n_g, device=dev)
        i_idx = torch.arange(N_z, device=dev)
        kk, ii = torch.meshgrid(k_idx, i_idx, indexing='ij')  # (n_g, N_z)
        kk = kk.ravel()   # (n_g*N_z,)
        ii = ii.ravel()   # (n_g*N_z,)

        h_k = h_item[kk]   # (n_g*N_z, 2*hidden)
        h_i = h_scen[ii]   # (n_g*N_z, 2*hidden)

        # ea est ordonné par meshgrid(k_idx, i_idx, indexing='ij') — même ordre que kk/ii
        # → ea[j] correspond exactement à la paire (kk[j], ii[j])
        ea_pairs = ea   # (n_g*N_z, 4m)

        ctx = data.context
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        ctx_exp = ctx.expand(n_g * N_z, -1)   # (n_g*N_z, 6)

        inp = torch.cat([h_k, h_i, ea_pairs, ctx_exp], dim=-1)   # (n_g*N_z, 4h+4m+6)
        lam = self.head(inp).squeeze(-1)                # (n_g*N_z,)
        return lam
