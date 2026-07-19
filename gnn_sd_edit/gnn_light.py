#!/usr/bin/env python3
"""
gnn_light.py — Variante allégée de gnn.gnn_v1ld_scriptable.GNNScriptable
(architecture sd_edit) pour réduire le coût d'inférence en B&B.

Deux changements par rapport à GNNScriptable, rien d'autre :
  - 1 seule couche de message passing (conv1 seul ; conv2 + son skip
    h1 sont retirés) au lieu de 2 (conv1 puis conv2 + skip(h1)).
  - hidden=64 par défaut au lieu de 128.

Le reste (embed, pooling item/scénario/graphe, tête, contexte 9-dim,
center_lam) est identique à GNNScriptable — cf. gnn_sd_edit/train_v1LD_sd_edit.py
et gnn/gnn_v1ld_scriptable.py pour la référence complète. Réutilise
GatedGraphConvScriptable telle quelle (pas de duplication).

Nouvelle architecture => pas de state_dict compatible avec un checkpoint
sd_edit existant (hidden et nombre de couches diffèrent) : ce module sert
pour l'instant à comparer le COÛT D'INFÉRENCE (dépend de la forme des
tenseurs et du nombre d'opérations, pas des poids), pas la qualité de
borne — un entraînement dédié serait nécessaire avant tout usage en B&B.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn.gnn_v1ld_scriptable import GatedGraphConvScriptable  # noqa: E402


class GNNLightScriptable(nn.Module):
    """
    Port allégé de GNNScriptable : 1 couche de message passing (conv1),
    hidden=64 par défaut. Même signature forward(X, edge_index, edge_weight,
    context, n_g) -> lam (V,) brut (non mis à l'échelle par c_scale/N).
    """

    def __init__(self, m: int, hidden: int = 64, use_max_pool: bool = False, center_lam: bool = False):
        super().__init__()
        self.m = m
        self.hidden = hidden
        self.use_max_pool = use_max_pool
        self.center_lam = center_lam
        self.CONTEXT_DIM = 9   # attribut d'instance (pas de classe) : requis par TorchScript
        node_dim = 7 * m + 5

        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.norm_embed = nn.LayerNorm(hidden)
        self.conv1 = GatedGraphConvScriptable(hidden, 2)
        self.norm1 = nn.LayerNorm(hidden)
        self.proj = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.ReLU(),
            nn.Linear(4 * hidden, 2 * hidden),
        )
        self.norm_ctx = nn.LayerNorm(self.CONTEXT_DIM)

        pool_dim = 4 * hidden if use_max_pool else 2 * hidden
        head_in = 2 * hidden + pool_dim + pool_dim + 2 * hidden + self.CONTEXT_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                context: torch.Tensor, n_g: int) -> torch.Tensor:
        v = X.size(0)
        n_z = v // n_g

        ctx_norm = self.norm_ctx(context.unsqueeze(0))   # (1, 9)

        h = self.norm_embed(self.embed(X))
        h_v = self.proj(self.norm1(self.conv1(h, edge_index, edge_weight)))   # (V, 2h)

        h_v_3d = h_v.view(n_z, n_g, h_v.size(-1))
        h_item_mean = h_v_3d.mean(0)
        h_scen_mean = h_v_3d.mean(1)
        if self.use_max_pool:
            h_item_max, _ = h_v_3d.max(0)
            h_scen_max, _ = h_v_3d.max(1)
            h_item = torch.cat([h_item_mean, h_item_max], dim=-1)
            h_scen = torch.cat([h_scen_mean, h_scen_max], dim=-1)
        else:
            h_item = h_item_mean
            h_scen = h_scen_mean

        h_item_exp = h_item.unsqueeze(0).expand(n_z, n_g, h_item.size(-1)).reshape(v, h_item.size(-1))
        h_scen_exp = h_scen.unsqueeze(1).expand(n_z, n_g, h_scen.size(-1)).reshape(v, h_scen.size(-1))

        g_pool = h_v.mean(0, keepdim=True)                # (1, 2h)
        g_exp = g_pool.expand(v, g_pool.size(-1))
        ctx_exp = ctx_norm.expand(v, self.CONTEXT_DIM)

        cat = torch.cat([h_v, h_item_exp, h_scen_exp, g_exp, ctx_exp], dim=-1)
        h_out = self.head(cat).squeeze(-1)
        if self.center_lam:
            h_out_2d = h_out.view(n_z, n_g)
            h_out = (h_out_2d - h_out_2d.mean(0, keepdim=True)).reshape(-1)
        return h_out


class GNNLight(nn.Module):
    """
    Port entraînable (torch_geometric, autograd) de GNNLightScriptable —
    même relation que gnn_sd_edit.train_v1LD_sd_edit.GNN vis-à-vis de
    gnn.gnn_v1ld_scriptable.GNNScriptable : ce module prend un objet
    pyg_data.Data (comme GNN) et sert à l'entraînement/l'overfit, pas à
    l'export TorchScript.

    Identique à gnn_sd_edit.train_v1LD_sd_edit.GNN sauf 1 couche de message
    passing (conv1 seul, pas de conv2/skip) et hidden=64 par défaut au lieu
    de 128 en production — cf. gnn_light.py (module) pour le rationale complet.
    """
    CONTEXT_DIM = 9

    def __init__(self, m: int = 2, hidden: int = 64, use_max_pool: bool = False):
        super().__init__()
        self.m            = m
        self.hidden       = hidden
        self.use_max_pool = use_max_pool
        node_dim    = 7 * m + 5

        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.norm_embed = nn.LayerNorm(hidden)
        self.conv1 = GatedGraphConv(hidden, num_layers=2)
        self.norm1 = nn.LayerNorm(hidden)
        self.proj  = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.ReLU(),
            nn.Linear(4 * hidden, 2 * hidden),
        )
        self.norm_ctx = nn.LayerNorm(self.CONTEXT_DIM)
        item_pool_dim = 4 * hidden if use_max_pool else 2 * hidden
        scen_pool_dim = 4 * hidden if use_max_pool else 2 * hidden

        head_in = 2 * hidden + item_pool_dim + scen_pool_dim + 2 * hidden + self.CONTEXT_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data, return_hidden: bool = False):
        x, ei, ew = data.x, data.edge_index, data.edge_weight
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        n_v      = x.size(0)
        dev      = x.device
        n_graphs = int(batch.max().item()) + 1

        ctx = data.context
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        ctx_norm = self.norm_ctx(ctx)                                  # (1 or B, 9)

        h_embed = self.norm_embed(self.embed(x))
        h1  = self.norm1(self.conv1(h_embed, ei, ew))
        h_v = self.proj(h1)                                        # (V, 2h)

        n_g = data.n_g if isinstance(data.n_g, int) else int(data.n_g.item())

        counts   = torch.bincount(batch, minlength=n_graphs)
        node_off = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        node_off[1:] = counts[:-1].cumsum(0)
        local_idx = torch.arange(n_v, dtype=torch.long, device=dev) - node_off[batch]

        item_off   = torch.arange(n_graphs, device=dev) * n_g
        item_batch = item_off[batch] + (local_idx % n_g)            # (V,)

        h_item_mean = global_mean_pool(h_v, item_batch)               # (Σ n_g, 2h)
        if self.use_max_pool:
            h_item_max = global_max_pool(h_v, item_batch)          # (Σ n_g, 2h)
            h_item = torch.cat([h_item_mean, h_item_max], dim=-1)  # (Σ n_g, 4h)
        else:
            h_item = h_item_mean                                    # (Σ n_g, 2h)
        h_item_exp = h_item[item_batch]                            # (V, 2h or 4h)

        n_scen_per = counts // n_g
        scen_off   = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        scen_off[1:] = n_scen_per[:-1].cumsum(0)
        scen_batch = scen_off[batch] + local_idx // n_g            # (V,)

        h_scen_mean = global_mean_pool(h_v, scen_batch)              # (Σ N_z, 2h)
        if self.use_max_pool:
            h_scen_max = global_max_pool(h_v, scen_batch)         # (Σ N_z, 2h)
            h_scen = torch.cat([h_scen_mean, h_scen_max], dim=-1) # (Σ N_z, 4h)
        else:
            h_scen = h_scen_mean                                   # (Σ N_z, 2h)
        h_scen_exp = h_scen[scen_batch]                            # (V, 2h or 4h)

        ctx_exp = ctx_norm[batch]                                          # (V, 9)

        g_pool  = global_mean_pool(h_v, batch)                        # (B, 2h)
        g_exp   = g_pool[batch]                                            # (V, 2h)

        cat     = torch.cat([h_v, h_item_exp, h_scen_exp, g_exp, ctx_exp], dim=-1)
        h_out   = self.head(cat).squeeze(-1)

        lam = h_out * data.c_scale / float(data.N)
        if return_hidden:
            extra = {"x": x, "h_embed": h_embed, "h1": h1,
                     "h_v": h_v, "h_item_mean": h_item_mean, "item_batch": item_batch,
                     "h_scen_mean": h_scen_mean, "scen_batch": scen_batch,
                     "g_pool": g_pool, "h_out": h_out}
            return lam, extra
        return lam
