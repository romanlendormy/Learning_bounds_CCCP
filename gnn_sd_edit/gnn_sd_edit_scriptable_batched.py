#!/usr/bin/env python3
"""
gnn_sd_edit_scriptable_batched.py — Variante BATCHÉE de
gnn_sd_edit.gnn_sld_scriptable.GNNScriptable : au lieu d'un seul graphe par appel
forward, accepte B graphes empilés en bloc-diagonal (nœuds du B&B traités par lots).

Pourquoi bloc-diagonal (edge_index/edge_weight dupliqués B fois avec offset
par graphe), pas une dimension batch native sur les arêtes : dans un batch
construit pendant UN run de B&B, tous les graphes partagent n_g (même
profondeur d'items, garanti par le bucketing par n_g) ET N (même instance en
cours de résolution) — donc la topologie ET les poids d'arêtes sont
IDENTIQUES pour tous les membres du batch. Le bloc-diagonal est l'approche la
plus simple et la moins risquée pour un premier test : GatedGraphConvScriptable
(gnn_sd_edit.gnn_sld_scriptable) est réutilisée SANS AUCUNE MODIFICATION — le
scatter/index_add_ générique traite le batch comme un seul grand graphe à
composantes déconnectées, ce qui est mathématiquement identique à B appels
séparés.

Mêmes noms de paramètres que GNNScriptable/gnn_sd_edit.train_SLD_sd_edit.GNN
→ load_state_dict(strict=True) fonctionne directement depuis un checkpoint de
production existant (aucun réentraînement requis pour valider le batching).

forward(X, edge_index, edge_weight, context, n_g, B) :
  X            : (B*V, node_dim)  — V = N_z * n_g, V identique pour tous les B
  edge_index   : (2, B*E)         — topologie dupliquée B fois avec offset b*V
  edge_weight  : (B*E,)           — poids dupliqués B fois (identiques par construction)
  context      : (B, 9)           — un contexte distinct par graphe du batch
  n_g, B       : int              — partagés par construction (bucketing par n_g)
  retour       : lam (B*V,) brut, non mis à l'échelle par c_scale/N (comme GNNScriptable)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn_sd_edit.gnn_sld_scriptable import GatedGraphConvScriptable  # noqa: E402


class GNNSdEditScriptableBatched(nn.Module):
    """
    Port batché de GNNScriptable (2 couches de message passing, hidden=128
    par défaut pour matcher le checkpoint sd_edit de production). Identique
    architecturalement — seule la mécanique de pooling gagne une dimension B.
    """

    def __init__(self, m: int, hidden: int = 128, use_max_pool: bool = True, center_lam: bool = True):
        super().__init__()
        self.m = m
        self.hidden = hidden
        self.use_max_pool = use_max_pool
        self.center_lam = center_lam
        self.CONTEXT_DIM = 9
        node_dim = 7 * m + 5

        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.norm_embed = nn.LayerNorm(hidden)
        self.conv1 = GatedGraphConvScriptable(hidden, 2)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = GatedGraphConvScriptable(hidden, 2)
        self.norm2 = nn.LayerNorm(hidden)
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
                context: torch.Tensor, n_g: int, B: int) -> torch.Tensor:
        v_total = X.size(0)
        v_per_graph = v_total // B
        n_z = v_per_graph // n_g
        dim_last = 2 * self.hidden

        ctx_norm = self.norm_ctx(context)   # (B, 9)

        h = self.norm_embed(self.embed(X))
        h1 = self.norm1(self.conv1(h, edge_index, edge_weight))
        h2 = self.norm2(self.conv2(h1, edge_index, edge_weight)) + h1
        h_v = self.proj(h2)                              # (B*V, 2h)

        h_v_4d = h_v.view(B, n_z, n_g, dim_last)
        h_item_mean = h_v_4d.mean(1)                      # (B, n_g, 2h)
        h_scen_mean = h_v_4d.mean(2)                       # (B, n_z, 2h)
        g_pool = h_item_mean.mean(1)                       # (B, 2h) — même moyenne globale, évite mean(dim=(1,2))

        if self.use_max_pool:
            h_item_max, _ = h_v_4d.max(1)
            h_scen_max, _ = h_v_4d.max(2)
            h_item = torch.cat([h_item_mean, h_item_max], dim=-1)   # (B, n_g, 4h)
            h_scen = torch.cat([h_scen_mean, h_scen_max], dim=-1)   # (B, n_z, 4h)
        else:
            h_item = h_item_mean
            h_scen = h_scen_mean

        item_dim = h_item.size(-1)
        scen_dim = h_scen.size(-1)

        h_item_exp = h_item.unsqueeze(1).expand(B, n_z, n_g, item_dim).reshape(v_total, item_dim)
        h_scen_exp = h_scen.unsqueeze(2).expand(B, n_z, n_g, scen_dim).reshape(v_total, scen_dim)
        g_exp = g_pool.unsqueeze(1).expand(B, v_per_graph, dim_last).reshape(v_total, dim_last)
        ctx_exp = ctx_norm.unsqueeze(1).expand(B, v_per_graph, self.CONTEXT_DIM).reshape(v_total, self.CONTEXT_DIM)

        cat = torch.cat([h_v, h_item_exp, h_scen_exp, g_exp, ctx_exp], dim=-1)
        h_out = self.head(cat).squeeze(-1)

        if self.center_lam:
            h_out_3d = h_out.view(B, n_z, n_g)
            h_out = (h_out_3d - h_out_3d.mean(1, keepdim=True)).reshape(-1)
        return h_out


def make_batched_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor, v_per_graph: int, B: int):
    """Duplique une topologie (2,E)/(E,) B fois avec offset b*v_per_graph par
    graphe — bloc-diagonal, aucune arête inter-graphe. Utilitaire pur Python
    (pas dans le forward scripté : construit une fois par shape, réutilisable
    tant que n_g/N/B ne changent pas)."""
    e = edge_index.size(1)
    offsets = (torch.arange(B, device=edge_index.device) * v_per_graph).view(B, 1, 1)
    ei_rep = edge_index.unsqueeze(0).expand(B, 2, e) + offsets           # (B,2,E)
    edge_index_batched = ei_rep.permute(1, 0, 2).reshape(2, B * e)
    edge_weight_batched = edge_weight.unsqueeze(0).expand(B, e).reshape(B * e)
    return edge_index_batched, edge_weight_batched


def export_batched(checkpoint_path: str, out_path: str, device_str: str = "cpu") -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt["args"]
    m, hidden = int(args["m"]), int(args["hidden"])
    use_max_pool = bool(args.get("use_max_pool", False))
    center_lam = bool(args.get("center_lam", False))

    model = GNNSdEditScriptableBatched(m, hidden, use_max_pool, center_lam)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    device = torch.device(device_str)
    model = model.to(device)

    scripted = torch.jit.script(model)
    scripted.save(out_path)
    print(f"[gnn_sd_edit_scriptable_batched] TorchScript exporté vers {out_path} "
          f"(device={device_str}, center_lam={center_lam})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = p.parse_args()
    export_batched(a.checkpoint, a.out, a.device)
