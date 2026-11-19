#!/usr/bin/env python3
"""
gnn_dld_sd_edit_scriptable_batched.py — Variante scriptable BATCHÉE de
train_DLD_sd_edit.GNN, sur le modèle de gnn_sd_edit_scriptable_batched.py
(SLD) : bloc-diagonal (B graphes empilés, topologie/poids identiques par
construction du bucketing par n_g — cf. docstring de ce dernier fichier pour
la justification complète, reprise ici sans modification).

Différences avec la version SLD batchée :
  - NODE_DIM=6 fixe (pas 7*m+5) — pas de paramètre `m` au constructeur,
    seulement à l'exécution (comme le modèle non-batché DLD).
  - Un pooling de plus (contrainte j, en plus de item k et scénario i) —
    donc `m` doit être passé au forward, en plus de n_g/B.
  - Ordre des nœuds dans UN graphe : i*(m*n_g) + j*n_g + k (k le plus
    rapide, cf. train_DLD_sd_edit.py::build_graph). Comme n_g, m ET N_z
    sont IDENTIQUES pour tous les graphes du batch (même raison que pour
    n_g côté SLD : bucketing par n_g implique m/N_z fixes puisque
    l'instance en cours de résolution est la même), un simple
    `.view(B, n_z, m, n_g, dim)` suffit pour regrouper les nœuds par
    (scénario, contrainte, item) sur tout le batch d'un coup — pas besoin
    de l'arithmétique d'index (bincount/local_idx) du forward non-batché
    (qui gère un PyG Batch hétérogène générique).
  - Rescale par c_scale/(N*m) au lieu de c_scale/N (cf. lagrangian_dld :
    p_ij = 1/(N*m), l'échelle naturelle de λ[i,j,k]).

Mêmes noms de paramètres que train_DLD_sd_edit.GNN → load_state_dict
(strict=True) fonctionne directement depuis le checkpoint de production
(aucun réentraînement requis pour valider le batching).

forward(X, edge_index, edge_weight, context, n_g, m, B) :
  X            : (B*V, 6)         — V = N_z * m * n_g, V identique pour tous les B
  edge_index   : (2, B*E)         — topologie dupliquée B fois avec offset b*V
  edge_weight  : (B*E,)           — poids dupliqués B fois (identiques par construction)
  context      : (B, 9)           — un contexte distinct par graphe du batch
  n_g, m, B    : int              — partagés par construction (bucketing par n_g)
  retour       : lam (B*V,) brut, NON mis à l'échelle par c_scale/(N*m)
                 (comme GNNSdEditScriptableBatched pour SLD — le rescale
                 reste à la charge de l'appelant C++, qui connaît c_scale/N/m
                 par nœud B&B et pas seulement au moment du forward).
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


class GNNDLDSdEditScriptableBatched(nn.Module):
    """
    Port batché de train_DLD_sd_edit.GNN. Identique architecturalement —
    seule la mécanique de pooling (3 vues au lieu de 2) gagne une dimension B.
    Pas de paramètre `m` au constructeur (NODE_DIM=6 fixe, cf. docstring de
    module) — `m` n'intervient qu'au forward, comme dans le modèle non-batché.
    """

    def __init__(self, hidden: int = 128, use_max_pool: bool = True):
        super().__init__()
        self.hidden = hidden
        self.use_max_pool = use_max_pool
        # Attributs d'instance (pas des attributs de classe) : requis pour que
        # TorchScript les reconnaisse (cf. GNNSdEditScriptableBatched, SLD,
        # même convention).
        self.NODE_DIM = 6
        self.CONTEXT_DIM = 9

        self.embed = nn.Sequential(
            nn.Linear(self.NODE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),        nn.ReLU(),
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
        # h_v(2h) + h_item(pool) + h_scen(pool) + h_cons(pool) + g_pool(2h) + ctx(9)
        head_in = 2 * hidden + 3 * pool_dim + 2 * hidden + self.CONTEXT_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                context: torch.Tensor, n_g: int, m: int, B: int) -> torch.Tensor:
        v_total = X.size(0)
        v_per_graph = v_total // B
        n_z = v_per_graph // (m * n_g)
        dim_last = 2 * self.hidden

        ctx_norm = self.norm_ctx(context)   # (B, 9)

        h = self.norm_embed(self.embed(X))
        h1 = self.norm1(self.conv1(h, edge_index, edge_weight))
        h2 = self.norm2(self.conv2(h1, edge_index, edge_weight)) + h1
        h_v = self.proj(h2)                                        # (B*V, 2h)

        # Ordre des nœuds : i*(m*n_g) + j*n_g + k → reshape direct, pas de
        # bincount/index arithmetic (cf. docstring de module).
        h_v_5d = h_v.view(B, n_z, m, n_g, dim_last)

        h_item_mean = h_v_5d.mean(2).mean(1)     # pool sur (i,j) → (B, n_g, 2h)
        h_scen_mean = h_v_5d.mean(3).mean(2)     # pool sur (j,k) → (B, n_z, 2h)
        h_cons_mean = h_v_5d.mean(3).mean(1)     # pool sur (i,k) → (B, m, 2h)
        g_pool      = h_item_mean.mean(1)         # (B, 2h)

        if self.use_max_pool:
            h_item_max = h_v_5d.amax(2).amax(1)
            h_scen_max = h_v_5d.amax(3).amax(2)
            h_cons_max = h_v_5d.amax(3).amax(1)
            h_item = torch.cat([h_item_mean, h_item_max], dim=-1)   # (B, n_g, 4h)
            h_scen = torch.cat([h_scen_mean, h_scen_max], dim=-1)   # (B, n_z, 4h)
            h_cons = torch.cat([h_cons_mean, h_cons_max], dim=-1)   # (B, m, 4h)
        else:
            h_item, h_scen, h_cons = h_item_mean, h_scen_mean, h_cons_mean

        item_dim = h_item.size(-1)
        scen_dim = h_scen.size(-1)
        cons_dim = h_cons.size(-1)

        h_item_exp = (h_item.view(B, 1, 1, n_g, item_dim)
                      .expand(B, n_z, m, n_g, item_dim).reshape(v_total, item_dim))
        h_scen_exp = (h_scen.view(B, n_z, 1, 1, scen_dim)
                      .expand(B, n_z, m, n_g, scen_dim).reshape(v_total, scen_dim))
        h_cons_exp = (h_cons.view(B, 1, m, 1, cons_dim)
                      .expand(B, n_z, m, n_g, cons_dim).reshape(v_total, cons_dim))
        g_exp   = g_pool.view(B, 1, 1, 1, dim_last).expand(B, n_z, m, n_g, dim_last).reshape(v_total, dim_last)
        ctx_exp = ctx_norm.view(B, 1, 1, 1, self.CONTEXT_DIM).expand(
            B, n_z, m, n_g, self.CONTEXT_DIM).reshape(v_total, self.CONTEXT_DIM)

        cat = torch.cat([h_v, h_item_exp, h_scen_exp, h_cons_exp, g_exp, ctx_exp], dim=-1)
        h_out = self.head(cat).squeeze(-1)
        return h_out


def export_batched(checkpoint_path: str, out_path: str, device_str: str = "cpu") -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt["args"]
    hidden = int(args["hidden"])
    use_max_pool = bool(args.get("use_max_pool", False))

    model = GNNDLDSdEditScriptableBatched(hidden, use_max_pool)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    device = torch.device(device_str)
    model = model.to(device)

    scripted = torch.jit.script(model)
    scripted.save(out_path)
    print(f"[gnn_dld_sd_edit_scriptable_batched] TorchScript exporté vers {out_path} "
          f"(device={device_str})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = p.parse_args()
    export_batched(a.checkpoint, a.out, a.device)
