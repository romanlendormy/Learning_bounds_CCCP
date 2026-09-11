#!/usr/bin/env python3
"""
gnn_sld_scriptable.py — Réimplémentation de gnn.train_gnn_sld.GNN en torch
pur (sans torch_geometric), scriptable via torch.jit.script, pour inférence
C++ via torch::jit::load() (cf. solver_bab/mkp/gnn_sld_native.cpp).

Pourquoi une réimplémentation et pas juste torch.jit.script(model) directement :
torch_geometric.data.Data utilise des méthodes à arguments variables
(*args, **kwargs) que TorchScript ne compile pas ("Compiled functions can't
take variable number of arguments"). GatedGraphConv/MessagePassing héritent
de cette incompatibilité par transitivité.

Cette classe prend des tenseurs explicites (X, edge_index, edge_weight,
context, n_g) au lieu d'un objet Data, et réimplémente GatedGraphConv à la
main (matmul par couche + index_add_ pour l'agrégation + GRUCell) — même
algorithme que torch_geometric.nn.GatedGraphConv (cf. sa docstring), vérifié
par la référence PyG plus bas. Le pooling item/scénario exploite l'invariant
"un seul graphe par appel B&B, nœuds ordonnés i*n_g+k" : un simple reshape
remplace le scatter par batch générique de PyG (valide uniquement en dehors
d'un batch — jamais utilisé pour l'entraînement, seulement l'inférence).

Les noms de paramètres (embed.*, conv1.weight, conv1.rnn.*, ...) sont
IDENTIQUES à gnn.train_gnn_sld.GNN, donc load_state_dict(strict=True)
fonctionne directement depuis un checkpoint existant.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


class GatedGraphConvScriptable(nn.Module):
    """Port de torch_geometric.nn.GatedGraphConv (aggr='add'), sans PyG."""

    def __init__(self, hidden: int, num_layers: int):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.weight = nn.Parameter(torch.zeros(num_layers, hidden, hidden))
        self.rnn = nn.GRUCell(hidden, hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        v = x.size(0)
        for l in range(self.num_layers):
            m = torch.matmul(x, self.weight[l])
            msg = edge_weight.unsqueeze(-1) * m.index_select(0, src)
            agg = torch.zeros((v, self.hidden), dtype=x.dtype, device=x.device)
            agg.index_add_(0, dst, msg)
            x = self.rnn(agg, x)
        return x


class GNNScriptable(nn.Module):
    """
    Port de gnn.train_gnn_sld.GNN pour un graphe UNIQUE (jamais batché —
    toujours le cas en B&B, un appel = un nœud = un graphe).

    forward(X, edge_index, edge_weight, context, n_g) -> lam (V,)
    lam est BRUT (pas encore mis à l'échelle par c_scale/N — fait par l'appelant,
    comme gnn_sld_native.cpp, car c_scale/N ne sont pas des poids du modèle).

    Le recentrage (center_lam, cf. gnn_inference_server.py::apply_center) est
    appliqué ICI, sur h_out avant mise à l'échelle — licite car centering et
    multiplication par un scalaire (c_scale/N, identique pour tout le graphe)
    commutent : (h - mean(h))·s = h·s - mean(h·s). Ça permet au module
    exporté d'être autonome (pas de méta-fichier séparé à lire côté C++).
    """

    def __init__(self, m: int, hidden: int, use_max_pool: bool, center_lam: bool):
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
                context: torch.Tensor, n_g: int) -> torch.Tensor:
        v = X.size(0)
        n_z = v // n_g

        ctx_norm = self.norm_ctx(context.unsqueeze(0))   # (1, 9)

        h = self.norm_embed(self.embed(X))
        h1 = self.norm1(self.conv1(h, edge_index, edge_weight))
        h2 = self.norm2(self.conv2(h1, edge_index, edge_weight)) + h1
        h_v = self.proj(h2)                              # (V, 2h)

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


def export(checkpoint_path: str, out_path: str, device_str: str = "cpu") -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt["args"]
    m, hidden = int(args["m"]), int(args["hidden"])
    use_max_pool = bool(args.get("use_max_pool", False))
    center_lam = bool(args.get("center_lam", False))

    model = GNNScriptable(m, hidden, use_max_pool, center_lam)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    device = torch.device(device_str)
    model = model.to(device)

    scripted = torch.jit.script(model)
    scripted.save(out_path)
    print(f"[gnn_sld_scriptable] TorchScript exporté vers {out_path} "
          f"(device={device_str}, center_lam={center_lam})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = p.parse_args()
    export(a.checkpoint, a.out, a.device)
