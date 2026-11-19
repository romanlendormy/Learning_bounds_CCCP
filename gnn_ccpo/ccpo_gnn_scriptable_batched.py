#!/usr/bin/env python3
"""
ccpo_gnn_scriptable_batched.py — Port TorchScript-able, BATCHÉ, de
ccpo_gnn_model.CCPOGNN (cf. ccpo_gnn_model.py) -- sur le patron exact de
gnn_sd_edit_scriptable_batched.py (MKP) : mêmes raisons (torch_geometric.data.Data
et GatedGraphConv/MessagePassing ne sont pas scriptables, cf. docstring
gnn_sld_scriptable.py), même stratégie de batching bloc-diagonal (B graphes
empilés, topologie dupliquée B fois avec offset, SANS arête inter-graphe).

Différence structurelle avec MKP à noter AVANT d'utiliser ce module dans le
futur driver B&B : côté MKP, le graphe contient TOUJOURS les N scénarios
(v_per_graph = N*n_g fixe pour toute l'instance). Côté CCPO, ccpo_graph.py
n'inclut QUE les scénarios ELIGIBLES (non presolve-infeasibles, cf.
ccpo_graph.build_node_features) -- mais comme la présolve est calculée UNE
FOIS par INSTANCE (jamais par nœud, cf. CCPOInstanceRecord.infeasible utilisé
identique à chaque nœud dans build_from_example), n_eligible est CONSTANT
pour TOUS les nœuds d'un même run B&B (une seule instance résolue à la fois,
comme solve_gecode_bab_mkp_gnn_batched côté MKP) -- donc le bucketing par
n_free SEUL suffit à garantir un v_per_graph uniforme dans un batch, exactement
comme le bucketing par n_g suffit côté MKP. Documenté ici car un bucketing par
(n_free, n_eligible) serait une erreur de sur-ingénierie non justifiée.

Mêmes noms de paramètres que CCPOGNN (embed.*, conv1.weight/conv1.rnn.*,
norm1.*, conv2.*, norm2.*, proj.*, norm_ctx.*, head.*) → load_state_dict
(strict=True) fonctionne directement depuis un checkpoint de production
existant (ex. ckpt_ccpo_gnn_mkpscheme_ep0030.pt), aucun réentraînement requis.

Convention de layout des nœuds (DOIT rester synchronisée avec
ccpo_graph.build_node_features) : pour un graphe, nœud d'indice
s*n_free + i correspond au (s-ième scénario ÉLIGIBLE, actif libre i) --
scénario-majeur, actif-mineur. C'est ce qui permet le pooling par simple
reshape (B, n_eligible, n_free, dim) ci-dessous, sans scatter générique.

forward(X, edge_index, context, n_free, n_eligible, B) :
  X          : (B*V, NODE_DIM)   — V = n_eligible * n_free, V identique pour tout le batch
  edge_index : (2, B*E)          — topologie dupliquée B fois avec offset b*V
  context    : (B, CONTEXT_DIM)  — un contexte distinct par membre du batch
  n_free, n_eligible, B : int    — partagés par construction (une instance, un n_free par bucket)
  retour     : h_out (B*V,) BRUT (pas encore mis à l'échelle par c_scale/N,
               comme ccpo_gnn_model.CCPOGNN.forward) -- l'appelant applique
               h_out * c_scale / N, cf. build_lambda_tensor/predict_lambda.

Pas d'edge_weight (aucune pondération d'arête côté CCPO, contrairement à la
variante DLD de MKP) -- GatedGraphConvScriptable réutilisée telle quelle avec
un vecteur de poids constant à 1 (cf. torch.ones dans make_batched_edges
ci-dessous), strictement équivalent à une agrégation non pondérée.
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

NODE_DIM = 10
CONTEXT_DIM = 11


class CCPOGNNScriptableBatched(nn.Module):
    """Port batché, scriptable, de ccpo_gnn_model.CCPOGNN. Mêmes noms de
    paramètres -- voir docstring du module pour le contrat de layout."""

    def __init__(self, hidden: int = 64, gnn_iters: int = 2, use_max_pool: bool = True,
                 node_dim: int = NODE_DIM, feat_version: int = 1, edge_mode: int = 0,
                 center_lam: bool = False, relu_h1: bool = True):
        super().__init__()
        self.hidden = hidden
        self.use_max_pool = use_max_pool
        # Attributs lus par le backend C++ (gnn_ccpo_native_batched.cpp) AVANT freeze
        # pour choisir le constructeur de features et les poids d'arêtes. Absents des
        # anciens exports -> le C++ retombe sur (1, 0) = comportement d'origine.
        self.feat_version = feat_version
        self.edge_mode = edge_mode
        self.center_lam = center_lam
        self.relu_h1 = relu_h1
        # CORRECTIF : TorchScript refuse de fermer sur une
        # variable GLOBALE de module referencee dans forward() ("python value
        # of type 'int' cannot be used as a value... closed over global
        # variable") -- doit etre un attribut self. (meme pattern que MKP :
        # self.CONTEXT_DIM dans GNNSdEditScriptableBatched).
        self.CONTEXT_DIM = CONTEXT_DIM

        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.conv1 = GatedGraphConvScriptable(hidden, gnn_iters)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = GatedGraphConvScriptable(hidden, gnn_iters)
        self.norm2 = nn.LayerNorm(hidden)

        self.proj = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.ReLU(),
            nn.Linear(4 * hidden, 2 * hidden),
        )
        self.h_v_dim = 2 * hidden
        self.norm_ctx = nn.LayerNorm(CONTEXT_DIM)

        pool_mult = 2 if use_max_pool else 1
        head_in = self.h_v_dim + self.h_v_dim * pool_mult + self.h_v_dim * pool_mult + self.h_v_dim + CONTEXT_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                context: torch.Tensor, n_free: int, n_eligible: int, B: int) -> torch.Tensor:
        v_total = X.size(0)
        v_per_graph = n_eligible * n_free
        dim_last = self.h_v_dim

        ctx_norm = self.norm_ctx(context)  # (B, CONTEXT_DIM)

        x = self.embed(X)
        h1 = self.norm1(self.conv1(x, edge_index, edge_weight))
        if self.relu_h1:
            h1 = torch.relu(h1)
        h2 = self.norm2(self.conv2(h1, edge_index, edge_weight)) + h1
        h_v = self.proj(h2)  # (B*V, 2h)

        h_v_4d = h_v.view(B, n_eligible, n_free, dim_last)
        h_asset_mean = h_v_4d.mean(1)   # (B, n_free, 2h) -- pool sur les scenarios, par actif
        h_scen_mean = h_v_4d.mean(2)    # (B, n_eligible, 2h) -- pool sur les actifs, par scenario
        g_pool = h_asset_mean.mean(1)   # (B, 2h) -- moyenne globale (meme valeur, cf. docstring MKP)

        if self.use_max_pool:
            h_asset_max, _ = h_v_4d.max(1)
            h_scen_max, _ = h_v_4d.max(2)
            h_asset = torch.cat([h_asset_mean, h_asset_max], dim=-1)
            h_scen = torch.cat([h_scen_mean, h_scen_max], dim=-1)
        else:
            h_asset = h_asset_mean
            h_scen = h_scen_mean

        asset_dim = h_asset.size(-1)
        scen_dim = h_scen.size(-1)

        h_asset_exp = h_asset.unsqueeze(1).expand(B, n_eligible, n_free, asset_dim).reshape(v_total, asset_dim)
        h_scen_exp = h_scen.unsqueeze(2).expand(B, n_eligible, n_free, scen_dim).reshape(v_total, scen_dim)
        g_exp = g_pool.unsqueeze(1).expand(B, v_per_graph, dim_last).reshape(v_total, dim_last)
        ctx_exp = ctx_norm.unsqueeze(1).expand(B, v_per_graph, self.CONTEXT_DIM).reshape(v_total, self.CONTEXT_DIM)

        combined = torch.cat([h_v, h_asset_exp, h_scen_exp, g_exp, ctx_exp], dim=-1)
        h_out = self.head(combined).squeeze(-1)
        if self.center_lam:
            h3 = h_out.view(B, n_eligible, n_free)
            h_out = (h3 - h3.mean(1, keepdim=True)).reshape(-1)
        return h_out


def make_batched_edges(edge_index: torch.Tensor, v_per_graph: int, B: int,
                       edge_weight: torch.Tensor | None = None):
    """Duplique une topologie (2,E) B fois avec offset b*v_per_graph -- bloc-
    diagonal, aucune arête inter-graphe (cf. make_batched_edges, MKP).
    Retourne aussi edge_weight : 1 partout si edge_weight=None (modèles d'origine),
    sinon les poids (E,) d'UN graphe dupliqués B fois."""
    e = edge_index.size(1)
    offsets = (torch.arange(B, device=edge_index.device) * v_per_graph).view(B, 1, 1)
    ei_rep = edge_index.unsqueeze(0).expand(B, 2, e) + offsets
    edge_index_batched = ei_rep.permute(1, 0, 2).reshape(2, B * e)
    if edge_weight is None:
        edge_weight_batched = torch.ones(B * e, dtype=torch.float32, device=edge_index.device)
    else:
        edge_weight_batched = edge_weight.to(torch.float32).unsqueeze(0).expand(B, e).reshape(B * e)
    return edge_index_batched, edge_weight_batched


def scriptable_from_args(args: dict) -> "CCPOGNNScriptableBatched":
    """Construit le module scriptable correspondant à ckpt["args"] (train_ccpo_gnn.py).
    Anciens checkpoints (sans feat_version/edge_mode/...) -> configuration d'origine."""
    from ccpo_graph import EDGE_MODES, node_dim_for
    fv = int(args.get("feat_version", 1))
    em = args.get("edge_mode") or ("sigma" if args.get("edge_weight_sigma", False) else "none")
    if em == "sigma":
        raise ValueError("edge_mode=sigma (ablation D) non porté en C++ -- export refusé")
    return CCPOGNNScriptableBatched(
        hidden=int(args["hidden"]), use_max_pool=bool(args.get("use_max_pool", True)),
        node_dim=node_dim_for(fv), feat_version=fv, edge_mode=EDGE_MODES[em],
        center_lam=bool(args.get("center_lam", False)), relu_h1=bool(args.get("relu_h1", True)))


def export_batched(checkpoint_path: str, out_path: str, device_str: str = "cpu") -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt["args"]
    hidden = int(args["hidden"])
    use_max_pool = bool(args.get("use_max_pool", True))

    model = scriptable_from_args(args)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    device = torch.device(device_str)
    model = model.to(device)

    scripted = torch.jit.script(model)
    scripted.save(out_path)
    print(f"[ccpo_gnn_scriptable_batched] TorchScript exporté vers {out_path} (device={device_str})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = p.parse_args()
    export_batched(a.checkpoint, a.out, a.device)
