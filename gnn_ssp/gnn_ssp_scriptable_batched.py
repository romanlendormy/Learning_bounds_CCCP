#!/usr/bin/env python3
"""
gnn_ssp_scriptable_batched.py — Réimplémentation de gnn_ssp.train_DLD_ssp.GNN
en torch pur (sans torch_geometric), scriptable via torch.jit.script, pour
inférence C++ via torch::jit::load() — mirror structurel de
gnn_sd_edit/gnn_sd_edit_scriptable_batched.py (MKP), adapté à SSP.

Pourquoi une réimplémentation et pas juste torch.jit.script(model) directement :
même raison que côté MKP (cf. gnn_sd_edit.gnn_sld_scriptable) — torch_geometric.data.Data
et GatedGraphConv/MessagePassing héritent de signatures à arguments variables
que TorchScript ne compile pas.

Différence structurelle avec MKP (à ne pas perdre de vue, cf. discussion
2026-08-18) : MKP n'a que DEUX vues de pooling (item, scénario) et un seul axe
qui varie par nœud B&B (n_g). SSP en a TROIS (item, scénario, période) —
l'axe "période" j∈[0,T) est un analogue direct de l'axe "contrainte" de MKP
(cf. train_DLD_ssp.py::GNN, docstring "h_scen_i / h_period_j"), mais T ne
rétrécit JAMAIS avec la profondeur B&B (contrairement à n_g et N_z) — c'est
une propriété fixe de l'instance. Conséquence pour le batching en amont
(Étape 4 du plan) : la condition de validité du bloc-diagonal ci-dessous
porte sur (n_g, N_z, T) ensemble, pas seulement n_g comme MKP — un bucketing
à trois clés (ou deux si on résout une seule instance à la fois, T restant
alors constant tout le run).

Pourquoi bloc-diagonal (edge_index/edge_weight dupliqués B fois avec offset
par graphe), pas une dimension batch native sur les arêtes : dans un batch
construit pendant UN run de B&B, tous les graphes partagent n_g, N_z ET T
(mêmes raisons que MKP — topologie ET poids d'arêtes identiques pour tous
les membres du batch). GatedGraphConvScriptable (gnn_sd_edit.gnn_sld_scriptable,
réutilisée sans modification) traite le batch comme un seul grand graphe à
composantes déconnectées — mathématiquement identique à B appels séparés.

Nœuds ordonnés i*(T*n_g) + j*n_g + k (scénario majeur, période, item — cf.
train_DLD_ssp.py::build_graph et le docstring de GNN.forward()), donc pour
un batch de B graphes empilés bout à bout : X a pour forme
(B*N_z*T*n_g, node_dim), et l'indice v_total // B // T // n_g redonne N_z.

Mêmes noms de paramètres que gnn_ssp.train_DLD_ssp.GNN → load_state_dict
(strict=True) fonctionne directement depuis un checkpoint de production
existant.

forward(X, edge_index, edge_weight, context, n_g, T, B) :
  X            : (B*V, node_dim)  — V = N_z * T * n_g, V identique pour tous les B
  edge_index   : (2, B*E)         — UNIQUEMENT les arêtes étoile inter-groupe
                                    depuis le 2026-08-20 (les arêtes clique
                                    intra-groupe sont calculées par formule
                                    fermée dans GatedGraphConvSspClique, plus
                                    besoin de les matérialiser -- cf. sa
                                    docstring). Toujours dupliqué B fois avec
                                    offset b*V.
  edge_weight  : (B*E,)           — poids dupliqués B fois (identiques par construction)
  context      : (B, 9)           — un contexte distinct par graphe du batch
  n_g, T, B    : int              — partagés par construction (bucketing par (n_g,N_z,T))
  retour       : lam (B*V, 4) brut — PAS mis à l'échelle par c_scale/(N*T) (fait
                 par l'appelant, comme GNNScriptable côté MKP : c_scale/N/T ne
                 sont pas des poids du modèle, cf. train_DLD_ssp.py:722-728
                 pour la formule de mise à l'échelle exacte à reproduire côté
                 appelant).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn_sd_edit.gnn_sld_scriptable import GatedGraphConvScriptable  # noqa: E402  (référence, gardée pour validation A/B)


class GatedGraphConvSspClique(nn.Module):
    """
    Variante SSP de GatedGraphConvScriptable (2026-08-20) : les arêtes clique
    intra-groupe (poids uniforme 1/G, cf. build_edges côté C++) sont
    remplacées par leur formule fermée -- identité algébrique EXACTE, pas une
    approximation, puisque le poids est constant :

        agg_clique[nœud] = Σ_{autres du groupe} (1/G)·m[autre]
                          = (group_sum[groupe] - m[nœud]) / G

    O(V) au lieu de O(G²·n_groups) pour cette partie, et plus besoin de
    matérialiser les arêtes clique côté C++ (gain mémoire direct, cf. OOM
    observés à grand N/T). Les arêtes passées ici (edge_index/edge_weight)
    ne contiennent donc plus QUE les arêtes étoile inter-groupe -- n_g (G)
    est passé explicitement pour reconstruire les groupes par un simple
    .view(), en exploitant l'ordonnancement de nœuds déjà utilisé partout
    ailleurs dans ce fichier (i*(T*n_g)+j*n_g+k -- n_g consécutifs = 1 groupe).

    Mêmes noms de paramètres que GatedGraphConvScriptable (weight, rnn.*) →
    load_state_dict(strict=True) fonctionne à l'identique depuis le même
    checkpoint, aucun réentraînement requis (cf. discussion 2026-08-20).
    """

    def __init__(self, hidden: int, num_layers: int):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.weight = nn.Parameter(torch.zeros(num_layers, hidden, hidden))
        self.rnn = nn.GRUCell(hidden, hidden)

    def forward(self, x: torch.Tensor, edge_index_star: torch.Tensor,
                edge_weight_star: torch.Tensor, n_g: int) -> torch.Tensor:
        src = edge_index_star[0]
        dst = edge_index_star[1]
        v = x.size(0)
        for l in range(self.num_layers):
            m = torch.matmul(x, self.weight[l])

            # Étoile inter-groupe : inchangé, via les arêtes explicites.
            msg = edge_weight_star.unsqueeze(-1) * m.index_select(0, src)
            agg = torch.zeros((v, self.hidden), dtype=x.dtype, device=x.device)
            agg.index_add_(0, dst, msg)

            # Clique intra-groupe : moyenne de groupe directe (voir docstring).
            m_grp = m.view(-1, n_g, self.hidden)                       # (n_groups, G, hidden)
            group_sum = m_grp.sum(dim=1, keepdim=True)                 # (n_groups, 1, hidden)
            agg_clique = ((group_sum - m_grp) / n_g).reshape(v, self.hidden)
            agg = agg + agg_clique

            x = self.rnn(agg, x)
        return x


class GNNSspScriptableBatched(nn.Module):
    """
    Port batché de gnn_ssp.train_DLD_ssp.GNN (3 vues de pooling : item,
    scénario pur, période pure — cf. docstring module). Architecturalement
    identique à l'original — seule la mécanique de pooling gagne une
    dimension B, et un axe (période) de plus que le pooling MKP.
    """

    NODE_DIM = 8
    N_SIGMA = 4

    def __init__(self, hidden: int = 64, use_max_pool: bool = False):
        super().__init__()
        self.hidden = hidden
        self.use_max_pool = use_max_pool
        self.CONTEXT_DIM = 9   # attribut d'instance (pas de classe) : requis par TorchScript
        node_dim = self.NODE_DIM

        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.norm_embed = nn.LayerNorm(hidden)
        self.conv1 = GatedGraphConvSspClique(hidden, 2)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = GatedGraphConvSspClique(hidden, 2)
        self.norm2 = nn.LayerNorm(hidden)
        self.proj = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.ReLU(),
            nn.Linear(4 * hidden, 2 * hidden),
        )
        self.norm_ctx = nn.LayerNorm(self.CONTEXT_DIM)

        pool_dim = 4 * hidden if use_max_pool else 2 * hidden
        head_in = 2 * hidden + 3 * pool_dim + 2 * hidden + self.CONTEXT_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, self.N_SIGMA),
        )

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                context: torch.Tensor, n_g: int, T: int, B: int) -> torch.Tensor:
        v_total = X.size(0)
        v_per_graph = v_total // B
        N_z = v_per_graph // (T * n_g)
        dim_last = 2 * self.hidden

        ctx_norm = self.norm_ctx(context)   # (B, 9)

        h = self.norm_embed(self.embed(X))
        h1 = self.norm1(self.conv1(h, edge_index, edge_weight, n_g))
        h2 = self.norm2(self.conv2(h1, edge_index, edge_weight, n_g)) + h1
        h_v = self.proj(h2)                                 # (B*V, 2h)

        # Nœuds ordonnés i*(T*n_g) + j*n_g + k → vue (B, N_z, T, n_g, 2h)
        h_v_5d = h_v.view(B, N_z, T, n_g, dim_last)

        # item (k)     : marginalise i puis j (deux .mean(dim) successifs,
        #                mathématiquement identique à un mean joint sur (i,j),
        #                cf. docstring module — évite mean(dim=(a,b)) par souci
        #                de cohérence de style avec la version MKP)
        h_item_mean = h_v_5d.mean(1).mean(1)                 # (B, n_g, 2h)
        # scénario (i) : marginalise j puis k
        h_scen_mean = h_v_5d.mean(2).mean(2)                 # (B, N_z, 2h)
        # période (j)  : marginalise i puis k
        h_period_mean = h_v_5d.mean(1).mean(2)               # (B, T, 2h)
        g_pool = h_item_mean.mean(1)                          # (B, 2h) — même trick que MKP

        if self.use_max_pool:
            h_item_max, _ = h_v_5d.max(1)
            h_item_max, _ = h_item_max.max(1)
            h_scen_max, _ = h_v_5d.max(2)
            h_scen_max, _ = h_scen_max.max(2)
            h_period_max, _ = h_v_5d.max(1)
            h_period_max, _ = h_period_max.max(2)
            h_item = torch.cat([h_item_mean, h_item_max], dim=-1)     # (B, n_g, 4h)
            h_scen = torch.cat([h_scen_mean, h_scen_max], dim=-1)     # (B, N_z, 4h)
            h_period = torch.cat([h_period_mean, h_period_max], dim=-1)  # (B, T, 4h)
        else:
            h_item = h_item_mean
            h_scen = h_scen_mean
            h_period = h_period_mean

        item_dim = h_item.size(-1)
        scen_dim = h_scen.size(-1)
        period_dim = h_period.size(-1)

        h_item_exp = (h_item.view(B, 1, 1, n_g, item_dim)
                      .expand(B, N_z, T, n_g, item_dim).reshape(v_total, item_dim))
        h_scen_exp = (h_scen.view(B, N_z, 1, 1, scen_dim)
                      .expand(B, N_z, T, n_g, scen_dim).reshape(v_total, scen_dim))
        h_period_exp = (h_period.view(B, 1, T, 1, period_dim)
                        .expand(B, N_z, T, n_g, period_dim).reshape(v_total, period_dim))
        g_exp = (g_pool.view(B, 1, 1, 1, dim_last)
                 .expand(B, N_z, T, n_g, dim_last).reshape(v_total, dim_last))
        ctx_exp = (ctx_norm.view(B, 1, 1, 1, self.CONTEXT_DIM)
                   .expand(B, N_z, T, n_g, self.CONTEXT_DIM).reshape(v_total, self.CONTEXT_DIM))

        cat = torch.cat([h_v, h_item_exp, h_scen_exp, h_period_exp, g_exp, ctx_exp], dim=-1)
        h_out = self.head(cat)                                # (B*V, 4) — brut, non mis à l'échelle
        return h_out


def make_batched_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor, v_per_graph: int, B: int):
    """Duplique une topologie (2,E)/(E,) B fois avec offset b*v_per_graph par
    graphe — bloc-diagonal, aucune arête inter-graphe. Identique à
    gnn_sd_edit_scriptable_batched.make_batched_edges (MKP)."""
    e = edge_index.size(1)
    offsets = (torch.arange(B, device=edge_index.device) * v_per_graph).view(B, 1, 1)
    ei_rep = edge_index.unsqueeze(0).expand(B, 2, e) + offsets
    edge_index_batched = ei_rep.permute(1, 0, 2).reshape(2, B * e)
    edge_weight_batched = edge_weight.unsqueeze(0).expand(B, e).reshape(B * e)
    return edge_index_batched, edge_weight_batched


def export_batched(checkpoint_path: str, out_path: str, device_str: str = "cpu") -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt["args"]
    hidden = int(args["hidden"])
    use_max_pool = bool(args.get("use_max_pool", False))

    model = GNNSspScriptableBatched(hidden, use_max_pool)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    device = torch.device(device_str)
    model = model.to(device)

    scripted = torch.jit.script(model)
    scripted.save(out_path)
    print(f"[gnn_ssp_scriptable_batched] TorchScript exporté vers {out_path} "
          f"(device={device_str}, use_max_pool={use_max_pool})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = p.parse_args()
    export_batched(a.checkpoint, a.out, a.device)
