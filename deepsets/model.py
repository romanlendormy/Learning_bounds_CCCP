"""
model.py — Variante "Deep Sets" du GNN v1LD (gnn.train_gnn_v1ld.GNN) : mêmes
node features (7m+5), même contexte (9-dim), même sortie (lambda par nœud
scénario x item), mais SANS message passing (pas de GatedGraphConv, pas
d'edge_index/edge_weight).

Motivation : dans gnn.train_gnn_v1ld.GNN, le graphe est constitué de deux
cliques complètes à poids uniformes (item_varies: 1/n_g, scenario_varies:
1/N_z) — cf. build_graph(). Avec des poids uniformes, le message passing sur
une clique complète calcule (via GRU) une variante de la moyenne par
ligne/colonne — exactement ce que h_item_mean/h_scen_mean calculent déjà en
aval par pooling direct. conv1+conv2 (4 rounds de GRU sur
~N_z*n_g*(n_g+N_z) arêtes) sont donc remplacés ici par un simple embed →
proj, sans edges du tout. Le reste (pooling item/scénario/graphe, tête, mise
à l'échelle finale) est identique.

Coût attendu : O(V) au lieu de O(V*(n_g+N_z)) par round — gain d'ordres de
grandeur sur le temps d'inférence par nœud B&B, au prix d'une perte possible
de raffinement croisé item<->scénario (2 hops dans la version GNN).

⚠️ Précédent de l'équipe (cf. docstring de GNN dans train_gnn_v1ld.py,
"on ne retire plus rien, on ajoute seulement") : une ablation antérieure de
g_pool + h_scen_i avait dégradé la borne malgré un ICC plus proche de la
cible. Cette version-ci retire une pièce différente (le message passing, pas
le pooling), mais le même risque s'applique — à valider par overfit_single
avant toute conclusion sur la qualité, pas seulement sur la latence.

Interface : DeepSetsGNN a exactement le même __init__(m, hidden, use_max_pool)
et forward(data, return_hidden=False) -> lam que GNN, donc interchangeable
dans les scripts existants (diag_overfit_*, train_scaled_v1ld.py, ...) en
remplaçant l'import.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool


class DeepSetsGNN(nn.Module):
    """
    Prédit λ[i,k] ∈ R par nœud (scénario libre i, item actif k) — sans edges.

    Architecture (7m+5 node features, context 9-dim, max_pool optionnel) :
      embed      : X(7m+5) → hidden → ReLU → hidden → ReLU → LayerNorm
      proj       : hidden → 4h → ReLU → 2h                       → h_v
      pool item  : nœud → item k [mean] (+ [max] si use_max_pool) → h_item_k
      pool scén  : nœud → scénario i [mean] (+ [max] si use_max_pool) → h_scen_i
      pool graph : nœud → graphe [mean]                          → g_pool (2h)
      norm_ctx   : LayerNorm sur les 9 features de contexte
      head       : [h_v | h_item_k | h_scen_i | g_pool | ctx] → h → ReLU → h → ReLU → 1

    Différence avec gnn.train_gnn_v1ld.GNN : conv1/conv2 (GatedGraphConv, 4
    rounds de message passing sur les cliques item/scénario) sont supprimés ;
    proj s'applique directement à l'embedding, pas à la sortie du message
    passing. Tout le reste (features, pooling, tête, scaling) est identique
    — voir docstring du module.
    """
    CONTEXT_DIM = 9  # [depth, frac_fixed, tightest, curr_cost, epsilon, n_fixed_z1/N, log(n_g), log(N_z), N_z/n_g]

    def __init__(self, m: int = 2, hidden: int = 64, use_max_pool: bool = False):
        super().__init__()
        self.m = m
        self.hidden = hidden
        self.use_max_pool = use_max_pool
        node_dim = 7 * m + 5

        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.norm_embed = nn.LayerNorm(hidden)
        self.proj = nn.Sequential(
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
        x = data.x
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        n_v = x.size(0)
        dev = x.device
        n_graphs = int(batch.max().item()) + 1

        ctx = data.context
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        ctx_norm = self.norm_ctx(ctx)                                  # (1 or B, 9)

        # ── Pas de message passing : embed puis proj directement ─────────────
        h_embed = self.norm_embed(self.embed(x))
        h_v = self.proj(h_embed)                                       # (V, 2h)

        # ── Pooling item-consensus (x̄_k) et scénario (h_scen_i) ───────────────
        # Identique à GNN.forward (train_gnn_v1ld.py) : nœuds ordonnés
        # i_idx*n_g + k_idx, n_g consécutifs = un scénario.
        n_g = data.n_g if isinstance(data.n_g, int) else int(data.n_g.item())

        counts = torch.bincount(batch, minlength=n_graphs)
        node_off = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        node_off[1:] = counts[:-1].cumsum(0)
        local_idx = torch.arange(n_v, dtype=torch.long, device=dev) - node_off[batch]

        item_off = torch.arange(n_graphs, device=dev) * n_g
        item_batch = item_off[batch] + (local_idx % n_g)                # (V,)

        h_item_mean = global_mean_pool(h_v, item_batch)
        if self.use_max_pool:
            h_item_max = global_max_pool(h_v, item_batch)
            h_item = torch.cat([h_item_mean, h_item_max], dim=-1)
        else:
            h_item = h_item_mean
        h_item_exp = h_item[item_batch]

        n_scen_per = counts // n_g
        scen_off = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        scen_off[1:] = n_scen_per[:-1].cumsum(0)
        scen_batch = scen_off[batch] + local_idx // n_g

        h_scen_mean = global_mean_pool(h_v, scen_batch)
        if self.use_max_pool:
            h_scen_max = global_max_pool(h_v, scen_batch)
            h_scen = torch.cat([h_scen_mean, h_scen_max], dim=-1)
        else:
            h_scen = h_scen_mean
        h_scen_exp = h_scen[scen_batch]

        ctx_exp = ctx_norm[batch]
        g_pool = global_mean_pool(h_v, batch)
        g_exp = g_pool[batch]

        cat = torch.cat([h_v, h_item_exp, h_scen_exp, g_exp, ctx_exp], dim=-1)
        h_out = self.head(cat).squeeze(-1)

        lam = h_out * data.c_scale / float(data.N)
        if return_hidden:
            extra = {"x": x, "h_embed": h_embed, "h_v": h_v,
                     "h_item_mean": h_item_mean, "item_batch": item_batch,
                     "h_scen_mean": h_scen_mean, "scen_batch": scen_batch,
                     "g_pool": g_pool, "h_out": h_out}
            return lam, extra
        return lam
