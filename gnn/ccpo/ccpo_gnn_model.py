"""
ccpo_gnn_model.py — Architecture GNN pour la prédiction de λ (CCPO).

Reprise quasi telle quelle de gnn/mkp/train_SLD_sd_edit.py::GNN (MKP) --
même squelette GatedGraphConv à 2 blocs + lecture par pooling multi-niveaux,
dimensions d'entrée adaptées à ccpo_graph.py (NODE_DIM=9, CONTEXT_DIM=11 au
lieu de 7m+5 et 9 côté MKP).

    embed      : X(NODE_DIM) → hidden → ReLU → hidden → ReLU → LayerNorm
    conv1      : GatedGraphConv(hidden, 2 it.) + LayerNorm
    conv2      : GatedGraphConv(hidden, 2 it.) + LayerNorm + skip(h1)
    proj       : hidden → 4h → ReLU → 2h                    → h_v
    pool actif : nœud → actif i [mean(+max)]  → h_actif_i     (= "pool item" MKP)
    pool scén  : nœud → scénario j [mean(+max)] → h_scen_j
    pool graphe: scénario → graphe [mean]      → g_pool
    norm_ctx   : LayerNorm sur les CONTEXT_DIM features de contexte
    head       : [h_v | h_actif | h_scen | g_pool | ctx] → h → ReLU → h → ReLU → 1

Sortie : lam = h_out * c_scale / N (même formule d'échelle que MKP).

── Décision de conception : λ fixé à 0 pour les scénarios presolve-infaisables ──
ccpo_graph.py exclut ces scénarios du graphe (aucun nœud) -- le GNN ne prédit
donc PAS de λ pour eux. C'est une simplification volontaire, PAS une
approximation qui casse la validité : la dualité faible tient pour N'IMPORTE
QUEL λ, y compris un sous-espace où certaines composantes sont fixées à 0
(cf. ccpo_sld.hpp). Le seul risque est de laisser une borne
un peu moins serrée que si ces λ étaient aussi optimisés -- jamais un risque
de solidité. La fonction predict_lambda() ci-dessous construit explicitement
la matrice (N,n) complète avec des zéros aux lignes infaisables, prête pour
evaluate_ccpo_lagrangian_c.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import scatter

from ccpo_graph import CONTEXT_DIM, NODE_DIM


class CCPOGNN(nn.Module):
    def __init__(self, node_dim: int = NODE_DIM, context_dim: int = CONTEXT_DIM,
                 hidden: int = 64, gnn_iters: int = 2, use_max_pool: bool = True,
                 use_edge_weight: bool = False, center_lam: bool = False, relu_h1: bool = True):
        """center_lam (option A) : sortie centrée sur les scénarios ÉLIGIBLES,
        pour chaque actif (h - h.mean sur E) -- cf. centering_A/MANIFEST.md : ce n'est
        PAS une reparamétrisation neutre quand |E| < N. relu_h1 : ReLU après norm1
        (True = modèles existants ; False = comme CCMKP). Défauts = comportement d'origine."""
        super().__init__()
        self.use_edge_weight = use_edge_weight
        self.center_lam = center_lam
        self.relu_h1 = relu_h1
        self.hidden = hidden
        self.use_max_pool = use_max_pool

        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.conv1 = GatedGraphConv(hidden, num_layers=gnn_iters)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = GatedGraphConv(hidden, num_layers=gnn_iters)
        self.norm2 = nn.LayerNorm(hidden)

        self.proj = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.ReLU(),
            nn.Linear(4 * hidden, 2 * hidden),
        )
        h_v_dim = 2 * hidden
        pool_mult = 2 if use_max_pool else 1
        self.h_v_dim = h_v_dim

        self.norm_ctx = nn.LayerNorm(context_dim)

        head_in = h_v_dim + h_v_dim * pool_mult + h_v_dim * pool_mult + h_v_dim + context_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _pool(self, h_v: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        mean = scatter(h_v, index, dim=0, reduce="mean")
        if not self.use_max_pool:
            return mean
        mx = scatter(h_v, index, dim=0, reduce="max")
        return torch.cat([mean, mx], dim=-1)

    def forward(self, data) -> torch.Tensor:
        """Retourne h_out brut, (n_nodes,) -- un scalaire par (scénario éligible,
        actif libre). La mise à l'échelle en λ se fait dans predict_lambda()."""
        x = self.embed(data.x)
        ew = data.edge_weight if self.use_edge_weight else None
        h1 = self.norm1(self.conv1(x, data.edge_index, ew))
        if self.relu_h1:
            h1 = F.relu(h1)
        h2 = self.norm2(self.conv2(h1, data.edge_index, ew)) + h1
        h_v = self.proj(h2)

        asset_id = data.asset_id
        scenario_id = data.scenario_id

        h_asset = self._pool(h_v, asset_id)
        h_scen = self._pool(h_v, scenario_id)
        h_scen_mean_only = scatter(h_v, scenario_id, dim=0, reduce="mean")
        g_pool = h_scen_mean_only.mean(dim=0, keepdim=True)

        ctx = self.norm_ctx(data.ctx)

        n_nodes = h_v.shape[0]
        combined = torch.cat([
            h_v,
            h_asset[asset_id],
            h_scen[scenario_id],
            g_pool.expand(n_nodes, -1),
            ctx.expand(n_nodes, -1),
        ], dim=-1)

        h_out = self.head(combined).squeeze(-1)
        if self.center_lam:
            n_elig = int(data.n_scenarios_eligible)
            h2d = h_out.view(n_elig, -1)
            h_out = (h2d - h2d.mean(dim=0, keepdim=True)).reshape(-1)
        return h_out

    def predict_lambda(self, data, N: int, n_free: int, c_scale: float) -> np.ndarray:
        """Reconstruit la matrice (N, n_free) complète, zéros aux lignes des
        scénarios presolve-infaisables (cf. docstring du module). Détache et
        renvoie un ndarray SEULEMENT pour l'inférence pure (C++) -- pour
        l'entraînement, utiliser forward() + ccpo_lagrangian() directement
        sur le tenseur (n_eligible*n_free,) pour garder le graphe autograd,
        cf. build_lambda_tensor().
        """
        with torch.no_grad():
            h_out = self.forward(data)
        lam_full = np.zeros((N, n_free), dtype=np.float64)
        h_out_np = (h_out.double() * c_scale / N).cpu().numpy()
        eligible = data.eligible_scenario_idx.cpu().numpy()
        for s, j in enumerate(eligible):
            lam_full[j] = h_out_np[s * n_free:(s + 1) * n_free]
        return lam_full

    def build_lambda_tensor(self, data, N: int, n_free: int, c_scale: float) -> torch.Tensor:
        """Version DIFFÉRENTIABLE de predict_lambda -- reconstruit un tenseur
        (N, n_free) avec gradient, lignes infaisables à 0 (constantes, pas de
        gradient nécessaire dessus). À utiliser directement avec
        ccpo_lagrangian() côté entraînement."""
        h_out = self.forward(data)
        lam_elig = h_out * c_scale / N  # (n_eligible*n_free,)
        lam_elig = lam_elig.view(data.n_scenarios_eligible, n_free)

        lam_full = lam_elig.new_zeros((N, n_free))
        lam_full[data.eligible_scenario_idx] = lam_elig
        return lam_full
