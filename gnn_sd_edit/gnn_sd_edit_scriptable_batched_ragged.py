#!/usr/bin/env python3
"""
gnn_sd_edit_scriptable_batched_ragged.py — Variante RAGGED (tailles N_z
hétérogènes dans un même batch) de GNNSdEditScriptableBatched
(gnn_sd_edit_scriptable_batched.py) — motivée par l'échec du bucketing
par (n_g, N_z) : fragmenter les buckets pour garantir un N_z uniforme casse
le taux de remplissage (56→23 sur K=64, cf. session), donc régresse plus que
le gain de resserrement de borne ne compense.

Ici, on revient au bucketing par n_g SEUL (aucune fragmentation) et on
transmet des graphes de TAILLE RÉELLEMENT DIFFÉRENTE (N_z_i propre à chaque
membre du batch, PAS de padding) empilés en bloc-diagonal — même principe que
GNNSdEditScriptableBatched pour la dimension B, étendu pour que chaque bloc
ait sa propre taille.

Pourquoi c'est possible SANS padding ni contamination : GatedGraphConvScriptable
(conv1/conv2, message passing) fait déjà du scatter/index_add_ générique sur
edge_index — un batch bloc-diagonal à tailles hétérogènes est mathématiquement
identique à B appels séparés, QUELLE QUE SOIT la taille de chaque bloc (déjà
noté dans gnn_sd_edit_scriptable_batched.py, jamais exploité jusqu'ici parce
que le SEUL point bloquant était le POOLING, pas le message passing).

SEULE différence avec GNNSdEditScriptableBatched : le pooling change de
`.view(B, n_z, n_g, dim).mean(...)` (suppose n_z uniforme) à un pooling par
scatter/index_add_ paramétré par des tenseurs d'index par nœud :
  - graph_id   (V_tot,) int64 ∈ [0, B)       — graphe d'appartenance
  - item_group (V_tot,) int64 ∈ [0, B*n_g)   — groupe (graphe, item) pour le
                                                pooling item-level (moyenne/max
                                                sur les scénarios, à item fixé)
  - scen_group (V_tot,) int64 ∈ [0, S_tot)   — groupe (graphe, scénario) pour
                                                le pooling scénario-level
                                                (moyenne/max sur les items, à
                                                scénario fixé) ; S_tot = Σ_b N_z_b

Mêmes noms/formes de paramètres que GNNSdEditScriptableBatched → hérite
directement (pas de duplication des couches) et load_state_dict(strict=True)
fonctionne à l'identique depuis le MÊME checkpoint, aucun réentraînement.

Validé numériquement (cf. validate_gnn_sd_edit_ragged.py) contre
GNNSdEditScriptableBatched sur un batch à N_z UNIFORME (cas où les deux
formulations doivent coïncider exactement, à l'arithmétique flottante près).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn_sd_edit.gnn_sd_edit_scriptable_batched import GNNSdEditScriptableBatched  # noqa: E402


class GNNSdEditScriptableBatchedRagged(GNNSdEditScriptableBatched):
    def forward(  # type: ignore[override]
        self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
        context: torch.Tensor, graph_id: torch.Tensor, item_group: torch.Tensor,
        scen_group: torch.Tensor, n_g: int, B: int, S_total: int
    ) -> torch.Tensor:
        v_total = X.size(0)
        dim_last = 2 * self.hidden
        Bn = B * n_g
        dev = X.device
        dt = X.dtype

        ctx_norm = self.norm_ctx(context)   # (B, 9)

        h = self.norm_embed(self.embed(X))
        h1 = self.norm1(self.conv1(h, edge_index, edge_weight))
        h2 = self.norm2(self.conv2(h1, edge_index, edge_weight)) + h1
        h_v = self.proj(h2)                                        # (v_total, 2h)

        ones_col = torch.ones((v_total, 1), dtype=dt, device=dev)

        # ── Pooling item-level : groupe = (graphe, item), moyenne/max sur les
        # scénarios (remplace h_v_4d.mean(1)/max(1) de la version dense). ────
        item_sum = torch.zeros((Bn, dim_last), dtype=dt, device=dev)
        item_sum.index_add_(0, item_group, h_v)
        item_cnt = torch.zeros((Bn, 1), dtype=dt, device=dev)
        item_cnt.index_add_(0, item_group, ones_col)
        h_item_mean_flat = item_sum / item_cnt.clamp(min=1.0)      # (B*n_g, 2h)

        # ── Pooling scénario-level : groupe = (graphe, scénario), moyenne/max
        # sur les items (remplace h_v_4d.mean(2)/max(2)). ───────────────────
        scen_sum = torch.zeros((S_total, dim_last), dtype=dt, device=dev)
        scen_sum.index_add_(0, scen_group, h_v)
        scen_cnt = torch.zeros((S_total, 1), dtype=dt, device=dev)
        scen_cnt.index_add_(0, scen_group, ones_col)
        h_scen_mean_flat = scen_sum / scen_cnt.clamp(min=1.0)      # (S_total, 2h)

        if self.use_max_pool:
            neg_inf = -1.0e30
            item_max_flat = torch.full((Bn, dim_last), neg_inf, dtype=dt, device=dev)
            item_max_flat.scatter_reduce_(
                0, item_group.unsqueeze(-1).expand(-1, dim_last), h_v,
                reduce="amax", include_self=True)
            scen_max_flat = torch.full((S_total, dim_last), neg_inf, dtype=dt, device=dev)
            scen_max_flat.scatter_reduce_(
                0, scen_group.unsqueeze(-1).expand(-1, dim_last), h_v,
                reduce="amax", include_self=True)
            h_item_flat = torch.cat([h_item_mean_flat, item_max_flat], dim=-1)   # (B*n_g, 4h)
            h_scen_flat = torch.cat([h_scen_mean_flat, scen_max_flat], dim=-1)   # (S_total, 4h)
        else:
            h_item_flat = h_item_mean_flat
            h_scen_flat = h_scen_mean_flat

        # g_pool = moyenne sur les items du pooling item-level MOYEN (PAS la
        # version concatenée max) — même définition que la version dense
        # (g_pool = h_item_mean.mean(1), avant concat avec h_item_max).
        g_pool = h_item_mean_flat.view(B, n_g, dim_last).mean(1)   # (B, 2h)

        item_dim = h_item_flat.size(-1)
        scen_dim = h_scen_flat.size(-1)

        h_item_exp = h_item_flat.index_select(0, item_group)   # (v_total, item_dim)
        h_scen_exp = h_scen_flat.index_select(0, scen_group)   # (v_total, scen_dim)
        g_exp   = g_pool.index_select(0, graph_id)              # (v_total, 2h)
        ctx_exp = ctx_norm.index_select(0, graph_id)            # (v_total, 9)

        cat = torch.cat([h_v, h_item_exp, h_scen_exp, g_exp, ctx_exp], dim=-1)
        h_out = self.head(cat).squeeze(-1)

        if self.center_lam:
            # Centre h_out par (graphe, item) sur l'axe scénario — même
            # sémantique que h_out.view(B,n_z,n_g) - mean(dim=1) de la version
            # dense, ici via le groupe item_group (constant sur l'axe scénario).
            out_sum = torch.zeros((Bn,), dtype=h_out.dtype, device=dev)
            out_sum.index_add_(0, item_group, h_out)
            ones1 = torch.ones((v_total,), dtype=h_out.dtype, device=dev)
            out_cnt = torch.zeros((Bn,), dtype=h_out.dtype, device=dev)
            out_cnt.index_add_(0, item_group, ones1)
            item_mean_out = out_sum / out_cnt.clamp(min=1.0)
            h_out = h_out - item_mean_out.index_select(0, item_group)

        return h_out


def export_ragged(checkpoint_path: str, out_path: str, device_str: str = "cpu") -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt["args"]
    m, hidden = int(args["m"]), int(args["hidden"])
    use_max_pool = bool(args.get("use_max_pool", False))
    center_lam = bool(args.get("center_lam", False))

    model = GNNSdEditScriptableBatchedRagged(m, hidden, use_max_pool, center_lam)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    device = torch.device(device_str)
    model = model.to(device)

    scripted = torch.jit.script(model)
    scripted.save(out_path)
    print(f"[gnn_sd_edit_scriptable_batched_ragged] TorchScript exporté vers {out_path} "
          f"(device={device_str}, center_lam={center_lam})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = p.parse_args()
    export_ragged(a.checkpoint, a.out, a.device)
