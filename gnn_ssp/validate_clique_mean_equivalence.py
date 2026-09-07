#!/usr/bin/env python3
"""
validate_clique_mean_equivalence.py — Vérifie que GatedGraphConvSspClique
(agrégation clique par formule fermée) produit EXACTEMENT (à la précision
flottante près) la même sortie que GatedGraphConvScriptable (agrégation
clique par arêtes explicites, poids uniforme 1/G) -- cf. discussion
2026-08-20, gnn_ssp/gnn_ssp_scriptable_batched.py.

Construit un petit graphe synthétique (plusieurs groupes de taille G, +
arêtes étoile), avec des poids ALÉATOIRES (pas besoin d'un vrai checkpoint --
c'est une identité algébrique valable pour N'IMPORTE QUELS poids), fait
tourner les deux implémentations avec les MÊMES poids copiés, et compare.

Usage : python validate_clique_mean_equivalence.py
"""
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_sd_edit.gnn_v1ld_scriptable import GatedGraphConvScriptable
from gnn_ssp.gnn_ssp_scriptable_batched import GatedGraphConvSspClique


def build_full_edges(G: int, n_groups: int):
    """Réplique exacte de build_edges() côté C++ (gnn_v1cld_native.cpp) : clique
    intra-groupe (poids 1/G) + étoile inter-groupe (poids 1.0, groupe 0 = hub)."""
    ii_src_list, ii_dst_list = [], []
    if G >= 2:
        tri = torch.triu_indices(G, G, offset=1)
        i1, i2 = tri[0], tri[1]
        grp_off = torch.arange(n_groups) * G
        fwd_src = (i1.unsqueeze(0) + grp_off.unsqueeze(1)).flatten()
        fwd_dst = (i2.unsqueeze(0) + grp_off.unsqueeze(1)).flatten()
        ii_src = torch.cat([fwd_src, fwd_dst])
        ii_dst = torch.cat([fwd_dst, fwd_src])
        ii_w = torch.full((ii_src.size(0),), 1.0 / G)
    else:
        ii_src = torch.empty(0, dtype=torch.long)
        ii_dst = torch.empty(0, dtype=torch.long)
        ii_w = torch.empty(0)

    if n_groups >= 2:
        k_all = torch.arange(G)
        ii = torch.arange(1, n_groups)
        fwd_src = (0 * G + k_all).unsqueeze(0).expand(ii.size(0), G).flatten()
        fwd_dst = (ii.unsqueeze(1) * G + k_all.unsqueeze(0)).flatten()
        ss_src = torch.cat([fwd_src, fwd_dst])
        ss_dst = torch.cat([fwd_dst, fwd_src])
        ss_w = torch.full((ss_src.size(0),), 1.0)
    else:
        ss_src = torch.empty(0, dtype=torch.long)
        ss_dst = torch.empty(0, dtype=torch.long)
        ss_w = torch.empty(0)

    edge_index_full = torch.stack([torch.cat([ii_src, ss_src]), torch.cat([ii_dst, ss_dst])])
    edge_weight_full = torch.cat([ii_w, ss_w])
    edge_index_star = torch.stack([ss_src, ss_dst])
    edge_weight_star = ss_w
    return edge_index_full, edge_weight_full, edge_index_star, edge_weight_star


def run_case(G: int, n_groups: int, hidden: int, num_layers: int, seed: int) -> float:
    torch.manual_seed(seed)
    v = G * n_groups
    x = torch.randn(v, hidden)

    ei_full, ew_full, ei_star, ew_star = build_full_edges(G, n_groups)

    old = GatedGraphConvScriptable(hidden, num_layers)
    new = GatedGraphConvSspClique(hidden, num_layers)
    # mêmes poids dans les deux
    with torch.no_grad():
        new.weight.copy_(old.weight)
        new.rnn.weight_ih.copy_(old.rnn.weight_ih)
        new.rnn.weight_hh.copy_(old.rnn.weight_hh)
        new.rnn.bias_ih.copy_(old.rnn.bias_ih)
        new.rnn.bias_hh.copy_(old.rnn.bias_hh)

    out_old = old(x, ei_full, ew_full)
    out_new = new(x, ei_star, ew_star, G)

    max_abs = (out_old - out_new).abs().max().item()
    max_rel = ((out_old - out_new).abs() / (out_old.abs() + 1e-8)).max().item()
    print(f"G={G:3d} n_groups={n_groups:3d} hidden={hidden:3d} layers={num_layers}  "
          f"max_abs_diff={max_abs:.3e}  max_rel_diff={max_rel:.3e}")
    return max_abs


def main():
    print("=== Validation clique->moyenne : GatedGraphConvScriptable (arêtes) "
          "vs GatedGraphConvSspClique (formule fermée) ===")
    worst = 0.0
    cases = [
        (1, 5, 8, 2, 0),    # G=1 : pas de clique du tout (cas limite)
        (2, 5, 8, 2, 1),    # G=2 : clique minimale (une seule paire)
        (4, 6, 16, 2, 2),   # cas typique petit
        (23, 400, 64, 2, 3),  # taille proche de la racine E4_T8_N50 (hidden réel=64)
        (48, 50, 64, 2, 4),  # G=E*T=48 (E=4,T=12 racine), hidden réel
    ]
    for G, n_groups, hidden, num_layers, seed in cases:
        d = run_case(G, n_groups, hidden, num_layers, seed)
        worst = max(worst, d)

    print(f"\nPire écart absolu observé sur tous les cas : {worst:.3e}")
    if worst < 1e-4:
        print("=== PASS (écart au niveau du bruit flottant, identité confirmée) ===")
    else:
        print("=== FAIL (écart anormalement grand -- bug d'implémentation à corriger) ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
