#!/usr/bin/env python3
"""
validate_gnn_sd_edit_ragged.py — Vérifie que GNNSdEditScriptableBatchedRagged
(pooling par scatter/index_add_, tailles N_z hétérogènes) donne EXACTEMENT
(à l'arithmétique flottante près) le même résultat que GNNSdEditScriptableBatched
(pooling dense .view(B,n_z,n_g,dim)) sur un batch à N_z UNIFORME — le cas où
les deux formulations doivent mathématiquement coïncider.

Puis un second test à N_z HÉTÉROGÈNE (2 graphes, tailles différentes) —
comparé à N appels séparés du modèle DENSE (B=1 à chaque fois), qui EST
correct par construction pour un seul graphe.

Usage :
  python3 validate_gnn_sd_edit_ragged.py <checkpoint.pt>
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn_sd_edit.gnn_sd_edit_scriptable_batched import GNNSdEditScriptableBatched
from gnn_sd_edit.gnn_sd_edit_scriptable_batched_ragged import GNNSdEditScriptableBatchedRagged


def load_both(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt["args"]
    m, hidden = int(args["m"]), int(args["hidden"])
    use_max_pool = bool(args.get("use_max_pool", False))
    center_lam = bool(args.get("center_lam", False))
    print(f"[config] m={m} hidden={hidden} use_max_pool={use_max_pool} center_lam={center_lam}")

    dense = GNNSdEditScriptableBatched(m, hidden, use_max_pool, center_lam)
    dense.load_state_dict(ckpt["model"], strict=True)
    dense.eval()

    ragged = GNNSdEditScriptableBatchedRagged(m, hidden, use_max_pool, center_lam)
    ragged.load_state_dict(ckpt["model"], strict=True)
    ragged.eval()
    return dense, ragged, m


def make_random_graph_block(n_z, n_g, node_dim, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_z * n_g, node_dim, generator=g)
    # item-clique (par scénario) + scenario-star (ancre i=0) — même topologie
    # que build_edges_sd_edit_cpp, juste construite ici en pur Python.
    src, dst, w = [], [], []
    for i in range(n_z):
        for k1 in range(n_g):
            for k2 in range(k1 + 1, n_g):
                a, b = i * n_g + k1, i * n_g + k2
                src += [a, b]; dst += [b, a]; w += [1.0 / n_g, 1.0 / n_g]
    for k in range(n_g):
        anchor = 0 * n_g + k
        for i2 in range(1, n_z):
            spoke = i2 * n_g + k
            src += [anchor, spoke]; dst += [spoke, anchor]; w += [1.0, 1.0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    return X, edge_index, edge_weight


def test_uniform(dense, ragged, m, n_g=6, n_z=8, B=5):
    node_dim = 7 * m + 5
    Xs, eis, ews = [], [], []
    for b in range(B):
        X, ei, ew = make_random_graph_block(n_z, n_g, node_dim, seed=100 + b)
        Xs.append(X); eis.append(ei); ews.append(ew)

    V = n_z * n_g
    X_all = torch.cat(Xs, dim=0)
    offsets = torch.arange(B) * V
    ei_all = torch.cat([eis[b] + offsets[b] for b in range(B)], dim=1)
    ew_all = torch.cat(ews, dim=0)
    context = torch.randn(B, 9)

    with torch.no_grad():
        lam_dense = dense(X_all, ei_all, ew_all, context, n_g, B)

    graph_id = torch.repeat_interleave(torch.arange(B), V)
    local_idx = torch.arange(V).repeat(B)
    local_i = local_idx // n_g
    local_k = local_idx % n_g
    item_group = graph_id * n_g + local_k
    scen_group = graph_id * n_z + local_i
    S_total = B * n_z

    with torch.no_grad():
        lam_ragged = ragged(X_all, ei_all, ew_all, context, graph_id, item_group,
                             scen_group, n_g, B, S_total)

    diff = (lam_dense - lam_ragged).abs()
    print(f"[uniforme] max|diff|={diff.max().item():.3e}  mean|diff|={diff.mean().item():.3e}")
    return diff.max().item()


def test_heterogeneous(dense, ragged, m, n_g=6, n_z_list=(5, 9, 3)):
    node_dim = 7 * m + 5
    B = len(n_z_list)
    Xs, eis, ews = [], [], []
    for b, n_z in enumerate(n_z_list):
        X, ei, ew = make_random_graph_block(n_z, n_g, node_dim, seed=200 + b)
        Xs.append(X); eis.append(ei); ews.append(ew)
    context = torch.randn(B, 9)

    # Référence : B appels séparés du modèle DENSE avec B=1 chacun.
    lam_ref_parts = []
    with torch.no_grad():
        for b, n_z in enumerate(n_z_list):
            lam_b = dense(Xs[b], eis[b], ews[b], context[b:b+1], n_g, 1)
            lam_ref_parts.append(lam_b)
    lam_ref = torch.cat(lam_ref_parts, dim=0)

    # Ragged : un seul appel batché à tailles hétérogènes.
    offsets = [0]
    for n_z in n_z_list:
        offsets.append(offsets[-1] + n_z * n_g)
    X_all = torch.cat(Xs, dim=0)
    ei_all = torch.cat([eis[b] + offsets[b] for b in range(B)], dim=1)
    ew_all = torch.cat(ews, dim=0)

    graph_id_parts, item_group_parts, scen_group_parts = [], [], []
    scen_offset = 0
    for b, n_z in enumerate(n_z_list):
        V = n_z * n_g
        local_idx = torch.arange(V)
        local_i = local_idx // n_g
        local_k = local_idx % n_g
        graph_id_parts.append(torch.full((V,), b, dtype=torch.long))
        item_group_parts.append(b * n_g + local_k)
        scen_group_parts.append(scen_offset + local_i)
        scen_offset += n_z
    graph_id = torch.cat(graph_id_parts)
    item_group = torch.cat(item_group_parts)
    scen_group = torch.cat(scen_group_parts)
    S_total = sum(n_z_list)

    with torch.no_grad():
        lam_ragged = ragged(X_all, ei_all, ew_all, context, graph_id, item_group,
                             scen_group, n_g, B, S_total)

    diff = (lam_ref - lam_ragged).abs()
    print(f"[hétérogène n_z={n_z_list}] max|diff|={diff.max().item():.3e}  mean|diff|={diff.mean().item():.3e}")
    return diff.max().item()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <checkpoint.pt>", file=sys.stderr)
        sys.exit(1)
    dense, ragged, m = load_both(sys.argv[1])
    d1 = test_uniform(dense, ragged, m)
    d2 = test_heterogeneous(dense, ragged, m)
    THRESH = 1e-4
    ok = d1 < THRESH and d2 < THRESH
    print(f"\n{'PASS' if ok else 'FAIL'} (seuil={THRESH:.0e})")
    sys.exit(0 if ok else 1)
