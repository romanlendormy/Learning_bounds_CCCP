#!/usr/bin/env python3
"""
validate_v1cld_scriptable_batched.py — Vérifie que
GNNV1CLDSdEditScriptableBatched (B=1) reproduit EXACTEMENT
train_v1CLD_sd_edit.GNN sur un vrai graphe issu des données d'entraînement,
avant tout export/usage C++. Cf. mêmes garanties déjà établies pour la
version V1LD batchée (aucun réentraînement requis pour valider le batching).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np

from gnn_sd_edit.train_v1CLD_sd_edit import GNN, load_instances, build_graph
from gnn_sd_edit.gnn_v1cld_sd_edit_scriptable_batched import GNNV1CLDSdEditScriptableBatched

CKPT = REPO_ROOT / "gnn_sd_edit/runs_v1cld/ckpt_v1cld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_v1cld_n10_20_500ep_ep0500.pt"
DATA = REPO_ROOT / "data/train_base_s10"


def main():
    ckpt = torch.load(CKPT, map_location="cpu")
    args = ckpt["args"]
    hidden, use_max_pool = int(args["hidden"]), bool(args.get("use_max_pool", False))
    print(f"hidden={hidden} use_max_pool={use_max_pool}")

    model_eager = GNN(hidden=hidden, use_max_pool=use_max_pool)
    missing, unexpected = model_eager.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model_eager.eval()
    print("[OK] state_dict chargé dans le modèle eager (non-batché)")

    model_batched = GNNV1CLDSdEditScriptableBatched(hidden=hidden, use_max_pool=use_max_pool)
    missing, unexpected = model_batched.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model_batched.eval()
    print("[OK] state_dict chargé dans le modèle batché")

    scripted = torch.jit.script(model_batched)
    print("[OK] torch.jit.script réussi")

    # Charge une vraie instance + un vrai nœud non-racine pour construire un graphe réaliste
    inst_f = DATA / "instances_n10_m5_N20.txt"
    nodes_f = DATA / "nodes_n10_m5_N20.txt"
    instances = load_instances(inst_f)
    node_rec = None
    with open(nodes_f) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split(",")]
            inst_id, depth = int(vals[0]), int(vals[1])
            if depth == 0 or inst_id not in instances:
                continue
            n, N = instances[inst_id]["n"], instances[inst_id]["N"]
            n_fmt_B, n_fmt_C = 5 + n + N, 3 + n + N
            if len(vals) == n_fmt_C:
                off = 3
            elif len(vals) >= n_fmt_B:
                off = 5
            else:
                continue
            x_mask = np.array(vals[off:off+n], dtype=np.int8)
            z_mask = np.array(vals[off+n:off+n+N], dtype=np.int8)
            if (x_mask == 2).sum() == 0 or (z_mask == 2).sum() == 0:
                continue
            node_rec = {"inst_id": inst_id, "depth": depth, "curr_cost": vals[2],
                        "x_mask": x_mask, "z_mask": z_mask}
            break
    assert node_rec is not None, "aucun nœud non-racine valide trouvé"
    inst = instances[node_rec["inst_id"]]

    # b_res : approximation simple pour ce test de validation (capacité résiduelle
    # = capacité originale — déjà (N, m), cf. load_instances ; le vrai calcul
    # incrémental est dans LazyGraphDataset, non nécessaire ici : on ne teste
    # que l'égalité batché vs non-batché, pas la valeur métier de b_res).
    node_rec["b_res"] = inst["b"].copy()

    data = build_graph(inst, node_rec)
    assert data is not None
    n_g, N_z, m = data.n_g, data.N_z, data.m
    print(f"graphe : n_g={n_g}  N_z={N_z}  m={m}  V={data.x.shape[0]}  E={data.edge_index.shape[1]}")

    with torch.no_grad():
        lam_eager = model_eager(data)                                    # (N_z*m*n_g,) déjà rescalé

    with torch.no_grad():
        h_out_batched = scripted(data.x, data.edge_index, data.edge_weight,
                                  data.context.unsqueeze(0), n_g, m, 1)   # (V,) brut, PAS rescalé
    lam_batched = h_out_batched * data.c_scale / (float(data.N) * float(m))

    diff = (lam_eager - lam_batched).abs()
    print(f"max|diff| = {diff.max().item():.3e}   mean|diff| = {diff.mean().item():.3e}")
    assert diff.max().item() < 1e-4, "DIVERGENCE batché vs non-batché — NE PAS utiliser tel quel"
    print("[OK] batché (B=1) == non-batché, à la précision flottante près")


if __name__ == "__main__":
    main()
