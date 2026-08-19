#!/usr/bin/env python3
"""
validate_v1cld_native_reference.py — Calcule la référence Python (build_graph
+ GNN + lagrangian_v1cld, train_v1CLD_sd_edit.py) pour le NŒUD RACINE (x/z
entièrement libres, b_res=b, curr_cost=0, depth=0) d'une instance donnée, à
comparer avec la sortie de test_gnn_v1cld_native_batched (même dataset,
même checkpoint) — cf. plan de validation du backend C++ V1CLD avant tout
branchement B&B.

Usage :
  python3 validate_v1cld_native_reference.py <dataset.txt> <checkpoint.pt>
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from gnn_sd_edit.train_v1CLD_sd_edit import GNN, build_graph, lagrangian_v1cld


def load_song_local(fp: str, rep: int = 0):
    rows = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(",")]
            if int(vals[4]) != rep:
                continue
            rows.append(vals)
    m = int(rows[0][2]); n = int(rows[0][3]); N = len(rows)
    A = np.zeros((N, m, n), dtype=np.float64)
    b = np.zeros((N, m), dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    for i, v in enumerate(rows):
        for j in range(m):
            for k in range(n):
                A[i, j, k] = v[5 + j * n + k]
        for j in range(m):
            b[i, j] = v[5 + m * n + j]
        if i == 0:
            for k in range(n):
                c[k] = v[5 + m * n + m + k]
    return {"n": n, "m": m, "N": N, "c": c, "b": b, "A": A, "epsilon": 0.10, "opt": 0.0}


def main():
    dataset, checkpoint = sys.argv[1], sys.argv[2]
    inst = load_song_local(dataset)
    print(f"n={inst['n']} m={inst['m']} N={inst['N']}")

    ckpt = torch.load(checkpoint, map_location="cpu")
    args = ckpt["args"]
    model = GNN(hidden=int(args["hidden"]), use_max_pool=bool(args.get("use_max_pool", False)))
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    node_rec = {
        "inst_id": 0, "depth": 0, "curr_cost": 0.0,
        "x_mask": np.full(inst["n"], 2, dtype=np.int8),
        "z_mask": np.full(inst["N"], 2, dtype=np.int8),
        "b_res": inst["b"].copy(),
    }
    data = build_graph(inst, node_rec)
    assert data is not None
    print(f"n_g={data.n_g} N_z={data.N_z} m={data.m} c_scale={float(data.c_scale):.6f}")

    with torch.no_grad():
        lam = model(data)
        l1 = lagrangian_v1cld(lam, data).item()

    print(f"ub_relative (= L1(lambda_GNN)) = {l1:.6f}")


if __name__ == "__main__":
    main()
