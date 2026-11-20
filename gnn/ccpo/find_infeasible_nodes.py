"""
find_infeasible_nodes.py — étiquette chaque exemple d'entraînement/VAL (même sélection que train_ccpo_gnn.py) selon
que son sous-arbre contient une solution.

Pour chaque nœud : force brute EXACTE sur les variables libres (toutes les affectations Σ = Q_res, <= k_res actives,
valeurs {0}∪[L,U]), compte des scénarios satisfaits sur la solution COMPLÈTE (variables fixées incluses), TOUS les
scénarios (indépendant du presolve). Statut :
  feasible   : au moins une affectation satisfait >= min_feas scénarios (v*_libre fini, donné) ;
  infeasible : aucune (v* = -inf) -> L(λ) non borné inférieurement sur ce nœud ;
  unknown    : énumération > --cap affectations (non tranché).
Racines (depth 0) : faisables par construction (l'instance a un optimum, même sous l'ancien modèle sur-contraint),
marquées feasible sans énumération.
Sortie JSONL, une ligne par exemple : n, N, inst, idx, depth, n_free, status, vstar_free, n_enum.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import train_ccpo_gnn as T  # noqa: E402
from ccpo_bruteforce import gen_free  # noqa: E402

CELLS = None


def label(t):
    ex, cap = t
    inst = CELLS[(ex.n, ex.N)].instances[ex.instance_id]
    node = CELLS[(ex.n, ex.N)].nodes_by_instance[ex.instance_id][ex.node_idx]
    q = node.q_mask
    free = np.where(q < 0)[0]
    base = dict(n=ex.n, N=ex.N, inst=ex.instance_id, idx=ex.node_idx, depth=ex.depth, n_free=int(len(free)))
    if ex.depth == 0:
        return {**base, "status": "feasible", "vstar_free": None, "n_enum": 0}
    v_fixed = np.where(q > 0, q, 0).astype(np.float64)
    Q_res = inst.Q - int(v_fixed.sum()); k_res = inst.k - int((q > 0).sum())
    X = gen_free(len(free), Q_res, k_res, inst.L, inst.U, cap)
    if X is None:
        return {**base, "status": "unknown", "vstar_free": None, "n_enum": -1}
    if len(X) == 0:
        return {**base, "status": "infeasible", "vstar_free": None, "n_enum": 0}
    full = np.tile(v_fixed, (len(X), 1)); full[:, free] = X
    cnt = np.zeros(len(X), dtype=np.int64)
    for j in range(inst.N):
        cnt += np.einsum("mi,ij,mj->m", full, inst.Sigma_dense[j], full) <= inst.B + 1e-9
    ok = cnt >= inst.min_feas
    if not ok.any():
        return {**base, "status": "infeasible", "vstar_free": None, "n_enum": int(len(X))}
    return {**base, "status": "feasible", "vstar_free": float((X[ok] @ inst.c_int[free]).max()), "n_enum": int(len(X))}


def _init(data_dir):
    global CELLS
    CELLS = T.load_all(Path(data_dir), node_suffix="_feasible")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--target_graphs", type=int, default=10000); ap.add_argument("--root_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--cap", type=int, default=3_000_000)
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"{a.out} existe : refus d'écraser")
    _init(a.data_dir)
    sel = T.stratified_select(CELLS, a.target_graphs, a.root_frac, random.Random(a.seed))
    # plus gros d'abord (équilibrage de charge)
    sel = sorted(sel, key=lambda e: -e.n)
    cnt = {}
    with Pool(a.procs) as pool, open(a.out, "x") as f:  # fork : CELLS hérité, partagé en copie à l'écriture
        for r in pool.imap_unordered(label, [(e, a.cap) for e in sel], chunksize=8):
            f.write(json.dumps(r) + "\n")
            cnt[(r["n"], r["N"], r["status"])] = cnt.get((r["n"], r["N"], r["status"]), 0) + 1
    for k in sorted(cnt):
        print(k, cnt[k])
    tot = {s: sum(v for k, v in cnt.items() if k[2] == s) for s in ("feasible", "infeasible", "unknown")}
    print("TOTAL", tot)


if __name__ == "__main__":
    main()
