#!/usr/bin/env python3
"""
train_gbdt_v1ld.py — Test rapide : GBDT (LightGBM) entraîné DIRECTEMENT sur la
borne lagrangienne V1LD (lagrangian_v1ld), sans passer par un GNN professeur.

Une ligne de la table = une paire (scénario libre i, item actif k) d'un nœud
B&B. Features = les mêmes 7m+5 features par paire + les 9 features de
contexte du graphe, exactement celles de gnn.train_gnn_v1ld.build_graph.

Contrairement à un GNN, ce modèle n'a pas de message passing : chaque ligne
est prédite indépendamment des autres lignes du même nœud B&B. Les features
existantes contiennent déjà beaucoup d'agrégats faits main (moyenne/std sur
scénarios, min/mean sur contraintes) qui jouent un peu ce rôle.

lagrangian_v1ld(lam, data) est différentiable en lam (utilisée pour entraîner
le GNN via autograd). On la réutilise ici telle quelle comme objectif custom
LightGBM : à chaque round de boosting, on calcule le gradient de la loss par
rapport aux prédictions courantes (via autograd, par groupe = nœud B&B, car la
loss couple les lignes d'un même nœud via Λ_sum/y*), et on fixe le hessien à 1
(la loss est linéaire par morceaux à cause des max(0,·) et du .detach() sur
ζ_i/y* — pas de hessien analytique non nul, comme pour la loss MAE dans
LightGBM). Chaque round revient donc à un pas de descente de gradient sur λ.

Comparaison : gap% = (L1(λ) - opt_résiduel) / |opt_résiduel| * 100, où
opt_résiduel = opt - curr_cost_raw (le graphe ne couvre que la partie encore
libre du B&B ; cf. build_graph et diag_overfit_single.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import lightgbm as lgb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn.train_gnn_v1ld import load_dataset, lagrangian_v1ld  # noqa: E402


def flatten_dataset(graphs: list) -> tuple[np.ndarray, np.ndarray]:
    """Table plate (X, group_sizes) : concatène x + contexte répété par ligne."""
    X_parts = []
    group_sizes = np.empty(len(graphs), dtype=np.int64)
    for gi, g in enumerate(graphs):
        v = g.x.shape[0]
        group_sizes[gi] = v
        ctx = np.broadcast_to(g.context.numpy(), (v, g.context.shape[0]))
        X_parts.append(np.concatenate([g.x.numpy(), ctx], axis=1))
    X = np.concatenate(X_parts, axis=0).astype(np.float64)
    return X, group_sizes


def group_offsets(group_sizes: np.ndarray) -> list[tuple[int, int]]:
    ends = np.cumsum(group_sizes)
    starts = ends - group_sizes
    return list(zip(starts.tolist(), ends.tolist()))


def make_objective(graphs: list, offsets: list[tuple[int, int]]):
    def objective(preds: np.ndarray, dataset: lgb.Dataset):
        grad = np.empty_like(preds, dtype=np.float64)
        hess = np.ones_like(preds, dtype=np.float64)
        for g, (s, e) in zip(graphs, offsets):
            lam = torch.tensor(preds[s:e], dtype=torch.float32, requires_grad=True)
            loss = lagrangian_v1ld(lam, g)
            (g_lam,) = torch.autograd.grad(loss, lam)
            grad[s:e] = g_lam.numpy()
        return grad, hess
    return objective


@torch.no_grad()
def eval_bound(graphs: list, offsets: list[tuple[int, int]], preds: np.ndarray) -> dict:
    """gap% par graphe = (L1(lam) - opt_residual) / |opt_residual| * 100."""
    gaps, l1_vals, opt_res_vals = [], [], []
    for g, (s, e) in zip(graphs, offsets):
        lam = torch.tensor(preds[s:e], dtype=torch.float32)
        l1 = lagrangian_v1ld(lam, g).item()
        opt_res = g.opt.item() - g.curr_cost_raw.item()
        gaps.append((l1 - opt_res) / max(abs(opt_res), 1e-6) * 100.0)
        l1_vals.append(l1)
        opt_res_vals.append(opt_res)
    return {
        "gap_mean":   float(np.mean(gaps)),
        "gap_median": float(np.median(gaps)),
        "gap_p90":    float(np.percentile(gaps, 90)),
        "n_graphs":   len(gaps),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=str(REPO_ROOT / "data/train_v1LD/instances_n10_gbab.txt"))
    ap.add_argument("--nodes",     default=str(REPO_ROOT / "data/train_v1LD/nodes_n10_gbab.txt"))
    ap.add_argument("--max_graphs", type=int, default=1500)
    ap.add_argument("--train_frac", type=float, default=0.9)
    ap.add_argument("--eval_graphs_max", type=int, default=50,
                     help="Nb de graphes éval utilisés pour le suivi périodique "
                          "(coûteux : ~0.25s/graphe via le sous-problème ζ)")
    ap.add_argument("--batch_graphs", type=int, default=64,
                     help="Mini-lot de graphes par step de boosting (mimique le "
                          "mini-batch SGD du GNN — évite d'appeler lagrangian_v1ld "
                          "sur tout le train set à chaque round)")
    ap.add_argument("--n_steps", type=int, default=400,
                     help="Nb de steps (= nb d'arbres ajoutés, 1 par mini-lot)")
    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--learning_rate", type=float, default=0.1)
    ap.add_argument("--min_data_in_leaf", type=int, default=50)
    ap.add_argument("--num_threads", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    t0 = time.time()
    graphs = load_dataset(args.instances, args.nodes, max_graphs=args.max_graphs)
    print(f"[gbdt] {len(graphs)} graphes chargés en {time.time()-t0:.1f}s", flush=True)

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(graphs))
    n_train = int(len(graphs) * args.train_frac)
    train_graphs = [graphs[i] for i in idx[:n_train]]
    eval_graphs  = [graphs[i] for i in idx[n_train:]][: args.eval_graphs_max]
    print(f"[gbdt] train_pool={len(train_graphs)}  eval={len(eval_graphs)}  "
          f"batch_graphs={args.batch_graphs}  n_steps={args.n_steps}", flush=True)

    X_eval, gs_eval = flatten_dataset(eval_graphs)
    off_eval = group_offsets(gs_eval)

    baseline_eval = eval_bound(eval_graphs, off_eval, np.zeros(X_eval.shape[0]))
    print(f"[gbdt] baseline λ=0 : gap_eval(median)={baseline_eval['gap_median']:+.2f}%  "
          f"gap_eval(mean)={baseline_eval['gap_mean']:+.2f}%", flush=True)

    base_params = dict(
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        min_data_in_leaf=args.min_data_in_leaf,
        num_threads=args.num_threads,
        verbose=-1,
        seed=args.seed,
    )

    history = []
    booster = None
    t1 = time.time()
    n_pool = len(train_graphs)
    for step in range(args.n_steps):
        batch_idx = rng.choice(n_pool, size=min(args.batch_graphs, n_pool), replace=False)
        batch_graphs = [train_graphs[i] for i in batch_idx]
        X_batch, gs_batch = flatten_dataset(batch_graphs)
        off_batch = group_offsets(gs_batch)

        dbatch = lgb.Dataset(X_batch, label=np.zeros(X_batch.shape[0]))
        params = dict(base_params, objective=make_objective(batch_graphs, off_batch))
        booster = lgb.train(params, dbatch, num_boost_round=1, init_model=booster)

        if (step + 1) % args.eval_every == 0 or step == 0:
            preds_eval = booster.predict(X_eval)
            m = eval_bound(eval_graphs, off_eval, preds_eval)
            m["step"] = step + 1
            m["elapsed_s"] = time.time() - t1
            history.append(m)
            print(f"[gbdt] step {step+1:4d}/{args.n_steps}  "
                  f"gap_eval(median)={m['gap_median']:+.2f}%  "
                  f"gap_eval(mean)={m['gap_mean']:+.2f}%  "
                  f"({m['elapsed_s']:.0f}s écoulées)", flush=True)

    train_time = time.time() - t1
    print(f"[gbdt] entraînement terminé en {train_time:.1f}s", flush=True)

    preds_eval_final = booster.predict(X_eval)
    final_eval = eval_bound(eval_graphs, off_eval, preds_eval_final)

    print("")
    print("=" * 70)
    print(f"[gbdt] FINAL")
    print(f"  eval  : gap_median={final_eval['gap_median']:+.2f}%  "
          f"gap_mean={final_eval['gap_mean']:+.2f}%  "
          f"(baseline λ=0 : {baseline_eval['gap_median']:+.2f}%)")
    print(f"  temps entraînement : {train_time:.1f}s  ({args.n_steps} steps, "
          f"{args.batch_graphs} graphes/step, train_pool={len(train_graphs)})")
    print("=" * 70)

    if args.out_json:
        out = dict(
            args=vars(args),
            baseline_eval=baseline_eval,
            final_eval=final_eval,
            history=history, train_time_s=train_time,
        )
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[gbdt] résultats écrits dans {args.out_json}")


if __name__ == "__main__":
    main()
