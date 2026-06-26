#!/usr/bin/env python3
"""
diag_overfit_100.py — Diagnostic de cohérence gradient sur 100 graphes diversifiés.

Entraîne 1 GNN partagé (même architecture que train_full_v1ld_multi) sur 100
graphes (n=10/15/20, depths variés) pendant N_EPOCHS epochs.

Métriques par instance à chaque eval_every epochs :
  - L1(λ) brute (borne supérieure)
  - |λ|_mean

En final :
  - Matrice cosine 20×20 des gradients ∂L1_i/∂θ (stratifiée par n)
  - Optionnel : comparaison L1_GNN vs sous-gradient de référence (T=100)

Sorties (out_dir/) :
  fig_convergence.png  fig_lambda.png  fig_cosine.png  [fig_quality.png]  metrics.json

Usage :
    cd <repo_root>
    python gnn/diag_overfit_100.py
    python gnn/diag_overfit_100.py --n_epochs 200 --eval_every 5
    python gnn/diag_overfit_100.py --skip_subgrad   # saute verify_v1ld (lent)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # évite overhead OpenMP sur login node

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn.train_gnn_v1ld import (
    load_raw_nodes, LazyGraphDataset, EagerGraphDataset,
    GNN, lagrangian_v1ld, verify_v1ld,
)


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnostic overfit 100 graphes — cohérence gradient V1LD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",     default=None,
                   help="Dossier train_base_s30. Défaut: <repo>/data/train_base_s30/")
    p.add_argument("--out_dir",      default=None,
                   help="Dossier de sortie. Défaut: <repo>/gnn/runs/diag_overfit_100/")
    p.add_argument("--n_list",        type=int,   nargs="+", default=[10],
                   help="Valeurs de n à charger (ex: 10 15 20). Défaut : [10] seulement.")
    p.add_argument("--n_graphs",     type=int,   default=100,
                   help="Nombre total de graphes (répartis uniformément entre les n de --n_list).")
    p.add_argument("--n_epochs",     type=int,   default=500)
    p.add_argument("--eval_every",   type=int,   default=10,
                   help="Fréquence d'évaluation par instance (epochs).")
    p.add_argument("--batch_size",   type=int,   default=32,
                   help="Accumulation gradient : 1 step Adam par batch_size graphes.")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--eta_min",      type=float, default=1e-5)
    p.add_argument("--hidden",       type=int,   default=64)
    p.add_argument("--m",            type=int,   default=5)
    p.add_argument("--max_norm",     type=float, default=0.0,
                   help="Clip global sur le gradient moyen du batch. 0 = pas de clip (défaut v12).")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--n_cosine",     type=int,   default=20,
                   help="Instances pour la matrice cosine (stratifié par n).")
    p.add_argument("--max_depth",    type=int,   default=4,
                   help="Depth max des nœuds sélectionnés.")
    p.add_argument("--max_load",     type=int,   default=10_000,
                   help="Limite de records chargés par fichier (pour la vitesse).")
    p.add_argument("--skip_subgrad",  action="store_true",
                   help="Sauter verify_v1ld pour toutes les instances.")
    p.add_argument("--subgrad_max_n", type=int, default=15,
                   help="N max pour verify_v1ld (ignore les instances avec n > valeur). "
                        "Défaut 15 : calcule n=10+15 uniquement (n=20 trop lent en Python B&B).")
    p.add_argument("--alpha_l2",     type=float, default=0.0,
                   help="Coefficient de régularisation L2 sur λ (loss += alpha * mean(λ²)).")
    p.add_argument("--lam_max",      type=float, default=0.0,
                   help="Borne tanh sur la sortie du GNN : λ ∈ (-lam_max, lam_max). 0 = non borné.")
    p.add_argument("--n_warmup",     type=int,   default=0,
                   help="Epochs de pré-entraînement supervisé (L2 sur λ* sous-gradient). 0 = désactivé.")
    p.add_argument("--warmup_lr",    type=float, default=1e-3,
                   help="Learning rate pour la phase warmup supervisé (défaut plus élevé que lr).")
    p.add_argument("--warmup_T",     type=int,   default=50,
                   help="Itérations sous-gradient pour générer les cibles du warmup.")
    p.add_argument("--use_max_pool", action="store_true",
                   help="Pool scénario = [mean | max] au lieu de [mean] seul "
                        "(capte les scénarios les plus contraints).")
    p.add_argument("--use_theta_attn", action="store_true",
                   help="Pool graphe attentionnel pondéré par θ_i (v20).")
    p.add_argument("--device",        default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Chargement et sélection stratifiée
# ─────────────────────────────────────────────────────────────────────────────

def load_and_select(
    data_dir: Path,
    m: int,
    n_per_group: int,
    max_depth: int,
    seed: int,
    max_load: int,
    n_list: list | None = None,
) -> tuple[dict, list]:
    """
    Charge instances/nodes pour les n de n_list (défaut [10, 15, 20]), renumérote
    les inst_ids pour éviter les collisions, puis sélectionne n_per_group
    enregistrements par groupe en diversifiant les depths (round-robin).

    Retourne (merged_instances, selected_records).
    """
    rng = random.Random(seed)
    if n_list is None:
        n_list = [10, 15, 20]
    merged_instances: dict = {}
    selected_all: list = []
    id_offset = 0

    for n_val in n_list:
        inst_file  = data_dir / f"instances_n{n_val}_m{m}.txt"
        nodes_file = data_dir / f"nodes_n{n_val}_m{m}.txt"
        if not inst_file.exists() or not nodes_file.exists():
            print(f"  WARN: fichiers manquants pour n={n_val}, ignoré")
            continue

        print(f"  n={n_val}: chargement ...", end=" ", flush=True)
        instances, records = load_raw_nodes(inst_file, nodes_file, max_records=max_load)

        # Renumbering pour éviter les collisions d'inst_id entre groupes
        id_map = {old: old + id_offset for old in instances}
        for old_id, inst in instances.items():
            merged_instances[id_map[old_id]] = inst
        for rec in records:
            rec["inst_id"] = id_map[rec["inst_id"]]
            rec["n_orig"]  = n_val  # annotation locale (non utilisée par build_graph)
        id_offset = max(merged_instances.keys(), default=-1) + 1

        # Filtrer par depth
        filtered = [r for r in records if r["depth"] <= max_depth]
        if not filtered:
            filtered = records
            print(f"WARN: aucun record avec depth≤{max_depth}, pas de filtre.", end=" ")

        # Round-robin sur les depth buckets pour diversifier
        by_depth: dict = defaultdict(list)
        for r in filtered:
            by_depth[r["depth"]].append(r)
        for bucket in by_depth.values():
            rng.shuffle(bucket)

        depths_sorted = sorted(by_depth.keys())
        pools = [list(by_depth[d]) for d in depths_sorted]
        chosen = []
        while len(chosen) < n_per_group:
            progress = False
            for pool in pools:
                if pool and len(chosen) < n_per_group:
                    chosen.append(pool.pop(0))
                    progress = True
            if not progress:
                break

        depth_counts = {d: sum(1 for r in chosen if r["depth"] == d) for d in depths_sorted}
        print(f"{len(chosen)} sélectionnés  depths: {dict(depth_counts)}")
        selected_all.extend(chosen)

    return merged_instances, selected_all


# ─────────────────────────────────────────────────────────────────────────────
# Gradient aplati (pour la matrice cosine)
# ─────────────────────────────────────────────────────────────────────────────

def gradient_vector(model: GNN, g, opt_val: float) -> np.ndarray:
    """Gradient de L1(λ)/opt par rapport à tous les paramètres (1 graphe)."""
    model.train()
    model.zero_grad()
    lam  = model(g)
    loss = lagrangian_v1ld(lam, g) / max(float(g.c_scale), 1.0)
    loss.backward()
    vecs = [p.grad.detach().flatten() for p in model.parameters()
            if p.grad is not None]
    model.zero_grad()
    return torch.cat(vecs).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

N_COLORS = {10: "#1f77b4", 15: "#ff7f0e", 20: "#2ca02c"}


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = (Path(args.data_dir) if args.data_dir
                else REPO_ROOT / "data" / "train_base_s30")
    out_dir  = (Path(args.out_dir) if args.out_dir
                else REPO_ROOT / "gnn" / "runs" / "diag_overfit_100")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[diag] device={device}  seed={args.seed}  out={out_dir}")

    # ── 1. Chargement ─────────────────────────────────────────────────────────
    n_list      = args.n_list
    n_groups    = len(n_list)
    n_per_group = args.n_graphs // n_groups
    print(f"[diag] Chargement — {n_per_group} graphes/groupe ×{n_groups}  (n={n_list}) :")
    merged_instances, selected_recs = load_and_select(
        data_dir, args.m, n_per_group, args.max_depth, args.seed, args.max_load,
        n_list=n_list,
    )
    print(f"[diag] {len(selected_recs)} enregistrements sélectionnés")

    # ── 2. Construction EagerGraphDataset ─────────────────────────────────────
    print("[diag] Construction graphes PyG ...", flush=True)
    lazy_ds  = LazyGraphDataset(merged_instances, selected_recs)
    eager_ds = EagerGraphDataset(lazy_ds)
    graphs_dev = [g.to(device) for g in eager_ds.graphs]
    n_total    = len(graphs_dev)

    n_origs = [int(g.n_orig.item()) if hasattr(g, "n_orig") else 0
               for g in graphs_dev]
    depths  = list(eager_ds._depths)      # aligné sur eager_ds.graphs
    opt_vals = [g.opt.item() for g in graphs_dev]

    print(f"[diag] {n_total} graphes sur {device}")
    for nv in n_list:
        cnt = sum(1 for n in n_origs if n == nv)
        print(f"  n={nv}: {cnt} graphes")

    # ── 3. Modèle + optimiseur ────────────────────────────────────────────────
    model     = GNN(m=args.m, hidden=args.hidden, lam_max=args.lam_max,
                    use_max_pool=args.use_max_pool,
                    use_theta_attn=args.use_theta_attn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=args.eta_min,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[diag] GNN : {n_params:,} params  lr={args.lr}  epochs={args.n_epochs}  "
          f"batch={args.batch_size}  max_norm={args.max_norm}  "
          f"use_max_pool={args.use_max_pool}  use_theta_attn={args.use_theta_attn}")

    # ── 4. Historique par instance ────────────────────────────────────────────
    hist_epochs: list          = []
    hist_loss:   dict          = {i: [] for i in range(n_total)}
    hist_lam:    dict          = {i: [] for i in range(n_total)}

    def eval_all() -> None:
        model.eval()
        with torch.no_grad():
            for gi, g in enumerate(graphs_dev):
                lam = model(g)
                hist_loss[gi].append(lagrangian_v1ld(lam, g).item())
                hist_lam[gi].append(lam.abs().mean().item())
        model.train()

    # ── 5. Warmup supervisé (L2 sur λ* sous-gradient) ────────────────────────
    if args.n_warmup > 0:
        print(f"\n[warmup] Génération cibles sous-gradient (T={args.warmup_T}) "
              f"sur {n_total} graphes ...", flush=True)
        t_wm = time.perf_counter()
        lam_targets = []
        for gi, g in enumerate(graphs_dev):
            g_cpu = g.cpu()
            _, lam_star = verify_v1ld(g_cpu, T=args.warmup_T, return_lam=True)
            lam_targets.append(
                torch.tensor(lam_star.astype(np.float32).flatten(),
                             dtype=torch.float32, device=device))
        print(f"[warmup] Cibles prêtes en {time.perf_counter()-t_wm:.1f}s  "
              f"|λ*| mean={np.mean([t.abs().mean().item() for t in lam_targets]):.4f}")

        opt_wm   = torch.optim.Adam(model.parameters(), lr=args.warmup_lr)
        sched_wm = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_wm, T_max=args.n_warmup, eta_min=1e-5)

        print(f"[warmup] {args.n_warmup} epochs supervisés (lr={args.warmup_lr}) ...",
              flush=True)
        t_wm = time.perf_counter()
        wm_indices = list(range(n_total))
        for wm_ep in range(1, args.n_warmup + 1):
            random.shuffle(wm_indices)
            mse_sum = 0.0
            for gi in wm_indices:
                g   = graphs_dev[gi]
                opt_wm.zero_grad()
                lam_pred = model(g)
                loss_wm  = ((lam_pred - lam_targets[gi]) ** 2).mean()
                loss_wm.backward()
                if args.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_norm=args.max_norm)
                opt_wm.step()
                mse_sum += loss_wm.item()
            sched_wm.step()
            if wm_ep % args.eval_every == 0 or wm_ep == args.n_warmup:
                # Mesure L1 lagrangienne pour suivre la qualité de borne
                model.eval()
                l1_vals = []
                with torch.no_grad():
                    for g in graphs_dev:
                        l1_vals.append(lagrangian_v1ld(model(g), g).item()
                                       / max(float(g.c_scale), 1.0))
                model.train()
                print(f"[warmup] ep {wm_ep:4d}/{args.n_warmup}  "
                      f"mse={mse_sum/n_total:.6f}  "
                      f"L1_norm={np.mean(l1_vals):.4f}  "
                      f"t={time.perf_counter()-t_wm:.0f}s", flush=True)
        print(f"[warmup] Terminé en {time.perf_counter()-t_wm:.0f}s\n", flush=True)

    # ── 6. Boucle d'entraînement Lagrangienne ─────────────────────────────────
    t_start  = time.perf_counter()
    indices  = list(range(n_total))

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        random.shuffle(indices)

        epoch_loss  = 0.0
        accum_count = 0
        accum_grads = None  # buffer des gradients clippés par instance

        for step_i, gi in enumerate(indices):
            g       = graphs_dev[gi]
            optimizer.zero_grad()
            lam  = model(g)
            loss = lagrangian_v1ld(lam, g) / max(float(g.c_scale), 1.0) + args.alpha_l2 * (lam ** 2).mean()
            loss.backward()

            # Clip par instance avant accumulation (max_norm=0 → désactivé)
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=args.max_norm)

            # Accumulation manuelle des gradients clippés
            if accum_grads is None:
                accum_grads = [
                    p.grad.detach().clone() if p.grad is not None else None
                    for p in model.parameters()
                ]
            else:
                for buf, p in zip(accum_grads, model.parameters()):
                    if buf is not None and p.grad is not None:
                        buf.add_(p.grad.detach())

            epoch_loss  += loss.item()
            accum_count += 1
            is_last = (step_i == n_total - 1)

            if (step_i + 1) % args.batch_size == 0 or is_last:
                optimizer.zero_grad()
                for buf, p in zip(accum_grads, model.parameters()):
                    if buf is not None:
                        p.grad = buf / accum_count
                optimizer.step()
                accum_count = 0
                accum_grads = None

        scheduler.step()
        mean_loss = epoch_loss / n_total

        if epoch % args.eval_every == 0 or epoch == 1:
            hist_epochs.append(epoch)
            eval_all()
            elapsed = time.perf_counter() - t_start
            # Résumé : L1 moyen par groupe de n
            by_n_loss: dict = defaultdict(list)
            for gi in range(n_total):
                by_n_loss[n_origs[gi]].append(hist_loss[gi][-1])
            summ = "  ".join(
                f"n{nv}={np.mean(v):.4f}({len(v)})"
                for nv, v in sorted(by_n_loss.items())
            )
            print(f"ep {epoch:4d}/{args.n_epochs}  loss_norm={mean_loss:.4f}  "
                  f"L1_raw [{summ}]  t={elapsed:.0f}s", flush=True)

    print(f"[diag] Entraînement terminé ({time.perf_counter() - t_start:.0f}s)")

    # ── 7. Évaluation finale ──────────────────────────────────────────────────
    model.eval()
    final_l1       = []
    final_lam_mean = []
    final_lam_max  = []

    with torch.no_grad():
        for g in graphs_dev:
            lam = model(g)
            final_l1.append(lagrangian_v1ld(lam, g).item())
            final_lam_mean.append(lam.abs().mean().item())
            final_lam_max.append(lam.abs().max().item())

    final_subgrad: list = [None] * n_total   # None pour les instances ignorées
    if not args.skip_subgrad:
        sg_todo = [gi for gi in range(n_total) if n_origs[gi] <= args.subgrad_max_n]
        print(f"[diag] Sous-gradients de référence (verify_v1ld, T=100) — "
              f"{len(sg_todo)}/{n_total} instances (n≤{args.subgrad_max_n}) ...", flush=True)
        t_sg = time.perf_counter()
        for j, gi in enumerate(sg_todo):
            g_cpu = graphs_dev[gi].cpu()   # verify_v1ld utilise numpy (CPU requis)
            final_subgrad[gi] = verify_v1ld(g_cpu, T=100)
            if (j + 1) % 10 == 0 or j == len(sg_todo) - 1:
                print(f"  [{j+1}/{len(sg_todo)}]  t={time.perf_counter()-t_sg:.0f}s",
                      flush=True)

    # ── 7b. Trackers v10 ─────────────────────────────────────────────────────
    # T1 : |λ| moyen par item (rang 0 = item le plus cher grâce au tri par c_k).
    #      Corrélation négative rang–|λ| attendue si rescaling c_k/N est efficace.
    # T2 : std de λ_ik sur les scénarios pour chaque item k.
    #      Valeur haute = le modèle différencie les scénarios (bénéfice de h_item_k).
    final_lam_by_item   = []  # (n_g,) par instance
    final_lam_std_scen  = []  # (n_g,) par instance
    model.eval()
    with torch.no_grad():
        for g in graphs_dev:
            lam_2d = model(g).view(g.N_z, g.n_g)           # (N_z, n_g)
            final_lam_by_item.append(lam_2d.abs().mean(dim=0).cpu().numpy().tolist())
            final_lam_std_scen.append(lam_2d.std(dim=0).cpu().numpy().tolist())

    # ── 8. Matrice cosine n_cosine × n_cosine ────────────────────────────────
    n_c = args.n_cosine
    n_per_cos = n_c // n_groups

    cosine_idxs: list = []
    by_n_idxs: dict = defaultdict(list)
    for gi, nv in enumerate(n_origs):
        by_n_idxs[nv].append(gi)
    for nv in sorted(by_n_idxs):
        cosine_idxs.extend(by_n_idxs[nv][:n_per_cos])
    cosine_idxs = cosine_idxs[:n_c]
    n_c = len(cosine_idxs)

    print(f"[diag] Gradients pour matrice cosine ({n_c}×{n_c}) ...", flush=True)
    model.train()
    grad_vecs: list = []
    for j, gi in enumerate(cosine_idxs):
        gv = gradient_vector(model, graphs_dev[gi], graphs_dev[gi].c_scale.item())
        grad_vecs.append(gv)
        if (j + 1) % 5 == 0:
            print(f"  [{j+1}/{n_c}]", flush=True)

    norms   = np.array([np.linalg.norm(v) for v in grad_vecs])
    cos_mat = np.zeros((n_c, n_c))
    for i in range(n_c):
        for j in range(n_c):
            if norms[i] > 0 and norms[j] > 0:
                cos_mat[i, j] = (np.dot(grad_vecs[i], grad_vecs[j])
                                 / (norms[i] * norms[j]))

    cos_labels = [f"n{n_origs[gi]}_d{depths[gi]}" for gi in cosine_idxs]
    cos_off_diag = (cos_mat.sum() - np.trace(cos_mat)) / max(n_c * (n_c - 1), 1)
    print(f"[diag] cos_mean_off_diag = {cos_off_diag:.4f}  "
          f"({'incohérent (< 0.1)' if abs(cos_off_diag) < 0.1 else 'cohérent'})")

    # T3 : cosine moyen par paire de depth (d0-d0, d0-d1, ..., d4-d4).
    # Aide à savoir si le conflit vient de paires intra-depth ou cross-depth.
    cos_depth_pairs: dict = {}
    for i_idx in range(n_c):
        for j_idx in range(n_c):
            if i_idx == j_idx:
                continue
            di = depths[cosine_idxs[i_idx]]
            dj = depths[cosine_idxs[j_idx]]
            key = f"d{min(di,dj)}_d{max(di,dj)}"
            cos_depth_pairs.setdefault(key, []).append(float(cos_mat[i_idx, j_idx]))
    cos_depth_mean = {k: float(np.mean(v)) for k, v in sorted(cos_depth_pairs.items())}

    # ── 9. Figures ────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 9})

    # 8a. Convergence L1 par instance
    fig, ax = plt.subplots(figsize=(13, 5))
    for gi in range(n_total):
        ax.plot(hist_epochs, hist_loss[gi],
                color=N_COLORS.get(n_origs[gi], "gray"), alpha=0.2, lw=0.7)
    for nv, color in sorted(N_COLORS.items()):
        grp = [gi for gi in range(n_total) if n_origs[gi] == nv]
        if grp:
            mat = np.array([hist_loss[gi] for gi in grp])
            ax.plot(hist_epochs, mat.mean(axis=0), color=color, lw=2.0,
                    label=f"n={nv} (moy., N={len(grp)})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1(λ) brute")
    ax.set_title("Convergence L1 par instance — 1 GNN partagé, 100 graphes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_convergence.png", dpi=150)
    plt.close(fig)

    # 8b. |λ|_mean par instance
    fig, ax = plt.subplots(figsize=(13, 5))
    for gi in range(n_total):
        ax.plot(hist_epochs, hist_lam[gi],
                color=N_COLORS.get(n_origs[gi], "gray"), alpha=0.2, lw=0.7)
    for nv, color in sorted(N_COLORS.items()):
        grp = [gi for gi in range(n_total) if n_origs[gi] == nv]
        if grp:
            mat = np.array([hist_lam[gi] for gi in grp])
            ax.plot(hist_epochs, mat.mean(axis=0), color=color, lw=2.0,
                    label=f"n={nv} (moy.)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|λ|_mean")
    ax.set_title("|λ| moyen par instance au fil des epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_lambda.png", dpi=150)
    plt.close(fig)

    # 8c. Matrice cosine heatmap
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cos_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="cosine similarity")
    tick_step = max(1, n_c // 10)
    ax.set_xticks(range(0, n_c, tick_step))
    ax.set_xticklabels(cos_labels[::tick_step], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(0, n_c, tick_step))
    ax.set_yticklabels(cos_labels[::tick_step], fontsize=7)
    # Séparateurs entre groupes
    for k in range(1, n_groups):
        s = k * n_per_cos - 0.5
        ax.axhline(s, color="black", lw=1.5)
        ax.axvline(s, color="black", lw=1.5)
    ax.set_title(
        f"Cosine sim. ∂L₁/∂θ — {n_c} instances (stratifiées par n)\n"
        f"Moy. hors-diag = {cos_off_diag:.3f}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cosine.png", dpi=150)
    plt.close(fig)

    # 9d. Qualité GNN vs sous-gradient (instances calculées seulement)
    sg_computed = [(gi, final_subgrad[gi]) for gi in range(n_total)
                   if final_subgrad[gi] is not None]
    if sg_computed:
        fig, ax = plt.subplots(figsize=(7, 6))
        for gi, sg in sg_computed:
            ax.scatter(sg, final_l1[gi],
                       color=N_COLORS.get(n_origs[gi], "gray"),
                       alpha=0.7, s=25, zorder=3)
        all_v = [sg for _, sg in sg_computed] + [final_l1[gi] for gi, _ in sg_computed]
        mn, mx = min(all_v), max(all_v)
        ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, label="y = x (identique)")
        for nv, color in sorted(N_COLORS.items()):
            ax.scatter([], [], color=color, s=40, label=f"n={nv}")
        ax.set_xlabel("L1 sous-gradient (T=100)")
        ax.set_ylabel("L1 GNN (final)")
        ax.set_title("Qualité GNN vs sous-gradient — après overfit 500 epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_quality.png", dpi=150)
        plt.close(fig)

    # ── 10. JSON ──────────────────────────────────────────────────────────────
    metrics = {
        "config":               vars(args),
        "n_graphs":             n_total,
        "n_params":             n_params,
        "hist_epochs":          hist_epochs,
        "cosine_mean_off_diag": float(cos_off_diag),
        "cosine_matrix":        cos_mat.tolist(),
        "cosine_labels":        cos_labels,
        "cosine_indices":       cosine_idxs,
        # T3 : cosine par paire de depth
        "cos_by_depth_pair":    cos_depth_mean,
        "by_instance": [
            {
                "idx":                  gi,
                "n_orig":               n_origs[gi],
                "depth":                depths[gi],
                "opt_val":              opt_vals[gi],
                "loss_hist":            hist_loss[gi],
                "lam_hist":             hist_lam[gi],
                "final_l1_gnn":         final_l1[gi],
                "final_lam_mean":       final_lam_mean[gi],
                "final_lam_max":        final_lam_max[gi],
                "final_l1_subgrad":     final_subgrad[gi] if final_subgrad else None,
                # T1 : |λ| moyen par item (trié par c_k desc)
                "lam_by_item":          final_lam_by_item[gi],
                # T2 : std de λ sur les scénarios par item
                "lam_std_over_scen":    final_lam_std_scen[gi],
            }
            for gi in range(n_total)
        ],
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── 11. Résumé final ──────────────────────────────────────────────────────
    print(f"\n[diag] ─── Résumé ───────────────────────────────────────")
    by_n_final: dict = defaultdict(list)
    for gi in range(n_total):
        by_n_final[n_origs[gi]].append(final_l1[gi])
    for nv, vals in sorted(by_n_final.items()):
        print(f"  n={nv:2d}  L1_final  mean={np.mean(vals):.4f}  "
              f"std={np.std(vals):.4f}  min={np.min(vals):.4f}  max={np.max(vals):.4f}")
    print(f"  cosine_mean_off_diag = {cos_off_diag:.4f}")

    # T3 : cosine par paire depth
    print(f"  Cosine par paire depth (hors-diag) :")
    for key, val in sorted(cos_depth_mean.items()):
        print(f"    {key}: {val:+.4f}")

    # T1 : corrélation rang-item / |λ| (attendu négatif : item cher → grand λ)
    lam_rank_corrs = []
    for lam_item in final_lam_by_item:
        if len(lam_item) > 2:
            corr = float(np.corrcoef(np.arange(len(lam_item)), lam_item)[0, 1])
            if not np.isnan(corr):
                lam_rank_corrs.append(corr)
    if lam_rank_corrs:
        print(f"  T1 corr rang–|λ|: {np.mean(lam_rank_corrs):+.3f}±{np.std(lam_rank_corrs):.3f}"
              f"  (attendu <0 : item cher → λ plus grand)")

    # T2 : std de λ sur les scénarios (bénéfice de h_item_k)
    all_stds = [v for inst in final_lam_std_scen for v in inst]
    if all_stds:
        print(f"  T2 std(λ) sur scénarios: mean={np.mean(all_stds):.4f}"
              f"  (↑ = modèle différencie mieux les scénarios)")

    if sg_computed:
        diffs = [final_l1[gi] - sg for gi, sg in sg_computed]
        print(f"  GNN - subgrad (n≤{args.subgrad_max_n})  "
              f"mean={np.mean(diffs):.4f}  max={np.max(diffs):.4f}  "
              f"pct_better={100*np.mean(np.array(diffs)<0):.1f}%")
    print(f"\n[diag] Fichiers dans {out_dir}/")
    figs = ["fig_convergence.png", "fig_lambda.png", "fig_cosine.png"]
    if final_subgrad:
        figs.append("fig_quality.png")
    print("  " + "  ".join(figs))
    print("  metrics.json")


if __name__ == "__main__":
    main()
