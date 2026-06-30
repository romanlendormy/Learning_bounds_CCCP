#!/usr/bin/env python3
"""
train_scaled_v1ld.py — Entraînement GNN V1LD sur ~10 000 graphes diversifiés.

Par rapport à train_full_v1ld_multi :
  - Sélection stratifiée : TOUTES les racines (depth=0) + depth>0 échantillonné
  - Multi-N pour n=10  : charge N=50/100/200/500 (train_N{N}/ + train_base_s30)
  - Warmup supervisé   : L2 sur λ* sous-gradient, sur les racines train seulement
  - depth_w0           : poids multiplicatif sur la loss des nœuds depth=0
  - Architecture v21   : FiLM, theta_attn, max_pool (flags optionnels)
  - lr_T0              : CosineAnnealingWarmRestarts (0 = standard CosineAnnealingLR)
  - Eval par depth     : gap par profondeur en plus du gap par n

Usage :
    python gnn/train_scaled_v1ld.py
    sbatch run_train_scaled.sh
"""

from __future__ import annotations

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
from torch_geometric.loader import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entraîne GNN V1LD sur ~10k graphes stratifiés (n=10/15/20, N variés).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Données ───────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",    default=None,
                   help="Racine contenant train_base_s30/, train_N50/, etc. Défaut: <repo>/data/")
    p.add_argument("--out_dir",     default=None,
                   help="Dossier de sortie. Défaut: <repo>/gnn/runs/")
    p.add_argument("--n_list",      type=int, nargs="+", default=[10, 15, 20],
                   help="Valeurs de n à charger.")
    p.add_argument("--n_scens_n10", type=int, nargs="+", default=[50, 100, 200, 500],
                   help="Valeurs de N à charger pour n=10 (multi-N). "
                        "Pour n=15/20, seul N=500 (train_base_s30) est utilisé.")
    p.add_argument("--m",           type=int, default=5)
    p.add_argument("--target_graphs", type=int, default=10_000,
                   help="Nombre total de graphes dans le dataset sélectionné.")
    p.add_argument("--max_load",    type=int, default=60_000,
                   help="Limite de records chargés par fichier avant la sélection stratifiée.")
    p.add_argument("--train_frac",  type=float, default=0.8)
    p.add_argument("--seed",        type=int, default=42)

    # ── Modèle ────────────────────────────────────────────────────────────────
    p.add_argument("--hidden",         type=int,   default=128)
    p.add_argument("--use_max_pool",   action="store_true")
    p.add_argument("--use_theta_attn", action="store_true")
    p.add_argument("--use_film",       action="store_true")
    p.add_argument("--lam_max",        type=float, default=0.0)

    # ── Entraînement ──────────────────────────────────────────────────────────
    p.add_argument("--n_epochs",    type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--eta_min",     type=float, default=1e-5)
    p.add_argument("--lr_T0",       type=int,   default=0,
                   help="T_0 pour CosineAnnealingWarmRestarts. 0 = CosineAnnealingLR.")
    p.add_argument("--lr_T_mult",   type=int,   default=1,
                   help="Facteur multiplicatif des cycles CAWR : T_mult=2 double la durée "
                        "à chaque restart. Ignoré si lr_T0=0.")
    p.add_argument("--lr_warmup_ep", type=int,  default=0,
                   help="Warmup Lagrangien (epochs) : LR monte linéairement de eta_min à lr, "
                        "puis cosinus jusqu'à eta_min sur le reste. 0 = désactivé.")
    p.add_argument("--no_scheduler", action="store_true",
                   help="LR constant, aucun scheduler.")
    p.add_argument("--max_norm",    type=float, default=0.5)
    p.add_argument("--weight_decay",type=float, default=0.0)
    p.add_argument("--alpha_l2",    type=float, default=0.0,
                   help="Régularisation L2 sur λ.")
    p.add_argument("--center_lam",  action="store_true",
                   help="Centrage des λ sur les scénarios après la sortie du GNN : "
                        "λ_ik ← λ_ik − (1/N_z) Σ_i λ_ik. Élimine la dérive de jauge.")
    p.add_argument("--depth_w0",    type=float, default=1.0,
                   help="Poids sur la loss des nœuds depth=0. 1.0 = désactivé.")
    p.add_argument("--depth_alpha", type=float, default=0.0,
                   help="Exposant du poids par depth : w=(depth+1)^alpha. "
                        "0=uniforme ; <0=favorise la racine. Remplacé par depth_w0 si depth_w0>1.")

    # ── Warmup supervisé (roots train seulement) ──────────────────────────────
    p.add_argument("--n_warmup",    type=int,   default=10,
                   help="Epochs de warmup supervisé L2 sur λ* (tous graphes n≤warmup_max_n). 0 = désactivé.")
    p.add_argument("--warmup_lr",   type=float, default=1e-3)
    p.add_argument("--warmup_T",    type=int,   default=50,
                   help="Itérations sous-gradient pour générer λ* cibles.")
    p.add_argument("--warmup_max_n",type=int,   default=15,
                   help="n max pour le warmup (n=20 trop lent en Python B&B).")

    # ── Évaluation & infrastructure ───────────────────────────────────────────
    p.add_argument("--eval_sample", type=int,   default=400,
                   help="Nombre max de graphes test pour l'éval gap.")
    p.add_argument("--eval_mixed",  action="store_true",
                   help="Eval set stratifié par depth (même distribution que train) "
                        "plutôt que racines uniquement. Rend loss_eval comparable à loss_train.")
    p.add_argument("--eval_every",  type=int,   default=5)
    p.add_argument("--keep_ckpts",  type=int,   default=3)
    p.add_argument("--num_workers", type=int,   default=3)
    p.add_argument("--run_tag",     type=str,   default="")
    p.add_argument("--device",      default="auto", choices=["auto", "cpu", "cuda"])

    # ── Figure qualité ────────────────────────────────────────────────────────
    p.add_argument("--skip_quality",  action="store_true",
                   help="Ne pas calculer la figure qualité GNN vs sous-gradient en fin d'entraînement.")
    p.add_argument("--quality_T",     type=int,   default=100,
                   help="Itérations sous-gradient pour fig_quality (défaut 100).")
    p.add_argument("--eval_sg_T",     type=int,   default=50,
                   help="Itérations sous-gradient pour la baseline pct_vs_sg durant l'éval (défaut 50).")
    p.add_argument("--quality_max_n", type=int,   default=15,
                   help="n max pour fig_quality (n=20 trop lent en Python B&B). Défaut 15.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Chargement multi-N / multi-n
# ─────────────────────────────────────────────────────────────────────────────

def discover_pairs(
    data_root: Path,
    n_list: list[int],
    n_scens_n10: list[int],
    m: int,
) -> list[tuple[Path, Path, int, int]]:
    """
    Retourne [(inst_file, nodes_file, n_val, N_val)] pour chaque combinaison
    disponible sur disque.

    n=10 : cherche dans train_N{N}/ pour N in n_scens_n10 (N=500 → train_base_s30)
    n!=10 : cherche uniquement dans train_base_s30 (N=500)
    """
    def src_dir(N: int) -> Path:
        return data_root / ("train_base_s30" if N == 500 else f"train_N{N}")

    pairs = []
    for n in sorted(n_list):
        n_vals = n_scens_n10 if n == 10 else [500]
        for N in sorted(n_vals):
            d = src_dir(N)
            inst_f  = d / f"instances_n{n}_m{m}.txt"
            nodes_f = d / f"nodes_n{n}_m{m}.txt"
            if inst_f.exists() and nodes_f.exists():
                pairs.append((inst_f, nodes_f, n, N))
            else:
                print(f"  WARN: absent — n={n} N={N} ({d.name})")
    return pairs


def load_and_merge(
    pairs: list[tuple[Path, Path, int, int]],
    max_load: int,
) -> tuple[dict, list]:
    """
    Charge les (inst_file, nodes_file) de toutes les paires, renumérote les
    inst_ids pour éviter les collisions, annote chaque record avec n_orig et N_orig.
    """
    merged_instances: dict = {}
    merged_records:   list = []

    for inst_f, nodes_f, n_val, N_val in pairs:
        print(f"  n={n_val} N={N_val}: chargement ...", end=" ", flush=True)
        instances, records = load_raw_nodes(inst_f, nodes_f, max_records=max_load)

        offset = max(merged_instances.keys(), default=-1) + 1
        id_map = {old: old + offset for old in instances}
        for old_id, inst in instances.items():
            merged_instances[id_map[old_id]] = inst
        for rec in records:
            nr = dict(rec)
            nr["inst_id"] = id_map[rec["inst_id"]]
            nr["n_orig"]  = n_val
            nr["N_orig"]  = N_val
            merged_records.append(nr)

        depth_cnt = defaultdict(int)
        for r in records:
            depth_cnt[r["depth"]] += 1
        print(f"{len(records)} records  depths:{dict(sorted(depth_cnt.items()))}")

    return merged_instances, merged_records


# ─────────────────────────────────────────────────────────────────────────────
# Sélection stratifiée
# ─────────────────────────────────────────────────────────────────────────────

def select_stratified(
    all_records: list,
    target_graphs: int,
    seed: int,
) -> list:
    """
    Garde TOUTES les racines (depth=0).
    Complète jusqu'à target_graphs avec les depth>0, en round-robin sur
    les buckets (n_orig, depth) pour maintenir la diversité.
    """
    rng = random.Random(seed)

    roots     = [r for r in all_records if r["depth"] == 0]
    non_roots = [r for r in all_records if r["depth"] >  0]

    budget = max(0, target_graphs - len(roots))

    # Buckets stratifiés
    buckets: dict = defaultdict(list)
    for r in non_roots:
        buckets[(r.get("n_orig", 0), r.get("N_orig", 0), r["depth"])].append(r)
    for lst in buckets.values():
        rng.shuffle(lst)

    pools  = {k: list(v) for k, v in sorted(buckets.items())}
    chosen: list = []
    while len(chosen) < budget:
        progress = False
        for k in list(pools.keys()):
            if pools[k] and len(chosen) < budget:
                chosen.append(pools[k].pop(0))
                progress = True
        if not progress:
            break

    selected = roots + chosen

    # Stats
    by_depth: dict = defaultdict(int)
    by_n:     dict = defaultdict(int)
    by_N:     dict = defaultdict(int)
    for r in selected:
        by_depth[r["depth"]] += 1
        by_n[r.get("n_orig", 0)] += 1
        by_N[r.get("N_orig", 0)] += 1
    print(f"[select] {len(roots)} racines + {len(chosen)} non-racines = {len(selected)} total")
    print(f"  par depth : {dict(sorted(by_depth.items()))}")
    print(f"  par n     : {dict(sorted(by_n.items()))}")
    print(f"  par N     : {dict(sorted(by_N.items()))}")
    return selected


def select_eval_mixed(
    test_records: list,
    eval_sample: int,
    seed: int,
) -> list:
    """
    Sélection stratifiée par depth pour l'eval set, sur les records test uniquement.
    Round-robin sur les depths pour reproduire la même distribution que le train set.
    """
    rng = random.Random(seed)
    by_depth: dict = defaultdict(list)
    for r in test_records:
        by_depth[r["depth"]].append(r)

    pools = {d: list(v) for d, v in sorted(by_depth.items())}
    for lst in pools.values():
        rng.shuffle(lst)

    chosen: list = []
    while len(chosen) < eval_sample:
        progress = False
        for d in list(pools.keys()):
            if pools[d] and len(chosen) < eval_sample:
                chosen.append(pools[d].pop(0))
                progress = True
        if not progress:
            break

    by_d: dict = defaultdict(int)
    for r in chosen:
        by_d[r["depth"]] += 1
    print(f"[eval_mixed] {len(chosen)} graphes  par depth: {dict(sorted(by_d.items()))}")
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# Split train / test au niveau instance
# ─────────────────────────────────────────────────────────────────────────────

def split_by_instance(
    instances: dict,
    all_ds: LazyGraphDataset,
    train_frac: float,
    seed: int,
) -> tuple[LazyGraphDataset, LazyGraphDataset]:
    inst_ids = np.array(sorted(instances.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(inst_ids)
    n_tr = int(len(inst_ids) * train_frac)
    train_ids = set(inst_ids[:n_tr].tolist())
    test_ids  = set(inst_ids[n_tr:].tolist())
    return all_ds.filter_by_inst_ids(train_ids), all_ds.filter_by_inst_ids(test_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Warmup supervisé (roots train seulement)
# ─────────────────────────────────────────────────────────────────────────────

def warmup_on_roots(
    model: GNN,
    train_ds: LazyGraphDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Warmup L2 sur λ* sous-gradient, sur tous les graphes train avec n <= warmup_max_n."""
    root_recs = [r for r in train_ds.raw_records
                 if r.get("n_orig", 0) <= args.warmup_max_n]
    if not root_recs:
        print("[warmup] Aucun graphe éligible — skip")
        return

    root_ds = LazyGraphDataset(train_ds.instances, root_recs)
    print(f"[warmup] Construction EagerGraphDataset ({len(root_ds)} graphes) ...", flush=True)
    eager = EagerGraphDataset(root_ds)
    graphs = [eager[i].to(device) for i in range(len(eager))]
    n_wm = len(graphs)

    print(f"[warmup] Génération cibles sous-gradient (T={args.warmup_T}) ...", flush=True)
    t0 = time.perf_counter()
    lam_targets = []
    for g in graphs:
        g.cpu()  # in-place dans PyG
        _, lam_star = verify_v1ld(g, T=args.warmup_T, return_lam=True)
        lam_targets.append(
            torch.tensor(lam_star.astype(np.float32).flatten(),
                         dtype=torch.float32, device=device))
        g.to(device)  # remettre sur GPU après verify
    print(f"[warmup] Cibles prêtes en {time.perf_counter()-t0:.1f}s  "
          f"|λ*|_mean={np.mean([t.abs().mean().item() for t in lam_targets]):.4f}")

    opt_wm   = torch.optim.Adam(model.parameters(), lr=args.warmup_lr)
    sched_wm = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_wm, T_max=args.n_warmup, eta_min=1e-5)
    idx_list = list(range(n_wm))

    print(f"[warmup] {args.n_warmup} epochs (lr={args.warmup_lr}) ...", flush=True)
    t0 = time.perf_counter()
    for ep in range(1, args.n_warmup + 1):
        random.shuffle(idx_list)
        mse_sum = 0.0
        for gi in idx_list:
            g = graphs[gi]
            opt_wm.zero_grad()
            lam_pred = model(g)
            loss_wm  = ((lam_pred - lam_targets[gi]) ** 2).mean()
            loss_wm.backward()
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            opt_wm.step()
            mse_sum += loss_wm.item()
        sched_wm.step()
        if ep % max(1, args.n_warmup // 10) == 0 or ep == args.n_warmup:
            model.eval()
            with torch.no_grad():
                l1_vals = [lagrangian_v1ld(model(g), g).item() / max(float(g.c_scale), 1.0)
                           for g in graphs]
            model.train()
            print(f"[warmup] ep {ep:4d}/{args.n_warmup}  mse={mse_sum/n_wm:.6f}  "
                  f"L1_norm={float(np.mean(l1_vals)):.4f}  "
                  f"t={time.perf_counter()-t0:.0f}s", flush=True)
    print(f"[warmup] Terminé en {time.perf_counter()-t0:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_l1_zero(graphs: list, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return np.array([lagrangian_v1ld(torch.zeros(g.N_z * g.n_g, device=device), g).item()
                         for g in graphs])


def eval_epoch(graphs, l1_zeros, model, device, center_lam=False):
    """Retourne (l1_mean, t2_mean, pct_better) sur les graphes eval."""
    model.eval()
    l1_vals, t2_vals, n_better = [], [], 0
    with torch.no_grad():
        for g, l0 in zip(graphs, l1_zeros):
            lam    = model(g)
            if center_lam:
                lam = _apply_center(lam, int(g.N_z), int(g.n_g))
            l1_gnn = lagrangian_v1ld(lam, g).item()
            lam_2d = lam.view(int(g.N_z), int(g.n_g))
            l1_vals.append(l1_gnn)
            t2_vals.append(lam_2d.std(dim=0).mean().item())
            if l1_gnn < l0:
                n_better += 1
    model.train()
    return (float(np.mean(l1_vals)), float(np.mean(t2_vals)),
            100.0 * n_better / max(len(graphs), 1))


def eval_loss_epoch(graphs, model, device, center_lam=False):
    """Loss normalisée (même formule que train, sans depth_w0) sur les graphes eval."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for g in graphs:
            lam = model(g)
            if center_lam:
                lam = _apply_center(lam, int(g.N_z), int(g.n_g))
            c_scale = max(float(g.c_scale), 1.0)
            total += lagrangian_v1ld(lam, g).item() / c_scale
    model.train()
    return total / max(len(graphs), 1)


def eval_by_n(graphs, l1_zeros, model, device, center_lam=False):
    model.eval()
    by_n: dict = {}
    with torch.no_grad():
        for g, l0 in zip(graphs, l1_zeros):
            n_key  = int(g.n_orig.item()) if hasattr(g, "n_orig") else -1
            lam    = model(g)
            if center_lam:
                lam = _apply_center(lam, int(g.N_z), int(g.n_g))
            l1_gnn = lagrangian_v1ld(lam, g).item()
            by_n.setdefault(n_key, {"l1": [], "better": []})
            by_n[n_key]["l1"].append(l1_gnn)
            by_n[n_key]["better"].append(int(l1_gnn < l0))
    model.train()
    return {n: {"l1_mean": float(np.mean(v["l1"])),
                "pct_better": float(100.0 * np.mean(v["better"])),
                "count": len(v["l1"])}
            for n, v in sorted(by_n.items())}


def eval_by_depth(graphs, l1_zeros, model, device, center_lam=False):
    """L1 et pct_better par profondeur B&B."""
    model.eval()
    by_d: dict = {}
    with torch.no_grad():
        for g, l0 in zip(graphs, l1_zeros):
            n_orig = int(g.n_orig.item()) if hasattr(g, "n_orig") else 10
            d_key  = int(round(g.context[0].item() * n_orig))
            lam    = model(g)
            if center_lam:
                lam = _apply_center(lam, int(g.N_z), int(g.n_g))
            l1_gnn = lagrangian_v1ld(lam, g).item()
            by_d.setdefault(d_key, {"l1": [], "better": []})
            by_d[d_key]["l1"].append(l1_gnn)
            by_d[d_key]["better"].append(int(l1_gnn < l0))
    model.train()
    return {d: {"l1_mean": float(np.mean(v["l1"])),
                "pct_better": float(100.0 * np.mean(v["better"])),
                "count": len(v["l1"])}
            for d, v in sorted(by_d.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def ckpt_path(out_dir: Path, run_name: str, epoch: int) -> Path:
    return out_dir / f"ckpt_{run_name}_ep{epoch:04d}.pt"


def save_checkpoint(out_dir, run_name, epoch, model, optimizer, scheduler,
                    history, args, saved, keep, keep_every=10):
    path = ckpt_path(out_dir, run_name, epoch)
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "history": history, "args": vars(args)}, path)
    if epoch % keep_every == 0:
        return
    saved.append(path)
    if len(saved) > keep:
        saved.pop(0).unlink(missing_ok=True)


def find_latest_checkpoint(out_dir: Path, run_name: str) -> Path | None:
    ckpts = sorted(out_dir.glob(f"ckpt_{run_name}_ep*.pt"))
    return ckpts[-1] if ckpts else None


def save_history(path: Path, history: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _apply_center(lam: torch.Tensor, N_z: int, n_g: int) -> torch.Tensor:
    """λ_ik ← λ_ik − (1/N_z) Σ_i λ_ik  (centrage sur les scénarios, différentiable)."""
    lam_2d = lam.view(N_z, n_g)
    return (lam_2d - lam_2d.mean(dim=0, keepdim=True)).reshape(-1)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data"
    out_dir   = Path(args.out_dir)  if args.out_dir  else REPO_ROOT / "gnn" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    # ── 1. Découverte des fichiers ─────────────────────────────────────────────
    log("Découverte des données ...")
    pairs = discover_pairs(data_root, args.n_list, args.n_scens_n10, args.m)
    if not pairs:
        log("ERREUR : aucun fichier trouvé — vérifiez --data_dir et --n_list")
        sys.exit(1)
    for inst_f, nodes_f, n_val, N_val in pairs:
        log(f"  n={n_val}  N={N_val}  {nodes_f.name}")

    # ── 2. Chargement & fusion ─────────────────────────────────────────────────
    log("Chargement ...")
    instances, all_records = load_and_merge(pairs, args.max_load)
    log(f"  {len(instances)} instances  {len(all_records)} records bruts")

    # ── 3. Sélection stratifiée ────────────────────────────────────────────────
    selected = select_stratified(all_records, args.target_graphs, args.seed)
    all_ds   = LazyGraphDataset(instances, selected)

    # ── 4. Split train / test ──────────────────────────────────────────────────
    train_ds, test_ds = split_by_instance(instances, all_ds, args.train_frac, args.seed)
    log(f"split {args.train_frac:.0%}/{1-args.train_frac:.0%} — "
        f"train={len(train_ds)}  test={len(test_ds)}")

    # Stats split
    for label, ds in [("train", train_ds), ("test", test_ds)]:
        by_d = defaultdict(int)
        for r in ds.raw_records:
            by_d[r["depth"]] += 1
        log(f"  {label} par depth: {dict(sorted(by_d.items()))}")

    # ── 5. Ensemble d'évaluation : stratifié par depth ou racines uniquement ──
    if args.eval_mixed:
        eval_recs = select_eval_mixed(test_ds.raw_records, args.eval_sample, args.seed)
        eval_desc = f"{len(eval_recs)} graphes test (depths mixtes)"
    else:
        root_test_recs = [r for r in test_ds.raw_records if r["depth"] == 0]
        n_eval = min(args.eval_sample, len(root_test_recs))
        random.Random(args.seed).shuffle(root_test_recs)
        eval_recs = root_test_recs[:n_eval]
        eval_desc = f"{len(eval_recs)} racines test"
    eval_lazy  = LazyGraphDataset(instances, eval_recs)
    log(f"Construction eval set ({eval_desc}) ...")
    t0 = time.time()
    eager_eval = EagerGraphDataset(eval_lazy)
    eval_graphs = [eager_eval[i].to(device) for i in range(len(eager_eval))]
    log(f"  fait en {time.time()-t0:.1f}s")

    log("Précalcul L1(λ=0) ...")
    l1_zeros = compute_l1_zero(eval_graphs, device)
    log(f"  mean L1(0) = {np.mean(l1_zeros):.2f}")

    log(f"Précalcul baseline sous-gradient eval (T={args.eval_sg_T}, {len(eval_graphs)} graphes) ...")
    t0_sg = time.time()
    l1_sg_eval = np.array([verify_v1ld(g.cpu(), T=args.eval_sg_T) for g in eval_graphs])
    for g in eval_graphs:
        g.to(device)
    log(f"  fait en {time.time()-t0_sg:.1f}s  mean L1_sg(T={args.eval_sg_T}) = {np.mean(l1_sg_eval):.2f}")

    # ── 6. Modèle ──────────────────────────────────────────────────────────────
    model = GNN(m=args.m, hidden=args.hidden, lam_max=args.lam_max,
                use_max_pool=args.use_max_pool,
                use_theta_attn=args.use_theta_attn,
                use_film=args.use_film).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        sched_desc = "ConstantLR"
    elif args.lr_warmup_ep > 0:
        n_decay = max(args.n_epochs - args.lr_warmup_ep, 1)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=args.eta_min / args.lr,
                    end_factor=1.0,
                    total_iters=args.lr_warmup_ep,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=n_decay, eta_min=args.eta_min,
                ),
            ],
            milestones=[args.lr_warmup_ep],
        )
        sched_desc = f"LinearWarmup({args.lr_warmup_ep})+CosineDecay"
    elif args.lr_T0 > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.lr_T0, T_mult=args.lr_T_mult, eta_min=args.eta_min)
        sched_desc = f"CAWR(T0={args.lr_T0},Tmult={args.lr_T_mult})"
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.n_epochs, eta_min=args.eta_min)
        sched_desc = "CosineAnnealingLR"

    n_params = sum(p.numel() for p in model.parameters())
    log(f"GNN — m={args.m}  hidden={args.hidden}  params={n_params:,}  "
        f"use_max_pool={args.use_max_pool}  use_theta_attn={args.use_theta_attn}  "
        f"use_film={args.use_film}  scheduler={sched_desc}  depth_w0={args.depth_w0}")

    # ── 7. Run name & auto-resume ──────────────────────────────────────────────
    tag = f"scaled_n{'_'.join(str(n) for n in sorted(args.n_list))}"
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    run_name = (f"{tag}_ep{args.n_epochs}_lr{args.lr}_bs{args.batch_size}"
                f"_h{args.hidden}_seed{args.seed}{tag_suffix}")
    metrics_path = out_dir / f"metrics_{run_name}.json"
    fig_path     = out_dir / f"fig_{run_name}.png"
    log(f"run : {run_name}")

    history: dict = {
        "epochs_loss": [], "loss": [], "loss_std": [], "l1_train": [],
        "lam_abs_mean": [], "lam_abs_max": [], "t2_mean": [],
        "grad_norm_max": [], "grad_norm_mean": [],
        "epochs_eval": [], "l1_eval": [], "loss_eval": [], "t2_eval": [], "pct_better": [],
        "l1_by_n": [],
    }
    saved_ckpts: list[Path] = []
    start_epoch = 1

    latest = find_latest_checkpoint(out_dir, run_name)
    if latest:
        log(f"Checkpoint trouvé : {latest.name} — reprise automatique")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        history     = ckpt.get("history", history)
        for k in {"epochs_loss", "loss", "loss_std", "l1_train", "lam_abs_mean", "lam_abs_max",
                  "t2_mean", "grad_norm_max", "grad_norm_mean",
                  "epochs_eval", "l1_eval", "loss_eval", "t2_eval", "pct_better", "l1_by_n"}:
            if k not in history:
                history[k] = []
        saved_ckpts = [
            p for p in sorted(out_dir.glob(f"ckpt_{run_name}_ep*.pt"))
            if int(p.stem.rsplit("ep", 1)[-1]) % 10 != 0
        ]
        log(f"  reprise epoch {start_epoch}")
    else:
        log("Démarrage from scratch")

    # ── 8. Warmup supervisé ────────────────────────────────────────────────────
    if args.n_warmup > 0 and start_epoch == 1:
        warmup_on_roots(model, train_ds, args, device)

    # ── 9. DataLoader ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )
    n_train = len(train_ds)
    log(f"DataLoader — {n_train} graphes  workers={args.num_workers}")
    log(f"Entraînement — {args.n_epochs} epochs  batch={args.batch_size}  "
        f"lr={args.lr}  eta_min={args.eta_min}")

    t_start = time.time()

    # ── 10. Boucle principale ──────────────────────────────────────────────────
    for epoch in range(start_epoch, args.n_epochs + 1):
        model.train()
        epoch_loss = epoch_loss_sq = 0.0
        epoch_l1_train = 0.0
        accum_count = 0
        accum_grads = None
        epoch_gnorm = epoch_gnorm_sum = 0.0
        epoch_nsteps = 0
        lam_abs_sum = lam_abs_max_ep = t2_sum = 0.0

        t_ep = time.perf_counter()

        for step_i, g in enumerate(train_loader):
            g = g.to(device, non_blocking=True)

            optimizer.zero_grad()
            lam = model(g)
            if args.center_lam:
                lam = _apply_center(lam, int(g.N_z), int(g.n_g))

            with torch.no_grad():
                lam_d = lam.detach().abs()
                lam_abs_sum    += lam_d.mean().item()
                lam_abs_max_ep  = max(lam_abs_max_ep, lam_d.max().item())
                n_z_i = int(g.N_z) if isinstance(g.N_z, int) else int(g.N_z.item())
                n_g_i = int(g.n_g) if isinstance(g.n_g, int) else int(g.n_g.item())
                t2_sum += lam.detach().view(n_z_i, n_g_i).std(dim=0).mean().item()

            # Normalisation par c_scale (rend loss ≈ O(1)) + depth_w0 + alpha_l2
            c_scale = max(float(g.c_scale), 1.0)
            n_orig  = int(g.n_orig.item()) if hasattr(g, "n_orig") else 10
            depth_i = int(round(g.context[0].item() * n_orig))

            if args.depth_w0 > 1.0 and depth_i == 0:
                w = args.depth_w0
            elif args.depth_alpha != 0.0:
                w = (depth_i + 1) ** args.depth_alpha
            else:
                w = 1.0

            l1_raw = lagrangian_v1ld(lam, g)
            epoch_l1_train += l1_raw.detach().item()
            loss = (l1_raw / c_scale
                    + args.alpha_l2 * (lam ** 2).mean()) * w
            loss.backward()

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            epoch_gnorm      = max(epoch_gnorm, gn.item())
            epoch_gnorm_sum += gn.item()

            if accum_grads is None:
                accum_grads = [p.grad.detach().clone() if p.grad is not None else None
                               for p in model.parameters()]
            else:
                for j, p in enumerate(model.parameters()):
                    if p.grad is not None and accum_grads[j] is not None:
                        accum_grads[j] += p.grad.detach()

            lv = loss.item()
            epoch_loss    += lv
            epoch_loss_sq += lv * lv
            accum_count   += 1
            is_last = (step_i == n_train - 1)

            if (step_i + 1) % args.batch_size == 0 or is_last:
                optimizer.zero_grad()
                for p, acc in zip(model.parameters(), accum_grads):
                    if acc is not None:
                        p.grad = acc / accum_count
                epoch_nsteps += 1
                optimizer.step()
                accum_grads = None
                accum_count = 0

        scheduler.step()
        mean_loss     = epoch_loss / n_train
        loss_std      = (max(epoch_loss_sq / n_train - mean_loss ** 2, 0.0)) ** 0.5
        l1_train      = epoch_l1_train / n_train
        lam_mean_ep   = lam_abs_sum / max(n_train, 1)
        t2_mean_ep    = t2_sum / max(n_train, 1)
        gnorm_mean_ep = epoch_gnorm_sum / max(epoch_nsteps, 1)
        ep_total      = time.perf_counter() - t_ep

        history["epochs_loss"].append(epoch)
        history["loss"].append(mean_loss)
        history["loss_std"].append(loss_std)
        history["l1_train"].append(l1_train)
        history["lam_abs_mean"].append(lam_mean_ep)
        history["lam_abs_max"].append(lam_abs_max_ep)
        history["t2_mean"].append(t2_mean_ep)
        history["grad_norm_max"].append(epoch_gnorm)
        history["grad_norm_mean"].append(gnorm_mean_ep)

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.n_epochs)
        eval_str = ""
        if do_eval:
            l1_eval, t2_eval, pct_vs_sg = eval_epoch(
                eval_graphs, l1_sg_eval, model, device, args.center_lam)
            l1_by_n = eval_by_n(eval_graphs, l1_sg_eval, model, device, args.center_lam)
            loss_eval = eval_loss_epoch(eval_graphs, model, device, args.center_lam)
            history["epochs_eval"].append(epoch)
            history["l1_eval"].append(l1_eval)
            history["loss_eval"].append(loss_eval)
            history["t2_eval"].append(t2_eval)
            history["pct_better"].append(pct_vs_sg)
            history["l1_by_n"].append(l1_by_n)
            eval_str = (f"  eval: L1={l1_eval:.4f}  loss_eval={loss_eval:.4f}"
                        f"  pct_vs_sg={pct_vs_sg:.1f}%  T2_eval={t2_eval:.4f}")

        elapsed   = time.time() - t_start
        done      = epoch - start_epoch + 1
        remaining = elapsed / done * (args.n_epochs - epoch) if done > 0 else 0
        log(f"ep {epoch:4d}/{args.n_epochs}  loss={mean_loss:.4f}  L1_train={l1_train:.2f}"
            f"  |λ|={lam_mean_ep:.3f}(max={lam_abs_max_ep:.2f})"
            f"  T2={t2_mean_ep:.4f}  gnorm={epoch_gnorm:.3f}"
            f"  lr={optimizer.param_groups[0]['lr']:.1e}"
            f"{eval_str}  t={ep_total:.0f}s  reste~{remaining/60:.0f}min")

        save_checkpoint(out_dir, run_name, epoch,
                        model, optimizer, scheduler,
                        history, args, saved_ckpts, args.keep_ckpts)
        if do_eval:
            save_history(metrics_path, {
                "run_name": run_name, "args": vars(args),
                "n_train": n_train, "n_test": len(test_ds),
                "n_eval_root": len(eval_graphs), "history": history,
            })

    # ── 11. Résumé & figure ────────────────────────────────────────────────────
    total_min = (time.time() - t_start) / 60
    log(f"Entraînement terminé en {total_min:.1f} min")
    log(f"Métriques : {metrics_path}")

    epochs_loss = history["epochs_loss"]
    epochs_eval = history["epochs_eval"]
    if len(epochs_loss) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].plot(epochs_loss, history["loss"], color="steelblue", lw=1.5)
        axes[0].fill_between(epochs_loss,
                             [l - s for l, s in zip(history["loss"], history["loss_std"])],
                             [l + s for l, s in zip(history["loss"], history["loss_std"])],
                             color="steelblue", alpha=0.15)
        axes[0].set(xlabel="Epoch", ylabel="Loss (L1/c_scale)", title="Loss train")
        axes[0].grid(True, alpha=0.3)
        if epochs_eval:
            axes[1].plot(epochs_eval, history["l1_eval"], color="tomato", lw=1.5)
            axes[1].set(xlabel="Epoch", ylabel="L1(λ_GNN)", title="L1 eval (racines test)")
            axes[1].grid(True, alpha=0.3)
            axes[2].plot(epochs_eval, history["pct_better"], color="mediumseagreen", lw=1.5)
            axes[2].axhline(50, color="gray", ls=":", lw=1)
            axes[2].set(xlabel="Epoch", ylabel="pct_better (%)",
                        title="% eval : L1_GNN < L1(λ=0)")
            axes[2].grid(True, alpha=0.3)
        plt.suptitle(f"GNN V1LD scaled — {run_name}", fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log(f"Figure : {fig_path}")

    # ── 12. Figure qualité GNN vs sous-gradient ───────────────────────────────
    if not args.skip_quality:
        _N_COLORS = {10: "#1f77b4", 15: "#ff7f0e", 20: "#2ca02c"}
        quality_eligible = [
            (g, int(g.n_orig.item()) if hasattr(g, "n_orig") else 0)
            for g in eval_graphs
            if (int(g.n_orig.item()) if hasattr(g, "n_orig") else 0) <= args.quality_max_n
        ]
        if quality_eligible:
            log(f"[quality] verify_v1ld (T={args.quality_T}) "
                f"sur {len(quality_eligible)} graphes eval (n≤{args.quality_max_n}) ...")
            t0 = time.time()
            q_subgrad, q_gnn, q_n = [], [], []
            model.eval()
            dev = next(model.parameters()).device
            for j, (g, n_val) in enumerate(quality_eligible):
                g_cpu = g.cpu()
                sg = verify_v1ld(g_cpu, T=args.quality_T)
                g_dev = g.to(dev)
                with torch.no_grad():
                    lam_q = model(g_dev)
                    if args.center_lam:
                        lam_q = _apply_center(lam_q, int(g_dev.N_z), int(g_dev.n_g))
                    gnn_l1 = lagrangian_v1ld(lam_q, g_dev).item()
                q_subgrad.append(sg)
                q_gnn.append(gnn_l1)
                q_n.append(n_val)
                if (j + 1) % 50 == 0 or j == len(quality_eligible) - 1:
                    log(f"  [{j+1}/{len(quality_eligible)}]  t={time.time()-t0:.0f}s")
            log(f"[quality] Terminé en {time.time()-t0:.0f}s")

            diffs = [gnn - sg for sg, gnn in zip(q_subgrad, q_gnn)]
            log(f"[quality] GNN-subgrad: mean={np.mean(diffs):.4f}  "
                f"max={np.max(diffs):.4f}  "
                f"pct_better={100*np.mean(np.array(diffs)<0):.1f}%")

            fig_q_path = out_dir / f"fig_quality_{run_name}.png"
            fig, ax = plt.subplots(figsize=(7, 6))
            for g_i, (sg, gnn, nv) in enumerate(zip(q_subgrad, q_gnn, q_n)):
                ax.scatter(sg, gnn,
                           color=_N_COLORS.get(nv, "gray"), alpha=0.7, s=25, zorder=3)
            all_v = q_subgrad + q_gnn
            mn, mx = min(all_v), max(all_v)
            ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, label="y = x (identique)")
            for nv, color in sorted(_N_COLORS.items()):
                if any(n == nv for n in q_n):
                    ax.scatter([], [], color=color, s=40, label=f"n={nv}")
            ax.set_xlabel(f"L1 sous-gradient (T={args.quality_T})")
            ax.set_ylabel("L1 GNN (final)")
            ax.set_title(
                f"Qualité GNN vs sous-gradient — {run_name}\n"
                f"GNN−subgrad : mean={np.mean(diffs):.4f}  "
                f"pct_better={100*np.mean(np.array(diffs)<0):.1f}%"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_q_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            log(f"Figure qualité : {fig_q_path}")
        else:
            log("[quality] Aucun graphe éligible — augmenter --quality_max_n")


if __name__ == "__main__":
    main()
