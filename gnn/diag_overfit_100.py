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
import re
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
    load_raw_nodes, LazyGraphDataset,
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
                   help="Dossier de données fusionnées. Défaut: <repo>/data/train_base_s30/")
    p.add_argument("--stratify_N",  action="store_true", default=False,
                   help="Sélectionner uniformément parmi les valeurs de N présentes dans les données "
                        "(round-robin par depth indépendant par groupe N).")
    p.add_argument("--out_dir",      default=None,
                   help="Dossier de sortie. Défaut: <repo>/gnn/runs/diag_overfit_100/")
    p.add_argument("--n_list",        type=int,   nargs="+", default=[10],
                   help="Valeurs de n à charger (ex: 10 15 20). Défaut : [10] seulement.")
    p.add_argument("--n_graphs",     type=int,   default=100,
                   help="Nombre total de graphes (répartis uniformément entre les n de --n_list).")
    p.add_argument("--n_epochs",     type=int,   default=500)
    p.add_argument("--eval_every",   type=int,   default=10,
                   help="Fréquence d'évaluation par instance (epochs).")
    p.add_argument("--eval_set_every", type=int, default=0,
                   help="Fréquence d'évaluation sur le set d'éval (epochs). "
                        "0 = même fréquence que --eval_every.")
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
    p.add_argument("--max_instances", type=int,  default=0,
                   help="Restreint le nombre d'instances de base distinctes utilisées par "
                        "groupe N (avant le split train/eval et le ré-échantillonnage par "
                        "profondeur B&B). 0 = pas de restriction. Sert à isoler l'effet de la "
                        "diversité d'instances sous-jacentes du nombre de graphes d'entraînement.")
    p.add_argument("--skip_subgrad",  action="store_true",
                   help="Sauter verify_v1ld pour toutes les instances.")
    p.add_argument("--subgrad_max_n", type=int, default=15,
                   help="N max pour verify_v1ld (ignore les instances avec n > valeur). "
                        "Défaut 15 : calcule n=10+15 uniquement (n=20 trop lent en Python B&B).")
    p.add_argument("--subgrad_full", action="store_true",
                   help="Lance verify_v1ld jusqu'à convergence (precision+patience) sur toutes "
                        "les instances et log |λ*|avg + T2_std(λ*) convergé. Lent mais exact.")
    p.add_argument("--subgrad_full_T_max", type=int, default=2000,
                   help="Nombre max d'itérations pour --subgrad_full (défaut 2000).")
    p.add_argument("--subgrad_full_patience", type=int, default=30,
                   help="Patience (--subgrad_full) : arrêt si pas d'amélioration pendant N iter.")
    p.add_argument("--subgrad_full_precision", type=float, default=0.1,
                   help="Seuil d'amélioration minimale (--subgrad_full) pour réinitialiser la patience.")
    p.add_argument("--beta1",         type=float, default=0.9,
                   help="Momentum β₁ d'Adam pour la phase Lagrangienne (défaut 0.9 ; 0.0 = sans momentum).")
    p.add_argument("--alpha_l2",     type=float, default=0.0,
                   help="Coefficient de régularisation L2 sur λ (loss += alpha * mean(λ²)).")
    p.add_argument("--depth_alpha",  type=float, default=0.0,
                   help="Exposant du poids par depth : w = (depth+1)^alpha. "
                        "0 = uniforme ; >0 = favorise les nœuds profonds ; <0 = favorise la racine.")
    p.add_argument("--lam_max",      type=float, default=0.0,
                   help="Borne tanh sur la sortie du GNN : λ ∈ (-lam_max, lam_max). 0 = non borné.")
    p.add_argument("--n_warmup",     type=int,   default=0,
                   help="Epochs de pré-entraînement supervisé (L2 sur λ* sous-gradient). 0 = désactivé.")
    p.add_argument("--warmup_lr",    type=float, default=1e-3,
                   help="Learning rate pour la phase warmup supervisé (défaut plus élevé que lr).")
    p.add_argument("--warmup_max_n", type=int,   default=100,
                   help="N max des graphes utilisés dans le warmup supervisé (défaut: 100). "
                        "Évite la lenteur des grands graphes (N=500 → 1.9M edges). 0 = tous les graphes.")
    p.add_argument("--use_max_pool", action="store_true",
                   help="Pool scénario = [mean | max] au lieu de [mean] seul "
                        "(capte les scénarios les plus contraints).")
    p.add_argument("--lr_T0",    type=int, default=0,
                   help="Période T_0 pour CosineAnnealingWarmRestarts. "
                        "0 = CosineAnnealingLR standard (défaut).")
    p.add_argument("--lr_T_mult", type=int, default=1,
                   help="Facteur multiplicatif des cycles CAWR : T_mult=2 double la durée "
                        "à chaque restart (restarts rares en fin d'entraînement). "
                        "Ignoré si lr_T0=0.")
    p.add_argument("--lr_warmup_ep", type=int, default=0,
                   help="Warmup Lagrangien (epochs) : LR monte linéairement de eta_min à lr, "
                        "puis cosinus jusqu'à eta_min sur le reste. 0 = désactivé. "
                        "Recommandé pour le scaling : pas de restarts, convergence propre.")
    p.add_argument("--no_scheduler", action="store_true",
                   help="LR constant, aucun scheduler.")
    p.add_argument("--center_lam", action="store_true",
                   help="Centrage des λ sur les scénarios après la sortie du GNN : "
                        "λ_ik ← λ_ik − (1/N_z) Σ_i λ_ik. Élimine la dérive de jauge.")
    p.add_argument("--no_c_scale_norm", action="store_true",
                   help="Désactive la normalisation de la loss par c_scale "
                        "(ablation : sans ça, l'échelle de la loss varie avec N).")
    p.add_argument("--save_checkpoint", action="store_true",
                   help="Sauvegarde model_state_dict (checkpoint.pt) et le λ final complet "
                        "par instance (lambdas_final.pt) dans out_dir, pour ré-évaluation "
                        "hors-ligne (ex. raffinement post-hoc via verify_v1ld(lam_init=...)).")
    p.add_argument("--device",        default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Chargement et sélection stratifiée
# ─────────────────────────────────────────────────────────────────────────────

def _round_robin_select(records: list, n_target: int, max_depth: int,
                        rng: random.Random) -> list:
    """Sélection stratifiée round-robin sur les buckets de depth."""
    filtered = [r for r in records if r["depth"] <= max_depth]
    if not filtered:
        filtered = records
    by_depth: dict = defaultdict(list)
    for r in filtered:
        by_depth[r["depth"]].append(r)
    for bucket in by_depth.values():
        rng.shuffle(bucket)
    pools = [list(by_depth[d]) for d in sorted(by_depth)]
    chosen = []
    while len(chosen) < n_target:
        progress = False
        for pool in pools:
            if pool and len(chosen) < n_target:
                chosen.append(pool.pop(0))
                progress = True
        if not progress:
            break
    return chosen


def load_and_select_split(
    data_dir: Path,
    m: int,
    n_per_group_train: int,
    max_depth: int,
    seed: int,
    max_load: int,
    n_list: list,
    eval_frac: float = 0.2,
    stratify_by_N: bool = False,
    max_instances: int = 0,
) -> tuple[dict, list, list]:
    """
    Split déterministe train/eval par inst_id (aucune instance commune).
    eval_frac=0.2 réserve 20% des instances à l'éval.

    stratify_by_N=True : sélectionne n_per_group_train // K records par valeur de N
    distincte présente dans les données (K = nombre de valeurs N détectées), en
    appliquant un round-robin sur les depths indépendamment pour chaque groupe N.

    max_instances > 0 : restreint le pool d'instances de base (avant split train/eval)
    à ce nombre, par groupe N — pour isoler l'effet de la diversité d'instances du
    nombre de graphes d'entraînement (ré-échantillonnés par profondeur B&B).

    Retourne (merged_instances, train_records, eval_records).
    """
    rng = random.Random(seed)
    if n_list is None:
        n_list = [10, 15, 20]
    merged_instances: dict = {}
    train_all: list = []
    eval_all_recs: list = []
    id_offset = 0

    def _depth_counts(recs):
        return {d: sum(1 for r in recs if r["depth"] == d)
                for d in sorted(set(r["depth"] for r in recs))} if recs else {}

    def _load_split_select(inst_f, nodes_f, max_recs, n_val, Nv, n_train, n_eval_target):
        """Load one (inst_f, nodes_f) pair, remap ids, split train/eval, run round-robin."""
        nonlocal id_offset
        instances_s, records_s = load_raw_nodes(inst_f, nodes_f, max_records=max_recs)

        inst_ids_s = sorted(instances_s.keys())
        rng_s = random.Random(seed ^ (n_val * 31337) ^ (Nv * 7919))
        rng_s.shuffle(inst_ids_s)
        if max_instances and max_instances > 0 and len(inst_ids_s) > max_instances:
            print(f"[max_instances] N={Nv}: restreint à {max_instances} instances de base "
                  f"(sur {len(inst_ids_s)} disponibles)", flush=True)
            inst_ids_s = inst_ids_s[:max_instances]
        allowed_ids_s = set(inst_ids_s)
        n_eval_s = max(1, round(len(inst_ids_s) * eval_frac))
        eval_ids_s = set(inst_ids_s[:n_eval_s])

        id_map_s = {old: old + id_offset for old in instances_s}
        for old_id, inst in instances_s.items():
            merged_instances[id_map_s[old_id]] = inst
        id_offset = max(id_map_s.values(), default=id_offset - 1) + 1

        train_s, eval_s = [], []
        for rec in records_s:
            orig_id = rec["inst_id"]
            if orig_id not in allowed_ids_s:
                continue
            new_rec = {**rec, "inst_id": id_map_s[orig_id], "n_orig": n_val}
            (eval_s if orig_id in eval_ids_s else train_s).append(new_rec)
        del records_s

        chosen_t = _round_robin_select(train_s, n_train, max_depth, rng)
        chosen_e = _round_robin_select(eval_s,  n_eval_target, max_depth, rng)
        print(f"train={len(chosen_t)} depths:{_depth_counts(chosen_t)}  "
              f"eval={len(chosen_e)} depths:{_depth_counts(chosen_e)}  "
              f"(inst: {len(inst_ids_s)-n_eval_s} / {n_eval_s})")
        del train_s, eval_s
        return chosen_t, chosen_e

    for n_val in n_list:
        inst_file  = data_dir / f"instances_n{n_val}_m{m}.txt"
        nodes_file = data_dir / f"nodes_n{n_val}_m{m}.txt"
        if not inst_file.exists() or not nodes_file.exists():
            print(f"  WARN: fichiers manquants pour n={n_val}, ignoré")
            continue

        if stratify_by_N:
            # Look for per-N files written by merge_datasets.py
            per_n_insts = sorted(data_dir.glob(f"instances_n{n_val}_m{m}_N*.txt"))
            if per_n_insts:
                # Memory-efficient path: load each N group separately (20k records max)
                # so peak memory per group ≤ ~175 MB instead of ~2 GB for all groups.
                per_n_max_load = 20_000
                n_N = len(per_n_insts)
                n_per_N_train = max(1, n_per_group_train // n_N)
                n_per_N_eval  = max(1, round(n_per_group_train * eval_frac) // n_N)

                for inst_f in per_n_insts:
                    m_match = re.search(r"_N(\d+)\.txt$", inst_f.name)
                    Nv = int(m_match.group(1))
                    nodes_f = data_dir / f"nodes_n{n_val}_m{m}_N{Nv}.txt"
                    if not nodes_f.exists():
                        print(f"  WARN: {nodes_f.name} manquant, ignoré")
                        continue
                    print(f"  n={n_val} N={Nv}: chargement (max {per_n_max_load}) ... ",
                          end="", flush=True)
                    chosen_t, chosen_e = _load_split_select(
                        inst_f, nodes_f, per_n_max_load, n_val, Nv,
                        n_per_N_train, n_per_N_eval)
                    train_all.extend(chosen_t)
                    eval_all_recs.extend(chosen_e)

                print(f"  → total train={len(train_all)}  eval={len(eval_all_recs)}")
                continue  # per-N done; skip combined-file path

            # Fallback: per-N files absent — warn and fall through to combined loading
            print(f"  WARN: fichiers per-N absents pour n={n_val}, "
                  f"chargement combiné (risque OOM si MAX_LOAD élevé)")

        # Combined-file path (stratify_by_N=False, or per-N files absent)
        print(f"  n={n_val}: chargement ...", end=" ", flush=True)
        instances, records = load_raw_nodes(inst_file, nodes_file, max_records=max_load)

        inst_ids = sorted(instances.keys())
        rng_split = random.Random(seed ^ (n_val * 31337))
        rng_split.shuffle(inst_ids)
        n_eval_inst = max(1, round(len(inst_ids) * eval_frac))
        eval_ids = set(inst_ids[:n_eval_inst])

        id_map = {old: old + id_offset for old in instances}
        for old_id, inst in instances.items():
            merged_instances[id_map[old_id]] = inst
        id_offset = max(id_map.values(), default=id_offset - 1) + 1

        train_recs_n: list = []
        eval_recs_n: list = []
        for rec in records:
            orig_id = rec["inst_id"]
            new_rec = {**rec, "inst_id": id_map[orig_id], "n_orig": n_val}
            if orig_id in eval_ids:
                eval_recs_n.append(new_rec)
            else:
                train_recs_n.append(new_rec)
        del records

        if stratify_by_N:
            by_N_train: dict = defaultdict(list)
            for rec in train_recs_n:
                by_N_train[merged_instances[rec["inst_id"]]["N"]].append(rec)
            by_N_eval: dict = defaultdict(list)
            for rec in eval_recs_n:
                by_N_eval[merged_instances[rec["inst_id"]]["N"]].append(rec)

            n_N = max(1, len(by_N_train))
            n_per_N_train = max(1, n_per_group_train // n_N)
            n_per_N_eval  = max(1, round(n_per_group_train * eval_frac) // n_N)

            for N_val in sorted(by_N_train):
                chosen_t = _round_robin_select(by_N_train[N_val], n_per_N_train, max_depth, rng)
                chosen_e = _round_robin_select(by_N_eval.get(N_val, []), n_per_N_eval, max_depth, rng)
                print(f"    N={N_val}: train={len(chosen_t)} depths:{_depth_counts(chosen_t)}  "
                      f"eval={len(chosen_e)} depths:{_depth_counts(chosen_e)}")
                train_all.extend(chosen_t)
                eval_all_recs.extend(chosen_e)
            print(f"  → total train={len(train_all)}  eval={len(eval_all_recs)}  "
                  f"(inst split: {len(inst_ids)-n_eval_inst} train / {n_eval_inst} eval)")
        else:
            n_per_group_eval = max(1, round(n_per_group_train * eval_frac))
            train_chosen = _round_robin_select(train_recs_n, n_per_group_train, max_depth, rng)
            eval_chosen  = _round_robin_select(eval_recs_n,  n_per_group_eval,  max_depth, rng)
            print(f"train={len(train_chosen)} depths:{_depth_counts(train_chosen)}  "
                  f"eval={len(eval_chosen)} depths:{_depth_counts(eval_chosen)}  "
                  f"(inst split: {len(inst_ids)-n_eval_inst} train / {n_eval_inst} eval)")
            train_all.extend(train_chosen)
            eval_all_recs.extend(eval_chosen)

    return merged_instances, train_all, eval_all_recs


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

def _apply_center(lam: torch.Tensor, N_z: int, n_g: int) -> torch.Tensor:
    """λ_ik ← λ_ik − (1/N_z) Σ_i λ_ik  (centrage sur les scénarios, op. linéaire différentiable)."""
    lam_2d = lam.view(N_z, n_g)
    return (lam_2d - lam_2d.mean(dim=0, keepdim=True)).reshape(-1)


def gradient_vector(model: GNN, g, opt_val: float, do_center: bool = False) -> np.ndarray:
    """Gradient de L1(λ)/opt par rapport à tous les paramètres (1 graphe)."""
    model.train()
    model.zero_grad()
    lam  = model(g)
    if do_center:
        lam = _apply_center(lam, int(g.N_z), int(g.n_g))
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
N_SCEN_COLORS = {50: "#9467bd", 100: "#1f77b4", 200: "#ff7f0e", 250: "#d62728", 500: "#2ca02c"}


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

    # ── 1. Chargement (split train/eval par instance) ─────────────────────────
    n_list      = args.n_list
    n_groups    = len(n_list)
    n_per_group = args.n_graphs // n_groups
    eval_set_freq = args.eval_set_every if args.eval_set_every > 0 else args.eval_every
    print(f"[diag] Chargement — {n_per_group} graphes/groupe ×{n_groups}  (n={n_list})"
          f"  eval_frac=20%  eval_set_every={eval_set_freq} :")
    merged_instances, train_recs, eval_recs = load_and_select_split(
        data_dir, args.m, n_per_group, args.max_depth, args.seed, args.max_load,
        n_list=n_list, eval_frac=0.2,
        stratify_by_N=args.stratify_N,
        max_instances=args.max_instances,
    )
    print(f"[diag] {len(train_recs)} train / {len(eval_recs)} eval enregistrements")

    # ── 2. Accès paresseux aux graphes PyG (construction à la volée) ──────────
    # Note : les graphes N=500 ont O(N_z²) ≈ 1,9M arêtes → ~39 MB/graphe, donc
    # un cache éager de ~1000 graphes N=500 (train+eval) coûte ~40-50 GB de
    # RAM. La construction elle-même est bon marché (mesurée ~0.13s/graphe
    # N=500, depth=0) — négligeable face au forward+backward du GNN sur ces
    # mêmes graphes (~6.3s/instance, cf. discussion protocole). Le lazy
    # loading règle donc le problème mémoire à coût quasi nul (~2% overhead).
    print("[diag] Datasets lazy (graphes construits à la volée) ...", flush=True)
    lazy_ds  = LazyGraphDataset(merged_instances, train_recs)
    n_total  = len(train_recs)

    lazy_eval = LazyGraphDataset(merged_instances, eval_recs)
    n_eval    = len(eval_recs)

    def get_graph(gi: int):
        return lazy_ds[gi].to(device)

    def get_eval_graph(ei: int):
        return lazy_eval[ei].to(device)

    # Métadonnées scalaires depuis les records (pas besoin de boucler sur les graphes)
    depths   = [rec["depth"]          for rec in train_recs]
    n_origs  = [rec.get("n_orig", 10) for rec in train_recs]
    N_vals   = [merged_instances[rec["inst_id"]]["N"] for rec in train_recs]
    opt_vals = [float(merged_instances[rec["inst_id"]]["opt"]) for rec in train_recs]

    print(f"[diag] train={n_total} graphes  eval={n_eval} graphes  device={device}")
    for nv in n_list:
        cnt_tr = sum(1 for n in n_origs if n == nv)
        cnt_ev = sum(1 for rec in eval_recs if rec.get("n_orig", 10) == nv)
        print(f"  n={nv}: train={cnt_tr}  eval={cnt_ev}")

    # ── 3. Modèle + optimiseur ────────────────────────────────────────────────
    model     = GNN(m=args.m, hidden=args.hidden, lam_max=args.lam_max,
                    use_max_pool=args.use_max_pool).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0,
                                 betas=(args.beta1, 0.999))
    if args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        sched_desc = "ConstantLR"
    elif args.lr_warmup_ep > 0:
        # Warmup linéaire → cosinus jusqu'à eta_min, sans restarts (recommandé pour le scaling)
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
            optimizer, T_0=args.lr_T0, T_mult=args.lr_T_mult, eta_min=args.eta_min,
        )
        sched_desc = f"CAWR(T0={args.lr_T0},Tmult={args.lr_T_mult})"
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.n_epochs, eta_min=args.eta_min,
        )
        sched_desc = "CosineAnnealingLR"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[diag] GNN : {n_params:,} params  lr={args.lr}  epochs={args.n_epochs}  "
          f"batch={args.batch_size}  max_norm={args.max_norm}  "
          f"use_max_pool={args.use_max_pool}  "
          f"scheduler={sched_desc}  depth_alpha={args.depth_alpha}")

    # ── 4. Historique par instance (train) + agrégat eval ────────────────────
    l1_ref: dict = {}   # L1 de référence sous-gradient (T=50) par graphe (clé=gi), rempli en 5.

    hist_epochs:     list = []
    hist_loss:       dict = {i: [] for i in range(n_total)}
    hist_lam:        dict = {i: [] for i in range(n_total)}
    hist_lam_std:    dict = {i: [] for i in range(n_total)}  # T2 : std(λ) sur scénarios
    hist_lam_gauge:  dict = {i: [] for i in range(n_total)}  # ‖λ̄_·k‖ — direction de jauge
    hist_lam_resid:  dict = {i: [] for i in range(n_total)}  # ‖λ_ik − λ̄_·k‖ — info utile

    hist_eval_epochs: list = []
    hist_eval_loss:   list = []  # L1 moyen sur l'eval set
    hist_eval_lam:    list = []  # |λ|avg moyen sur l'eval set

    # Historique gradient norm (diagnostic de stabilité, comparable entre configs)
    hist_gnorm_mean: list = []
    hist_gnorm_max:  list = []

    # Historique par N (nb de scénarios) — comparaison multi-N entre configs
    N_set    = sorted(set(N_vals))
    hist_by_N: dict = {
        Nv: {"epochs": [], "l1_mean": [], "l1_std": [], "pct_better_T50": []}
        for Nv in N_set
    }
    # Écart L1 max-min entre groupes N à chaque epoch d'éval (déséquilibre multi-N)
    hist_l1_spread:     list = []  # absolu
    hist_l1_spread_rel: list = []  # relatif (/ moyenne globale)

    def eval_all() -> None:
        model.eval()
        with torch.no_grad():
            for gi in range(n_total):
                g   = get_graph(gi)
                lam = model(g)
                if args.center_lam:
                    lam = _apply_center(lam, int(g.N_z), int(g.n_g))
                lam_2d    = lam.view(int(g.N_z), int(g.n_g))
                lam_gauge = lam_2d.mean(dim=0)
                lam_resid = lam_2d - lam_gauge
                hist_loss[gi].append(lagrangian_v1ld(lam, g).item())
                hist_lam[gi].append(lam.abs().mean().item())
                hist_lam_std[gi].append(lam_2d.std(dim=0).mean().item())
                hist_lam_gauge[gi].append(lam_gauge.abs().mean().item())
                hist_lam_resid[gi].append(lam_resid.abs().mean().item())
        model.train()

    def eval_set() -> tuple[float, float]:
        """Évalue le modèle sur les instances d'éval (distinctes du train). Retourne (L1_mean, |λ|avg_mean)."""
        model.eval()
        l1_vals, lam_vals = [], []
        with torch.no_grad():
            for ei in range(n_eval):
                g   = get_eval_graph(ei)
                lam = model(g)
                if args.center_lam:
                    lam = _apply_center(lam, int(g.N_z), int(g.n_g))
                l1_vals.append(lagrangian_v1ld(lam, g).item())
                lam_vals.append(lam.abs().mean().item())
        model.train()
        return float(np.mean(l1_vals)), float(np.mean(lam_vals))

    # ── 5. Référence sous-gradient T=50 ───────────────────────────────────────
    # Calculée systématiquement (indépendamment de n_warmup) : c'est elle qui
    # sert de référence au "(X%)" affiché dans les logs par epoch (L1 [N50=...(X%)]).
    # Réutilisée comme cibles supervisées si le warmup est activé (5b).
    wm_max_n = args.warmup_max_n
    wm_indices_all = [gi for gi in range(n_total)
                      if wm_max_n == 0 or N_vals[gi] <= wm_max_n]
    wm_n_graphs = len(wm_indices_all)
    n_filter_msg = (f"N≤{wm_max_n}" if wm_max_n > 0 else "tous") + f" → {wm_n_graphs} graphes"
    print(f"\n[ref] Génération référence sous-gradient T=50 "
          f"sur {wm_n_graphs} graphes ({n_filter_msg}) ...", flush=True)
    t_wm = time.perf_counter()
    lam_targets = {}
    nz_ng_cache: dict = {}   # (N_z, n_g) par gi — évite de reconstruire le graphe pour t2_star
    for gi in wm_indices_all:
        g = get_graph(gi)
        _, lam_star = verify_v1ld(g.cpu(), T=50, return_lam=True)
        lam_t = torch.tensor(lam_star.astype(np.float32).flatten(),
                             dtype=torch.float32, device=device)
        lam_targets[gi] = lam_t
        l1_ref[gi]      = lagrangian_v1ld(lam_t, g).item()
        nz_ng_cache[gi] = (int(g.N_z), int(g.n_g))
    t2_star = [
        lam_targets[gi].view(*nz_ng_cache[gi]).std(dim=0).mean().item()
        for gi in wm_indices_all
    ]
    print(f"[ref] Cibles T=50 prêtes en {time.perf_counter()-t_wm:.1f}s  "
          f"|λ*| mean={np.mean([lam_targets[gi].abs().mean().item() for gi in wm_indices_all]):.4f}  "
          f"T2_std(λ*)={np.mean(t2_star):.4f}")
    l1_ref_by_N = defaultdict(list)
    for gi in wm_indices_all:
        l1_ref_by_N[N_vals[gi]].append(l1_ref[gi])
    ref_summ = "  ".join(
        f"N{Nv}={np.mean(v):.1f}" for Nv, v in sorted(l1_ref_by_N.items())
    )
    print(f"[ref] L1 référence sous-grad (T=50) : {ref_summ}", flush=True)

    # ── 5b. Warmup supervisé (optionnel — réutilise les cibles T=50 ci-dessus) ─
    if args.n_warmup > 0:
        opt_wm   = torch.optim.Adam(model.parameters(), lr=args.warmup_lr)
        sched_wm = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_wm, T_max=args.n_warmup, eta_min=1e-5)

        print(f"[warmup] {args.n_warmup} epochs supervisés (lr={args.warmup_lr}) ...",
              flush=True)
        t_wm = time.perf_counter()
        wm_indices = list(wm_indices_all)
        for wm_ep in range(1, args.n_warmup + 1):
            random.shuffle(wm_indices)
            mse_sum = 0.0
            for gi in wm_indices:
                g   = get_graph(gi)
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
                model.eval()
                l1_vals = []
                with torch.no_grad():
                    for gi in wm_indices_all:
                        g = get_graph(gi)
                        l1_vals.append(lagrangian_v1ld(model(g), g).item()
                                       / max(float(g.c_scale), 1.0))
                model.train()
                print(f"[warmup] ep {wm_ep:4d}/{args.n_warmup}  "
                      f"mse={mse_sum/wm_n_graphs:.6f}  "
                      f"L1_norm={np.mean(l1_vals):.4f}  "
                      f"t={time.perf_counter()-t_wm:.0f}s", flush=True)
        print(f"[warmup] Terminé en {time.perf_counter()-t_wm:.0f}s\n", flush=True)

    # ── 5b. Convergence complète sous-gradient (optionnel) ────────────────────
    if args.subgrad_full:
        print(f"[subgrad_full] Convergence sous-gradient "
              f"(T_max={args.subgrad_full_T_max}, patience={args.subgrad_full_patience}, "
              f"precision={args.subgrad_full_precision}) "
              f"sur {n_total} instances ...", flush=True)
        t_sf = time.perf_counter()
        sf_l1, sf_lam_avg, sf_lam_max, sf_t2, sf_iters = [], [], [], [], []
        for gi in range(n_total):
            g_cpu = get_graph(gi).cpu()
            l1_sg, n_it, lam_sg = verify_v1ld(
                g_cpu,
                T=args.subgrad_full_T_max,
                precision=args.subgrad_full_precision,
                patience=args.subgrad_full_patience,
                return_lam=True,
                return_n_iters=True,
            )
            lam_2d = lam_sg.reshape(int(g_cpu.N_z), int(g_cpu.n_g))
            abs_lam = np.abs(lam_2d)
            sf_l1.append(l1_sg)
            sf_lam_avg.append(abs_lam.mean())
            sf_lam_max.append(abs_lam.max())
            sf_t2.append(lam_2d.std(axis=0).mean())
            sf_iters.append(n_it)
            if (gi + 1) % 10 == 0 or gi == n_total - 1:
                print(f"  [{gi+1}/{n_total}]  t={time.perf_counter()-t_sf:.0f}s",
                      flush=True)
        sf_lam_max_arr = np.array(sf_lam_max)
        print(
            f"[subgrad_full] Terminé en {time.perf_counter()-t_sf:.0f}s\n"
            f"  L1_sg      : mean={np.mean(sf_l1):.4f}  std={np.std(sf_l1):.4f}\n"
            f"  |λ*|avg    : mean={np.mean(sf_lam_avg):.4f}  std={np.std(sf_lam_avg):.4f}\n"
            f"  |λ*|max    : mean={sf_lam_max_arr.mean():.4f}  p50={np.percentile(sf_lam_max_arr,50):.4f}"
            f"  p95={np.percentile(sf_lam_max_arr,95):.4f}  p99={np.percentile(sf_lam_max_arr,99):.4f}"
            f"  max={sf_lam_max_arr.max():.4f}\n"
            f"  T2_std(λ*) : mean={np.mean(sf_t2):.4f}  std={np.std(sf_t2):.4f}\n"
            f"  iters      : mean={np.mean(sf_iters):.1f}  max={np.max(sf_iters)}",
            flush=True,
        )
    else:
        sf_l1 = sf_lam_avg = sf_t2 = sf_iters = None

    # ── 6. Boucle d'entraînement Lagrangienne ─────────────────────────────────
    t_start  = time.perf_counter()
    indices  = list(range(n_total))

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        random.shuffle(indices)

        epoch_loss  = 0.0
        accum_count = 0
        accum_grads = None  # buffer des gradients clippés par instance
        epoch_gnorm_sum = 0.0
        epoch_gnorm_max = 0.0
        epoch_gnorm_n   = 0

        for step_i, gi in enumerate(indices):
            g       = get_graph(gi)
            optimizer.zero_grad()
            lam  = model(g)
            if args.center_lam:
                lam = _apply_center(lam, int(g.N_z), int(g.n_g))
            w = (depths[gi] + 1) ** args.depth_alpha
            c_norm = 1.0 if args.no_c_scale_norm else max(float(g.c_scale), 1.0)
            loss = (lagrangian_v1ld(lam, g) / c_norm + args.alpha_l2 * (lam ** 2).mean()) * w
            loss.backward()

            # Clip par instance avant accumulation (max_norm<=0 → pas de clip réel,
            # mais on calcule quand même la norme pour le diagnostic de stabilité)
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.max_norm if args.max_norm > 0 else float("inf"))
            epoch_gnorm_sum += gn.item()
            epoch_gnorm_max  = max(epoch_gnorm_max, gn.item())
            epoch_gnorm_n   += 1

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

        do_train_eval = (epoch % args.eval_every == 0)
        do_eval_set   = (epoch % eval_set_freq == 0)

        gnorm_mean_ep = epoch_gnorm_sum / max(epoch_gnorm_n, 1)
        hist_gnorm_mean.append(gnorm_mean_ep)
        hist_gnorm_max.append(epoch_gnorm_max)

        if do_train_eval:
            hist_epochs.append(epoch)
            eval_all()
            elapsed = time.perf_counter() - t_start

            # Résumé par N (nombre de scénarios) — le "(X%)" fait référence à la
            # borne sous-gradient T=50 calculée en 5. (l1_ref), toujours disponible
            # que le warmup soit activé ou non.
            by_N_loss:   dict = defaultdict(list)
            by_N_better: dict = defaultdict(list)
            for gi in range(n_total):
                Nv = N_vals[gi]
                l1_gi = hist_loss[gi][-1]
                by_N_loss[Nv].append(l1_gi)
                if gi in l1_ref:
                    by_N_better[Nv].append(l1_gi < l1_ref[gi])
            summ_parts = []
            by_N_means = {}
            for Nv in sorted(by_N_loss):
                l1_mean = float(np.mean(by_N_loss[Nv]))
                l1_std  = float(np.std(by_N_loss[Nv]))
                by_N_means[Nv] = l1_mean
                hist_by_N[Nv]["epochs"].append(epoch)
                hist_by_N[Nv]["l1_mean"].append(l1_mean)
                hist_by_N[Nv]["l1_std"].append(l1_std)
                if l1_ref and Nv in by_N_better:
                    pct = 100.0 * np.mean(by_N_better[Nv])
                    hist_by_N[Nv]["pct_better_T50"].append(pct)
                    summ_parts.append(f"N{Nv}={l1_mean:.1f}({pct:.0f}%)")
                else:
                    hist_by_N[Nv]["pct_better_T50"].append(None)
                    summ_parts.append(f"N{Nv}={l1_mean:.1f}")
            summ = "  ".join(summ_parts)

            # Déséquilibre entre groupes N (utile pour comparer des configs :
            # un écart qui grandit = le modèle apprend inégalement selon N)
            means_list = list(by_N_means.values())
            spread     = float(max(means_list) - min(means_list)) if len(means_list) > 1 else 0.0
            spread_rel = spread / max(np.mean(means_list), 1e-9) if len(means_list) > 1 else 0.0
            hist_l1_spread.append(spread)
            hist_l1_spread_rel.append(spread_rel)

            lam_mean_ep  = float(np.mean([hist_lam[gi][-1]       for gi in range(n_total)]))
            lam_max_ep   = float(np.max( [hist_lam[gi][-1]       for gi in range(n_total)]))
            lam_std_ep   = float(np.mean([hist_lam_std[gi][-1]   for gi in range(n_total)]))
            gauge_ep     = float(np.mean([hist_lam_gauge[gi][-1] for gi in range(n_total)]))
            resid_ep     = float(np.mean([hist_lam_resid[gi][-1] for gi in range(n_total)]))

            eval_suffix = ""
            if do_eval_set:
                ev_l1, ev_lam = eval_set()
                hist_eval_epochs.append(epoch)
                hist_eval_loss.append(ev_l1)
                hist_eval_lam.append(ev_lam)
                eval_suffix = f"  |eval| L1={ev_l1:.4f} |λ|={ev_lam:.3f}"

            print(f"ep {epoch:4d}/{args.n_epochs}  loss_norm={mean_loss:.4f}  "
                  f"L1 [{summ}]  spread={spread:.1f}({spread_rel:.1%})  "
                  f"|λ|avg={lam_mean_ep:.3f}(max={lam_max_ep:.3f})  "
                  f"T2_std={lam_std_ep:.4f}  gauge={gauge_ep:.4f}  resid={resid_ep:.4f}  "
                  f"gnorm={gnorm_mean_ep:.3f}(max={epoch_gnorm_max:.3f})"
                  f"{eval_suffix}  t={elapsed:.0f}s", flush=True)

        elif do_eval_set:
            ev_l1, ev_lam = eval_set()
            hist_eval_epochs.append(epoch)
            hist_eval_loss.append(ev_l1)
            hist_eval_lam.append(ev_lam)
            elapsed = time.perf_counter() - t_start
            print(f"ep {epoch:4d}/{args.n_epochs}  loss_norm={mean_loss:.4f}  "
                  f"|eval| L1={ev_l1:.4f} |λ|={ev_lam:.3f}  t={elapsed:.0f}s", flush=True)

    print(f"[diag] Entraînement terminé ({time.perf_counter() - t_start:.0f}s)")

    # ── 7. Évaluation finale (+ 7b trackers v10, même passe — évite de
    #      reconstruire deux fois chaque graphe) ───────────────────────────────
    model.eval()
    final_l1            = []
    final_lam_mean       = []
    final_lam_max        = []
    final_lam_by_item    = []
    final_lam_std_scen   = []
    lam_dump: dict = {}   # rempli seulement si --save_checkpoint

    with torch.no_grad():
        for gi in range(n_total):
            g   = get_graph(gi)
            lam = model(g)
            if args.center_lam:
                lam = _apply_center(lam, int(g.N_z), int(g.n_g))
            final_l1.append(lagrangian_v1ld(lam, g).item())
            final_lam_mean.append(lam.abs().mean().item())
            final_lam_max.append(lam.abs().max().item())
            lam_2d = lam.view(g.N_z, g.n_g)                 # (N_z, n_g)
            final_lam_by_item.append(lam_2d.abs().mean(dim=0).cpu().numpy().tolist())
            final_lam_std_scen.append(lam_2d.std(dim=0).cpu().numpy().tolist())
            if args.save_checkpoint:
                # Auto-suffisant : inclut A_g/b_res_g/c_g/epsilon/n_fixed_z1/c_scale
                # pour pouvoir ré-évaluer lagrangian_v1ld/verify_v1ld hors-ligne sans
                # dépendre des fichiers de données d'origine (souvent sur SLURM_TMPDIR,
                # éphémères après la fin du job).
                lam_dump[gi] = {
                    "lam_2d": lam_2d.detach().cpu().numpy().astype("float32"),
                    "N": int(g.N), "N_z": int(g.N_z), "n_g": int(g.n_g),
                    "n_orig": int(n_origs[gi]), "depth": int(depths[gi]),
                    "A_g": g.A_g.cpu().numpy().astype("float32"),
                    "b_res_g": g.b_res_g.cpu().numpy().astype("float32"),
                    "c_g": g.c_g.cpu().numpy().astype("float32"),
                    "c_scale": float(g.c_scale),
                    "epsilon": float(g.epsilon),
                    "n_fixed_z1": int(g.n_fixed_z1),
                    "opt": float(g.opt),
                }

    if args.save_checkpoint:
        torch.save({"model_state_dict": model.state_dict(),
                    "args": vars(args)}, Path(args.out_dir) / "checkpoint.pt")
        torch.save(lam_dump, Path(args.out_dir) / "lambdas_final.pt")
        print(f"[diag] Checkpoint + λ finaux sauvegardés dans {args.out_dir}", flush=True)

    final_subgrad: list = [None] * n_total   # None pour les instances ignorées
    if not args.skip_subgrad:
        sg_todo = [gi for gi in range(n_total) if n_origs[gi] <= args.subgrad_max_n]
        print(f"[diag] Sous-gradients de référence (verify_v1ld, T=100) — "
              f"{len(sg_todo)}/{n_total} instances (n≤{args.subgrad_max_n}) ...", flush=True)
        t_sg = time.perf_counter()
        for j, gi in enumerate(sg_todo):
            g_cpu = get_graph(gi).cpu()   # verify_v1ld utilise numpy (CPU requis)
            final_subgrad[gi] = verify_v1ld(g_cpu, T=100)
            if (j + 1) % 10 == 0 or j == len(sg_todo) - 1:
                print(f"  [{j+1}/{len(sg_todo)}]  t={time.perf_counter()-t_sg:.0f}s",
                      flush=True)

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
        g  = get_graph(gi)
        gv = gradient_vector(model, g, g.c_scale.item(),
                             do_center=args.center_lam)
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

    # 8a. Convergence L1 train + eval (même axes)
    fig, ax = plt.subplots(figsize=(13, 5))
    for gi in range(n_total):
        ax.plot(hist_epochs, hist_loss[gi],
                color=N_COLORS.get(n_origs[gi], "gray"), alpha=0.15, lw=0.6)
    for nv, color in sorted(N_COLORS.items()):
        grp = [gi for gi in range(n_total) if n_origs[gi] == nv]
        if grp:
            mat = np.array([hist_loss[gi] for gi in grp])
            ax.plot(hist_epochs, mat.mean(axis=0), color=color, lw=2.0,
                    label=f"train n={nv} (moy., N={len(grp)})")
    if hist_eval_epochs:
        ax.plot(hist_eval_epochs, hist_eval_loss, "k--", lw=2.0,
                label=f"eval (N={n_eval})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1(λ) brute")
    ax.set_title("Convergence L1 — train vs eval (instances différentes)")
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

    # 8b'. Décomposition λ = jauge + résidu
    fig, (ax_g, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    for gi in range(n_total):
        col = N_COLORS.get(n_origs[gi], "gray")
        ax_g.plot(hist_epochs, hist_lam_gauge[gi], color=col, alpha=0.2, lw=0.7)
        ax_r.plot(hist_epochs, hist_lam_resid[gi], color=col, alpha=0.2, lw=0.7)
    for nv, color in sorted(N_COLORS.items()):
        grp = [gi for gi in range(n_total) if n_origs[gi] == nv]
        if grp:
            mat_g = np.array([hist_lam_gauge[gi] for gi in grp])
            mat_r = np.array([hist_lam_resid[gi] for gi in grp])
            ax_g.plot(hist_epochs, mat_g.mean(axis=0), color=color, lw=2.0,
                      label=f"n={nv} (moy.)")
            ax_r.plot(hist_epochs, mat_r.mean(axis=0), color=color, lw=2.0,
                      label=f"n={nv} (moy.)")
    ax_g.set_xlabel("Epoch")
    ax_g.set_ylabel("|λ̄_·k| (jauge)")
    ax_g.set_title("Composante de jauge  λ̄_·k = mean_i(λ_ik)\n"
                   "(si ↑ = divergence inoffensive pour la borne)")
    ax_g.legend(); ax_g.grid(True, alpha=0.3)
    ax_r.set_xlabel("Epoch")
    ax_r.set_ylabel("|λ_ik − λ̄_·k| (résidu)")
    ax_r.set_title("Résidu  λ_ik − λ̄_·k  — information utile à la borne\n"
                   "(si ↑ = instabilité d'optimisation réelle)")
    ax_r.legend(); ax_r.grid(True, alpha=0.3)
    fig.suptitle("Décomposition λ = jauge + résidu", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_lambda_decomp.png", dpi=150)
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
                       color=N_SCEN_COLORS.get(N_vals[gi], "gray"),
                       alpha=0.7, s=25, zorder=3)
        all_v = [sg for _, sg in sg_computed] + [final_l1[gi] for gi, _ in sg_computed]
        mn, mx = min(all_v), max(all_v)
        ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, label="y = x (identique)")
        for Nv, color in sorted(N_SCEN_COLORS.items()):
            if any(N_vals[gi] == Nv for gi, _ in sg_computed):
                ax.scatter([], [], color=color, s=40, label=f"N={Nv}")
        ax.set_xlabel("L1 sous-gradient (T=100)")
        ax.set_ylabel("L1 GNN (final)")
        ax.set_title("Qualité GNN vs sous-gradient — après overfit 500 epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_quality.png", dpi=150)
        plt.close(fig)

    # ── 10. JSON ──────────────────────────────────────────────────────────────
    ev_l1_final, ev_lam_final = eval_set()
    total_train_time_s = time.perf_counter() - t_start
    metrics = {
        "config":               vars(args),
        "n_graphs_train":       n_total,
        "n_graphs_eval":        n_eval,
        "n_params":             n_params,
        "total_train_time_s":   float(total_train_time_s),
        "hist_epochs":          hist_epochs,
        "eval_set": {
            "epochs":    hist_eval_epochs,
            "l1_mean":   hist_eval_loss,
            "lam_mean":  hist_eval_lam,
            "final_l1":  ev_l1_final,
            "final_lam": ev_lam_final,
        },
        # Historique gradient norm (stabilité de l'optimisation, comparable entre configs)
        "grad_norm_mean_hist": hist_gnorm_mean,
        "grad_norm_max_hist":  hist_gnorm_max,
        # Historique par N (nb de scénarios) — clé = N (str, JSON n'autorise pas les clés int)
        "train_by_N": {
            str(Nv): hist_by_N[Nv] for Nv in N_set
        },
        # Déséquilibre L1 entre groupes N par epoch (0 = tous les N appris également)
        "l1_spread_hist":     hist_l1_spread,
        "l1_spread_rel_hist": hist_l1_spread_rel,
        # L1 de référence sous-gradient T=50 par instance (même référence que les "(%)" des logs)
        "l1_ref_T50": {str(gi): v for gi, v in l1_ref.items()},
        "cosine_mean_off_diag": float(cos_off_diag),
        "cosine_matrix":        cos_mat.tolist(),
        "cosine_labels":        cos_labels,
        "cosine_indices":       cosine_idxs,
        # T3 : cosine par paire de depth
        "cos_by_depth_pair":    cos_depth_mean,
        # Convergence complète sous-gradient (None si --subgrad_full non activé)
        "subgrad_full": None if sf_l1 is None else {
            "l1_mean":         float(np.mean(sf_l1)),
            "l1_std":          float(np.std(sf_l1)),
            "lam_avg_mean":    float(np.mean(sf_lam_avg)),
            "lam_avg_std":     float(np.std(sf_lam_avg)),
            "lam_max_mean":    float(sf_lam_max_arr.mean()),
            "lam_max_p50":     float(np.percentile(sf_lam_max_arr, 50)),
            "lam_max_p95":     float(np.percentile(sf_lam_max_arr, 95)),
            "lam_max_p99":     float(np.percentile(sf_lam_max_arr, 99)),
            "lam_max_max":     float(sf_lam_max_arr.max()),
            "t2_std_mean":     float(np.mean(sf_t2)),
            "t2_std_std":      float(np.std(sf_t2)),
            "iters_mean":      float(np.mean(sf_iters)),
            "iters_max":       int(np.max(sf_iters)),
        },
        "by_instance": [
            {
                "idx":                  gi,
                "n_orig":               n_origs[gi],
                "N":                    N_vals[gi],
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
                # Décomposition jauge/résidu au fil des epochs
                "lam_gauge_hist":       hist_lam_gauge[gi],
                "lam_resid_hist":       hist_lam_resid[gi],
            }
            for gi in range(n_total)
        ],
        # Moyennes globales jauge/résidu par epoch
        "gauge_mean_hist": [
            float(np.mean([hist_lam_gauge[gi][t] for gi in range(n_total)]))
            for t in range(len(hist_epochs))
        ],
        "resid_mean_hist": [
            float(np.mean([hist_lam_resid[gi][t] for gi in range(n_total)]))
            for t in range(len(hist_epochs))
        ],
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── 11. Résumé final ──────────────────────────────────────────────────────
    print(f"\n[diag] ─── Résumé ───────────────────────────────────────")
    by_N_final: dict = defaultdict(list)
    for gi in range(n_total):
        by_N_final[N_vals[gi]].append(final_l1[gi])
    for Nv, vals in sorted(by_N_final.items()):
        print(f"  N={Nv:3d}  L1_final  mean={np.mean(vals):.1f}  "
              f"std={np.std(vals):.1f}  min={np.min(vals):.1f}  max={np.max(vals):.1f}")
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

    # Décomposition jauge / résidu (epoch finale)
    if hist_epochs:
        g_final = float(np.mean([hist_lam_gauge[gi][-1] for gi in range(n_total)]))
        r_final = float(np.mean([hist_lam_resid[gi][-1] for gi in range(n_total)]))
        ratio   = g_final / (r_final + 1e-12)
        print(f"  Décomposition finale  gauge={g_final:.4f}  resid={r_final:.4f}  "
              f"ratio gauge/resid={ratio:.2f}")
        print(f"  → {'JAUGE domine (divergence inoffensive)' if ratio > 2 else 'résidu domine (instabilité réelle)' if ratio < 0.5 else 'équilibré — vérifier les courbes'}")

    if sg_computed:
        by_N_sg: dict = defaultdict(list)
        for gi, sg in sg_computed:
            by_N_sg[N_vals[gi]].append((final_l1[gi], sg))
        print(f"  GNN vs sous-grad T=100 (n≤{args.subgrad_max_n}) — par N :")
        all_diffs = []
        for Nv in sorted(by_N_sg):
            pairs = by_N_sg[Nv]
            diffs_N = [l1 - sg for l1, sg in pairs]
            all_diffs.extend(diffs_N)
            pct = 100.0 * np.mean(np.array(diffs_N) < 0)
            l1_mean = np.mean([l1 for l1, _ in pairs])
            sg_mean = np.mean([sg for _, sg in pairs])
            print(f"    N={Nv:3d}  L1_gnn={l1_mean:.1f}  L1_sg={sg_mean:.1f}  "
                  f"pct_better={pct:.1f}%  ({len(pairs)} inst)")
        pct_all = 100.0 * np.mean(np.array(all_diffs) < 0)
        print(f"    ALL   mean_diff={np.mean(all_diffs):.1f}  pct_better={pct_all:.1f}%")

    print(f"\n  ── Eval set (N={n_eval}, instances distinctes) ──────────────")
    print(f"  L1_eval final : {ev_l1_final:.4f}  |λ|avg : {ev_lam_final:.4f}")
    train_l1_final = float(np.mean(final_l1))
    print(f"  L1_train final: {train_l1_final:.4f}  "
          f"gap eval-train = {ev_l1_final - train_l1_final:+.4f}")

    print(f"\n[diag] Fichiers dans {out_dir}/")
    figs = ["fig_convergence.png", "fig_lambda.png", "fig_cosine.png"]
    if final_subgrad:
        figs.append("fig_quality.png")
    print("  " + "  ".join(figs))
    print("  metrics.json")


if __name__ == "__main__":
    main()
