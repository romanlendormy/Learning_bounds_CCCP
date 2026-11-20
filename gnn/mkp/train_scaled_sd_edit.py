#!/usr/bin/env python3
"""
train_scaled_sd_edit.py — Entraînement du GNN SLD (graphe sd_edit) sur ~10 000
graphes diversifiés (n=10/20, N=20/50/100, data/train_base_s10).

Modèle, borne et vérification : train_SLD_sd_edit.py. Ce script gère le
chargement des données, la sélection stratifiée, le split train/test par
instance, le warmup supervisé sur les racines, la boucle d'entraînement,
l'évaluation et la reprise depuis le dernier checkpoint.

Usage :
    python gnn/mkp/train_scaled_sd_edit.py --data_dir data --out_dir <dossier>
    sbatch gnn/mkp/run_train_sld.sh
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
REPO_ROOT  = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gnn.mkp.train_SLD_sd_edit import (
    load_instances, LazyGraphDataset, EagerGraphDataset,
    GNN, lagrangian_sld, verify_sld,
)
from torch_geometric.loader import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entraîne GNN SLD sd_edit sur ~10k graphes stratifiés (n=10/20, N variés).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Données ───────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",    default=None,
                   help="Racine contenant train_base_s10/, train_N50/, etc. Défaut: <repo>/data/")
    p.add_argument("--out_dir",     default=None,
                   help="Dossier de sortie. Défaut: <repo>/gnn/mkp/runs/")
    p.add_argument("--n_list",      type=int, nargs="+", default=[10, 20],
                   help="Valeurs de n à charger.")
    p.add_argument("--n_scens",     type=int, nargs="+", default=[20, 50, 100],
                   help="Valeurs de N à charger, appliquées à TOUS les n de "
                        "--n_list (cf. discover_pairs — avant, seul n=10 "
                        "balayait plusieurs N, n!=10 était figé à N=500).")
    p.add_argument("--m",           type=int, default=5)
    p.add_argument("--target_graphs", type=int, default=10_000,
                   help="Nombre total de graphes dans le dataset sélectionné.")
    p.add_argument("--root_frac",   type=float, default=0.05,
                   help="Fraction du budget par combinaison (n,N) réservée aux racines "
                        "(depth=0). Avant, TOUTES les racines étaient gardées "
                        "inconditionnellement (jusqu'à 50k+ sur les données actuelles, "
                        "5x le target_graphs par défaut) — plafonné ici à une fraction "
                        "délibérée, cohérente avec la fréquence réelle d'appel de la "
                        "racine en B&B (une fois par instance).")
    p.add_argument("--reservoir_k", type=int, default=2,
                   help="Nb max de nœuds non-racine gardés PAR INSTANCE, "
                        "échantillonnés par reservoir sampling sur un seul "
                        "passage du fichier complet (jamais tronqué) — "
                        "chaque instance a une chance égale d'être vue, "
                        "contrairement à une troncature par nb de lignes "
                        "(--max_load avant) qui ne voyait jamais la fin du "
                        "fichier pour les gros fichiers (n=20 : ~200 nœuds/"
                        "instance en moyenne, un fichier de 2M lignes tronqué "
                        "à 60k ne couvrait que les ~300 premières instances "
                        "sur 9927+ disponibles).")
    p.add_argument("--train_frac",  type=float, default=0.8)
    p.add_argument("--seed",        type=int, default=42)

    # ── Modèle ────────────────────────────────────────────────────────────────
    p.add_argument("--hidden",         type=int,   default=128)
    p.add_argument("--use_max_pool",   action="store_true")

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
    p.add_argument("--warmup_max_n",type=int,   default=20,
                   help="n max pour le warmup — 20 par défaut pour couvrir n=10 ET "
                        "n=20 (le B&B C++ rend n=20 assez rapide, contrairement à "
                        "l'ancien B&B Python).")

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
    n_scens: list[int],
    m: int,
) -> list[tuple[Path, Path, int, int]]:
    """
    Retourne [(inst_file, nodes_file, n_val, N_val)] pour chaque combinaison
    (n, N) de n_list × n_scens disponible sur disque. Deux layouts supportés,
    essayés dans cet ordre :

      1. train_base_s10/instances_n{n}_m{m}_N{N}.txt — nouveau layout (un seul
         dossier, N dans le nom de fichier), utilisé pour n=10/20 × N=20/50/100.
      2. train_N{N}/instances_n{n}_m{m}.txt (train_base_s30/ si N=500) —
         ancien layout, un dossier par N.

    Contrairement à avant, la même liste n_scens s'applique à TOUS les n de
    n_list (avant, seul n=10 balayait plusieurs N ; n!=10 était figé à
    N=500) — nécessaire pour scaler n=10 ET n=20 sur N=20/50/100.
    """
    pairs = []
    for n in sorted(n_list):
        for N in sorted(n_scens):
            s10_inst  = data_root / "train_base_s10" / f"instances_n{n}_m{m}_N{N}.txt"
            s10_nodes = data_root / "train_base_s10" / f"nodes_n{n}_m{m}_N{N}.txt"
            if s10_inst.exists() and s10_nodes.exists():
                pairs.append((s10_inst, s10_nodes, n, N))
                continue

            d = data_root / ("train_base_s30" if N == 500 else f"train_N{N}")
            inst_f  = d / f"instances_n{n}_m{m}.txt"
            nodes_f = d / f"nodes_n{n}_m{m}.txt"
            if inst_f.exists() and nodes_f.exists():
                pairs.append((inst_f, nodes_f, n, N))
            else:
                print(f"  WARN: absent — n={n} N={N} (ni train_base_s10, ni {d.name})")
    return pairs


def _parse_full_node_record(line: str, inst: dict) -> dict | None:
    """Parse complet d'une ligne nodes_*.txt (formats B et C, cf. load_raw_nodes
    dans train_SLD_sd_edit.py — logique dupliquée ici volontairement pour ne
    pas toucher ce module partagé par les autres scripts diag)."""
    vals = [float(x) for x in line.split(",")]
    inst_id   = int(vals[0])
    depth     = int(vals[1])
    curr_cost = vals[2]
    n, N = inst["n"], inst["N"]
    n_fmt_B = 5 + n + N
    n_fmt_C = 3 + n + N

    if len(vals) == n_fmt_C:
        x_mask = np.array(vals[3 : 3 + n], dtype=np.int8)
        z_mask = np.array(vals[3 + n : 3 + n + N], dtype=np.int8)
    elif len(vals) >= n_fmt_B:
        x_mask = np.array(vals[5 : 5 + n], dtype=np.int8)
        z_mask = np.array(vals[5 + n : 5 + n + N], dtype=np.int8)
    else:
        return None

    if (x_mask == 2).sum() == 0 or (z_mask == 2).sum() == 0:
        return None

    return {"inst_id": inst_id, "depth": depth, "curr_cost": curr_cost,
            "x_mask": x_mask, "z_mask": z_mask}


def stream_reservoir_by_instance(
    nodes_file: Path,
    instances: dict,
    reservoir_k: int,
    rng: random.Random,
) -> tuple[list, list]:
    """
    Un seul passage sur nodes_file, jamais tronqué : chaque instance a une
    chance égale d'être représentée, quelle que soit sa position dans le
    fichier (contrairement à une troncature par nb de lignes, cf. --reservoir_k
    dans parse_args).

    - Racines (depth=0) : toutes gardées (une par instance, peu nombreuses).
    - Non-racines : au plus `reservoir_k` par instance, par reservoir
      sampling classique (algo R) — échantillon uniforme parmi TOUTES les
      occurrences de cette instance dans le fichier, pas seulement les
      premières rencontrées.

    Le parsing complet (numpy x_mask/z_mask) n'est fait que pour les lignes
    effectivement retenues par le reservoir ; les lignes rejetées ne coûtent
    que deux int() (inst_id, depth) — nécessaire pour rester praticable sur
    des fichiers de plusieurs millions de lignes.
    """
    roots: list = []
    reservoirs: dict[int, list] = {}
    seen_non_root: dict[int, int] = defaultdict(int)
    skipped = 0

    with open(nodes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c1 = line.index(",")
            c2 = line.index(",", c1 + 1)
            inst_id = int(line[:c1])
            depth   = int(line[c1 + 1 : c2])

            inst = instances.get(inst_id)
            if inst is None:
                skipped += 1
                continue

            if depth == 0:
                rec = _parse_full_node_record(line, inst)
                if rec is not None:
                    roots.append(rec)
                else:
                    skipped += 1
                continue

            seen_non_root[inst_id] += 1
            c   = seen_non_root[inst_id]
            cur = reservoirs.get(inst_id)
            if cur is None or len(cur) < reservoir_k:
                # Encore en phase de remplissage — un enregistrement invalide
                # (rec=None, ex. nœud sans item/scénario libre) ne compte pas
                # comme un slot rempli, on continue à essayer sur les lignes
                # suivantes de cette instance plutôt que de planter sur un
                # index qui n'existe pas encore.
                rec = _parse_full_node_record(line, inst)
                if rec is not None:
                    reservoirs.setdefault(inst_id, []).append(rec)
            else:
                # Réservoir plein (taille reservoir_k) : remplacement classique
                # (algo R). `c` inclut les lignes invalides déjà vues — légère
                # approximation si elles sont fréquentes, négligeable sinon.
                j = rng.randint(1, c)
                if j <= reservoir_k:
                    rec = _parse_full_node_record(line, inst)
                    if rec is not None:
                        reservoirs[inst_id][j - 1] = rec

    non_root = [r for lst in reservoirs.values() for r in lst]
    return roots, non_root


def load_and_merge(
    pairs: list[tuple[Path, Path, int, int]],
    reservoir_k: int,
    seed: int,
) -> tuple[dict, list]:
    """
    Charge les (inst_file, nodes_file) de toutes les paires par reservoir
    sampling (stream_reservoir_by_instance, un seul passage par fichier —
    voit TOUTES les instances, pas seulement un préfixe du fichier),
    renumérote les inst_ids pour éviter les collisions, annote chaque record
    avec n_orig et N_orig.
    """
    rng = random.Random(seed)
    merged_instances: dict = {}
    merged_records:   list = []

    for inst_f, nodes_f, n_val, N_val in pairs:
        print(f"  n={n_val} N={N_val}: chargement (reservoir k={reservoir_k}) ...",
              end=" ", flush=True)
        instances = load_instances(inst_f)
        roots, non_root = stream_reservoir_by_instance(nodes_f, instances, reservoir_k, rng)
        records = roots + non_root

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
        n_inst_seen = len({r["inst_id"] for r in records})
        print(f"{len(roots)} racines + {len(non_root)} non-racines "
              f"({n_inst_seen} instances vues sur {len(instances)} dispo)  "
              f"depths:{dict(sorted(depth_cnt.items()))}")

    return merged_instances, merged_records


# ─────────────────────────────────────────────────────────────────────────────
# Sélection stratifiée
# ─────────────────────────────────────────────────────────────────────────────

def select_stratified(
    all_records: list,
    target_graphs: int,
    root_frac: float,
    seed: int,
) -> list:
    """
    Budget réparti à PARTS ÉGALES entre les combinaisons (n_orig, N_orig)
    présentes dans all_records — pas proportionnellement à leur volume brut,
    qui reflète un budget de collecte (combien d'instances on a eu le temps
    de résoudre), pas une importance relative entre (n, N).

    Dans chaque combinaison :
      - racines (depth=0) plafonnées à `root_frac * budget_combo` — avant,
        TOUTES les racines étaient gardées inconditionnellement, ce qui
        représente ~50k+ racines sur les données actuelles (5x un
        target_graphs=10k), écrasant silencieusement le budget demandé et
        biaisant massivement vers un état jamais revisité en B&B réel (la
        racine n'est appelée qu'une fois par instance, contre des milliers
        de fois pour les nœuds profonds) ;
      - non-racines échantillonnées par round-robin PAR INSTANCE (pas par
        depth comme avant) : un enregistrement aléatoire par instance et par
        tour, en priorisant les instances pas encore utilisées pour une
        racine. Ça (a) préserve la distribution de depth telle que collectée
        (chaque instance apporte son propre mélange, agrégé ça approxime la
        fréquence de visite d'un vrai B&B, contrairement à un quota uniforme
        par depth) et (b) maximise le nombre d'instances distinctes
        représentées — pas de "racine + tous ses sous-nœuds".
    """
    rng = random.Random(seed)

    by_combo: dict = defaultdict(list)
    for r in all_records:
        by_combo[(r.get("n_orig", 0), r.get("N_orig", 0))].append(r)

    combos = sorted(by_combo.keys())
    n_combos = len(combos) or 1
    per_combo_budget = target_graphs // n_combos

    selected: list = []
    for combo in combos:
        recs      = by_combo[combo]
        roots     = [r for r in recs if r["depth"] == 0]
        non_roots = [r for r in recs if r["depth"] >  0]
        rng.shuffle(roots)

        root_budget    = min(len(roots), round(per_combo_budget * root_frac))
        nonroot_budget = max(0, per_combo_budget - root_budget)

        chosen_roots  = roots[:root_budget]
        used_inst_ids = {r["inst_id"] for r in chosen_roots}

        by_inst: dict = defaultdict(list)
        for r in non_roots:
            by_inst[r["inst_id"]].append(r)
        for lst in by_inst.values():
            rng.shuffle(lst)

        # Instances pas encore utilisées par une racine en premier, pour
        # maximiser le nombre total d'instances distinctes de ce combo.
        inst_ids = list(by_inst.keys())
        rng.shuffle(inst_ids)
        inst_ids.sort(key=lambda i: i in used_inst_ids)

        idx_per_inst: dict = {i: 0 for i in inst_ids}
        chosen_nonroot: list = []
        while len(chosen_nonroot) < nonroot_budget:
            progress = False
            for i in inst_ids:
                lst = by_inst[i]
                pos = idx_per_inst[i]
                if pos < len(lst) and len(chosen_nonroot) < nonroot_budget:
                    chosen_nonroot.append(lst[pos])
                    idx_per_inst[i] = pos + 1
                    progress = True
            if not progress:
                break

        selected.extend(chosen_roots)
        selected.extend(chosen_nonroot)

        combo_inst_ids = used_inst_ids | {r["inst_id"] for r in chosen_nonroot}
        print(f"  combo n={combo[0]:<3} N={combo[1]:<4} budget={per_combo_budget:6d}  "
              f"racines={len(chosen_roots):5d}/{len(roots):<6d}  "
              f"non-racines={len(chosen_nonroot):6d}/{len(non_roots):<8d}  "
              f"instances_distinctes={len(combo_inst_ids)}")

    # Stats globales
    by_depth: dict = defaultdict(int)
    by_n:     dict = defaultdict(int)
    by_N:     dict = defaultdict(int)
    all_inst_ids: set = set()
    for r in selected:
        by_depth[r["depth"]] += 1
        by_n[r.get("n_orig", 0)] += 1
        by_N[r.get("N_orig", 0)] += 1
        all_inst_ids.add(r["inst_id"])
    print(f"[select] {len(selected)} graphes au total, {len(all_inst_ids)} instances distinctes")
    print(f"  par depth : {dict(sorted(by_depth.items()))}")
    print(f"  par n     : {dict(sorted(by_n.items()))}")
    print(f"  par N     : {dict(sorted(by_N.items()))}")
    return selected


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
        _, lam_star = verify_sld(g, T=args.warmup_T, return_lam=True)
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
                l1_vals = [lagrangian_sld(model(g), g).item() / max(float(g.c_scale), 1.0)
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
        return np.array([lagrangian_sld(torch.zeros(g.N_z * g.n_g, device=device), g).item()
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
            l1_gnn = lagrangian_sld(lam, g).item()
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
            total += lagrangian_sld(lam, g).item() / c_scale
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
            l1_gnn = lagrangian_sld(lam, g).item()
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
            l1_gnn = lagrangian_sld(lam, g).item()
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
    out_dir   = Path(args.out_dir)  if args.out_dir  else REPO_ROOT / "gnn" / "mkp" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    # ── 1. Découverte des fichiers ─────────────────────────────────────────────
    log("Découverte des données ...")
    pairs = discover_pairs(data_root, args.n_list, args.n_scens, args.m)
    if not pairs:
        log("ERREUR : aucun fichier trouvé — vérifiez --data_dir et --n_list")
        sys.exit(1)
    for inst_f, nodes_f, n_val, N_val in pairs:
        log(f"  n={n_val}  N={N_val}  {nodes_f.name}")

    # ── 2. Chargement & fusion ─────────────────────────────────────────────────
    log("Chargement ...")
    instances, all_records = load_and_merge(pairs, args.reservoir_k, args.seed)
    log(f"  {len(instances)} instances  {len(all_records)} records bruts")

    # ── 3. Sélection stratifiée ────────────────────────────────────────────────
    selected = select_stratified(all_records, args.target_graphs, args.root_frac, args.seed)
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

    # ── 5. Ensemble d'évaluation : même méthode que le train, ou racines uniquement ──
    # select_stratified appliqué au pool test (déjà séparé par instance, donc
    # aucune fuite) — garantit que train et eval suivent EXACTEMENT la même
    # méthodologie (budget égal par combo n×N, plafond racine, diversité par
    # instance), sinon un écart train/eval peut refléter une différence de
    # distribution plutôt qu'une vraie différence de généralisation.
    if args.eval_mixed:
        eval_recs = select_stratified(test_ds.raw_records, args.eval_sample, args.root_frac, args.seed)
        eval_desc = f"{len(eval_recs)} graphes test (même méthode que train)"
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
    l1_sg_eval = np.array([verify_sld(g.cpu(), T=args.eval_sg_T) for g in eval_graphs])
    for g in eval_graphs:
        g.to(device)
    log(f"  fait en {time.time()-t0_sg:.1f}s  mean L1_sg(T={args.eval_sg_T}) = {np.mean(l1_sg_eval):.2f}")

    # ── 6. Modèle ──────────────────────────────────────────────────────────────
    model = GNN(m=args.m, hidden=args.hidden,
                       use_max_pool=args.use_max_pool).to(device)
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
        f"use_max_pool={args.use_max_pool}  "
        f"scheduler={sched_desc}  depth_w0={args.depth_w0}")

    # ── 7. Run name & auto-resume ──────────────────────────────────────────────
    # run_name NE contient PAS n_epochs : c'est l'identité de la session
    # d'entraînement (utilisée pour retrouver les checkpoints), pas la cible
    # d'epochs. Avant, ep{args.n_epochs} en faisait partie — reprendre un
    # entraînement en visant PLUS d'epochs (ex. 100 → 150) changeait le nom,
    # find_latest_checkpoint ne retrouvait plus les checkpoints existants et
    # repartait de zéro SILENCIEUSEMENT (pas d'erreur, juste un mauvais départ).
    tag = f"sdedit_scaled_n{'_'.join(str(n) for n in sorted(args.n_list))}"
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    run_name = (f"{tag}_lr{args.lr}_bs{args.batch_size}"
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
        start_epoch = ckpt["epoch"] + 1
        # `scheduler` a déjà été (re)construit juste au-dessus avec le
        # --n_epochs COURANT (pas restauré depuis le checkpoint) : si on
        # reprend pour viser plus d'epochs qu'à l'origine (ex. 100 → 150),
        # charger l'état sauvegardé recollerait sur l'ancien T_max et ferait
        # remonter le LR au lieu de continuer sa décroissance. On "rattrape"
        # plutôt le scheduler frais en rejouant ses pas jusqu'à start_epoch-1
        # — la courbe suit alors la NOUVELLE cible du début à la fin.
        for _ in range(start_epoch - 1):
            scheduler.step()
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

            l1_raw = lagrangian_sld(lam, g)
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
            by_n_str = "  ".join(
                f"n={n_key}: L1={v['l1_mean']:.2f} pct_vs_sg={v['pct_better']:.1f}% (N={v['count']})"
                for n_key, v in l1_by_n.items())
            log(f"  [généralisation par n] {by_n_str}")

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
        plt.suptitle(f"GNN SLD sd_edit scaled — {run_name}", fontsize=10)
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
            log(f"[quality] verify_sld (T={args.quality_T}) "
                f"sur {len(quality_eligible)} graphes eval (n≤{args.quality_max_n}) ...")
            t0 = time.time()
            q_subgrad, q_gnn, q_n = [], [], []
            model.eval()
            dev = next(model.parameters()).device
            for j, (g, n_val) in enumerate(quality_eligible):
                g_cpu = g.cpu()
                sg = verify_sld(g_cpu, T=args.quality_T)
                g_dev = g.to(dev)
                with torch.no_grad():
                    lam_q = model(g_dev)
                    if args.center_lam:
                        lam_q = _apply_center(lam_q, int(g_dev.N_z), int(g_dev.n_g))
                    gnn_l1 = lagrangian_sld(lam_q, g_dev).item()
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
                f"Qualité GNN vs sous-gradient (sd_edit) — {run_name}\n"
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
