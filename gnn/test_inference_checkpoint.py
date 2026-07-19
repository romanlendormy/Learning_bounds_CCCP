#!/usr/bin/env python3
"""
test_inference_checkpoint.py — Charge un checkpoint GNN V1LD et vérifie sa cohérence
par inférence sur une ou plusieurs instances test (jamais vues à l'entraînement).

Reproduit exactement le pipeline de données de train_scaled_v1ld.py (mêmes
n_list/n_scens_n10/m/seed/train_frac, lus depuis les args sauvegardés dans le
checkpoint) pour retrouver le split train/test au niveau instance, puis pour
chaque instance sélectionnée :

  1. Charge le modèle depuis le checkpoint (architecture lue dans ckpt["args"]).
  2. Sélectionne --num_instances instance(s) test n / N (par défaut n=10, N=500,
     racine depth=0), ou une instance précise via --inst_id.
  3. Inférence GNN -> λ, calcule L1(λ_GNN) et |λ|_mean.
  4. Sous-gradient depuis λ=0 (via verify_v1ld), deux baselines :
       - convergé avec patience (T_max/precision/patience, comme dans
         diag_overfit_single.py)
       - budget fixe (--subgrad50_T itérations, 50 par défaut)

Si --n n'est pas dans le n_list d'entraînement du checkpoint (ex. n=20 pour un
ckpt entraîné sur n=10 uniquement), le split train/test ne s'applique pas : le
modèle n'a vu AUCUNE instance à ce n, donc toutes les instances disponibles sur
disque pour ce (n, N) sont valides comme test (généralisation pure, pas de
risque de fuite) — elles sont chargées directement, hors pipeline de split.

Avec --num_instances > 1, un tableau récapitulatif par instance est affiché puis
les moyennes sur l'ensemble des instances (L1(λ=0), L1(λ_GNN), L1(sous-gradient
convergé), L1(sous-gradient à budget fixe), % instances où le GNN bat chaque
baseline).

Le --data_dir du checkpoint pointe vers un SLURM_TMPDIR éphémère (les données sont
copiées là pour le training) : ce script retombe sur <repo>/data par défaut, sauf
si --data_dir est passé explicitement.

Usage :
    python gnn/test_inference_checkpoint.py --ckpt <chemin_vers.pt>
    python gnn/test_inference_checkpoint.py --ckpt <chemin_vers.pt> --inst_id 1234
    python gnn/test_inference_checkpoint.py --ckpt <chemin_vers.pt> --n 10 --N 500 --num_instances 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn.train_gnn_v1ld import GNN, LazyGraphDataset, lagrangian_v1ld, verify_v1ld
from gnn.train_scaled_v1ld import (
    discover_pairs, load_and_merge, split_by_instance,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test de cohérence : inférence GNN sur une instance test + comparaison sous-gradient.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="Chemin vers le checkpoint (.pt).")
    p.add_argument("--data_dir", default=None,
                   help="Racine des données (train_base_s30/, train_N50/, ...). "
                        "Défaut : <repo>/data (le data_dir du checkpoint pointe vers "
                        "un SLURM_TMPDIR qui n'existe plus).")
    p.add_argument("--n", type=int, default=10, help="n de l'instance à tester.")
    p.add_argument("--N", type=int, default=500, help="N (nb de scénarios) de l'instance à tester.")
    p.add_argument("--depth", type=int, default=0,
                   help="Profondeur B&B du nœud à tester (0 = racine, l'instance complète).")
    p.add_argument("--inst_id", type=int, default=None,
                   help="Force un inst_id précis (ignore --n/--N/--depth pour la sélection, "
                        "vérifie seulement qu'il est dans le split test).")
    p.add_argument("--pick_index", type=int, default=0,
                   help="Index de départ (dans la liste triée des candidats) des instances test choisies.")
    p.add_argument("--num_instances", type=int, default=1,
                   help="Nombre d'instances test consécutives (à partir de --pick_index) à évaluer.")
    p.add_argument("--subgrad50_T", type=int, default=50,
                   help="Nb d'itérations de la baseline sous-gradient à budget fixe.")
    p.add_argument("--subgrad_conv_T_max", type=int, default=500,
                   help="Nb max d'itérations de la baseline sous-gradient convergé.")
    p.add_argument("--subgrad_conv_precision", type=float, default=0.1,
                   help="Amélioration minimale pour réinitialiser la patience (convergence).")
    p.add_argument("--subgrad_conv_patience", type=int, default=30,
                   help="Nb d'itérations sans amélioration avant arrêt (convergence).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def load_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "args" not in ckpt:
        raise ValueError(f"Checkpoint {ckpt_path} ne contient pas 'args' — "
                          "impossible de reconstruire architecture/split.")
    return ckpt


def build_model(ckpt_args: dict, state_dict: dict, device: torch.device) -> GNN:
    model = GNN(
        m=ckpt_args["m"],
        hidden=ckpt_args["hidden"],
        lam_max=ckpt_args.get("lam_max", 0.0),
        use_max_pool=ckpt_args.get("use_max_pool", False),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _apply_center(lam: torch.Tensor, N_z: int, n_g: int) -> torch.Tensor:
    lam_2d = lam.view(N_z, n_g)
    return (lam_2d - lam_2d.mean(dim=0, keepdim=True)).reshape(-1)


def build_extra_pair(data_dir: Path, n: int, N: int, m: int) -> tuple[Path, Path, int, int]:
    """
    Construit le (inst_file, nodes_file, n, N) pour un n absent du n_list
    d'entraînement du checkpoint (ex. n=20 pour tester la généralisation
    d'un ckpt entraîné sur n=10 uniquement). Même convention de répertoire
    que discover_pairs.src_dir, mais sans la restriction n!=10 -> N=500.
    """
    d = data_dir / ("train_base_s30" if N == 500 else f"train_N{N}")
    inst_f = d / f"instances_n{n}_m{m}.txt"
    nodes_f = d / f"nodes_n{n}_m{m}.txt"
    if not (inst_f.exists() and nodes_f.exists()):
        raise FileNotFoundError(f"Données introuvables pour n={n} N={N} m={m} sous {d}.")
    return inst_f, nodes_f, n, N


def rebuild_split(ckpt_args: dict, data_dir: Path):
    """Reproduit le split train/test instance-level de train_scaled_v1ld.py."""
    pairs = discover_pairs(data_dir, ckpt_args["n_list"], ckpt_args["n_scens_n10"], ckpt_args["m"])
    if not pairs:
        raise FileNotFoundError(
            f"Aucun fichier de données trouvé sous {data_dir} pour "
            f"n_list={ckpt_args['n_list']} n_scens_n10={ckpt_args['n_scens_n10']} m={ckpt_args['m']}."
        )
    instances, all_records = load_and_merge(pairs, ckpt_args["max_load"])
    all_ds = LazyGraphDataset(instances, all_records)
    train_ds, test_ds = split_by_instance(instances, all_ds, ckpt_args["train_frac"], ckpt_args["seed"])
    return instances, train_ds, test_ds


def select_test_records(instances: dict, test_ds: LazyGraphDataset, args: argparse.Namespace) -> list[dict]:
    """
    NB: le tag de répertoire ("N_orig", ex. train_base_s30 -> 500) ne reflète PAS
    toujours le vrai nombre de scénarios de l'instance — train_base_s30 mélange en
    réalité des instances avec N réel in {50, 250, 500, ...}. On filtre donc sur le
    champ réel instances[inst_id]["N"], pas sur le tag de répertoire.

    Retourne une liste d'au plus args.num_instances enregistrements (1 si
    --inst_id est passé), pris à partir de --pick_index dans la liste triée
    des candidats du split TEST.
    """
    if args.inst_id is not None:
        cands = [r for r in test_ds.raw_records
                 if r["inst_id"] == args.inst_id and r["depth"] == args.depth]
        if not cands:
            raise ValueError(
                f"inst_id={args.inst_id} depth={args.depth} absent du split TEST "
                "(soit inconnu, soit dans le split train — pas d'inférence sur une "
                "instance vue à l'entraînement)."
            )
        return [cands[0]]

    cands = [r for r in test_ds.raw_records
             if r["depth"] == args.depth
             and instances[r["inst_id"]]["n"] == args.n
             and instances[r["inst_id"]]["N"] == args.N]
    if not cands:
        raise ValueError(f"Aucune instance test n={args.n} N={args.N} depth={args.depth} trouvée.")
    cands = sorted(cands, key=lambda r: r["inst_id"])
    if args.pick_index >= len(cands):
        raise ValueError(f"pick_index={args.pick_index} hors limites ({len(cands)} candidats).")
    end = args.pick_index + args.num_instances
    selected = cands[args.pick_index:end]
    if len(selected) < args.num_instances:
        print(f"[warn] seulement {len(selected)}/{args.num_instances} instances test "
              f"disponibles pour n={args.n} N={args.N} depth={args.depth}.")
    return selected


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path, device)
    ckpt_args = ckpt["args"]
    print(f"[ckpt] {ckpt_path.name}  epoch={ckpt['epoch']}")
    print(f"[ckpt] arch: m={ckpt_args['m']} hidden={ckpt_args['hidden']} "
          f"use_max_pool={ckpt_args.get('use_max_pool')} "
          f"center_lam={ckpt_args.get('center_lam')}")

    model = build_model(ckpt_args, ckpt["model"], device)

    data_dir = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data"

    if args.n in ckpt_args["n_list"]:
        print(f"[split] reconstruction du split train/test depuis {data_dir} "
              f"(seed={ckpt_args['seed']}, train_frac={ckpt_args['train_frac']}) ...")
        instances, train_ds, test_ds = rebuild_split(ckpt_args, data_dir)
        print(f"[split] instances: train={len(set(r['inst_id'] for r in train_ds.raw_records))}  "
              f"test={len(set(r['inst_id'] for r in test_ds.raw_records))}")
        eval_ds = test_ds
    else:
        print(f"[split] n={args.n} absent du n_list d'entraînement {ckpt_args['n_list']} -> "
              f"généralisation pure (le modèle n'a vu AUCUNE instance à ce n), toutes les "
              f"instances n={args.n} N={args.N} disponibles sur disque sont valides comme test.")
        pair = build_extra_pair(data_dir, args.n, args.N, ckpt_args["m"])
        instances, all_records = load_and_merge([pair], ckpt_args["max_load"])
        eval_ds = LazyGraphDataset(instances, all_records)
        print(f"[split] instances n={args.n} N={args.N} chargées : {len(instances)}")

    records = select_test_records(instances, eval_ds, args)
    if not records:
        raise SystemExit("Aucune instance sélectionnée.")

    results = []
    for k, rec in enumerate(records):
        inst = instances[rec["inst_id"]]
        print(f"\n########## Instance {k + 1}/{len(records)} "
              f"(inst_id={rec['inst_id']}) ##########")
        provenance = "confirmé absent du split train" if args.n in ckpt_args["n_list"] \
            else f"n={args.n} jamais entraîné (généralisation)"
        print(f"[instance] inst_id={rec['inst_id']}  n={inst['n']}  N={inst['N']}  "
              f"depth={rec['depth']}  opt={inst.get('opt', float('nan')):.2f}  ({provenance})")

        g = LazyGraphDataset(instances, [rec])[0].to(device)

        with torch.no_grad():
            lam = model(g)
            if ckpt_args.get("center_lam", False):
                lam = _apply_center(lam, int(g.N_z), int(g.n_g))
            l1_gnn = lagrangian_v1ld(lam, g).item()
            lam_mean = lam.abs().mean().item()
            lam_max = lam.abs().max().item()

        lam0 = torch.zeros(int(g.N_z) * int(g.n_g), device=device)
        l1_zero = lagrangian_v1ld(lam0, g).item()

        print("=== Inférence GNN ===")
        print(f"  L1(λ=0)        = {l1_zero:.4f}")
        print(f"  L1(λ_GNN)      = {l1_gnn:.4f}")
        print(f"  |λ_GNN|_mean   = {lam_mean:.4f}")
        print(f"  |λ_GNN|_max    = {lam_max:.4f}")

        g_cpu = g.cpu()
        sg50_best = verify_v1ld(g_cpu, T=args.subgrad50_T)
        sg_conv_best, sg_conv_iters = verify_v1ld(
            g_cpu, T=args.subgrad_conv_T_max,
            precision=args.subgrad_conv_precision,
            patience=args.subgrad_conv_patience,
            return_n_iters=True,
        )
        gnn_beats_sg50 = l1_gnn <= sg50_best
        gnn_beats_sgconv = l1_gnn <= sg_conv_best

        print(f"=== Sous-gradient (λ=0) vs borne GNN ===")
        print(f"  L1(sous-grad, {args.subgrad50_T} it. fixes)              = {sg50_best:.4f}")
        print(f"  L1(sous-grad convergé, prec={args.subgrad_conv_precision}, "
              f"pat={args.subgrad_conv_patience}, {sg_conv_iters} it.) = {sg_conv_best:.4f}")

        verdict = "GNN meilleur (ou égal) que L1(λ=0)" if l1_gnn <= l1_zero else "GNN moins bon que L1(λ=0) — INCOHÉRENT"
        print(f"  Résumé : {verdict} ; "
              f"{'GNN bat le SG@50' if gnn_beats_sg50 else 'SG@50 bat le GNN'} ; "
              f"{'GNN bat le SG convergé' if gnn_beats_sgconv else 'SG convergé bat le GNN'}")

        results.append({
            "inst_id": rec["inst_id"], "n": inst["n"], "N": inst["N"],
            "l1_zero": l1_zero, "l1_gnn": l1_gnn,
            "lam_mean": lam_mean, "lam_max": lam_max,
            "sg50_best": sg50_best,
            "sg_conv_best": sg_conv_best, "sg_conv_iters": sg_conv_iters,
            "gnn_beats_sg50": gnn_beats_sg50, "gnn_beats_sgconv": gnn_beats_sgconv,
            "coherent": l1_gnn <= l1_zero,
        })

    print(f"\n{'=' * 90}\n=== Tableau récapitulatif ({len(results)} instance(s)) ===\n{'=' * 90}")
    header = (f"{'inst_id':>8}  {'L1(λ=0)':>12}  {'L1(λ_GNN)':>12}  {'SG@50':>12}  "
              f"{'SG_conv':>12}  {'conv_it':>8}  {'GNN<=50':>8}  {'GNN<=conv':>10}")
    print(header)
    for r in results:
        print(f"{r['inst_id']:>8}  {r['l1_zero']:>12.4f}  {r['l1_gnn']:>12.4f}  "
              f"{r['sg50_best']:>12.4f}  {r['sg_conv_best']:>12.4f}  {r['sg_conv_iters']:>8}  "
              f"{str(r['gnn_beats_sg50']):>8}  {str(r['gnn_beats_sgconv']):>10}")

    n_r = len(results)
    mean = lambda key: sum(r[key] for r in results) / n_r
    pct_coherent = 100.0 * sum(r["coherent"] for r in results) / n_r
    pct_beats_sg50 = 100.0 * sum(r["gnn_beats_sg50"] for r in results) / n_r
    pct_beats_sgconv = 100.0 * sum(r["gnn_beats_sgconv"] for r in results) / n_r

    print(f"\n=== Moyennes sur {n_r} instance(s) ===")
    print(f"  L1(λ=0)_mean       = {mean('l1_zero'):.4f}")
    print(f"  L1(λ_GNN)_mean     = {mean('l1_gnn'):.4f}")
    print(f"  |λ_GNN|_mean_mean  = {mean('lam_mean'):.4f}")
    print(f"  |λ_GNN|_max_mean   = {mean('lam_max'):.4f}")
    print(f"  L1(SG@{args.subgrad50_T})_mean          = {mean('sg50_best'):.4f}")
    print(f"  L1(SG convergé)_mean       = {mean('sg_conv_best'):.4f}  "
          f"(it. moyennes = {mean('sg_conv_iters'):.1f})")
    print(f"  % instances GNN <= L1(λ=0) (cohérent, borne resserrée) = {pct_coherent:.1f}%")
    print(f"  % instances où GNN bat SG@{args.subgrad50_T}         = {pct_beats_sg50:.1f}%")
    print(f"  % instances où GNN bat SG convergé      = {pct_beats_sgconv:.1f}%")


if __name__ == "__main__":
    main()
