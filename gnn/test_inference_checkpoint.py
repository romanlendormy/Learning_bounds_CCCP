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
  4. Sous-gradient depuis λ=0, T_max itérations : cherche la première itération t
     où la meilleure borne sous-gradient vue égale ou dépasse L1(λ_GNN) (borne
     considérée "meilleure" si plus basse, comme dans train_scaled_v1ld.eval_epoch).

Avec --num_instances > 1, un tableau récapitulatif par instance est affiché puis
les moyennes sur l'ensemble des instances (L1(λ=0), L1(λ_GNN), |λ|_mean/max,
% instances où le GNN bat ou égale le sous-gradient en T_max itérations).

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

from gnn.train_gnn_v1ld import GNN, LazyGraphDataset, lagrangian_v1ld
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
    p.add_argument("--T_max", type=int, default=500,
                   help="Nombre max d'itérations de sous-gradient pour rattraper la borne GNN.")
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
        use_theta_attn=ckpt_args.get("use_theta_attn", False),
        use_film=ckpt_args.get("use_film", False),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _apply_center(lam: torch.Tensor, N_z: int, n_g: int) -> torch.Tensor:
    lam_2d = lam.view(N_z, n_g)
    return (lam_2d - lam_2d.mean(dim=0, keepdim=True)).reshape(-1)


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


def subgrad_iters_to_match(data, target: float, T_max: int) -> tuple[int | None, float, np.ndarray]:
    """
    Sous-gradient V1LD depuis λ=0 (même algorithme que verify_v1ld dans
    train_gnn_v1ld.py), en suivant la meilleure borne à chaque itération.

    Retourne (n_iters, best_final, trace) :
      n_iters   : première itération (1-indexée) où best <= target, ou None si
                  jamais atteint en T_max itérations.
      best_final: meilleure borne sous-gradient vue sur les T_max itérations.
      trace     : historique de la meilleure borne par itération (pour diagnostic).
    """
    from gnn.train_gnn_v1ld import _USE_CPP_BNB
    if _USE_CPP_BNB:
        from gnn.train_gnn_v1ld import _cpp_solve_zeta_batch
    else:
        from gnn.train_gnn_v1ld import _solve_zeta_multi_bb

    N_z, n_g, N = data.N_z, data.n_g, data.N
    n_fixed_z1 = data.n_fixed_z1
    epsilon = data.epsilon.item()
    A_np = data.A_g.numpy()
    b_np = data.b_res_g.numpy()
    c_np = data.c_g.numpy()

    p_unif = 1.0 / N
    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_z1 * p_unif)

    lam = np.zeros((N_z, n_g), dtype=np.float64)
    best = np.inf
    trace = np.empty(T_max, dtype=np.float64)
    n_iters_hit = None

    for t in range(T_max):
        Lsum = lam.sum(axis=0)
        cost = c_np + lam / p_unif - Lsum

        x_theta = (cost > 0).astype(np.float64)
        theta_vals = np.sum(np.maximum(cost, 0), axis=1)

        if _USE_CPP_BNB:
            x_zeta = _cpp_solve_zeta_batch(
                cost.astype(np.float64), A_np.astype(np.float64), b_np.astype(np.float64)
            ).astype(np.float64)
        else:
            x_zeta = np.zeros_like(x_theta)
            for i in range(N_z):
                _, xz = _solve_zeta_multi_bb(
                    cost[i].astype(np.float32), A_np[i].astype(np.float32), b_np[i].astype(np.float32))
                x_zeta[i] = xz.astype(np.float64)

        zeta_vals = np.sum(cost * x_zeta, axis=1)
        Delta = np.maximum(0.0, theta_vals - zeta_vals)

        order = np.argsort(Delta)
        y_star = np.zeros(N_z)
        cumprob = 0.0
        for idx in order:
            if required_prob <= 0.0:
                break
            y_star[idx] = 1.0
            cumprob += p_unif
            if cumprob >= required_prob - 1e-9:
                break

        L1 = p_unif * theta_vals.sum() - p_unif * (y_star * Delta).sum()
        best = min(best, L1)
        trace[t] = best
        if n_iters_hit is None and best <= target:
            n_iters_hit = t + 1

        x_bar = np.zeros(n_g)
        for i in range(N_z):
            xi = x_zeta[i] if y_star[i] > 0.5 else x_theta[i]
            x_bar += p_unif * xi
        step = 1.0 / np.sqrt(t + 1)
        for i in range(N_z):
            xi = x_zeta[i] if y_star[i] > 0.5 else x_theta[i]
            lam[i] += step * (x_bar - xi)

    return n_iters_hit, best, trace


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path, device)
    ckpt_args = ckpt["args"]
    print(f"[ckpt] {ckpt_path.name}  epoch={ckpt['epoch']}")
    print(f"[ckpt] arch: m={ckpt_args['m']} hidden={ckpt_args['hidden']} "
          f"use_max_pool={ckpt_args.get('use_max_pool')} "
          f"use_theta_attn={ckpt_args.get('use_theta_attn')} "
          f"use_film={ckpt_args.get('use_film')} "
          f"center_lam={ckpt_args.get('center_lam')}")

    model = build_model(ckpt_args, ckpt["model"], device)

    data_dir = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data"
    print(f"[split] reconstruction du split train/test depuis {data_dir} "
          f"(seed={ckpt_args['seed']}, train_frac={ckpt_args['train_frac']}) ...")
    instances, train_ds, test_ds = rebuild_split(ckpt_args, data_dir)
    print(f"[split] instances: train={len(set(r['inst_id'] for r in train_ds.raw_records))}  "
          f"test={len(set(r['inst_id'] for r in test_ds.raw_records))}")

    records = select_test_records(instances, test_ds, args)
    if not records:
        raise SystemExit("Aucune instance sélectionnée.")

    results = []
    for k, rec in enumerate(records):
        inst = instances[rec["inst_id"]]
        print(f"\n########## Instance {k + 1}/{len(records)} "
              f"(inst_id={rec['inst_id']}) ##########")
        print(f"[instance] inst_id={rec['inst_id']}  n={inst['n']}  N={inst['N']}  "
              f"depth={rec['depth']}  opt={inst.get('opt', float('nan')):.2f}  "
              f"(confirmé absent du split train)")

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

        print(f"=== Sous-gradient (λ=0, T_max={args.T_max}) vs borne GNN ===")
        n_iters, sg_best, trace = subgrad_iters_to_match(g.cpu(), target=l1_gnn, T_max=args.T_max)
        gnn_beats_sg = l1_gnn <= sg_best
        print(f"  meilleure borne sous-gradient en {args.T_max} it. = {sg_best:.4f}")
        if n_iters is not None:
            print(f"  itérations sous-gradient pour égaler/dépasser la borne GNN : {n_iters}")
        else:
            gap_pct = 100.0 * (sg_best - l1_gnn) / abs(l1_gnn) if l1_gnn else float("nan")
            print(f"  borne GNN NON atteinte en {args.T_max} itérations "
                  f"(écart restant : {sg_best - l1_gnn:+.4f}, {gap_pct:+.2f}%)")

        verdict = "GNN meilleur (ou égal) que L1(λ=0)" if l1_gnn <= l1_zero else "GNN moins bon que L1(λ=0) — INCOHÉRENT"
        print(f"  Résumé : {verdict} ; "
              f"{'GNN bat le sous-gradient en T_max it.' if gnn_beats_sg else 'sous-gradient bat le GNN en T_max it.'}")

        results.append({
            "inst_id": rec["inst_id"], "n": inst["n"], "N": inst["N"],
            "l1_zero": l1_zero, "l1_gnn": l1_gnn,
            "lam_mean": lam_mean, "lam_max": lam_max,
            "n_iters": n_iters, "sg_best": sg_best,
            "gnn_beats_sg": gnn_beats_sg,
            "coherent": l1_gnn <= l1_zero,
        })

    print(f"\n{'=' * 78}\n=== Tableau récapitulatif ({len(results)} instance(s)) ===\n{'=' * 78}")
    header = f"{'inst_id':>8}  {'L1(λ=0)':>12}  {'L1(λ_GNN)':>12}  {'|λ|_mean':>9}  {'|λ|_max':>9}  {'sg_best':>12}  {'n_iters':>8}  {'GNN<=SG':>8}"
    print(header)
    for r in results:
        n_iters_str = str(r["n_iters"]) if r["n_iters"] is not None else f">{args.T_max}"
        print(f"{r['inst_id']:>8}  {r['l1_zero']:>12.4f}  {r['l1_gnn']:>12.4f}  "
              f"{r['lam_mean']:>9.4f}  {r['lam_max']:>9.4f}  {r['sg_best']:>12.4f}  "
              f"{n_iters_str:>8}  {str(r['gnn_beats_sg']):>8}")

    n_r = len(results)
    mean = lambda key: sum(r[key] for r in results) / n_r
    pct_coherent = 100.0 * sum(r["coherent"] for r in results) / n_r
    pct_beats_sg = 100.0 * sum(r["gnn_beats_sg"] for r in results) / n_r
    matched = [r["n_iters"] for r in results if r["n_iters"] is not None]

    print(f"\n=== Moyennes sur {n_r} instance(s) ===")
    print(f"  L1(λ=0)_mean       = {mean('l1_zero'):.4f}")
    print(f"  L1(λ_GNN)_mean     = {mean('l1_gnn'):.4f}")
    print(f"  |λ_GNN|_mean_mean  = {mean('lam_mean'):.4f}")
    print(f"  |λ_GNN|_max_mean   = {mean('lam_max'):.4f}")
    print(f"  sg_best_mean (T_max={args.T_max}) = {mean('sg_best'):.4f}")
    print(f"  % instances GNN <= L1(λ=0) (cohérent, borne resserrée) = {pct_coherent:.1f}%")
    print(f"  % instances où GNN bat le sous-gradient en T_max={args.T_max} it. = {pct_beats_sg:.1f}%")
    if matched:
        print(f"  itérations sous-gradient moyennes pour rattraper le GNN "
              f"(sur {len(matched)}/{n_r} instances où il y arrive) = {sum(matched) / len(matched):.1f}")
    else:
        print(f"  aucune instance où le sous-gradient rattrape le GNN en T_max={args.T_max} itérations.")


if __name__ == "__main__":
    main()
