#!/usr/bin/env python3
"""
gen_quality_by_depth.py — Qualité de la borne GNN vs sous-gradient (T fixe),
en fonction de la profondeur B&B, sur des nœuds échantillonnés dans data_s10.

Recharge le pipeline de données de train_scaled_v1ld.py (mêmes n_list/n_scens/m/
seed/reservoir_k/target_graphs/root_frac/train_frac, lus depuis ckpt["args"])
pour reproduire exactement le split train/test au niveau instance, puis
échantillonne des nœuds à TOUTES profondeurs (contrairement à gen_quality_fig.py,
qui ne regarde que la racine). Pour chaque nœud :
  - L1(λ_GNN)                         : inférence one-shot du GNN
  - L1(sous-gradient, T=--sg_T, cold start λ=0)
puis gap relatif : gap% = (L1_GNN − L1_SG) / |L1_SG| * 100
  (négatif = GNN meilleur/plus resserré ; positif = GNN moins bon).

Le --data_dir du checkpoint pointe vers un SLURM_TMPDIR éphémère (copié là pour
l'entraînement) : ce script retombe sur <repo>/data par défaut.

Usage :
    python gnn/gen_quality_by_depth.py --ckpt <checkpoint.pt>
    python gnn/gen_quality_by_depth.py --ckpt <checkpoint.pt> --split test --n_points 400 --sg_T 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn.train_scaled_v1ld import (
    discover_pairs, load_and_merge, select_stratified, split_by_instance,
)
from gnn.train_gnn_v1ld import GNN, LazyGraphDataset, lagrangian_v1ld, verify_v1ld

_N_COLORS = {10: "#1f77b4", 15: "#ff7f0e", 20: "#2ca02c"}
_N_MARKERS = {20: "o", 50: "s", 100: "^"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gap relatif GNN vs sous-gradient(T) par profondeur B&B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data_dir", type=Path, default=None,
                   help="Défaut : <repo>/data.")
    p.add_argument("--split", choices=["train", "test", "all"], default="test")
    p.add_argument("--n_points", type=int, default=400,
                   help="Nombre de nœuds échantillonnés (toutes profondeurs confondues).")
    p.add_argument("--sg_T", type=int, default=50,
                   help="Itérations de sous-gradient (cold start λ=0) de référence.")
    p.add_argument("--sample_seed", type=int, default=123,
                   help="Graine du sous-échantillonnage (indépendante de la graine d'entraînement).")
    p.add_argument("--out_dir", type=Path, default=None, help="Défaut : dossier du checkpoint.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def build_model(ck: dict, state_dict: dict, device: torch.device) -> GNN:
    model = GNN(m=ck["m"], hidden=ck["hidden"], use_max_pool=ck.get("use_max_pool", False)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def apply_center(lam: torch.Tensor, N_z: int, n_g: int) -> torch.Tensor:
    lam_2d = lam.view(N_z, n_g)
    return (lam_2d - lam_2d.mean(dim=0, keepdim=True)).reshape(-1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"[ckpt] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    ck = ckpt["args"]
    print(f"[ckpt] epoch={ckpt['epoch']}  n_list={ck['n_list']}  n_scens={ck['n_scens']}  "
          f"hidden={ck['hidden']}  center_lam={ck.get('center_lam')}")

    model = build_model(ck, ckpt["model"], device)

    data_dir = args.data_dir if args.data_dir else REPO_ROOT / "data"
    print(f"[data] découverte sous {data_dir} ...")
    pairs = discover_pairs(data_dir, ck["n_list"], ck["n_scens"], ck["m"])
    if not pairs:
        sys.exit(f"ERREUR : aucune donnée trouvée sous {data_dir}")

    instances, all_records = load_and_merge(pairs, ck["reservoir_k"], ck["seed"])
    selected = select_stratified(all_records, ck["target_graphs"], ck["root_frac"], ck["seed"])
    all_ds = LazyGraphDataset(instances, selected)
    train_ds, test_ds = split_by_instance(instances, all_ds, ck["train_frac"], ck["seed"])
    src_ds = {"train": train_ds, "test": test_ds, "all": all_ds}[args.split]
    n_inst = len(set(r["inst_id"] for r in src_ds.raw_records))
    print(f"[split] {args.split} : {len(src_ds)} nœuds disponibles ({n_inst} instances)")

    records = list(src_ds.raw_records)
    random.Random(args.sample_seed).shuffle(records)
    if args.n_points and len(records) > args.n_points:
        records = records[: args.n_points]
    print(f"[eval] {len(records)} nœuds échantillonnés (sg_T={args.sg_T})")

    ds = LazyGraphDataset(instances, records)
    depths, gaps, ns, Ns, l1_gnns, l1_sgs, inst_ids = [], [], [], [], [], [], []
    t0 = time.time()
    for j, rec in enumerate(records):
        g = ds[j].to(device)
        with torch.no_grad():
            lam = model(g)
            if ck.get("center_lam", False):
                lam = apply_center(lam, int(g.N_z), int(g.n_g))
            l1_gnn = lagrangian_v1ld(lam, g).item()
        g_cpu = g.cpu()
        l1_sg = verify_v1ld(g_cpu, T=args.sg_T)
        gap_pct = (l1_gnn - l1_sg) / abs(l1_sg) * 100 if abs(l1_sg) > 1e-9 else float("nan")

        depths.append(rec["depth"])
        gaps.append(gap_pct)
        ns.append(rec.get("n_orig", instances[rec["inst_id"]]["n"]))
        Ns.append(rec.get("N_orig", instances[rec["inst_id"]]["N"]))
        l1_gnns.append(l1_gnn)
        l1_sgs.append(l1_sg)
        inst_ids.append(rec["inst_id"])

        if (j + 1) % 25 == 0 or j == len(records) - 1:
            print(f"  [{j + 1}/{len(records)}]  t={time.time() - t0:.0f}s")

    depths = np.array(depths)
    gaps = np.array(gaps)
    ns = np.array(ns)
    Ns = np.array(Ns)

    valid = ~np.isnan(gaps)
    pct_better = 100 * np.mean(gaps[valid] < 0)
    print(f"[quality] {valid.sum()}/{len(gaps)} points valides  "
          f"gap moyen={np.nanmean(gaps):+.2f}%  médian={np.nanmedian(gaps):+.2f}%  "
          f"GNN meilleur (gap<0) : {pct_better:.1f}%")

    # ── Figure ────────────────────────────────────────────────────────────────
    out_dir = args.out_dir if args.out_dir else args.ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.ckpt.stem}_{args.split}_sgT{args.sg_T}"
    fig_path = out_dir / f"fig_quality_by_depth_{tag}.png"

    rng = np.random.default_rng(args.sample_seed)
    jitter = rng.uniform(-0.15, 0.15, size=len(depths))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0, color="0.4", lw=1.2, ls="--", zorder=2,
               label=f"gap = 0 (GNN = sous-grad T={args.sg_T})")

    for n_val in sorted(set(ns.tolist())):
        color = _N_COLORS.get(n_val, "gray")
        for N_val in sorted(set(Ns[ns == n_val].tolist())):
            mask = (ns == n_val) & (Ns == N_val) & valid
            if not mask.any():
                continue
            marker = _N_MARKERS.get(N_val, "x")
            ax.scatter(depths[mask] + jitter[mask], gaps[mask],
                       color=color, marker=marker, alpha=0.35, s=22,
                       linewidths=0, zorder=3)

    # Tendance agrégée : moyenne par (n, depth)
    for n_val in sorted(set(ns.tolist())):
        color = _N_COLORS.get(n_val, "gray")
        mask_n = (ns == n_val) & valid
        if not mask_n.any():
            continue
        depth_vals = sorted(set(depths[mask_n].tolist()))
        means = [gaps[mask_n & (depths == d)].mean() for d in depth_vals]
        ax.plot(depth_vals, means, color=color, lw=2.2, marker="D", ms=6,
                zorder=4, label=f"n={n_val} (moyenne)")

    for N_val, marker in sorted(_N_MARKERS.items()):
        if N_val in set(Ns.tolist()):
            ax.scatter([], [], color="0.3", marker=marker, s=40, label=f"N={N_val}")

    ax.set_xlabel("Profondeur B&B")
    ax.set_ylabel(f"Gap relatif GNN vs sous-gradient T={args.sg_T}  (%)")
    ax.set_title(
        f"Qualité GNN vs sous-gradient (T={args.sg_T}) par profondeur — [{args.split}]\n"
        f"{tag}  (epoch={ckpt['epoch']})\n"
        f"gap moyen={np.nanmean(gaps):+.2f}%  médian={np.nanmedian(gaps):+.2f}%  "
        f"GNN meilleur (gap<0) : {pct_better:.1f}% ({int(valid.sum())} pts)"
    )
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[quality] Figure : {fig_path}")

    # ── Données brutes ───────────────────────────────────────────────────────
    data_path = out_dir / f"data_quality_by_depth_{tag}.json"
    with open(data_path, "w") as f:
        json.dump({
            "ckpt": str(args.ckpt), "epoch": ckpt["epoch"], "split": args.split,
            "sg_T": args.sg_T, "n_points": len(records),
            "summary": {
                "gap_mean": float(np.nanmean(gaps)),
                "gap_median": float(np.nanmedian(gaps)),
                "pct_gnn_better": float(pct_better),
            },
            "points": [
                {"inst_id": int(i), "depth": int(d), "n": int(n), "N": int(N),
                 "l1_gnn": float(lg), "l1_sg": float(ls), "gap_pct": float(gp)}
                for i, d, n, N, lg, ls, gp in
                zip(inst_ids, depths, ns, Ns, l1_gnns, l1_sgs, gaps)
            ],
        }, f, indent=2)
    print(f"[quality] Données : {data_path}")


if __name__ == "__main__":
    main()
