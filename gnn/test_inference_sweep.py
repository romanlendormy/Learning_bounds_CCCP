#!/usr/bin/env python3
"""
test_inference_sweep.py — Balaie plusieurs (N, depth) pour un même checkpoint GNN
V1LD et produit un CSV brut, un CSV agrégé, et une figure récapitulative.

Réutilise les briques de test_inference_checkpoint.py (chargement modèle,
reconstruction du split train/test, inférence + sous-gradient par instance)
en ne rechargeant le modèle et les données qu'une seule fois pour tout le
balayage, plutôt que via des jobs séparés par (N, depth).

Usage :
    python gnn/test_inference_sweep.py --ckpt <chemin.pt> \
        --N_list 50 100 200 500 --depths 0 1 2 3 4 5 6 7 8 9 \
        --num_instances 10 --T_max 500 --out_dir gnn/runs/test_inference_sweep
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn.train_gnn_v1ld import LazyGraphDataset, lagrangian_v1ld
from gnn.test_inference_checkpoint import (
    load_checkpoint, build_model, rebuild_split, select_test_records,
    subgrad_iters_to_match, _apply_center,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Balayage (N, depth) : inférence GNN + comparaison sous-gradient.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="Chemin vers le checkpoint (.pt).")
    p.add_argument("--data_dir", default=None, help="Racine des données. Défaut : <repo>/data.")
    p.add_argument("--n", type=int, default=10, help="n de l'instance à tester.")
    p.add_argument("--N_list", type=int, nargs="+", default=[50, 100, 200, 500],
                   help="Valeurs de N (nb de scénarios) à balayer.")
    p.add_argument("--depths", type=int, nargs="+", default=list(range(10)),
                   help="Profondeurs B&B à balayer (0 = racine).")
    p.add_argument("--num_instances", type=int, default=10,
                   help="Nombre d'instances test par cellule (N, depth).")
    p.add_argument("--pick_index", type=int, default=0)
    p.add_argument("--T_max", type=int, default=500)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out_dir", default=str(REPO_ROOT / "gnn" / "runs" / "test_inference_sweep"))
    return p.parse_args()


def evaluate_cell(model, instances, test_ds, device, n, N, depth, num_instances, pick_index, T_max):
    cell_args = SimpleNamespace(
        n=n, N=N, depth=depth, inst_id=None,
        pick_index=pick_index, num_instances=num_instances,
    )
    try:
        records = select_test_records(instances, test_ds, cell_args)
    except ValueError as e:
        print(f"[skip] n={n} N={N} depth={depth} : {e}")
        return []

    ckpt_center = model.center_lam if hasattr(model, "center_lam") else False
    results = []
    for rec in records:
        inst = instances[rec["inst_id"]]
        g = LazyGraphDataset(instances, [rec])[0].to(device)

        with torch.no_grad():
            lam = model(g)
            if ckpt_center:
                lam = _apply_center(lam, int(g.N_z), int(g.n_g))
            l1_gnn = lagrangian_v1ld(lam, g).item()
            lam_mean = lam.abs().mean().item()
            lam_max = lam.abs().max().item()

        lam0 = torch.zeros(int(g.N_z) * int(g.n_g), device=device)
        l1_zero = lagrangian_v1ld(lam0, g).item()

        n_iters, sg_best, _trace = subgrad_iters_to_match(g.cpu(), target=l1_gnn, T_max=T_max)
        gnn_beats_sg = l1_gnn <= sg_best

        results.append({
            "N": N, "depth": depth, "inst_id": rec["inst_id"], "n": inst["n"],
            "l1_zero": l1_zero, "l1_gnn": l1_gnn,
            "lam_mean": lam_mean, "lam_max": lam_max,
            "n_iters": n_iters, "sg_best": sg_best,
            "gnn_beats_sg": gnn_beats_sg,
            "coherent": l1_gnn <= l1_zero,
        })
    return results


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["N"], r["depth"])].append(r)

    agg = []
    for (N, depth), cell in sorted(by_cell.items()):
        n_r = len(cell)
        gap_pct = [100.0 * (c["l1_zero"] - c["l1_gnn"]) / abs(c["l1_zero"]) if c["l1_zero"] else float("nan")
                   for c in cell]
        matched = [c["n_iters"] for c in cell if c["n_iters"] is not None]
        agg.append({
            "N": N, "depth": depth, "num_instances": n_r,
            "l1_zero_mean": sum(c["l1_zero"] for c in cell) / n_r,
            "l1_gnn_mean": sum(c["l1_gnn"] for c in cell) / n_r,
            "gap_pct_mean": sum(gap_pct) / n_r,
            "pct_coherent": 100.0 * sum(c["coherent"] for c in cell) / n_r,
            "pct_beats_sg": 100.0 * sum(c["gnn_beats_sg"] for c in cell) / n_r,
            "mean_iters_to_match": (sum(matched) / len(matched)) if matched else float("nan"),
            "pct_matched": 100.0 * len(matched) / n_r,
        })
    return agg


def plot_summary(agg: list[dict], N_list: list[int], depths: list[int], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_N = {N: {a["depth"]: a for a in agg if a["N"] == N} for N in N_list}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [
        ("gap_pct_mean", "Écart L1(λ=0) → L1(λ_GNN) (%)", axes[0, 0]),
        ("pct_coherent", "% instances cohérentes (GNN ≤ λ=0)", axes[0, 1]),
        ("pct_beats_sg", "% instances où GNN bat le sous-gradient (T_max)", axes[1, 0]),
        ("mean_iters_to_match", "Itérations sous-gradient moy. pour rattraper GNN", axes[1, 1]),
    ]
    for key, title, ax in metrics:
        for N in N_list:
            xs = [d for d in depths if d in by_N[N]]
            ys = [by_N[N][d][key] for d in xs]
            ax.plot(xs, ys, marker="o", label=f"N={N}")
        ax.set_xlabel("depth")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Sweep inférence GNN V1LD — L1(λ=0) vs L1(λ_GNN) vs sous-gradient, par (N, depth)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"[plot] figure sauvegardée dans {out_png}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path, device)
    ckpt_args = ckpt["args"]
    print(f"[ckpt] {ckpt_path.name}  epoch={ckpt['epoch']}")

    model = build_model(ckpt_args, ckpt["model"], device)
    model.center_lam = ckpt_args.get("center_lam", False)

    data_dir = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data"
    print(f"[split] reconstruction du split train/test depuis {data_dir} ...")
    instances, train_ds, test_ds = rebuild_split(ckpt_args, data_dir)
    print(f"[split] instances: train={len(set(r['inst_id'] for r in train_ds.raw_records))}  "
          f"test={len(set(r['inst_id'] for r in test_ds.raw_records))}")

    all_rows = []
    for N in args.N_list:
        for depth in args.depths:
            print(f"\n=== n={args.n} N={N} depth={depth} ===")
            rows = evaluate_cell(
                model, instances, test_ds, device,
                args.n, N, depth, args.num_instances, args.pick_index, args.T_max,
            )
            print(f"  {len(rows)} instance(s) évaluée(s).")
            all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("Aucune instance évaluée sur l'ensemble du balayage.")

    out_dir = Path(args.out_dir)
    raw_csv = out_dir / "sweep_raw.csv"
    agg_csv = out_dir / "sweep_agg.csv"
    out_png = out_dir / "sweep_summary.png"

    write_csv(raw_csv, all_rows, [
        "N", "depth", "inst_id", "n", "l1_zero", "l1_gnn",
        "lam_mean", "lam_max", "n_iters", "sg_best", "gnn_beats_sg", "coherent",
    ])
    print(f"\n[csv] brut : {raw_csv}")

    agg = aggregate(all_rows)
    write_csv(agg_csv, agg, [
        "N", "depth", "num_instances", "l1_zero_mean", "l1_gnn_mean", "gap_pct_mean",
        "pct_coherent", "pct_beats_sg", "mean_iters_to_match", "pct_matched",
    ])
    print(f"[csv] agrégé : {agg_csv}")

    print(f"\n{'N':>6}  {'depth':>5}  {'n_inst':>6}  {'gap%':>8}  {'%coher':>7}  {'%beatSG':>8}  {'it_moy':>7}")
    for a in agg:
        it_str = f"{a['mean_iters_to_match']:.1f}" if a["mean_iters_to_match"] == a["mean_iters_to_match"] else "n/a"
        print(f"{a['N']:>6}  {a['depth']:>5}  {a['num_instances']:>6}  {a['gap_pct_mean']:>8.2f}  "
              f"{a['pct_coherent']:>7.1f}  {a['pct_beats_sg']:>8.1f}  {it_str:>7}")

    plot_summary(agg, args.N_list, args.depths, out_png)


if __name__ == "__main__":
    main()
