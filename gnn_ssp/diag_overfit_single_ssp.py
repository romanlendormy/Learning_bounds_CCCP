#!/usr/bin/env python3
"""
diag_overfit_single_ssp.py — Overfit du GNN V1LD (SSP chance-constrained,
cf. train_v1LD_ssp.py) sur un seul nœud B&B (racine par défaut) d'une
instance E/T/N donnée.

Miroir de gnn_sd_edit/diag_overfit_single.py (MKP), simplifié : ζ/θ côté SSP
passent par CP-SAT (SSPShiftProblem, cf. bnb_solver_ssp.py) et coûtent
plusieurs secondes par forward (N_z appels theta+zeta), donc pas de balayage
"convergence complète" ni d'équivalence warm-start ici — seulement les deux
références demandées (sous-gradient cold-start T=50 et T=100), obtenues en UN
seul run à T=100 (history_out) pour ne pas payer deux fois le coût CP-SAT.

Sorties (out_dir/) : fig_convergence.png, metrics.json

Usage :
    cd <repo_root>
    python gnn_ssp/diag_overfit_single_ssp.py --E 4 --T 8 --N 20 --n_epochs 300
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # évite l'overhead OpenMP sur login/compute node

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from train_v1LD_ssp import (
    load_raw_nodes, LazyGraphDataset,
    GNN, lagrangian_v1ld_ssp, verify_v1ld_ssp, _apply_center_ssp,
    SSP_SIGMA_SIZE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overfit du GNN V1LD SSP sur un seul nœud B&B (racine).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--E", type=int, default=4, help="Nb d'employés.")
    p.add_argument("--T", type=int, default=8, help="Horizon temporel.")
    p.add_argument("--N", type=int, default=20, help="Nb de scénarios de l'instance.")
    p.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "train_ssp_base_s10"),
                   help="Dossier contenant instances_E{E}_T{T}_N{N}.txt / nodes_E{E}_T{T}_N{N}.txt.")
    p.add_argument("--max_load", type=int, default=2000,
                   help="Limite de lignes lues dans nodes_*.txt (les nœuds racine "
                        "apparaissent tôt dans le fichier, pas besoin de tout charger).")
    p.add_argument("--graph_idx", type=int, default=0,
                   help="Index du nœud racine (depth=0) choisi parmi ceux filtrés.")

    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eta_min", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=5,
                   help="Fréquence d'enregistrement de l'historique (epochs).")

    p.add_argument("--subgrad_T", type=int, default=100,
                   help="Nb max d'itérations sous-gradient (cold start) — les valeurs "
                        "à T=50 et T=subgrad_T sont extraites d'un seul run.")
    p.add_argument("--subgrad_T50", type=int, default=50)

    p.add_argument("--out_dir", default=None,
                   help="Dossier de sortie. Défaut : "
                        "<repo>/gnn_ssp/runs/overfit_single/E{E}_T{T}_N{N}_inst{id}_idx{graph_idx}/")
    args = p.parse_args()
    return args


def main() -> None:
    args = parse_args()
    t_start = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")  # ζ/θ CP-SAT dominent — aucun intérêt à un GPU ici

    # ── 1. Chargement + sélection du nœud racine ───────────────────────────
    src_dir    = Path(args.data_dir)
    inst_file  = src_dir / f"instances_E{args.E}_T{args.T}_N{args.N}.txt"
    nodes_file = src_dir / f"nodes_E{args.E}_T{args.T}_N{args.N}.txt"
    print(f"[diag] E={args.E} T={args.T} N={args.N}  ->  {src_dir}", flush=True)
    if not inst_file.exists() or not nodes_file.exists():
        sys.exit(f"[diag] Fichiers manquants dans {src_dir}")

    instances, records = load_raw_nodes(inst_file, nodes_file, max_records=args.max_load)
    records_root = [r for r in records if r["depth"] == 0]
    print(f"[diag] {len(records_root)} nœud(s) racine (depth=0) trouvé(s) "
          f"parmi les {len(records)} premiers enregistrements.", flush=True)
    if not records_root:
        sys.exit("[diag] Aucun nœud racine trouvé — augmentez --max_load.")

    ds = LazyGraphDataset(instances, records_root)
    g = ds[args.graph_idx]
    inst_id = records_root[args.graph_idx]["inst_id"]
    inst = instances[inst_id]
    print(f"[diag] Instance choisie : inst_id={inst_id}  G={g.n_g}  N_z={g.N_z}/{g.N}  "
          f"E={g.E}  T={g.T}  opt={g.opt.item():.4f}", flush=True)

    out_dir = (Path(args.out_dir) if args.out_dir else
               REPO_ROOT / "gnn_ssp" / "runs" / "overfit_single" /
               f"E{args.E}_T{args.T}_N{args.N}_inst{inst_id}_idx{args.graph_idx}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diag] out_dir = {out_dir}", flush=True)

    # ── 2. Références : cost_lower(λ=0) et sous-gradient cold-start ────────
    opt_val = g.opt.item()
    with torch.no_grad():
        lam0    = torch.zeros(g.N_z * g.n_g * SSP_SIGMA_SIZE)
        cl_init = -lagrangian_v1ld_ssp(lam0, g, inst).item()
    print(f"[diag] cost_lower(λ=0) = {cl_init:.4f}  opt = {opt_val:.4f}", flush=True)

    t0 = time.time()
    hist_ref: list = []
    best_T = verify_v1ld_ssp(g, inst, T=args.subgrad_T, v_lower=-opt_val, history_out=hist_ref)
    print(f"[diag] Sous-gradient cold-start jusqu'à T={args.subgrad_T} : "
          f"{time.time() - t0:.1f}s", flush=True)
    if len(hist_ref) < args.subgrad_T:
        sys.exit(f"[diag] hist_ref trop court ({len(hist_ref)} < {args.subgrad_T}) — bug ?")

    cl_ref50  = -hist_ref[args.subgrad_T50 - 1]
    cl_ref100 = -hist_ref[args.subgrad_T - 1]
    print(f"[diag] Sous-gradient T={args.subgrad_T50}  = {cl_ref50:.4f}", flush=True)
    print(f"[diag] Sous-gradient T={args.subgrad_T}   = {cl_ref100:.4f}", flush=True)

    # ── 3. Modèle + entraînement (overfit sur ce seul graphe) ──────────────
    model = GNN(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[diag] GNN SSP : {n_params:,} paramètres  hidden={args.hidden}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=args.eta_min)

    hist_epochs, hist_cl = [], []
    t0 = time.time()
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lam = model(g)
        lam = _apply_center_ssp(lam, int(g.N_z), int(g.n_g))
        loss = lagrangian_v1ld_ssp(lam, g, inst)
        loss.backward()
        optimizer.step()
        scheduler.step()

        cl = -loss.item()
        if epoch % args.eval_every == 0 or epoch == 1:
            hist_epochs.append(epoch)
            hist_cl.append(cl)
            elapsed = time.time() - t0
            gap_opt = (cl - opt_val) / opt_val * 100 if opt_val else float("nan")
            print(f"ep {epoch:5d}/{args.n_epochs}  cost_lower={cl:10.4f}  "
                  f"gap_vs_opt={gap_opt:+6.2f}%  elapsed={elapsed:7.1f}s", flush=True)

    final_cl = hist_cl[-1]
    print(f"\n[diag] Borne finale (cost_lower) : {final_cl:.4f}", flush=True)
    print(f"  vs cost_lower(λ=0) : {cl_init:.4f}", flush=True)
    print(f"  vs sous-grad T={args.subgrad_T50}  : {cl_ref50:.4f}  "
          f"({'mieux' if final_cl > cl_ref50 else 'moins bon'})", flush=True)
    print(f"  vs sous-grad T={args.subgrad_T} : {cl_ref100:.4f}  "
          f"({'mieux' if final_cl > cl_ref100 else 'moins bon'})", flush=True)
    print(f"  vs opt              : {opt_val:.4f}", flush=True)

    # ── 4. Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(hist_epochs, hist_cl, color="steelblue", lw=1.6, marker=".", ms=4,
            label="cost_lower(λ_GNN) — overfit")
    ax.axhline(opt_val, color="green", lw=1.5, ls="-", label=f"opt = {opt_val:.2f}")
    ax.axhline(cl_ref50, color="gray", lw=1.4, ls="--",
               label=f"Sous-gradient T={args.subgrad_T50} = {cl_ref50:.2f}")
    ax.axhline(cl_ref100, color="dimgray", lw=1.4, ls="--",
               label=f"Sous-gradient T={args.subgrad_T} = {cl_ref100:.2f}")
    ax.scatter([hist_epochs[-1]], [final_cl], color="steelblue", s=60, zorder=5)
    ax.annotate(f"  {final_cl:.2f}", (hist_epochs[-1], final_cl),
                va="center", fontsize=9, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("cost_lower(λ) — borne inférieure sur le coût")
    ax.set_title(
        f"Overfit SSP — instance E={args.E}, T={args.T}, N={args.N} (inst_id={inst_id})\n"
        f"GNN (hidden={args.hidden}), lr={args.lr}, epochs={args.n_epochs}"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = out_dir / "fig_convergence.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n[diag] Figure sauvegardée : {fig_path}", flush=True)

    # ── 5. Metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "args": vars(args),
        "instance": {"inst_id": inst_id, "G": g.n_g, "N_z": int(g.N_z), "N": int(g.N),
                     "E": int(g.E), "T": int(g.T), "opt": opt_val},
        "references": {"cl_init": cl_init, "cl_ref50": cl_ref50, "cl_ref100": cl_ref100,
                        "hist_subgrad": [-v for v in hist_ref]},
        "final_cl": final_cl,
        "history": {"epochs": hist_epochs, "cost_lower": hist_cl},
        "total_wall_time_s": time.time() - t_start,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[diag] Metrics sauvegardées : {metrics_path}", flush=True)
    print(f"[diag] Temps total : {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
