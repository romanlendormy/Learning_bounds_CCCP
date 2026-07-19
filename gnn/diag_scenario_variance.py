#!/usr/bin/env python3
"""
diag_scenario_variance.py — Localise l'étage du GNN V1LD où la différenciation
inter-scénarios de λ s'effondre à mesure que N grandit.

Contexte : sur overfit_single (gnn/diag_overfit_single.py, gnn_sd_edit/...), le
ratio de diversité résid_GNN/résid_conv décroît de ~3 (N=20) à ~1 (N=100) pour
les DEUX architectures (clique homogène et étoile sd_edit) — signal que le
goulot n'est pas la topologie des arêtes inter-scénarios. Ce script mesure,
à chaque étage caché (x, h_embed, h1, h2, h_v, sortie λ), la part de variance
inter-scénarios (ICC = variance_between_scénarios / variance_totale, ANOVA à un
facteur) sur les représentations nodales — à l'initialisation (avant tout
entraînement, signal purement structurel) ET après overfit (500 epochs, mêmes
réglages que diag_overfit_single.py) — pour localiser à quel étage et à quel
degré (structurel vs appris) le signal scénario-spécifique se dilue.

ICC(H, scen) = Σ_scén n_g·‖μ_scén − μ‖² / Σ_nœud ‖H_nœud − μ‖²
  = 1 si les scénarios sont parfaitement séparés (toute la variance est
    inter-scénarios), → 0 si tous les scénarios sont indiscernables au sein
    de cet étage.

Usage :
    python gnn/diag_scenario_variance.py --arch homogene --data_dir data/train_base_s10 \
        --n_val 10 --n_scen 20
    python gnn/diag_scenario_variance.py --arch sd_edit --data_dir data/train_base_s10 \
        --n_val 10 --n_scen 100
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from collect.merge_datasets import src_dir_for, valid_inst_ids_for_N

STAGES = ["x", "h_embed", "h1", "h2", "h_v"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Localise le goulot de différenciation inter-scénarios du GNN V1LD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--arch", choices=["homogene", "sd_edit"], default="homogene",
                   help="homogene = gnn.train_gnn_v1ld (clique inter-scénarios), "
                        "sd_edit = gnn_sd_edit.train_v1LD_sd_edit (étoile).")
    p.add_argument("--n_scen", type=int, default=100)
    p.add_argument("--n_val", type=int, default=10)
    p.add_argument("--m_val", type=int, default=5)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_depth", type=int, default=0)
    p.add_argument("--max_load", type=int, default=5000)
    p.add_argument("--graph_idx", type=int, default=0)

    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--no_max_pool", action="store_true")
    p.add_argument("--no_center_lam", action="store_true")
    p.add_argument("--no_c_scale_norm", action="store_true")
    p.add_argument("--max_norm", type=float, default=2.0)

    p.add_argument("--n_epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eta_min", type=float, default=1e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=20)

    p.add_argument("--subgrad_full_T_max", type=int, default=500)
    p.add_argument("--subgrad_full_patience", type=int, default=30)
    p.add_argument("--subgrad_full_precision", type=float, default=1.0)

    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def apply_center(lam: torch.Tensor, N_z: int, n_g: int) -> torch.Tensor:
    lam_2d = lam.view(N_z, n_g)
    return (lam_2d - lam_2d.mean(dim=0, keepdim=True)).reshape(-1)


def icc(H: torch.Tensor, scen_batch: torch.Tensor, N_z: int) -> float:
    """Part de variance nodale expliquée par l'appartenance au scénario (ANOVA à
    un facteur, groupes de taille égale n_g). H : (n_nodes, d) ou (n_nodes,)."""
    if H.dim() == 1:
        H = H.unsqueeze(-1)
    H = H.double()
    mu = H.mean(dim=0, keepdim=True)                                   # (1,d)
    sums   = torch.zeros(N_z, H.shape[1], dtype=torch.double)
    counts = torch.zeros(N_z, dtype=torch.double)
    sums.index_add_(0, scen_batch, H)
    counts.index_add_(0, scen_batch, torch.ones_like(scen_batch, dtype=torch.double))
    mu_scen = sums / counts.clamp(min=1).unsqueeze(-1)                  # (N_z,d)
    mu_scen_exp = mu_scen[scen_batch]                                   # (n_nodes,d)
    total   = ((H - mu) ** 2).sum().item()
    between = ((mu_scen_exp - mu) ** 2).sum().item()
    return between / total if total > 1e-12 else float("nan")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.arch == "homogene":
        from gnn.train_gnn_v1ld import (
            load_raw_nodes, LazyGraphDataset, GNN, lagrangian_v1ld, verify_v1ld,
        )
    else:
        from gnn_sd_edit.train_v1LD_sd_edit import (
            load_raw_nodes, LazyGraphDataset, GNN, lagrangian_v1ld, verify_v1ld,
        )

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                           else ("cpu" if args.device == "auto" else args.device))

    # ── 1. Chargement instance (identique à diag_overfit_single.py) ────────
    if args.data_dir:
        src_dir    = Path(args.data_dir)
        inst_file  = src_dir / f"instances_n{args.n_val}_m{args.m_val}_N{args.n_scen}.txt"
        nodes_file = src_dir / f"nodes_n{args.n_val}_m{args.m_val}_N{args.n_scen}.txt"
    else:
        src_dir    = src_dir_for(args.n_scen, REPO_ROOT / "data")
        inst_file  = src_dir / f"instances_n{args.n_val}_m{args.m_val}.txt"
        nodes_file = src_dir / f"nodes_n{args.n_val}_m{args.m_val}.txt"
    print(f"[diag] arch={args.arch}  n_scen={args.n_scen}  ->  {src_dir}  device={device}")
    if not inst_file.exists() or not nodes_file.exists():
        sys.exit(f"[diag] Fichiers manquants dans {src_dir}")

    instances, records = load_raw_nodes(inst_file, nodes_file, max_records=args.max_load)
    valid_ids = valid_inst_ids_for_N(inst_file, args.n_scen)
    records_N = [r for r in records if valid_ids is None or r["inst_id"] in valid_ids]
    if args.max_depth >= 0:
        filtered = [r for r in records_N if r["depth"] <= args.max_depth]
        records_N = filtered if filtered else records_N
    if not records_N:
        sys.exit("[diag] Aucun nœud trouvé.")

    ds = LazyGraphDataset(instances, records_N)
    g  = ds[args.graph_idx]
    inst_id = records_N[args.graph_idx]["inst_id"]
    depth   = int(g.context[0].item())
    N_z, n_g = int(g.N_z), int(g.n_g)
    print(f"[diag] Instance : inst_id={inst_id}  depth={depth}  n_g={n_g}  N_z={N_z}/{g.N}  m={g.m}")

    out_dir = (Path(args.out_dir) if args.out_dir else
               REPO_ROOT / "gnn" / "runs" / "diag_scenario_variance" /
               f"{args.arch}_n{args.n_val}_N{args.n_scen}")
    out_dir.mkdir(parents=True, exist_ok=True)

    g = g.to(device)
    center_lam   = not args.no_center_lam
    c_scale_norm = not args.no_c_scale_norm
    model_cfg = dict(m=g.m, hidden=args.hidden, use_max_pool=not args.no_max_pool)
    model = GNN(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[diag] GNN {args.arch} : {n_params:,} paramètres  cfg={model_cfg}")

    # ── 2. Référence sous-gradient convergé (pour ICC-cible) ───────────────
    g_cpu = g.cpu()
    l1_conv, lam_conv = verify_v1ld(
        g_cpu, T=args.subgrad_full_T_max,
        precision=args.subgrad_full_precision, patience=args.subgrad_full_patience,
        return_lam=True,
    )
    scen_batch_ref = torch.arange(N_z).repeat_interleave(n_g)
    icc_target = icc(torch.from_numpy(lam_conv.reshape(-1)), scen_batch_ref, N_z)
    print(f"[diag] Sous-gradient convergé = {l1_conv:.2f}  ICC(λ_conv) = {icc_target:.4f}")

    # ── 3. Snapshot ICC à l'initialisation (avant tout entraînement) ───────
    # scen_batch_ref réutilisé (indépendant du pooling interne du modèle, qui
    # groupe désormais par item — cf. train_gnn_v1ld.py) : ce diagnostic pose
    # toujours la question "différencie-t-on les scénarios ?".
    scen_batch = scen_batch_ref
    model.eval()
    with torch.no_grad():
        lam_init, extra_init = model(g, return_hidden=True)
    icc_init = {s: icc(extra_init[s], scen_batch, N_z) for s in STAGES}
    icc_init["lam"] = icc(lam_init, scen_batch, N_z)
    print(f"[diag] ICC à l'init : " + "  ".join(f"{s}={icc_init[s]:.4f}" for s in STAGES + ['lam']))

    # ── 4. Entraînement (identique à diag_overfit_single.py) ───────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=args.eta_min)

    hist_epochs, hist_l1 = [], []
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lam = model(g)
        if center_lam:
            lam = apply_center(lam, N_z, n_g)
        c_norm = max(float(g.c_scale), 1.0) if c_scale_norm else 1.0
        loss = lagrangian_v1ld(lam, g) / c_norm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=args.max_norm if args.max_norm > 0 else float("inf"))
        optimizer.step()
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == 1:
            with torch.no_grad():
                l1_raw = lagrangian_v1ld(lam, g).item()
            hist_epochs.append(epoch); hist_l1.append(l1_raw)
            print(f"ep {epoch:5d}/{args.n_epochs}  L1={l1_raw:10.3f}", flush=True)

    # ── 5. Snapshot ICC après overfit ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        lam_final, extra_final = model(g, return_hidden=True)
        if center_lam:
            lam_final = apply_center(lam_final, N_z, n_g)
    icc_trained = {s: icc(extra_final[s], scen_batch, N_z) for s in STAGES}
    icc_trained["lam"] = icc(lam_final, scen_batch, N_z)
    print(f"[diag] ICC après overfit ({args.n_epochs} ep) : "
          + "  ".join(f"{s}={icc_trained[s]:.4f}" for s in STAGES + ['lam']))
    print(f"[diag] ICC cible (sous-gradient convergé) : {icc_target:.4f}")

    # ── 6. Figure ────────────────────────────────────────────────────────────
    stages_all = STAGES + ["lam"]
    xs = list(range(len(stages_all)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [icc_init[s] for s in stages_all], "o--", color="gray", label="à l'initialisation")
    ax.plot(xs, [icc_trained[s] for s in stages_all], "o-", color="steelblue", lw=2, label="après overfit")
    ax.axhline(icc_target, color="red", ls=":", label=f"cible (sous-grad. convergé) = {icc_target:.3f}")
    ax.set_xticks(xs); ax.set_xticklabels(stages_all)
    ax.set_ylabel("ICC (part de variance inter-scénarios)")
    ax.set_title(f"Différenciation inter-scénarios par étage — {args.arch}, "
                 f"n={args.n_val}, N={args.n_scen} (n_g={n_g}, N_z={N_z})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = out_dir / "fig_icc_by_stage.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag] Figure : {fig_path}")

    # ── 7. Metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "args": vars(args),
        "instance": {"inst_id": inst_id, "depth": depth, "n_g": n_g, "N_z": N_z,
                     "N": int(g.N), "m": g.m},
        "icc_init": icc_init, "icc_trained": icc_trained, "icc_target": icc_target,
        "l1_conv": l1_conv,
        "history": {"epochs": hist_epochs, "l1": hist_l1},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[diag] Metrics : {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
