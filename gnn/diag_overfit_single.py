#!/usr/bin/env python3
"""
diag_overfit_single.py — Overfit du GNN V1LD sur une seule instance (données ablation N).

Version script de `experiment_overfit_ablationN.ipynb` : mêmes données, même
modèle, mêmes réglages. Objectif : suivre la progression de la borne
L1(λ_GNN) époque par époque sur UNE instance unique, comparée à L1(λ=0) et à
la référence sous-gradient (T=50, la même que celle loguée pendant
l'ablation N — voir gnn/diag_overfit_100.py et run_ablation_N_rorqual.sh).

- charge une seule instance (un seul nœud B&B, racine depth=0 par défaut) ;
- N (nombre de scénarios) est sélectionnable (--n_scen 50 / 200 / 500), résolu
  vers data/train_N{N}/ ou data/train_base_s30/ (pour N=500) exactement comme
  collect/merge_datasets.py::src_dir_for utilisé pour fusionner les données
  de l'ablation N ;
- utilise le même GNN (classe GNN de gnn/train_gnn_v1ld.py) avec la même
  configuration que l'ablation N (hidden=64, use_max_pool, center_lam,
  clip de gradient max_norm=2.0, loss normalisée par
  c_scale) — pas la config nue utilisée par experiment_gnn_v1LD_base.ipynb.

Sorties (out_dir/) : fig_convergence.png, metrics.json

Usage :
    cd <repo_root>
    python gnn/diag_overfit_single.py --n_scen 500
    python gnn/diag_overfit_single.py --n_scen 50 --n_epochs 500 --eval_every 10
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # évite l'overhead OpenMP sur login node

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

from gnn.train_gnn_v1ld import (
    load_raw_nodes, LazyGraphDataset,
    GNN, lagrangian_v1ld, verify_v1ld,
)
from deepsets.model import DeepSetsGNN
from collect.merge_datasets import src_dir_for, valid_inst_ids_for_N


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overfit du GNN V1LD (config ablation N) sur une seule instance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Sélection de l'instance ────────────────────────────────────────────
    p.add_argument("--n_scen", type=int, default=500,
                   help="Nombre de scénarios N de l'instance (50, 200 ou 500 — "
                        "résolu via collect/merge_datasets.py::src_dir_for, "
                        "sauf si --data_dir est fourni).")
    p.add_argument("--n_val", type=int, default=10, help="n (nb d'items).")
    p.add_argument("--m_val", type=int, default=5, help="m (nb de contraintes).")
    p.add_argument("--data_dir", default=None,
                   help="Dossier contenant directement "
                        "instances_n{n_val}_m{m_val}_N{n_scen}.txt / "
                        "nodes_n{n_val}_m{m_val}_N{n_scen}.txt (ex. "
                        "data/train_base_s10). Si fourni, remplace la "
                        "résolution via src_dir_for.")
    p.add_argument("--max_depth", type=int, default=0,
                   help="Ne garder que les nœuds de profondeur <= ce seuil "
                        "(0 = racine uniquement). -1 = pas de filtre.")
    p.add_argument("--max_load", type=int, default=5000,
                   help="Limite de lignes lues dans nodes_*.txt (accélère le chargement).")
    p.add_argument("--graph_idx", type=int, default=0,
                   help="Index du nœud choisi parmi ceux filtrés.")

    # ── Modèle (config ablation N par défaut) ──────────────────────────────
    p.add_argument("--model_type", default="gnn", choices=["gnn", "deepsets"],
                   help="gnn = GatedGraphConv (train_gnn_v1ld.GNN, message "
                        "passing sur cliques item/scénario) ; deepsets = "
                        "deepsets.model.DeepSetsGNN (même embed/pooling/tête, "
                        "sans message passing).")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--no_max_pool", action="store_true",
                   help="Désactive use_max_pool (activé par défaut, comme l'ablation N).")
    p.add_argument("--use_rich_pool", action="store_true",
                   help="g_pool = concat([mean, max, attn(θ)]) (6h) au lieu "
                        "d'une seule stratégie (2h) — désactivé par défaut.")
    p.add_argument("--use_factored_head", action="store_true",
                   help="λ_ik = gauge_k + resid_ik au lieu de la tête "
                        "partagée [h_v|h_scen|g_pool|ctx] — cf. "
                        "NOTES_N_scaling_investigation.md, 'Piste retenue'. "
                        "Ignore l'effet de --use_rich_pool "
                        "sur la tête (ce flag ne pilote que g_pool, "
                        "inutilisé ici).")
    p.add_argument("--no_center_lam", action="store_true",
                   help="Désactive le centrage de λ (activé par défaut, comme l'ablation N).")
    p.add_argument("--no_c_scale_norm", action="store_true",
                   help="Désactive la normalisation de la loss par c_scale "
                        "(activée par défaut, comme l'ablation N).")
    p.add_argument("--max_norm", type=float, default=2.0,
                   help="Clip du gradient global par pas (0 = pas de clip).")

    # ── Entraînement ────────────────────────────────────────────────────────
    p.add_argument("--n_epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eta_min", type=float, default=1e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=20,
                   help="Fréquence d'enregistrement de l'historique (epochs).")

    # ── Références sous-gradient ────────────────────────────────────────────
    p.add_argument("--skip_subgrad_full", action="store_true",
                   help="Sauter la convergence complète du sous-gradient (lent pour N grand).")
    p.add_argument("--subgrad_full_T_max", type=int, default=500)
    p.add_argument("--subgrad_full_patience", type=int, default=30)
    p.add_argument("--subgrad_full_precision", type=float, default=0.1)

    # ── Sorties ─────────────────────────────────────────────────────────────
    p.add_argument("--out_dir", default=None,
                   help="Dossier de sortie. Défaut : "
                        "<repo>/gnn/runs/overfit_single/N{n_scen}_inst{id}_d{depth}_idx{graph_idx}/")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def apply_center(lam: torch.Tensor, N_z: int, n_g: int) -> torch.Tensor:
    """λ_ik ← λ_ik − (1/N_z) Σ_i λ_ik (centrage sur les scénarios)."""
    lam_2d = lam.view(N_z, n_g)
    return (lam_2d - lam_2d.mean(dim=0, keepdim=True)).reshape(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── 1. Chargement + sélection de l'instance ────────────────────────────
    if args.data_dir:
        src_dir    = Path(args.data_dir)
        inst_file  = src_dir / f"instances_n{args.n_val}_m{args.m_val}_N{args.n_scen}.txt"
        nodes_file = src_dir / f"nodes_n{args.n_val}_m{args.m_val}_N{args.n_scen}.txt"
    else:
        src_dir    = src_dir_for(args.n_scen, REPO_ROOT / "data")
        inst_file  = src_dir / f"instances_n{args.n_val}_m{args.m_val}.txt"
        nodes_file = src_dir / f"nodes_n{args.n_val}_m{args.m_val}.txt"
    print(f"[diag] n_scen={args.n_scen}  ->  {src_dir}  device={device}")
    if not inst_file.exists() or not nodes_file.exists():
        sys.exit(f"[diag] Fichiers manquants dans {src_dir}")

    instances, records = load_raw_nodes(inst_file, nodes_file, max_records=args.max_load)

    valid_ids = valid_inst_ids_for_N(inst_file, args.n_scen)
    records_N = [r for r in records if valid_ids is None or r["inst_id"] in valid_ids]
    if args.max_depth >= 0:
        filtered = [r for r in records_N if r["depth"] <= args.max_depth]
        records_N = filtered if filtered else records_N

    print(f"[diag] {len(records_N)} nœuds disponibles pour N={args.n_scen} "
          f"(depth<={args.max_depth if args.max_depth >= 0 else 'inf'})")
    if not records_N:
        sys.exit("[diag] Aucun nœud trouvé — augmentez --max_load ou --max_depth.")

    ds = LazyGraphDataset(instances, records_N)
    g  = ds[args.graph_idx]
    inst_id = records_N[args.graph_idx]["inst_id"]
    depth   = int(g.context[0].item())
    print(f"[diag] Instance choisie : inst_id={inst_id}  depth={depth}  "
          f"n_g={g.n_g}  N_z={g.N_z}/{g.N}  m={g.m}  opt={g.opt.item():.2f}")

    model_suffix = "" if args.model_type == "gnn" else f"_{args.model_type}"
    out_dir = (Path(args.out_dir) if args.out_dir else
               REPO_ROOT / "gnn" / "runs" / "overfit_single" /
               f"N{args.n_scen}_inst{inst_id}_d{depth}_idx{args.graph_idx}{model_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diag] out_dir = {out_dir}")

    g = g.to(device)

    # ── 2. Références : L1(λ=0) et sous-gradient ───────────────────────────
    with torch.no_grad():
        lam0    = torch.zeros(g.N_z * g.n_g, device=device)
        l1_init = lagrangian_v1ld(lam0, g).item()

    opt_val      = g.opt.item()
    curr_cost    = g.context[3].item()
    opt_residual = opt_val - curr_cost
    print(f"[diag] L1(λ=0)={l1_init:.2f}  opt={opt_val:.2f}  opt_résiduel={opt_residual:.2f}")

    g_cpu    = g.cpu()
    l1_ref50 = verify_v1ld(g_cpu, T=50)
    print(f"[diag] Sous-gradient T=50 = {l1_ref50:.2f}  "
          f"({(l1_init - l1_ref50) / l1_init * 100:.1f}% vs L1(0))")

    l1_conv     = None
    lam_conv_2d = None
    if not args.skip_subgrad_full:
        l1_conv, n_iters_conv, lam_conv = verify_v1ld(
            g_cpu, T=args.subgrad_full_T_max,
            precision=args.subgrad_full_precision,
            patience=args.subgrad_full_patience,
            return_n_iters=True, return_lam=True,
        )
        lam_conv_2d = lam_conv.reshape(int(g.N_z), int(g.n_g))
        print(f"[diag] Sous-gradient convergé ({n_iters_conv} it., "
              f"prec={args.subgrad_full_precision}, pat={args.subgrad_full_patience}) "
              f"= {l1_conv:.2f}")
    else:
        print("[diag] ATTENTION : --skip_subgrad_full actif — la baseline de "
              "convergence sous-gradient n'apparaîtra pas sur la figure.")

    # ── 3. Modèle (config ablation N par défaut) ───────────────────────────
    # NB : la classe GNN (train_gnn_v1ld.py) a été simplifiée (refonte
    # n/N-agnostique) et n'accepte plus que m/hidden/use_max_pool — les flags
    # ci-dessous sont désormais sans effet sur le modèle.
    if args.use_rich_pool or args.use_factored_head:
        print("[diag] ATTENTION : --use_rich_pool / --use_factored_head "
              "ignorés — non supportés par la classe GNN actuelle "
              "(architecture simplifiée, cf. refonte n/N-agnostique).")
    model_cfg = dict(
        m=g.m, hidden=args.hidden,
        use_max_pool=not args.no_max_pool,
    )
    center_lam   = not args.no_center_lam
    c_scale_norm = not args.no_c_scale_norm
    if args.use_factored_head and center_lam:
        print("[diag] use_factored_head=True : centrage géré en interne "
              "(résidu seulement, gauge_k libre) — désactivation du "
              "centrage externe apply_center, qui annulerait gauge_k "
              "(constant sur l'axe scénario, donc supprimé par une "
              "moyenne sur ce même axe).")
        center_lam = False

    model_cls = GNN if args.model_type == "gnn" else DeepSetsGNN
    model = model_cls(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[diag] model_type={args.model_type} : {n_params:,} paramètres  cfg={model_cfg}  "
          f"center_lam={center_lam}  c_scale_norm={c_scale_norm}  max_norm={args.max_norm}")

    # ── 4. Entraînement ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=args.eta_min)

    hist_epochs, hist_l1, hist_lam_mean, hist_gnorm, hist_lr = [], [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lam = model(g)
        if center_lam:
            lam = apply_center(lam, int(g.N_z), int(g.n_g))
        c_norm = max(float(g.c_scale), 1.0) if c_scale_norm else 1.0
        loss = lagrangian_v1ld(lam, g) / c_norm
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=args.max_norm if args.max_norm > 0 else float("inf"))
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            l1_raw   = lagrangian_v1ld(lam, g).item()  # borne réelle (non normalisée)
            lam_mean = lam.abs().mean().item()

        if epoch % args.eval_every == 0 or epoch == 1:
            hist_epochs.append(epoch)
            hist_l1.append(l1_raw)
            hist_lam_mean.append(lam_mean)
            hist_gnorm.append(gnorm.item())
            hist_lr.append(optimizer.param_groups[0]["lr"])

            gap_ref50 = (l1_raw - l1_ref50) / abs(l1_ref50) * 100 if l1_ref50 else float("nan")
            print(f"ep {epoch:5d}/{args.n_epochs}  L1={l1_raw:10.3f}  "
                  f"gap_vs_T50={gap_ref50:+6.1f}%  |λ|mean={lam_mean:7.4f}  "
                  f"gnorm={gnorm.item():7.3f}  lr={optimizer.param_groups[0]['lr']:.2e}",
                  flush=True)

    final_l1 = hist_l1[-1]
    print(f"\n[diag] Borne finale : {final_l1:.2f}")
    print(f"  vs L1(λ=0)          : {l1_init:.2f}  ({(l1_init - final_l1) / l1_init * 100:+.1f}%)")
    print(f"  vs sous-grad T=50   : {l1_ref50:.2f}  "
          f"({'mieux' if final_l1 < l1_ref50 else 'moins bon'})")
    if l1_conv is not None:
        print(f"  vs sous-grad convergé: {l1_conv:.2f}  "
              f"({'mieux' if final_l1 < l1_conv else 'moins bon'})")

    # ── 5. Warm-start (GNN + quelques it. sous-gradient) ───────────────────
    model.eval()
    with torch.no_grad():
        lam_final = model(g)
        if center_lam:
            lam_final = apply_center(lam_final, int(g.N_z), int(g.n_g))
        lam_gnn_np = lam_final.cpu().numpy().reshape(int(g.N_z), int(g.n_g)).astype(np.float64)
    l1_warm_5 = verify_v1ld(g_cpu, T=5, lam_init=lam_gnn_np)
    print(f"  GNN + 5 it. subgrad [warm start] : {l1_warm_5:.2f}")

    # Balayage warm-start à plusieurs profondeurs T — teste si affiner le
    # point de départ GNN avec quelques dizaines d'itérations de sous-gradient
    # (peu coûteuses) suffit à passer sous l1_conv, en alternative à
    # continuer à augmenter n_epochs (rendements décroissants, cf. discussion).
    l1_warm_sweep = {}
    for T_warm in (10, 20, 50, 100):
        l1_warm_sweep[T_warm] = verify_v1ld(g_cpu, T=T_warm, lam_init=lam_gnn_np)
    warm_str = "  ".join(f"T={t}:{v:.2f}" for t, v in l1_warm_sweep.items())
    print(f"  Warm-start sweep : {warm_str}")

    lam_gnn_path = out_dir / "lam_gnn_final.npy"
    np.save(lam_gnn_path, lam_gnn_np)

    # ── 5b. Décomposition gauge/résidu : diversité de λ entre scénarios ─────
    # Teste l'hypothèse "goulot de pooling" (structurel) : cf. commentaire
    # équivalent dans gnn_bipartite/diag_overfit_single.py.
    lam_diversity = None
    if lam_conv_2d is not None:
        def _gauge_resid(lam_2d: np.ndarray) -> tuple[float, float]:
            gauge = lam_2d.mean(axis=0)
            resid = lam_2d - gauge[None, :]
            return float(np.abs(gauge).mean()), float(np.abs(resid).mean())

        gauge_gnn,  resid_gnn  = _gauge_resid(lam_gnn_np)
        gauge_conv, resid_conv = _gauge_resid(lam_conv_2d)
        ratio = resid_gnn / resid_conv if resid_conv > 0 else float("nan")
        lam_diversity = {
            "gauge_gnn": gauge_gnn, "resid_gnn": resid_gnn,
            "gauge_conv": gauge_conv, "resid_conv": resid_conv,
            "resid_ratio_gnn_over_conv": ratio,
        }
        print(f"  Diversité λ (résidu scénario-spécifique) : "
              f"GNN={resid_gnn:.5f}  sous-grad_conv={resid_conv:.5f}  "
              f"ratio={ratio:.3f}  (gauge GNN={gauge_gnn:.4f} / conv={gauge_conv:.4f})")
        if ratio < 0.5:
            print("  -> ratio << 1 : le GNN sous-différencie fortement les scénarios "
                  "(signal en faveur du goulot de pooling structurel).")

    # ── 6. Figure : progression de la borne ────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax1.plot(hist_epochs, hist_l1, color="steelblue", lw=1.5, marker=".", ms=3,
             label="L1(λ_GNN)")
    ax1.axhline(l1_init, color="orange", lw=1.5, ls="--", label=f"L1(λ=0) = {l1_init:.0f}")
    ax1.axhline(l1_ref50, color="gray", lw=1.2, ls=":",
                label=f"Sous-gradient T=50 (cold start) = {l1_ref50:.0f}")
    # Baseline principale (comme experiment_gnn_v1LD_base.ipynb) : convergence
    # sous-gradient à patience/précision données — toujours affichée sauf si
    # --skip_subgrad_full a été explicitement passé.
    if l1_conv is not None:
        ax1.axhline(l1_conv, color="red", lw=1.8, ls="--", zorder=4,
                    label=f"Sous-gradient convergé (prec={args.subgrad_full_precision}, "
                          f"pat={args.subgrad_full_patience}) = {l1_conv:.0f}")
    else:
        ax1.text(0.02, 0.02,
                  "ATTENTION : baseline de convergence sous-gradient non calculée "
                  "(--skip_subgrad_full)",
                  transform=ax1.transAxes, fontsize=8, color="red", va="bottom")
    ax1.axhline(opt_residual, color="green", lw=1.5, ls=":",
                label=f"opt_résiduel = {opt_residual:.0f}")
    ax1.scatter([hist_epochs[-1]], [final_l1], color="steelblue", s=60, zorder=5)
    ax1.annotate(f"  {final_l1:.0f}", (hist_epochs[-1], final_l1),
                 va="center", fontsize=9, color="steelblue")
    ax1.set_ylabel("L1(λ) résiduelle — borne supérieure")
    ax1.set_title(
        f"Overfit instance unique — N={args.n_scen}, depth={depth}, n_g={g.n_g}, "
        f"N_z={g.N_z}, m={g.m}\n"
        f"GNN config ablation-N (hidden={args.hidden}, "
        f"max_pool={model_cfg['use_max_pool']}, center_lam={center_lam}), "
        f"lr={args.lr}, epochs={args.n_epochs}"
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(hist_epochs, hist_lr, color="gray", lw=1.5)
    ax2b = ax2.twinx()
    ax2b.plot(hist_epochs, hist_gnorm, color="firebrick", lw=1.0, alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LR", color="gray")
    ax2b.set_ylabel("‖grad‖", color="firebrick")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "fig_convergence.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[diag] Figure sauvegardée : {fig_path}")

    # ── 7. Metrics JSON ─────────────────────────────────────────────────────
    metrics = {
        "args": vars(args),
        "instance": {
            "inst_id": inst_id, "depth": depth, "n_g": g.n_g, "N_z": int(g.N_z),
            "N": int(g.N), "m": g.m, "opt": opt_val, "curr_cost": curr_cost,
            "opt_residual": opt_residual,
        },
        "model_type": args.model_type,
        "model_cfg": model_cfg,
        "center_lam": center_lam, "c_scale_norm": c_scale_norm,
        "references": {
            "l1_init": l1_init, "l1_ref50": l1_ref50, "l1_conv": l1_conv,
            "l1_warm_5": l1_warm_5,
            "l1_warm_sweep": {str(t): v for t, v in l1_warm_sweep.items()},
        },
        "final_l1": final_l1,
        "lam_diversity": lam_diversity,
        "lam_gnn_path": str(lam_gnn_path),
        "history": {
            "epochs": hist_epochs, "l1": hist_l1, "lam_mean": hist_lam_mean,
            "gnorm": hist_gnorm, "lr": hist_lr,
        },
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[diag] Metrics sauvegardées : {metrics_path}")
    print(f"\n[diag] Tout est dans : {out_dir}")


if __name__ == "__main__":
    main()
