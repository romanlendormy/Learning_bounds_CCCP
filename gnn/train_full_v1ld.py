#!/usr/bin/env python3
"""
train_full_v1ld.py — Entraînement GNN V1LD sur dataset complet.

Usage :
    python train_full_v1ld.py [options]

Exemples :
    # local
    python gnn/train_full_v1ld.py --dataset train_v1LD --n_epochs 50

    # Compute Canada
    python train_full_v1ld.py \\
        --dataset train_v1LD --n_val 10 --n_epochs 200 \\
        --data_dir $SLURM_TMPDIR/data --out_dir $SCRATCH/results/run1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # backend non-interactif (cluster sans affichage)
import matplotlib.pyplot as plt

# Racine du repo = parent du dossier gnn/
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn.train_gnn_v1ld import load_dataset, GNN, lagrangian_v1ld


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Entraîne le GNN V1LD sur l'intégralité d'un dataset collecté.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    g_data = p.add_argument_group("Dataset")
    g_data.add_argument("--dataset",  default="train_v1LD",
                        choices=["train_v1LD", "train_base"],
                        help="Sous-dossier du dataset.")
    g_data.add_argument("--n_val",    type=int, default=10,
                        help="Taille des instances (suffixe des fichiers).")
    g_data.add_argument("--data_dir", default=None,
                        help="Dossier contenant train_v1LD/ ou train_base/. "
                             "Défaut : <repo>/data/")
    g_data.add_argument("--out_dir",  default=None,
                        help="Dossier de sortie (checkpoints, figures, métriques). "
                             "Défaut : <repo>/gnn/")

    # Split
    g_split = p.add_argument_group("Split train/test")
    g_split.add_argument("--train_frac", type=float, default=0.8,
                         help="Fraction des instances allouées au train.")
    g_split.add_argument("--seed", type=int, default=42)

    # Entraînement
    g_train = p.add_argument_group("Entraînement")
    g_train.add_argument("--n_epochs",   type=int,   default=50,
                         help="Nombre d'epochs (1 epoch = 1 pass complet du train set).")
    g_train.add_argument("--batch_size", type=int,   default=32,
                         help="Gradient accumulation : 1 step optimizer par B graphes.")
    g_train.add_argument("--lr",         type=float, default=1e-3)
    g_train.add_argument("--eta_min",    type=float, default=1e-5,
                         help="LR minimal pour le scheduler cosine.")
    g_train.add_argument("--hidden",     type=int,   default=64,
                         help="Dimension cachée du GNN.")
    g_train.add_argument("--resume",     default=None,
                         help="Chemin vers un checkpoint .pt pour reprendre.")

    # Logging / checkpoint
    g_log = p.add_argument_group("Logging")
    g_log.add_argument("--log_every",  type=int, default=5,
                       help="Afficher la loss tous les N epochs.")
    g_log.add_argument("--save_every", type=int, default=10,
                       help="Sauvegarder un checkpoint tous les N epochs (0=off).")

    # Évaluation
    g_eval = p.add_argument_group("Évaluation")
    g_eval.add_argument("--eval_every",         type=int, default=10,
                        help="Évaluer le gap test tous les N epochs (0=off).")
    g_eval.add_argument("--eval_sample_size",   type=int, default=200,
                        help="Graphes test pour l'éval intermédiaire.")
    g_eval.add_argument("--final_eval_n_train", type=int, default=1000,
                        help="Graphes train pour l'éval finale (0 = tout le train).")

    # Device
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "cuda", "auto"])

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_l1_zero(graphs, device):
    """L1(λ=0) pour chaque graphe, sans backward."""
    out = []
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            lam = torch.zeros(g.N_z * g.n_g, device=device)
            out.append(lagrangian_v1ld(lam, g).item())
    return np.array(out)


def eval_gap(graphs, l1_0s, model, device):
    """
    gaps[i] = (curr_cost + L1_GNN − opt) / opt × 100   (≥0 si borne valide)
    imps[i] = (L1_0 − L1_GNN) / L1_0 × 100            (>0 = GNN améliore vs λ=0)
    """
    model.eval()
    gaps, imps = [], []
    with torch.no_grad():
        for g, l0 in zip(graphs, l1_0s):
            g      = g.to(device)
            lam    = model(g)
            l1_gnn = lagrangian_v1ld(lam, g).item()
            curr_cost = g.context[3].item()
            opt       = g.opt.item()
            gaps.append((curr_cost + l1_gnn - opt) / max(abs(opt), 1e-9) * 100)
            imps.append((l0 - l1_gnn)              / max(abs(l0),  1e-9) * 100)
    return np.array(gaps), np.array(imps)


def stats(arr):
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90":    float(np.percentile(arr, 90)),
        "max":    float(np.max(arr)),
        "min":    float(np.min(arr)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Chemins ───────────────────────────────────────────────────────────────
    data_root = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data"
    out_dir   = Path(args.out_dir)  if args.out_dir  else REPO_ROOT / "gnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix     = "_gbab" if args.dataset == "train_v1LD" else ""
    inst_file  = data_root / args.dataset / f"instances_n{args.n_val}{suffix}.txt"
    nodes_file = data_root / args.dataset / f"nodes_n{args.n_val}{suffix}.txt"

    run_name = (f"{args.dataset}_n{args.n_val}"
                f"_ep{args.n_epochs}_lr{args.lr}_bs{args.batch_size}"
                f"_h{args.hidden}_seed{args.seed}")
    ckpt_path    = out_dir / f"ckpt_{run_name}.pt"
    metrics_path = out_dir / f"metrics_{run_name}.json"
    fig_path     = out_dir / f"fig_{run_name}.png"

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    log(f"run       : {run_name}")
    log(f"device    : {device}")
    log(f"inst_file : {inst_file}")
    log(f"out_dir   : {out_dir}")

    assert inst_file.exists(),  f"Introuvable : {inst_file}"
    assert nodes_file.exists(), f"Introuvable : {nodes_file}"

    # ── Chargement ────────────────────────────────────────────────────────────
    log("Chargement du dataset...")
    t0 = time.time()
    all_graphs = load_dataset(inst_file, nodes_file)
    log(f"  {len(all_graphs)} graphes chargés en {time.time()-t0:.1f}s")

    # ── Split par inst_id ─────────────────────────────────────────────────────
    inst_ids = np.array(sorted({int(g.inst_id.item()) for g in all_graphs}))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(inst_ids)
    n_tr = int(len(inst_ids) * args.train_frac)
    train_ids = set(inst_ids[:n_tr].tolist())
    test_ids  = set(inst_ids[n_tr:].tolist())
    assert not (train_ids & test_ids), "Fuite train/test !"

    train_graphs = [g for g in all_graphs if int(g.inst_id.item()) in train_ids]
    test_graphs  = [g for g in all_graphs if int(g.inst_id.item()) in test_ids]
    log(f"split {args.train_frac:.0%}/{1-args.train_frac:.0%} — "
        f"train={len(train_ids)} inst / {len(train_graphs)} graphes  |  "
        f"test={len(test_ids)} inst / {len(test_graphs)} graphes")

    # ── Baseline L1(λ=0) — pré-calculé une seule fois ────────────────────────
    log(f"Pré-calcul L1(λ=0) — test ({len(test_graphs)} graphes)...")
    t0 = time.time()
    l1_0_test = compute_l1_zero(test_graphs, device)
    log(f"  {time.time()-t0:.1f}s  mean={np.mean(l1_0_test):.2f}")

    n_eval_tr = args.final_eval_n_train or len(train_graphs)
    n_eval_tr = min(n_eval_tr, len(train_graphs))
    rng2 = np.random.default_rng(args.seed + 1)
    eval_tr_idx = rng2.choice(len(train_graphs), n_eval_tr, replace=False)
    eval_tr     = [train_graphs[i] for i in eval_tr_idx]

    log(f"Pré-calcul L1(λ=0) — train sample ({n_eval_tr} graphes)...")
    t0 = time.time()
    l1_0_train = compute_l1_zero(eval_tr, device)
    log(f"  {time.time()-t0:.1f}s  mean={np.mean(l1_0_train):.2f}")

    # ── Modèle ────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    model     = GNN(m=train_graphs[0].m, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=args.eta_min)

    start_epoch = 1
    if args.resume:
        log(f"Reprise depuis {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        log(f"  reprise à l'epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    log(f"GNN — m={train_graphs[0].m}  hidden={args.hidden}  "
        f"params={n_params:,}")
    log(f"Entraînement — {args.n_epochs} epochs  "
        f"batch_size={args.batch_size}")

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    history_loss     = []   # (epoch, mean_loss)
    history_test_gap = []   # (epoch, mean_gap, median_gap)
    indices  = np.arange(len(train_graphs))
    t_start  = time.time()

    for epoch in range(start_epoch, args.n_epochs + 1):
        model.train()
        np.random.shuffle(indices)
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step_i, idx in enumerate(indices):
            g    = train_graphs[idx].to(device)
            lam  = model(g)
            loss = lagrangian_v1ld(lam, g)
            (loss / args.batch_size).backward()
            epoch_loss += loss.item()

            if (step_i + 1) % args.batch_size == 0 or step_i == len(indices) - 1:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        mean_loss = epoch_loss / len(train_graphs)
        history_loss.append((epoch, mean_loss))

        if args.log_every and epoch % args.log_every == 0:
            elapsed   = time.time() - t_start
            done      = epoch - start_epoch + 1
            remaining = elapsed / done * (args.n_epochs - epoch)
            log(f"epoch {epoch:4d}/{args.n_epochs}  "
                f"loss={mean_loss:9.2f}  "
                f"lr={optimizer.param_groups[0]['lr']:.1e}  "
                f"elapsed={elapsed/60:.1f}min  reste~{remaining/60:.0f}min")

        if args.eval_every and epoch % args.eval_every == 0:
            _s  = test_graphs[:args.eval_sample_size]
            _l0 = l1_0_test[:args.eval_sample_size]
            g_arr, i_arr = eval_gap(_s, _l0, model, device)
            history_test_gap.append(
                (epoch, float(np.mean(g_arr)), float(np.median(g_arr))))
            log(f"  eval gap test  mean={np.mean(g_arr):+.1f}%  "
                f"median={np.median(g_arr):+.1f}%  "
                f"imp_L0={np.mean(i_arr):+.1f}%")
            model.train()

        if args.save_every and epoch % args.save_every == 0:
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args)}, ckpt_path)

    total_time = time.time() - t_start
    log(f"Entraînement terminé en {total_time/60:.1f} min")
    torch.save({"epoch": args.n_epochs,
                "model": model.state_dict(),
                "args":  vars(args)}, ckpt_path)
    log(f"Checkpoint final : {ckpt_path}")

    # ── Évaluation finale ─────────────────────────────────────────────────────
    model.eval()
    log(f"Éval finale — test ({len(test_graphs)} graphes)...")
    t0 = time.time()
    gaps_test, imps_test = eval_gap(test_graphs, l1_0_test, model, device)
    log(f"  {time.time()-t0:.1f}s")

    log(f"Éval finale — train sample ({n_eval_tr} graphes)...")
    t0 = time.time()
    gaps_train, imps_train = eval_gap(eval_tr, l1_0_train, model, device)
    log(f"  {time.time()-t0:.1f}s")

    metrics = {
        "run_name": run_name,
        "args": vars(args),
        "n_train_graphs": len(train_graphs),
        "n_test_graphs":  len(test_graphs),
        "train_time_min": round(total_time / 60, 2),
        "history_loss":     history_loss,
        "history_test_gap": history_test_gap,
        "gap_train":    stats(gaps_train),
        "gap_test":     stats(gaps_test),
        "imp_l0_train": stats(imps_train),
        "imp_l0_test":  stats(imps_test),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Métriques JSON : {metrics_path}")

    # Résumé console
    print()
    print(f"{'':6}  {'mean gap':>10}  {'median gap':>11}  "
          f"{'p90 gap':>9}  {'imp L0':>9}")
    print("─" * 55)
    for label, gaps, imps in [("TRAIN", gaps_train, imps_train),
                               ("TEST ", gaps_test,  imps_test)]:
        s = stats(gaps)
        print(f"{label}  {s['mean']:+9.2f}%  {s['median']:+10.2f}%  "
              f"{s['p90']:+8.2f}%  {np.mean(imps):+8.2f}%")
    print(f"\n  fraction valide (gap≥0) : "
          f"train={(gaps_train >= 0).mean()*100:.1f}%  "
          f"test={(gaps_test   >= 0).mean()*100:.1f}%")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax  = axes[0]
    ax2 = ax.twinx()
    ep_h  = [h[0] for h in history_loss]
    l1_h  = [h[1] for h in history_loss]
    W     = max(1, len(l1_h) // 15)
    l1_sm = np.convolve(l1_h, np.ones(W) / W, mode="valid")
    ax.plot(ep_h, l1_h, alpha=0.25, color="steelblue", lw=0.8)
    ax.plot(ep_h[W-1:], l1_sm, color="steelblue", lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("L1(λ_GNN) moyen", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    if history_test_gap:
        ep_tg = [h[0] for h in history_test_gap]
        mg_tg = [h[1] for h in history_test_gap]
        ax2.plot(ep_tg, mg_tg, "r--o", ms=5, lw=1.5, label="gap test moyen (%)")
        ax2.set_ylabel("gap test (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Convergence"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.boxplot([gaps_train, gaps_test],
               labels=[f"Train\n({n_eval_tr})", f"Test\n({len(test_graphs)})"],
               patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.45),
               medianprops=dict(color="red", lw=2),
               flierprops=dict(marker=".", ms=3, alpha=0.4))
    ax.axhline(0, color="green", ls="--", lw=1.5, label="opt (gap=0)")
    ax.set_ylabel("(bound_global − opt) / opt  ×100 (%)")
    ax.set_title("Gap relatif (≥0 = borne valide)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[2]
    all_i = np.concatenate([imps_train, imps_test])
    lo, hi = np.percentile(all_i, 1), np.percentile(all_i, 99)
    bins = np.linspace(lo, hi, 45)
    ax.hist(imps_train, bins=bins, alpha=0.55, color="steelblue",
            label=f"Train  μ={np.mean(imps_train):+.1f}%", density=True)
    ax.hist(imps_test,  bins=bins, alpha=0.55, color="tomato",
            label=f"Test   μ={np.mean(imps_test):+.1f}%",  density=True)
    ax.axvline(0, color="black", ls="--", lw=1.2, label="λ=0 baseline")
    ax.set_xlabel("(L1₀ − L1_GNN) / L1₀  ×100 (%)")
    ax.set_ylabel("Densité")
    ax.set_title("Amélioration vs L1(λ=0)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"GNN V1LD — {args.dataset}  n={args.n_val}  "
        f"ep={args.n_epochs}  lr={args.lr}  bs={args.batch_size}  h={args.hidden}",
        fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    log(f"Figure : {fig_path}")


if __name__ == "__main__":
    main()
