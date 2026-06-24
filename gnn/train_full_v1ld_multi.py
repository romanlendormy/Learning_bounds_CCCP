#!/usr/bin/env python3
"""
train_full_v1ld_multi.py — Entraînement GNN V1LD multi-dataset (n=10,15,20).

Charge tous les fichiers instances_n{n}_m{m}.txt / nodes_n{n}_m{m}.txt du
dossier train_base_s30/ pour les valeurs de n indiquées (même m, GNN fixé à m).

Fonctionnalités :
  - Multi-dataset  : mélange n=10/15/20, N variés dans le même entraînement
  - Checkpoint     : sauvegarde à chaque epoch, conserve les K derniers
  - Auto-resume    : détecte automatiquement le dernier checkpoint du run
  - Gap par epoch  : gap relatif moyen sur nœuds racines test, loggé chaque epoch
  - Historique     : loss + gap écrits en JSON après chaque epoch (safe-write)

Usage :
    python gnn/train_full_v1ld_multi.py --n_list 10 15 20 --n_epochs 100
    sbatch run_train_gnn_multi.sh
"""

from __future__ import annotations

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
sys.path.insert(0, str(REPO_ROOT))

from gnn.train_gnn_v1ld import (
    load_raw_nodes, LazyGraphDataset, EagerGraphDataset, GNN, lagrangian_v1ld,
)
from torch_geometric.loader import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entraîne GNN V1LD sur plusieurs (n, N) simultanément.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Sélection des fichiers ────────────────────────────────────────────────
    # Mode explicite (prioritaire) : paires inst/nodes directement
    p.add_argument("--inst_files",  nargs="+", default=None,
                   help="Chemins explicites vers les fichiers instances (même ordre que --nodes_files).")
    p.add_argument("--nodes_files", nargs="+", default=None,
                   help="Chemins explicites vers les fichiers nodes (même ordre que --inst_files).")
    # Mode auto (fallback) : découverte par n dans train_base_s30/
    p.add_argument("--n_list",    nargs="+", type=int, default=[10, 15, 20],
                   help="(mode auto) Valeurs de n à charger.")
    p.add_argument("--m",         type=int,   default=5,
                   help="Valeur de m (architecture GNN).")
    p.add_argument("--data_dir",  default=None,
                   help="(mode auto) Dossier contenant train_base_s30/. Défaut : <repo>/data/")
    p.add_argument("--out_dir",   default=None,
                   help="Dossier de sortie (checkpoints, métriques). Défaut : <repo>/gnn/runs/")

    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed",       type=int,   default=42)

    p.add_argument("--n_epochs",   type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=32,
                   help="Accumulation de gradient : 1 step optimizer par batch_size graphes.")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--eta_min",    type=float, default=1e-5)
    p.add_argument("--hidden",     type=int,   default=64)

    p.add_argument("--eval_sample", type=int,  default=300,
                   help="Nombre de nœuds racines test pour le gap.")
    p.add_argument("--eval_every",  type=int,  default=5,
                   help="Fréquence (en epochs) du calcul du gap et de l'enregistrement dans l'historique.")
    p.add_argument("--keep_ckpts",  type=int,  default=3,
                   help="Nombre de checkpoints récents à conserver.")
    p.add_argument("--num_workers",  type=int,   default=3)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="L2 régularisation sur les poids (0 = désactivé).")
    p.add_argument("--max_norm",     type=float, default=0.5,
                   help="Norme max pour le clipping du gradient (par instance).")
    p.add_argument("--run_tag",      type=str,   default="",
                   help="Suffixe ajouté au run_name pour distinguer les architectures (ex: v2).")
    p.add_argument("--device",       default="auto",
                   choices=["auto", "cpu", "cuda"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Chargement multi-fichiers
# ─────────────────────────────────────────────────────────────────────────────

def load_multi(file_pairs: list[tuple[Path, Path]]) -> tuple[dict, list]:
    """
    Charge plusieurs (inst_file, nodes_file), fusionne en renumérisant les
    inst_id pour éviter les collisions entre fichiers.
    """
    merged_instances: dict = {}
    merged_records:   list = []

    for inst_file, nodes_file in file_pairs:
        instances, records = load_raw_nodes(inst_file, nodes_file)
        offset = max(merged_instances.keys(), default=-1) + 1
        id_map = {old: old + offset for old in instances}
        for old_id, inst in instances.items():
            merged_instances[id_map[old_id]] = inst
        for rec in records:
            nr = dict(rec)
            nr["inst_id"] = id_map[rec["inst_id"]]
            merged_records.append(nr)

    return merged_instances, merged_records


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires dataset
# ─────────────────────────────────────────────────────────────────────────────

def root_subset(dataset: LazyGraphDataset) -> LazyGraphDataset:
    recs = [r for r in dataset.raw_records if r["depth"] == 0]
    return LazyGraphDataset(dataset.instances, recs)


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_l1_zero(graphs: list, device: torch.device) -> np.ndarray:
    """L1(λ=0) sur des graphes déjà transférés sur device."""
    out = []
    with torch.no_grad():
        for g in graphs:
            lam = torch.zeros(g.N_z * g.n_g, device=device)
            out.append(lagrangian_v1ld(lam, g).item())
    return np.array(out)


def eval_epoch(
    graphs_on_device: list,
    l1_zeros: np.ndarray,
    model: GNN,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Évalue le gap et l'amélioration vs λ=0 sur les graphes pré-chargés.

    gap[i]  = (L1_GNN - opt) / |opt| * 100   (pour racines : curr_cost=0)
    impr[i] = (L1_0 - L1_GNN) / |L1_0| * 100

    Retourne (mean_gap, median_gap, mean_impr).
    """
    model.eval()
    gaps, imprs = [], []
    with torch.no_grad():
        for g, l0 in zip(graphs_on_device, l1_zeros):
            lam    = model(g)
            l1_gnn = lagrangian_v1ld(lam, g).item()
            opt    = g.opt.item()
            # curr_cost_raw présent depuis v2 ; fallback sur context[3]*opt pour compatibilité
            if hasattr(g, "curr_cost_raw"):
                curr = g.curr_cost_raw.item()
            else:
                curr = g.context[3].item() * opt
            gaps.append((curr + l1_gnn - opt) / max(abs(opt), 1e-9) * 100)
            imprs.append((l0 - l1_gnn)        / max(abs(l0),  1e-9) * 100)
    model.train()
    return float(np.mean(gaps)), float(np.median(gaps)), float(np.mean(imprs))


def eval_by_n(
    graphs_on_device: list,
    l1_zeros: np.ndarray,
    model: GNN,
    device: torch.device,
) -> dict:
    """Gap et amélioration moyens par taille de problème (n_orig = 10 / 15 / 20)."""
    model.eval()
    by_n: dict = {}
    with torch.no_grad():
        for g, l0 in zip(graphs_on_device, l1_zeros):
            n_key  = int(g.n_orig.item()) if hasattr(g, "n_orig") else -1
            lam    = model(g)
            l1_gnn = lagrangian_v1ld(lam, g).item()
            opt    = g.opt.item()
            if hasattr(g, "curr_cost_raw"):
                curr = g.curr_cost_raw.item()
            else:
                curr = g.context[3].item() * opt
            gap    = (curr + l1_gnn - opt) / max(abs(opt), 1e-9) * 100
            impr   = (l0  - l1_gnn)        / max(abs(l0),  1e-9) * 100
            if n_key not in by_n:
                by_n[n_key] = {"gaps": [], "imprs": []}
            by_n[n_key]["gaps"].append(gap)
            by_n[n_key]["imprs"].append(impr)
    model.train()
    return {
        n: {
            "gap_mean": float(np.mean(v["gaps"])),
            "gap_med":  float(np.median(v["gaps"])),
            "impr":     float(np.mean(v["imprs"])),
            "count":    len(v["gaps"]),
        }
        for n, v in sorted(by_n.items())
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def ckpt_path(out_dir: Path, run_name: str, epoch: int) -> Path:
    return out_dir / f"ckpt_{run_name}_ep{epoch:04d}.pt"


def save_checkpoint(
    out_dir: Path, run_name: str, epoch: int,
    model: GNN, optimizer, scheduler,
    history: dict, args: argparse.Namespace,
    saved: list[Path], keep: int,
    keep_every: int = 10,
) -> None:
    path = ckpt_path(out_dir, run_name, epoch)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history":   history,
        "args":      vars(args),
    }, path)
    # Checkpoints multiples de keep_every : jamais supprimés
    if epoch % keep_every == 0:
        return
    saved.append(path)
    if len(saved) > keep:
        old = saved.pop(0)
        old.unlink(missing_ok=True)


def find_latest_checkpoint(out_dir: Path, run_name: str) -> Path | None:
    ckpts = sorted(out_dir.glob(f"ckpt_{run_name}_ep*.pt"))
    return ckpts[-1] if ckpts else None


def save_history(path: Path, history: dict) -> None:
    """Écriture atomique (tmp + rename) pour éviter la corruption en cas de crash."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data"
    out_dir   = Path(args.out_dir)  if args.out_dir  else REPO_ROOT / "gnn" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── Fichiers à charger ────────────────────────────────────────────────────
    if args.inst_files and args.nodes_files:
        if len(args.inst_files) != len(args.nodes_files):
            log("ERREUR : --inst_files et --nodes_files doivent avoir le même nombre d'éléments.")
            sys.exit(1)
        pairs = [(Path(i), Path(n)) for i, n in zip(args.inst_files, args.nodes_files)]
        tag   = "_".join(Path(nf).stem for nf in args.nodes_files)
    else:
        s30_dir = data_root / "train_base_s30"
        pairs   = []
        for n in sorted(args.n_list):
            inst_f  = s30_dir / f"instances_n{n}_m{args.m}.txt"
            nodes_f = s30_dir / f"nodes_n{n}_m{args.m}.txt"
            if not inst_f.exists() or not nodes_f.exists():
                log(f"  WARN : fichiers absents pour n={n} m={args.m} — ignoré")
                continue
            pairs.append((inst_f, nodes_f))
        ns_str = "-".join(str(n) for n in sorted(args.n_list))
        tag    = f"n{ns_str}_m{args.m}"

    if not pairs:
        log("ERREUR : aucun fichier trouvé.")
        sys.exit(1)

    for inst_f, nodes_f in pairs:
        if not inst_f.exists() or not nodes_f.exists():
            log(f"ERREUR : fichier absent — {inst_f}  ou  {nodes_f}")
            sys.exit(1)
        log(f"  {nodes_f.name}")

    # ── Run name ──────────────────────────────────────────────────────────────
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    run_name = (f"multi_{tag}_ep{args.n_epochs}"
                f"_lr{args.lr}_bs{args.batch_size}_h{args.hidden}_seed{args.seed}{tag_suffix}")
    metrics_path = out_dir / f"metrics_{run_name}.json"
    fig_path     = out_dir / f"fig_{run_name}.png"

    log(f"run    : {run_name}")
    log(f"device : {device}"
        + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    log(f"out    : {out_dir}")

    # ── Chargement & merge ────────────────────────────────────────────────────
    log("Chargement des datasets...")
    t0 = time.time()
    instances, raw_records = load_multi(pairs)
    all_dataset = LazyGraphDataset(instances, raw_records)
    log(f"  {len(instances)} instances  {len(all_dataset)} graphes  "
        f"en {time.time()-t0:.1f}s")

    # ── Split train / test par inst_id ────────────────────────────────────────
    inst_ids = np.array(sorted(instances.keys()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(inst_ids)
    n_tr     = int(len(inst_ids) * args.train_frac)
    train_ids = set(inst_ids[:n_tr].tolist())
    test_ids  = set(inst_ids[n_tr:].tolist())

    train_ds = all_dataset.filter_by_inst_ids(train_ids)
    test_ds  = all_dataset.filter_by_inst_ids(test_ids)
    log(f"split {args.train_frac:.0%}/{1-args.train_frac:.0%} — "
        f"train {len(train_ds)} graphes  test {len(test_ds)} graphes")

    # ── Ensemble d'évaluation : racines test, pré-chargé sur device ──────────
    root_test = root_subset(test_ds)
    n_eval    = min(args.eval_sample, len(root_test))
    root_test = root_test.subset(0, n_eval)
    log(f"Préconstruction eval set ({n_eval} racines test) ...")
    t0 = time.time()
    eager_eval = EagerGraphDataset(root_test)
    eval_graphs = [eager_eval[i].to(device) for i in range(len(eager_eval))]
    log(f"  fait en {time.time()-t0:.1f}s")

    log("Précalcul L1(λ=0) sur eval set ...")
    t0 = time.time()
    l1_zeros = compute_l1_zero(eval_graphs, device)
    log(f"  mean L1(0) = {np.mean(l1_zeros):.2f}  ({time.time()-t0:.1f}s)")

    # ── Modèle ────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    model     = GNN(m=args.m, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=args.eta_min)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"GNN — m={args.m}  hidden={args.hidden}  params={n_params:,}")

    # ── Auto-resume ───────────────────────────────────────────────────────────
    start_epoch = 1
    history: dict = {
        "epochs_loss":    [], "loss":          [], "loss_std":      [],
        "lam_abs_mean":   [], "lam_abs_max":   [],
        "grad_norm_max":  [], "grad_norm_mean": [],
        "epochs_gap":     [], "gap_mean":       [], "gap_median":    [],
        "impr_mean":      [], "gap_by_n":       [],
    }
    saved_ckpts: list[Path] = []

    latest = find_latest_checkpoint(out_dir, run_name)
    if latest:
        log(f"Checkpoint trouvé : {latest.name} — reprise automatique")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        history     = ckpt.get("history", history)
        for _k in ["grad_norm_max", "grad_norm_mean", "loss_std",
                   "lam_abs_mean", "lam_abs_max", "gap_by_n"]:
            if _k not in history:
                history[_k] = []
        # Reconstruire la liste rolling (exclut les périodiques, jamais supprimés)
        saved_ckpts = [
            p for p in sorted(out_dir.glob(f"ckpt_{run_name}_ep*.pt"))
            if int(p.stem.rsplit("ep", 1)[-1]) % 10 != 0
        ]
        log(f"  reprise epoch {start_epoch}  "
            f"({len(saved_ckpts)} checkpoints existants)")
    else:
        log("Aucun checkpoint — démarrage from scratch")

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )
    log(f"DataLoader — {len(train_ds)} graphes  "
        f"workers={args.num_workers}  pin_memory={device.type == 'cuda'}")
    log(f"Entraînement — {args.n_epochs} epochs  batch_size={args.batch_size}  "
        f"lr={args.lr}  eta_min={args.eta_min}")
    log("")

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    t_start = time.time()

    # ── Boucle principale ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.n_epochs + 1):
        model.train()
        epoch_loss    = 0.0
        epoch_loss_sq = 0.0               # pour l'écart-type inter-graphes
        accum_count   = 0
        accum_grads   = None              # gradients clippés accumulés par instance
        epoch_gnorm   = 0.0               # max norme gradient pré-clip (par instance)
        epoch_gnorm_sum = 0.0             # pour la moyenne
        epoch_nsteps  = 0                 # nb de steps optimizer
        lam_abs_sum   = 0.0               # Σ mean(|λ|) — un scalaire par graphe
        lam_abs_max   = 0.0               # max global |λ| sur l'epoch
        loss_by_n: dict  = {}             # loss cumulée  par n_orig
        count_by_n: dict = {}             # nb de graphes par n_orig

        t_load = t_transfer = t_forward = t_loss = t_backward = t_optim = 0.0
        t_ep   = time.perf_counter()
        t_iter = t_ep
        n_graphs  = len(train_ds)
        log_every = max(1, n_graphs // 10)   # ≈ 10 lignes de progress par epoch

        for step_i, g in enumerate(train_loader):
            t0 = time.perf_counter();  t_load     += t0 - t_iter

            g = g.to(device, non_blocking=True); _sync()
            t1 = time.perf_counter();  t_transfer += t1 - t0

            # Gradient isolé par instance (zero_grad avant chaque backward)
            optimizer.zero_grad()
            lam = model(g); _sync()
            t2 = time.perf_counter();  t_forward  += t2 - t1

            # λ stats — détaché, O(n_g × N_z) en mémoire, négligeable
            with torch.no_grad():
                lam_d = lam.detach().abs()
                lam_abs_sum += lam_d.mean().item()
                lam_abs_max  = max(lam_abs_max, lam_d.max().item())

            # ── Normalisation par opt : rend la loss ≈ O(1) quelle que soit la
            #    taille du graphe (n=10/15/20) et évite l'amplification par N.
            opt_val = max(abs(g.opt.item()), 1.0)
            loss = lagrangian_v1ld(lam, g) / opt_val; _sync()
            t3 = time.perf_counter();  t_loss     += t3 - t2

            loss.backward(); _sync()
            t4 = time.perf_counter();  t_backward += t4 - t3

            # Clip par instance (avant accumulation) — borne max_norm par graphe
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            epoch_gnorm      = max(epoch_gnorm, gn.item())
            epoch_gnorm_sum += gn.item()

            # Accumuler les gradients clippés
            if accum_grads is None:
                accum_grads = [p.grad.detach().clone() if p.grad is not None else None
                               for p in model.parameters()]
            else:
                for j, p in enumerate(model.parameters()):
                    if p.grad is not None and accum_grads[j] is not None:
                        accum_grads[j] += p.grad.detach()

            lv             = loss.item()
            epoch_loss    += lv
            epoch_loss_sq += lv * lv
            accum_count   += 1
            n_key = int(g.n_orig.item()) if hasattr(g, "n_orig") else 0
            loss_by_n[n_key]  = loss_by_n.get(n_key,  0.0) + lv
            count_by_n[n_key] = count_by_n.get(n_key, 0)   + 1
            is_last = (step_i == n_graphs - 1)

            if (step_i + 1) % args.batch_size == 0 or is_last:
                # Appliquer la moyenne des gradients clippés
                optimizer.zero_grad()
                for p, acc in zip(model.parameters(), accum_grads):
                    if acc is not None:
                        p.grad = acc / accum_count
                epoch_nsteps += 1
                optimizer.step(); _sync()
                t5 = time.perf_counter(); t_optim += t5 - t4; t_iter = t5
                accum_grads = None
                accum_count = 0
            else:
                t_iter = t4

            # Progress intra-epoch — 10 lignes par epoch (≈ toutes les 2-3 min)
            if (step_i + 1) % log_every == 0:
                pct     = 100.0 * (step_i + 1) / n_graphs
                avg_l   = epoch_loss  / (step_i + 1)
                avg_lam = lam_abs_sum / (step_i + 1)
                t_now   = time.perf_counter()
                log(f"  [{pct:5.1f}%  {step_i+1:>6}/{n_graphs}]"
                    f"  loss={avg_l:.4f}"
                    f"  |λ|avg={avg_lam:.3f}  |λ|max={lam_abs_max:.2f}"
                    f"  t={t_now - t_ep:.0f}s")

        scheduler.step()
        mean_loss     = epoch_loss / n_graphs
        loss_std      = (max(epoch_loss_sq / n_graphs - mean_loss ** 2, 0.0)) ** 0.5
        lam_mean_ep   = lam_abs_sum / max(n_graphs, 1)
        gnorm_mean_ep = epoch_gnorm_sum / max(epoch_nsteps, 1)
        ep_total      = time.perf_counter() - t_ep

        size_str = "  ".join(
            f"n{n}={loss_by_n[n] / count_by_n[n]:.4f}({count_by_n[n]})"
            for n in sorted(loss_by_n) if count_by_n.get(n, 0) > 0
        )

        # ── Historique loss (chaque epoch) ───────────────────────────────────
        history["epochs_loss"].append(epoch)
        history["loss"].append(mean_loss)
        history["loss_std"].append(loss_std)
        history["lam_abs_mean"].append(lam_mean_ep)
        history["lam_abs_max"].append(lam_abs_max)
        history["grad_norm_max"].append(epoch_gnorm)
        history["grad_norm_mean"].append(gnorm_mean_ep)

        # ── Gap sur racines test (toutes les eval_every epochs) ──────────────
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.n_epochs)
        if do_eval:
            gap_mean, gap_med, impr = eval_epoch(eval_graphs, l1_zeros, model, device)
            gap_by_n = eval_by_n(eval_graphs, l1_zeros, model, device)
            history["epochs_gap"].append(epoch)
            history["gap_mean"].append(gap_mean)
            history["gap_median"].append(gap_med)
            history["impr_mean"].append(impr)
            history["gap_by_n"].append(gap_by_n)

        # ── Log ───────────────────────────────────────────────────────────────
        elapsed   = time.time() - t_start
        done      = epoch - start_epoch + 1
        remaining = elapsed / done * (args.n_epochs - epoch) if done > 0 else 0
        gap_str   = (f"gap={gap_mean:+7.2f}% (med {gap_med:+.2f}%)  impr={impr:+.1f}%  "
                     if do_eval else "")
        log(
            f"ep {epoch:4d}/{args.n_epochs}  "
            f"loss={mean_loss:.4f}±{loss_std:.4f}  "
            f"{gap_str}"
            f"|λ|={lam_mean_ep:.3f}(max={lam_abs_max:.2f})  "
            f"gnorm={epoch_gnorm:.3f}(avg={gnorm_mean_ep:.3f})  "
            f"lr={optimizer.param_groups[0]['lr']:.1e}  "
            f"ep={ep_total:.1f}s  "
            f"reste~{remaining/60:.0f}min"
        )
        log(f"  [sizes ] {size_str}")
        if do_eval:
            by_n_str = "  ".join(
                f"n{n}: gap={v['gap_mean']:+.1f}%(med {v['gap_med']:+.1f}%) impr={v['impr']:+.1f}% ({v['count']})"
                for n, v in sorted(gap_by_n.items())
            )
            log(f"  [eval  ] {by_n_str}")
        log(
            f"  [timing] load={t_load:.1f}s({100*t_load/ep_total:.0f}%)  "
            f"xfer={t_transfer:.1f}s  "
            f"fwd={t_forward:.1f}s({100*t_forward/ep_total:.0f}%)  "
            f"bnb={t_loss:.1f}s({100*t_loss/ep_total:.0f}%)  "
            f"bwd={t_backward:.1f}s({100*t_backward/ep_total:.0f}%)  "
            f"opt={t_optim:.1f}s"
        )

        # ── Checkpoint (keep derniers K) ──────────────────────────────────────
        save_checkpoint(
            out_dir, run_name, epoch,
            model, optimizer, scheduler,
            history, args, saved_ckpts, args.keep_ckpts,
        )

        # ── Sauvegarde incrémentale de l'historique (à chaque eval) ──────────
        if do_eval:
            save_history(metrics_path, {
                "run_name":    run_name,
                "args":        vars(args),
                "n_train":     len(train_ds),
                "n_test":      len(test_ds),
                "n_eval_root": n_eval,
                "history":     history,
            })

    # ── Résumé final ──────────────────────────────────────────────────────────
    total_min = (time.time() - t_start) / 60
    log(f"Entraînement terminé en {total_min:.1f} min")
    log(f"Dernier checkpoint : {saved_ckpts[-1] if saved_ckpts else 'N/A'}")
    log(f"Métriques JSON     : {metrics_path}")

    final_gap  = history["gap_mean"][-1]  if history["gap_mean"]  else float("nan")
    final_loss = history["loss"][-1]      if history["loss"]      else float("nan")
    log(f"Epoch finale — loss_norm={final_loss:.4f}  gap={final_gap:+.2f}%")

    # ── Figure ────────────────────────────────────────────────────────────────
    epochs_loss = history["epochs_loss"]
    epochs_gap  = history["epochs_gap"]
    if len(epochs_loss) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        ax.plot(epochs_loss, history["loss"],
                color="steelblue", lw=1.5, label="L1 train")
        ax.set_xlabel("Epoch"); ax.set_ylabel("L1(λ_GNN)/opt moyen")
        ax.set_title("Loss d'entraînement"); ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        ax.plot(epochs_gap, history["gap_mean"],
                color="tomato", lw=1.5, label="gap moyen (%)")
        ax.plot(epochs_gap, history["gap_median"],
                color="coral", lw=1, ls="--", label="gap médian (%)")
        ax.axhline(0, color="green", ls=":", lw=1.2, label="opt (0%)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Gap relatif (%)")
        ax.set_title("Gap (racines test)"); ax.grid(True, alpha=0.3)
        ax.legend()

        plt.suptitle(f"GNN V1LD multi — {tag}", fontsize=11)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log(f"Figure : {fig_path}")


if __name__ == "__main__":
    main()
