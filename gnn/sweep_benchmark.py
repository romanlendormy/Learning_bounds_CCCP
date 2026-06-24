#!/usr/bin/env python3
"""
sweep_benchmark.py — Sweep de timing : entraînement + inférence GNN V1LD.

Pour chaque (n, m, N) dans {10,15,20} × {5,10,15} × {50,100,250} :
  1. Génère 30 instances CC-MKP (epsilon aléatoire ∈ {0.1, 0.2, 0.3})
     via bab_mkp_gecode_base_collect (Format C).
  2. Sélectionne 30 noeuds racines + 98 sous-noeuds aléatoires = 128 total.
  3. Entraîne GNN V1LD pour 1 epoch (batch_size=32, C++ solver pour ζ_i),
     exactement comme train_full_v1ld.py.
  4. Mesure le temps d'inférence par nœud B&B (forward pass seul, GPU chaud).
  5. Log les timings progressivement et sauvegarde sweep_results.json.

Usage :
    python gnn/sweep_benchmark.py [options]
    sbatch run_sweep_benchmark.sh
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# ── Chemins ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "collect"))

from gnn.train_gnn_v1ld import (          # noqa: E402
    load_raw_nodes,
    LazyGraphDataset,
    EagerGraphDataset,
    GNN,
    lagrangian_v1ld,
    _USE_CPP_BNB,
)
from collect_large_base import generate_song_tempfile  # noqa: E402

# ── Grille de sweep ────────────────────────────────────────────────────────────
NS      = [10, 15, 20]
MS      = [5,  10, 15]
NS_SCEN = [50, 100, 250]
EPSILONS = [0.1, 0.2, 0.3]

N_INST      = 30   # instances à générer par (n, m, N)
N_ROOT_KEEP = 30   # noeuds racines à garder (un par instance)
N_SUB_KEEP  = 98   # sous-noeuds aléatoires à garder
N_TOTAL     = N_ROOT_KEEP + N_SUB_KEEP  # = 128

BATCH_SIZE = 32
LR         = 1e-3
ETA_MIN    = 1e-5
HIDDEN     = 64

N_WARMUP = 10   # forward passes de chauffe avant chronométrage
N_TRIALS = 100  # forward passes chronométrés


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)


def _pct(part: float, total: float) -> str:
    return f"{100 * part / total:.0f}%" if total > 1e-9 else "n/a"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Collecte via C++
# ─────────────────────────────────────────────────────────────────────────────

def collect_instances(
    n: int, m: int, N: int,
    binary: Path, outdir: Path,
    time_limit: int, max_depth: int,
    master_seed: int,
) -> tuple[Path, Path, float]:
    """
    Génère N_INST instances CC-MKP et collecte leurs noeuds B&B.

    Epsilon assigné aléatoirement dans {0.1, 0.2, 0.3} pour chaque instance.
    Le solveur C++ (bab_mkp_gecode_base_collect) produit Format C :
      inst_id, depth, curr_cost, x_mask[n], z_mask[N]

    Retourne (inst_file, nodes_file, elapsed_s).
    """
    rng      = np.random.default_rng(master_seed ^ (n * 100_003 + m * 1_009 + N))
    epsilons = rng.choice(EPSILONS, size=N_INST, replace=True)

    outdir.mkdir(parents=True, exist_ok=True)
    # Le binaire C++ appende "_base" au chemin outdir passé en argument
    outdir_base = Path(str(outdir) + "_base")
    inst_file  = outdir_base / f"instances_n{n}.txt"
    nodes_file = outdir_base / f"nodes_n{n}.txt"

    t0   = time.perf_counter()
    n_ok = 0

    for i in range(N_INST):
        tmpfile = generate_song_tempfile(n, m, N, seed=i)
        cmd = [
            str(binary), tmpfile,
            str(float(epsilons[i])),
            str(time_limit),
            str(outdir),
            str(max_depth),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(tmpfile)
        if r.returncode == 0:
            n_ok += 1
        else:
            log(f"    WARN seed={i} eps={epsilons[i]:.1f}: rc={r.returncode}")

    elapsed = time.perf_counter() - t0
    log(f"  collect: {n_ok}/{N_INST} instances OK en {elapsed:.1f}s — "
        f"{_count_lines(inst_file)} inst, {_count_lines(nodes_file)} noeuds")
    return inst_file, nodes_file, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# 2. Assemblage du dataset 128
# ─────────────────────────────────────────────────────────────────────────────

def build_sweep_dataset(
    inst_file: Path, nodes_file: Path, rng_seed: int
) -> tuple[dict, list]:
    """
    Sélectionne N_ROOT_KEEP racines + N_SUB_KEEP sous-noeuds aléatoires.
    Retourne (instances_dict, raw_records_selected).
    """
    instances, raw_records = load_raw_nodes(inst_file, nodes_file)

    roots    = [r for r in raw_records if r["depth"] == 0]
    subnodes = [r for r in raw_records if r["depth"] >  0]

    rng = np.random.default_rng(rng_seed)

    roots_kept = roots[: min(N_ROOT_KEEP, len(roots))]

    n_sub = min(N_SUB_KEEP, len(subnodes))
    if n_sub > 0:
        idx      = rng.choice(len(subnodes), size=n_sub, replace=False)
        sub_kept = [subnodes[i] for i in sorted(idx)]
    else:
        sub_kept = []

    selected = roots_kept + sub_kept
    log(f"  dataset: {len(roots)} racines, {len(subnodes)} sous-noeuds "
        f"→ {len(roots_kept)} + {len(sub_kept)} = {len(selected)} graphes")
    if len(selected) < N_TOTAL:
        log(f"  NOTE: {len(selected)} < {N_TOTAL} graphes (moins de données que prévu)")

    return instances, selected


# ─────────────────────────────────────────────────────────────────────────────
# 3a. Entraînement : 1 epoch (protocole exact de train_full_v1ld.py)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    instances: dict, raw_records: list, m: int,
    device: torch.device, batch_size: int, num_workers: int,
) -> dict:
    """
    Adam + CosineAnnealingLR.

    Le DataLoader utilise batch_size=1 car lagrangian_v1ld et le pooling
    hiérarchique du GNN supposent un graphe unique (data.n_g et data.N_z
    sont des scalaires). L'accumulation de gradient sur `batch_size` étapes
    donne un batch effectif de 32 pour l'optimiseur — comportement identique
    à un vrai batch_size=32 pour les mises à jour des poids.

    Logs progressifs : imprimés après chaque optimizer.step() (toutes les
    `batch_size` graphes) pour suivre l'avancement même en cas de crash.
    """
    torch.manual_seed(42)
    model     = GNN(m=m, hidden=HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1, eta_min=ETA_MIN)

    dataset = LazyGraphDataset(instances, raw_records)
    loader  = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    model.train()
    optimizer.zero_grad()

    t_load = t_transfer = t_forward = t_loss = t_backward = t_optim = 0.0
    epoch_loss   = 0.0
    batch_loss   = 0.0   # perte cumulée sur le batch courant (pour le log)
    n_graphs     = len(dataset)
    n_opt_steps  = 0     # nombre d'optimizer.step() effectués

    t_start    = time.perf_counter()
    t_iter_end = t_start

    for step_i, g in enumerate(loader):
        t0 = time.perf_counter(); t_load += t0 - t_iter_end

        g = g.to(device); _sync()
        t1 = time.perf_counter(); t_transfer += t1 - t0

        lam = model(g); _sync()
        t2 = time.perf_counter(); t_forward += t2 - t1

        loss = lagrangian_v1ld(lam, g); _sync()
        t3 = time.perf_counter(); t_loss += t3 - t2

        (loss / batch_size).backward(); _sync()
        t4 = time.perf_counter(); t_backward += t4 - t3

        loss_val   = loss.item()
        epoch_loss += loss_val
        batch_loss += loss_val

        is_last = (step_i == n_graphs - 1)
        if (step_i + 1) % batch_size == 0 or is_last:
            optimizer.step(); optimizer.zero_grad(); _sync()
            t5 = time.perf_counter(); t_optim += t5 - t4; t_iter_end = t5

            n_opt_steps  += 1
            done          = step_i + 1
            elapsed       = t5 - t_start
            rate          = done / elapsed if elapsed > 1e-6 else 0.0
            eta           = (n_graphs - done) / rate if rate > 1e-6 else 0.0
            mean_bl       = batch_loss / min(batch_size, done % batch_size or batch_size)
            t_fwd_per     = t_forward  / done
            t_bnb_per     = t_loss     / done
            t_bwd_per     = t_backward / done

            log(f"    [{done:4d}/{n_graphs}] "
                f"{100*done/n_graphs:5.1f}% | "
                f"elapsed={elapsed:6.1f}s | eta~{eta:5.1f}s | "
                f"loss={mean_bl:.2f} | "
                f"fwd={t_fwd_per*1e3:.1f}ms "
                f"bnb={t_bnb_per*1e3:.1f}ms "
                f"bwd={t_bwd_per*1e3:.1f}ms /graph")

            batch_loss = 0.0
        else:
            t_iter_end = t4

    scheduler.step()
    total = time.perf_counter() - t_start

    return {
        "n_graphs":   n_graphs,
        "mean_loss":  epoch_loss / max(n_graphs, 1),
        "total_s":    total,
        "load_s":     t_load,
        "transfer_s": t_transfer,
        "forward_s":  t_forward,
        "loss_bnb_s": t_loss,
        "backward_s": t_backward,
        "optim_s":    t_optim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Timing d'inférence (forward pass seul, GPU chaud)
# ─────────────────────────────────────────────────────────────────────────────

def time_inference(
    instances: dict, raw_records: list, m: int,
    device: torch.device, n_warmup: int, n_trials: int,
) -> dict:
    """
    Mesure le forward pass GNN pour un nœud B&B individuel.
    Graphes pré-construits et pré-transférés sur GPU avant la mesure.
    """
    torch.manual_seed(42)
    model = GNN(m=m, hidden=HIDDEN).to(device)
    model.eval()

    lazy   = LazyGraphDataset(instances, raw_records)
    eager  = EagerGraphDataset(lazy)
    graphs = [eager[i].to(device) for i in range(len(eager))]

    if not graphs:
        return {"error": "aucun graphe valide"}

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    def _fwd(g) -> float:
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(g)
        _sync()
        return time.perf_counter() - t0

    for i in range(n_warmup):
        _fwd(graphs[i % len(graphs)])

    times = [_fwd(graphs[i % len(graphs)]) for i in range(n_trials)]
    t = np.array(times)

    return {
        "n_trials": n_trials,
        "median_s": float(np.median(t)),
        "mean_s":   float(np.mean(t)),
        "std_s":    float(np.std(t)),
        "min_s":    float(np.min(t)),
        "max_s":    float(np.max(t)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep timing GNN V1LD : entraînement (1 epoch) + inférence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--binary",
                   default="solver_bab/build/bab_mkp_gecode_base_collect",
                   help="Chemin vers le binaire C++ (relatif à la racine du repo "
                        "ou absolu).")
    p.add_argument("--time_limit", type=int, default=10,
                   help="Limite en secondes par instance B&B (-1 = illimité).")
    p.add_argument("--max_depth",  type=int, default=5,
                   help="Profondeur max de l'arbre B&B lors de la collecte.")
    p.add_argument("--num_workers", type=int, default=3,
                   help="Workers DataLoader (recommandé : CPUs alloués - 1).")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--device",  default="auto",
                   help="'auto', 'cuda', 'cpu', 'cuda:0', etc.")
    p.add_argument("--out_dir", default="results/sweep_benchmark",
                   help="Dossier de sortie (absolu ou relatif à la racine du repo).")
    p.add_argument("--quick", action="store_true",
                   help="Test rapide : une seule combinaison (n=10, m=5, N=50).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "sweep_results.json"

    log(f"{'='*65}")
    log("SWEEP BENCHMARK — GNN V1LD")
    log(f"{'='*65}")
    log(f"  device      : {device}"
        + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    log(f"  num_workers : {args.num_workers}")
    log(f"  time_limit  : {args.time_limit}s/instance  "
        f"max_depth : {args.max_depth}")
    log(f"  seed        : {args.seed}")
    log(f"  out_dir     : {out_dir}")

    if not _USE_CPP_BNB:
        log("\nERREUR : C++ B&B solver (bnb_solver.py) non chargé. "
            "Vérifier la compilation.")
        sys.exit(1)
    log("  C++ B&B solver : OK")

    binary = Path(args.binary)
    if not binary.is_absolute():
        binary = REPO_ROOT / binary
    if not binary.exists():
        log(f"\nERREUR : binaire introuvable : {binary}")
        sys.exit(1)

    combos = [(10, 5, 50)] if args.quick else [(n, m, N) for n in NS for m in MS for N in NS_SCEN]
    results: list[dict] = []

    for combo_i, (n, m, N) in enumerate(combos, start=1):
        log(f"\n{'='*65}")
        log(f"[{combo_i:2d}/{len(combos)}]  n={n}  m={m}  N={N}")
        log(f"{'='*65}")

        with tempfile.TemporaryDirectory(
                prefix=f"sweep_n{n}_m{m}_N{N}_") as tmpdir:

            # ── 1. Collecte ───────────────────────────────────────────────────
            log("\n── 1/3  Collecte ─────────────────────────────────────────")
            inst_f, nodes_f, t_collect = collect_instances(
                n, m, N, binary, Path(tmpdir),
                args.time_limit, args.max_depth, args.seed)

            if not inst_f.exists() or not nodes_f.exists():
                log("  SKIP : aucun fichier généré par le solveur.")
                continue

            # ── 2. Dataset ────────────────────────────────────────────────────
            log("\n── 2/3  Assemblage dataset ───────────────────────────────")
            instances, raw_records = build_sweep_dataset(
                inst_f, nodes_f, rng_seed=args.seed)

            if not raw_records:
                log("  SKIP : aucun enregistrement valide.")
                continue

            # ── 3a. Entraînement ──────────────────────────────────────────────
            log(f"\n── 3a/3  Entraînement (1 epoch, batch={BATCH_SIZE}) ──────")
            tr = train_one_epoch(
                instances, raw_records, m, device, BATCH_SIZE, args.num_workers)

            n_g  = tr["n_graphs"]
            tot  = tr["total_s"]
            proj = tot / n_g * 10_000 if n_g > 0 else float("nan")
            log(f"  {n_g} graphes — 1 epoch : {tot:.2f}s")
            log(f"    load={tr['load_s']:.2f}s ({_pct(tr['load_s'], tot)})  "
                f"transfer={tr['transfer_s']:.2f}s ({_pct(tr['transfer_s'], tot)})  "
                f"forward={tr['forward_s']:.2f}s ({_pct(tr['forward_s'], tot)})  "
                f"loss_bnb={tr['loss_bnb_s']:.2f}s ({_pct(tr['loss_bnb_s'], tot)})  "
                f"backward={tr['backward_s']:.2f}s ({_pct(tr['backward_s'], tot)})  "
                f"optim={tr['optim_s']:.2f}s ({_pct(tr['optim_s'], tot)})")
            log(f"  → projection 10k exemples : {proj:.0f}s  ({proj/60:.1f} min)")

            # ── 3b. Inférence ─────────────────────────────────────────────────
            log(f"\n── 3b/3  Inférence ({N_WARMUP} warmup, {N_TRIALS} essais) ──")
            inf = time_inference(
                instances, raw_records, m, device, N_WARMUP, N_TRIALS)

            if "error" not in inf:
                log(f"  median={inf['median_s']*1e3:.3f}ms  "
                    f"mean={inf['mean_s']*1e3:.3f}ms  "
                    f"std={inf['std_s']*1e3:.3f}ms  "
                    f"[{inf['min_s']*1e3:.3f}–{inf['max_s']*1e3:.3f}ms]")
            else:
                log(f"  ERREUR : {inf['error']}")

            # ── Sauvegarde intermédiaire ───────────────────────────────────────
            results.append({
                "n": n, "m": m, "N": N,
                "n_graphs":   n_g,
                "collect_s":  t_collect,
                "train":      tr,
                "infer":      inf,
                "proj_10k_s": proj,
            })
            with open(results_file, "w") as fh:
                json.dump(results, fh, indent=2)

    # ── Résumé final ──────────────────────────────────────────────────────────
    log(f"\n\n{'='*65}")
    log("RÉSUMÉ FINAL DU SWEEP")
    log(f"{'='*65}")
    log(f"{'n':>3} {'m':>3} {'N':>4} │ {'#graphes':>8} │ "
        f"{'epoch(s)':>9} │ {'proj10k(min)':>13} │ {'infer(ms)':>10}")
    log("─" * 60)
    for r in results:
        infer_ms = (f"{r['infer']['median_s']*1e3:.3f}"
                    if "median_s" in r["infer"] else "   ERR")
        log(f"{r['n']:>3} {r['m']:>3} {r['N']:>4} │ "
            f"{r['n_graphs']:>8} │ "
            f"{r['train']['total_s']:>9.2f} │ "
            f"{r['proj_10k_s']/60:>13.1f} │ "
            f"{infer_ms:>10}")

    log(f"\nRésultats JSON : {results_file}")


if __name__ == "__main__":
    main()
