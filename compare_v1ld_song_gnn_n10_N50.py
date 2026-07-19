#!/usr/bin/env python3
"""
compare_v1ld_song_gnn_n10_N50.py — Tableau comparatif de la borne v1LD sur une
instance n=10, m=5, N=50, epsilon=0.1 (data/train_base_s10/instances_n10_m5_N50.txt),
choisie dans le split TEST du checkpoint GNN (jamais vue à l'entraînement).

Trois méthodes comparées :
  A. Sous-gradient v1LD "maison" (verify_v1ld, port Python de
     solver_generic/v1ld_solver.cpp) — T=500 itérations, cold start λ=0,
     pas 1/sqrt(t+1), sans arrêt anticipé (précision/patience désactivés).
  B. Inférence GNN (checkpoint entraîné sur ~10k graphes, train_scaled_v1ld.py)
     — forward pass (device=cuda si dispo) puis UNE évaluation de la borne
     L1(λ_GNN) (lagrangian_v1ld : un seul appel à l'oracle ζ par scénario,
     pas de boucle sous-gradient).
  C. V1LD façon SONG et al. — reproduction fidèle de l'algorithme de
     external/CCdual/ccdual.cpp (fonctions lr1/V1LD, plan coupant de Kelley)
     mais avec OR-Tools GLOP à la place d'IBM CPLEX pour le maître LP : le
     binaire original ne peut pas être compilé sur ce cluster (dépendance
     totale à ilcplex/ilocplex.h, aucun module ni licence CPLEX disponible
     sur ce compte). Le LP maître est résolu exactement (simplex), donc pour
     un problème de cette taille (~n*N=500 variables duales) le résultat est
     numériquement équivalent à ce qu'aurait donné CPLEX — voir
     compare_v1ld_ccdual.py pour la validation initiale (n=20/N=100).

     Cette méthode C tourne dans un SOUS-PROCESSUS séparé
     (_kelley_song_worker.py) : gnn/_bnb_solver_lib.so (compilé avec
     -fopenmp, lié à libgomp.so.1 de CVMFS) et le runtime OpenMP embarqué
     par OR-Tools segfault au chargement s'ils cohabitent dans le même
     process Python (testé, indépendant de l'ordre d'import) — isolation
     par subprocess plutôt que de toucher au bnb_solver.py partagé par le
     reste du pipeline GNN.

Toutes les méthodes lisent EXACTEMENT la même instance (mêmes c/A/b/epsilon,
même optimum de référence `opt`, calculé par B&B exact) — donc les 3 valeurs
de borne sont directement comparables, et le gap relatif à l'optimal est
défini comme (borne - opt) / opt (les 3 méthodes donnent des bornes
SUPÉRIEURES valides sur le problème de maximisation, donc gap >= 0 attendu).

Usage :
    python compare_v1ld_song_gnn_n10_N50.py --ckpt <chemin_vers.pt> --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
sys.path.insert(0, str(REPO_ROOT))

from gnn.train_gnn_v1ld import GNN, LazyGraphDataset, lagrangian_v1ld, verify_v1ld
from gnn.train_scaled_v1ld import discover_pairs, load_and_merge, split_by_instance


def run_kelley_song_worker(inst: dict, max_iters: int, time_limit: float, tol: float,
                            lam_bound: float, method: str = "trustregion",
                            r_init: float = 20.0, r_min: float = 0.1,
                            r_max: float = 400.0, kappa: float = 0.1) -> dict:
    """Lance _kelley_song_worker.py dans un processus séparé (cf. docstring
    module) et récupère son résultat JSON. Le temps mesuré (time_s) vient du
    worker lui-même (n'inclut donc pas le coût de démarrage de l'interpréteur
    ni le dump/chargement du .npz)."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        npz_path = f.name
    np.savez(npz_path, n=inst["n"], m=inst["m"], N=inst["N"],
              epsilon=inst["epsilon"], c=inst["c"], b=inst["b"], A=inst["A"])
    try:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "_kelley_song_worker.py"),
             "--npz", npz_path, "--max_iters", str(max_iters),
             "--time_limit", str(time_limit), "--tol", str(tol),
             "--lam_bound", str(lam_bound), "--method", method,
             "--r_init", str(r_init), "--r_min", str(r_min),
             "--r_max", str(r_max), "--kappa", str(kappa)],
            capture_output=True, text=True, check=False,
        )
    finally:
        Path(npz_path).unlink(missing_ok=True)

    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"_kelley_song_worker.py a échoué (code={proc.returncode}):\n{proc.stdout}")
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON: ")), None)
    if line is None:
        raise RuntimeError(f"Pas de RESULT_JSON dans la sortie du worker:\n{proc.stdout}")
    return json.loads(line[len("RESULT_JSON: "):])


# ─────────────────────────────────────────────────────────────────────────────
# Sélection de l'instance test (même split que l'entraînement du checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--pick_index", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--subgrad_T", type=int, default=500)
    p.add_argument("--kelley_max_iters", type=int, default=2000)
    p.add_argument("--kelley_time_limit", type=float, default=300.0)
    p.add_argument("--kelley_tol", type=float, default=1e-5)
    p.add_argument("--kelley_lam_bound", type=float, default=50.0,
                    help="Boite [-x,x] pour les lambda du maitre LP Kelley — "
                         "50.0 = valeur fixe de ccdual.cpp:886 (pas mise à "
                         "l'échelle du coût, cf. _kelley_song_worker.py). "
                         "Ignoré si --kelley_method=trustregion.")
    p.add_argument("--kelley_method", choices=["naive", "trustregion"], default="trustregion",
                    help="naive = reproduction fidèle de ccdual.cpp (boîte fixe, sans "
                         "stabilisation). trustregion = bundle stabilisé (pas SONG "
                         "littéral, mais teste le meilleur v1LD atteignable par ce "
                         "type de méthode — cf. _kelley_song_worker.py).")
    p.add_argument("--kelley_r_init", type=float, default=20.0)
    p.add_argument("--kelley_r_min", type=float, default=0.1)
    p.add_argument("--kelley_r_max", type=float, default=400.0)
    p.add_argument("--kelley_kappa", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("[warn] --device cuda demandé mais torch.cuda.is_available()=False -> repli sur cpu",
              file=sys.stderr)

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt["args"]
    print(f"[ckpt] {ckpt_path.name}  epoch={ckpt['epoch']}  "
          f"n_list={ckpt_args['n_list']} n_scens={ckpt_args['n_scens']} "
          f"target_graphs={ckpt_args['target_graphs']}")

    data_dir = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data"

    print(f"[split] reconstruction du split train/test (seed={ckpt_args['seed']}, "
          f"train_frac={ckpt_args['train_frac']}, reservoir_k={ckpt_args['reservoir_k']}) ...")
    pairs = discover_pairs(data_dir, ckpt_args["n_list"], ckpt_args["n_scens"], ckpt_args["m"])
    instances, all_records = load_and_merge(pairs, ckpt_args["reservoir_k"], ckpt_args["seed"])
    all_ds = LazyGraphDataset(instances, all_records)
    train_ds, test_ds = split_by_instance(instances, all_ds, ckpt_args["train_frac"], ckpt_args["seed"])
    print(f"[split] instances: train={len(set(r['inst_id'] for r in train_ds.raw_records))}  "
          f"test={len(set(r['inst_id'] for r in test_ds.raw_records))}")

    cands = [r for r in test_ds.raw_records
             if r["depth"] == 0
             and instances[r["inst_id"]]["n"] == args.n
             and instances[r["inst_id"]]["N"] == args.N]
    cands = sorted(cands, key=lambda r: r["inst_id"])
    if not cands:
        raise SystemExit(f"Aucune instance test n={args.n} N={args.N} depth=0 trouvée.")
    if args.pick_index >= len(cands):
        raise SystemExit(f"pick_index={args.pick_index} hors limites ({len(cands)} candidats).")
    rec = cands[args.pick_index]
    inst = instances[rec["inst_id"]]
    opt = float(inst["opt"])

    print(f"\n[instance] inst_id={rec['inst_id']} (parmi {len(cands)} candidats test n={args.n} N={args.N})  "
          f"n={inst['n']} m={inst['m']} N={inst['N']} epsilon={inst['epsilon']}  opt(B&B exact)={opt:.4f}")
    print(f"[instance] confirmée absente du split train du checkpoint ({ckpt_path.name})")

    results = {}

    # ── A. Sous-gradient v1LD "maison" (verify_v1ld, T fixé, cold start) ──────
    g_cpu = LazyGraphDataset(instances, [rec])[0]
    t0 = time.perf_counter()
    v_subgrad = verify_v1ld(g_cpu, T=args.subgrad_T)
    t_subgrad = time.perf_counter() - t0
    results[f"A. Sous-gradient v1LD maison (T={args.subgrad_T}, cold start)"] = dict(
        value=v_subgrad, time_s=t_subgrad)

    # ── B. Inférence GNN (checkpoint 10k graphes) ─────────────────────────────
    model = GNN(m=ckpt_args["m"], hidden=ckpt_args["hidden"],
                use_max_pool=ckpt_args.get("use_max_pool", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    g_dev = LazyGraphDataset(instances, [rec])[0].to(device)

    with torch.no_grad():
        _ = model(g_dev)                      # warmup : init contexte CUDA / noyaux, non chronométré
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        lam = model(g_dev)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_forward = time.perf_counter() - t0

        if ckpt_args.get("center_lam", False):
            lam_2d = lam.view(int(g_dev.N_z), int(g_dev.n_g))
            lam = (lam_2d - lam_2d.mean(dim=0, keepdim=True)).reshape(-1)

        t1 = time.perf_counter()
        v_gnn = lagrangian_v1ld(lam, g_dev).item()
        t_eval = time.perf_counter() - t1

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    results[f"B. GNN (ckpt 10k graphes, {device_name})"] = dict(
        value=v_gnn, time_s=t_forward + t_eval,
        note=f"forward={t_forward*1000:.2f}ms + éval_borne(oracle ζ)={t_eval*1000:.2f}ms")

    # ── C. V1LD "SONG" (reproduction Kelley/GLOP de ccdual.cpp, subprocess) ──
    kelley = run_kelley_song_worker(inst, args.kelley_max_iters, args.kelley_time_limit,
                                     args.kelley_tol, args.kelley_lam_bound, args.kelley_method,
                                     args.kelley_r_init, args.kelley_r_min, args.kelley_r_max,
                                     args.kelley_kappa)
    conv_note = "convergé" if kelley["converged"] else "NON convergé (best_UB rapporté)"
    if kelley["method"] == "naive":
        boundary_note = " -- ATTENTION: max|lambda| colle a la borne" if kelley["at_boundary"] else ""
        label = f"C. V1LD SONG (repro fidèle Kelley/GLOP, {kelley['iters']} it.)"
        note = (f"{conv_note}; max|lambda|={kelley['max_abs_lam']:.2f} "
                f"(boîte=±{args.kelley_lam_bound}){boundary_note}")
    else:
        label = f"C. V1LD SONG (Kelley stabilisé trust-region, {kelley['iters']} it., {kelley['n_serious']} pas sérieux)"
        note = f"{conv_note}; certificat LB={kelley['lb_certificate']:.4f} (pas une repro fidèle de ccdual.cpp)"
    results[label] = dict(value=kelley["value"], time_s=kelley["time_s"], note=note)

    # ── Tableau récapitulatif ──────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"Instance n={inst['n']} m={inst['m']} N={inst['N']} epsilon={inst['epsilon']}  "
          f"inst_id={rec['inst_id']}  opt(B&B exact)={opt:.4f}")
    print(f"{'='*100}")
    header = f"{'Méthode':55s} {'v1LD (borne)':>14s} {'gap vs opt':>12s} {'temps (s)':>12s}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        gap_pct = 100.0 * (r["value"] - opt) / opt
        print(f"{name:55s} {r['value']:14.4f} {gap_pct:11.2f}% {r['time_s']:12.4f}")
        if "note" in r:
            print(f"{'':55s} note: {r['note']}")
    print("-" * len(header))
    print(f"opt (B&B exact, référence) = {opt:.4f}")
    print("Les 3 valeurs sont des bornes SUPÉRIEURES valides (dualité faible) : "
          "gap >= 0 attendu. Plus le gap est petit, plus la borne est resserrée.")


if __name__ == "__main__":
    main()
