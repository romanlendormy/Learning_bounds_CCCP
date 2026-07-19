#!/usr/bin/env python3
"""
compare_v1ld_song_gnn_mknap1_4.py — Même tableau comparatif que
compare_v1ld_song_gnn_n10_N50.py (sous-gradient "maison" T=500 vs inférence
GNN vs V1LD façon SONG/Kelley), mais sur l'instance réelle du benchmark SONG
et al. : data/knapsack/datasets-song/ccmk-song-mknap1_4_m5-S100.txt
(n=20, m=5, N=100, rep=0), pour vérifier si l'écart GNN vs sous-gradient
observé sur une instance n=10/N=50 générée (compare_v1ld_song_gnn_n10_N50.py)
se reproduit sur une VRAIE instance de benchmark, à n différent (n=20, dans
la distribution d'entraînement n_list=[10,20] mais jamais vue — ce fichier
n'a jamais fait partie du pipeline train_base_s10).

L'optimum exact de cette instance (opt=6070.0000, statut OPTIMAL confirmé
par Gecode BAB sans borne ET avec borne GNN, mêmes valeurs) a été établi
dans logs/slurm_cmp_ref_gnn_song_15517441.out (job 15517441, 2026-07-08,
compare_ref_gnn_song.sh) — passé ici en constante plutôt que recalculé.

Le format du fichier SONG est différent de celui de train_base_s10 : UNE
LIGNE PAR SCÉNARIO (pas par instance), sans epsilon/opt embarqués — voir
solver_bab/mkp/main_gecode.cpp::load_song_local pour le format de référence,
reproduit ici en Python (load_song_instance).

Comme cette instance n'a pas de nodes_*.txt (pas de nœud B&B précalculé),
le graphe racine (depth=0, rien de fixé) est construit directement via
build_graph(inst, node_rec) plutôt que via le pipeline LazyGraphDataset /
discover_pairs / split train-test (inutile ici : ce fichier n'a jamais fait
partie du dataset d'entraînement, donc automatiquement hors distribution
d'entraînement, peu importe le split).

Usage :
    python compare_v1ld_song_gnn_mknap1_4.py --ckpt <chemin_vers.pt> --device cuda
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

from gnn.train_gnn_v1ld import GNN, build_graph, lagrangian_v1ld, verify_v1ld

DEFAULT_SONG_FILE = REPO_ROOT / "data/knapsack/datasets-song/ccmk-song-mknap1_4_m5-S100.txt"
KNOWN_OPT = 6070.0000  # cf. logs/slurm_cmp_ref_gnn_song_15517441.out (statut OPTIMAL, ref ET gnn)


def load_song_instance(path: Path, rep: int = 0) -> dict:
    """Reproduit load_song_local() de solver_bab/mkp/main_gecode.cpp :
    une ligne par scénario [scen_id, ?, m, n, rep, A[m*n], b[m], c[n]]."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(",")]
            if int(vals[4]) != rep:
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"Aucun scénario pour rep={rep} dans {path}")

    m = int(rows[0][2]); n = int(rows[0][3]); N = len(rows)
    A = np.zeros((N, m, n), dtype=np.float32)
    b = np.zeros((N, m), dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    for i, v in enumerate(rows):
        for j in range(m):
            for k in range(n):
                A[i, j, k] = v[5 + j * n + k]
        for j in range(m):
            b[i, j] = v[5 + m * n + j]
        if i == 0:
            for k in range(n):
                c[k] = v[5 + m * n + m + k]
    return dict(n=n, m=m, N=N, c=c, b=b, A=A)


def build_root_record(inst: dict) -> dict:
    """Nœud racine B&B : rien de fixé (x_mask/z_mask tout à 2 = libre),
    b_res = b d'origine, curr_cost = 0, depth = 0."""
    return dict(
        x_mask=np.full(inst["n"], 2, dtype=np.int8),
        z_mask=np.full(inst["N"], 2, dtype=np.int8),
        b_res=inst["b"].copy(),
        curr_cost=0.0,
        depth=0,
    )


def run_kelley_song_worker(inst: dict, max_iters: int, time_limit: float, tol: float) -> dict:
    """Cf. compare_v1ld_song_gnn_n10_N50.py — isolation en subprocess pour
    éviter le conflit OpenMP entre gnn/_bnb_solver_lib.so et OR-Tools."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        npz_path = f.name
    np.savez(npz_path, n=inst["n"], m=inst["m"], N=inst["N"],
              epsilon=inst["epsilon"], c=inst["c"], b=inst["b"], A=inst["A"])
    try:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "_kelley_song_worker.py"),
             "--npz", npz_path, "--max_iters", str(max_iters),
             "--time_limit", str(time_limit), "--tol", str(tol)],
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--song_file", default=str(DEFAULT_SONG_FILE))
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument("--opt", type=float, default=KNOWN_OPT)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--subgrad_T", type=int, default=500)
    p.add_argument("--kelley_max_iters", type=int, default=300)
    p.add_argument("--kelley_time_limit", type=float, default=300.0)
    p.add_argument("--kelley_tol", type=float, default=1e-5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("[warn] --device cuda demandé mais torch.cuda.is_available()=False -> repli sur cpu",
              file=sys.stderr)

    inst = load_song_instance(Path(args.song_file), rep=args.rep)
    inst["epsilon"] = args.epsilon
    inst["opt"] = args.opt
    opt = args.opt

    print(f"[instance] {Path(args.song_file).name}  rep={args.rep}  "
          f"n={inst['n']} m={inst['m']} N={inst['N']} epsilon={inst['epsilon']}  "
          f"opt={opt:.4f} (connu, cf. logs/slurm_cmp_ref_gnn_song_15517441.out)")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt["args"]
    n_status = "n dans la distribution d'entrainement" if inst["n"] in ckpt_args["n_list"] \
        else "n HORS distribution"
    print(f"[ckpt] {ckpt_path.name}  epoch={ckpt['epoch']}  n_list={ckpt_args['n_list']}  "
          f"({n_status}, mais cette instance n'a jamais fait partie du dataset d'entrainement)")

    rec = build_root_record(inst)
    results = {}

    # ── A. Sous-gradient v1LD "maison" (verify_v1ld, T fixé, cold start) ──────
    g_cpu = build_graph(inst, rec)
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

    g_dev = build_graph(inst, rec).to(device)

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
    kelley = run_kelley_song_worker(inst, args.kelley_max_iters, args.kelley_time_limit, args.kelley_tol)
    conv_note = "convergé" if kelley["converged"] else "NON convergé (best_UB rapporté, pas le certificat LB)"
    results[f"C. V1LD SONG (repro Kelley/GLOP, {kelley['iters']} it.)"] = dict(
        value=kelley["value"], time_s=kelley["time_s"], note=conv_note)

    # ── Tableau récapitulatif ──────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"Instance {Path(args.song_file).name} rep={args.rep}  "
          f"n={inst['n']} m={inst['m']} N={inst['N']} epsilon={inst['epsilon']}  opt(B&B exact)={opt:.4f}")
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
