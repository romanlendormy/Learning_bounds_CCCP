#!/usr/bin/env python3
"""
sweep_bab.py — Sweep Branch & Bound CCMKP sur grille (n, m, N).

4 modes par triplet (n, m, N) :
  1. V1LD_fixed    — T_root=T_ROOT, T_node=T_NODE, pas de patience
  2. V1CLD_fixed   — T_root=T_ROOT, T_node=T_NODE, pas de patience
  3. V1LD_patience — convergence sous-gradient (tol=PATIENCE_TOL)
  4. V1CLD_patience

Les datasets Song-Pisinger manquants sont générés automatiquement.
Le binaire bab_mkp_gecode est compilé si absent.
Les résultats sont écrits ligne à ligne dans OUTPUT_CSV (reprise possible).

Usage :
    python sweep_bab.py
"""

import csv
import os
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# ── Paramètres sweep ───────────────────────────────────────────────────────────
N_VALS       = [5, 10, 20]      # nombre d'items. 30
M_VALS       = [2, 5, 10, 15]      # nombre de contraintes
S_VALS       = [50, 100, 250, 500]  # nombre de scénarios (N)
# N_VALS       = [5]      # nombre d'items
# M_VALS       = [2]      # nombre de contraintes
# S_VALS       = [50]  # nombre de scénarios (N)

# Paramètres B&B
EPSILON      = 0.10
T_ROOT       = 200   # itérations sous-gradient (cold start, nœud racine)
T_NODE       = 5     # itérations sous-gradient (warm-start, nœuds suivants)
TIME_LIMIT   = -1    # pas de limite de temps (en secondes)
REPLICATION  = 0     # index de réplication Song utilisé

# Paramètres mode patience
PATIENCE_TOL = 1  # tolérance convergence (rel_tol)

# Paramètres génération Pisinger
ALPHA        = 0.5               # tightness des capacités
CORRELATION  = "weakly_correlated"
SEED         = 42

# Chemins
OUTPUT_CSV    = ROOT / "results" / "sweep_bab.csv"
BAB_BIN       = ROOT / "solver_bab" / "build" / "bab_mkp_gecode"
PISINGER_DIR  = ROOT / "data" / "knapsack" / "pisinger"
DATASET_DIR   = ROOT / "data" / "knapsack" / "datasets-pisinger"
CREATE_SCRIPT = ROOT / "data" / "knapsack" / "create_database_knapsack_pisinger.py"

# ── Modes ──────────────────────────────────────────────────────────────────────
# (mode_str passé au binaire, patience, label CSV)
MODES = [
    ("v1ld",  False, "V1LD_fixed"),
    ("v1cld", False, "V1CLD_fixed"),
    ("v1ld",  True,  "V1LD_patience"),
    ("v1cld", True,  "V1CLD_patience"),
]

CSV_FIELDS = [
    "mode", "n", "m", "N", "rep",
    "T_root", "T_node", "patience", "patience_tol",
    "statut", "obj", "greedy", "best_ub",
    "gap_greedy_pct", "gap_dual_pct",
    "nodes", "naive_nodes", "pct_naive",
    "depth", "depth_max",
    "fails", "fails_bound", "fails_cp", "pct_fails_bound",
    "inc_imp", "more_calls", "ub_tight", "pct_ub_tight",
    "t_solve", "t_root_bound", "t_bound_total", "t_bab_pure", "ms_node",
    "t_wall",
]


# ── Compilation ────────────────────────────────────────────────────────────────

def compile_binary():
    if BAB_BIN.exists():
        print(f"[binaire] Trouvé : {BAB_BIN}", flush=True)
        return
    print("[binaire] Compilation de bab_mkp_gecode...", flush=True)
    build_dir = BAB_BIN.parent
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    subprocess.run(["make", "-j4"], cwd=build_dir, check=True)
    if not BAB_BIN.exists():
        raise FileNotFoundError(f"Compilation échouée : {BAB_BIN}")
    print(f"[binaire] Compilé : {BAB_BIN}", flush=True)


# ── Datasets ───────────────────────────────────────────────────────────────────

def _inst_name(n, m):
    alpha_pct = int(round(ALPHA * 100))
    corr_abbr = CORRELATION[:2]
    return f"pisinger-n{n}-m{m}-a{alpha_pct}-{corr_abbr}-r{SEED}"


def dataset_path(n, m, N):
    return DATASET_DIR / f"ccmk-song-{_inst_name(n, m)}-S{N}.txt"


def generate_missing_datasets():
    """Génère les datasets Song-Pisinger manquants (y compris m=15)."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Regrouper les S manquants par (n, m) pour minimiser les appels
    missing: dict[tuple, list[int]] = {}
    for n, m, N in product(N_VALS, M_VALS, S_VALS):
        if not dataset_path(n, m, N).exists():
            missing.setdefault((n, m), []).append(N)

    if not missing:
        print("[datasets] Tous les datasets existent.", flush=True)
        return

    n_files = sum(len(v) for v in missing.values())
    print(f"[datasets] Génération de {n_files} dataset(s) manquant(s)...", flush=True)

    for (n, m), ns in sorted(missing.items()):
        print(f"  n={n}, m={m}, S∈{ns}", flush=True)
        cmd = [
            sys.executable, str(CREATE_SCRIPT),
            "--n", str(n),
            "--m", str(m),
            "--alpha", str(ALPHA),
            "--correlation", CORRELATION,
            "--S_sizes", *[str(s) for s in ns],
            "--n_reps", "1",   # une seule réplication (rep=0)
            "--seed", str(SEED),
            "--orlib_dir", str(PISINGER_DIR),
            "--dataset_dir", str(DATASET_DIR),
        ]
        subprocess.run(cmd, check=True, cwd=ROOT)

    print("[datasets] Génération terminée.", flush=True)


# ── Runner B&B ─────────────────────────────────────────────────────────────────

def run_bab(n, m, N, mode_str, patience):
    ds = dataset_path(n, m, N)
    cmd = [
        str(BAB_BIN), str(ds),
        str(EPSILON), str(T_NODE), str(TIME_LIMIT),
        str(REPLICATION), str(T_ROOT), mode_str,
    ]
    if patience:
        cmd += ["patience", str(PATIENCE_TOL)]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t_wall = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(
            f"Erreur {proc.returncode}\n{proc.stderr.strip()}"
        )
    return proc.stdout, t_wall


# ── Parsing sortie binaire ─────────────────────────────────────────────────────

def parse_result(out):
    def pf(pat, default=float("nan")):
        m = re.search(pat, out)
        return float(m.group(1)) if m else default

    def pi(pat, default=-1):
        m = re.search(pat, out)
        return int(m.group(1)) if m else default

    st = re.search(r"Statut\s*:\s*(\S+)", out)
    return dict(
        statut        = st.group(1) if st else "?",
        obj           = pf(r"Valeur optimale\s*:\s*([\d.]+)"),
        greedy        = pf(r"Incumbent initial.*?:\s*([\d.]+)"),
        best_ub       = pf(r"UB finale\s*:\s*([\d.]+)"),
        nodes         = pi(r"N.+uds Gecode\s*:\s*(\d+)"),
        naive_nodes   = pi(r"N.+uds Gecode\s*:\s*\d+ / (\d+)"),
        depth         = pi(r"Profondeur max\s*:\s*(\d+)"),
        depth_max     = pi(r"Profondeur max\s*:\s*\d+ / (\d+)"),
        fails         = pi(r"Échecs totaux\s*:\s*(\d+)"),
        fails_bound   = pi(r"dont borne.*?:\s*(\d+)"),
        fails_cp      = pi(r"dont propagation CP\s*:\s*(\d+)"),
        inc_imp       = pi(r"Améliorations incumbent\s*:\s*(\d+)"),
        more_calls    = pi(r"Appels more\(\).*?:\s*(\d+)"),
        ub_tight      = pi(r"UB V1LD utile.*?:\s*(\d+)"),
        t_solve       = pf(r"Temps total\s*:\s*([\d.]+)"),
        t_root_bound  = pf(r"Temps borne root\s*:\s*([\d.]+)"),
        t_bound_total = pf(r"Temps borne total\s*:\s*([\d.]+)"),
        t_bab_pure    = pf(r"Temps BAB pur\s*:\s*([\d.]+)"),
        ms_node       = pf(r"Temps moyen par n.+ud\s*:\s*([\d.]+)"),
    )


def build_row(r, n, m, N, mode_label, patience, t_wall):
    obj, greedy, best_ub = r["obj"], r["greedy"], r["best_ub"]
    nodes, naive = r["nodes"], r["naive_nodes"]
    fails, mc = r["fails"], r["more_calls"]

    def safe_pct(a, b):
        return round(100.0 * a / b, 2) if b and b > 0 else float("nan")

    return {
        "mode"           : mode_label,
        "n"              : n,
        "m"              : m,
        "N"              : N,
        "rep"            : REPLICATION,
        "T_root"         : T_ROOT,
        "T_node"         : T_NODE,
        "patience"       : patience,
        "patience_tol"   : PATIENCE_TOL if patience else "",
        "statut"         : r["statut"],
        "obj"            : r["obj"],
        "greedy"         : r["greedy"],
        "best_ub"        : r["best_ub"],
        "gap_greedy_pct" : round(100.0 * (obj - greedy) / obj, 3) if obj > 0 else float("nan"),
        "gap_dual_pct"   : round(100.0 * (best_ub - obj) / obj, 3) if obj > 0 else float("nan"),
        "nodes"          : nodes,
        "naive_nodes"    : naive,
        "pct_naive"      : safe_pct(nodes, naive),
        "depth"          : r["depth"],
        "depth_max"      : r["depth_max"],
        "fails"          : fails,
        "fails_bound"    : r["fails_bound"],
        "fails_cp"       : r["fails_cp"],
        "pct_fails_bound": safe_pct(r["fails_bound"], fails),
        "inc_imp"        : r["inc_imp"],
        "more_calls"     : mc,
        "ub_tight"       : r["ub_tight"],
        "pct_ub_tight"   : safe_pct(r["ub_tight"], mc),
        "t_solve"        : r["t_solve"],
        "t_root_bound"   : r["t_root_bound"],
        "t_bound_total"  : r["t_bound_total"],
        "t_bab_pure"     : r["t_bab_pure"],
        "ms_node"        : r["ms_node"],
        "t_wall"         : round(t_wall, 3),
    }


# ── Reprise (skip déjà faits) ──────────────────────────────────────────────────

def load_done(csv_path):
    done = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    done.add((row["mode"], int(row["n"]), int(row["m"]), int(row["N"])))
                except (KeyError, ValueError):
                    pass
    return done


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    compile_binary()
    generate_missing_datasets()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(OUTPUT_CSV)
    is_new_file = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    total = len(N_VALS) * len(M_VALS) * len(S_VALS) * len(MODES)
    run_idx = 0
    t_sweep_start = time.perf_counter()

    print(f"\n{'═'*60}")
    print(f"  SWEEP BAB  —  {total} runs au total  ({len(done)} déjà faits)")
    print(f"{'═'*60}\n")

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new_file:
            writer.writeheader()

        # Ordre : petits n d'abord, puis m, N, mode
        for n, m, N in product(N_VALS, M_VALS, S_VALS):
            for mode_str, patience, mode_label in MODES:
                run_idx += 1
                key = (mode_label, n, m, N)

                if key in done:
                    print(f"[{run_idx:>4}/{total}] SKIP  {mode_label:<18}  n={n:2} m={m:2} N={N:3}", flush=True)
                    continue

                ds = dataset_path(n, m, N)
                if not ds.exists():
                    print(f"[{run_idx:>4}/{total}] MANQUANT  {ds.name}", flush=True)
                    continue

                print(f"[{run_idx:>4}/{total}] {mode_label:<18}  n={n:2} m={m:2} N={N:3}  ...",
                      end="  ", flush=True)

                t_run_start = time.perf_counter()
                try:
                    out, t_wall = run_bab(n, m, N, mode_str, patience)
                    r = parse_result(out)
                    row = build_row(r, n, m, N, mode_label, patience, t_wall)
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"{r['statut']:<8}  obj={r['obj']:10.1f}  "
                        f"nodes={r['nodes']:>8}  t={r['t_solve']:8.2f}s"
                    )
                except Exception as e:
                    t_elapsed = time.perf_counter() - t_run_start
                    print(f"ERREUR — {e}")
                    err_row = {k: "" for k in CSV_FIELDS}
                    err_row.update(
                        mode=mode_label, n=n, m=m, N=N, rep=REPLICATION,
                        T_root=T_ROOT, T_node=T_NODE,
                        patience=patience,
                        patience_tol=PATIENCE_TOL if patience else "",
                        statut="ERROR",
                        t_wall=round(t_elapsed, 3),
                    )
                    writer.writerow(err_row)
                    f.flush()

    t_total = time.perf_counter() - t_sweep_start
    print(f"\n{'═'*60}")
    print(f"  Sweep terminé en {t_total/3600:.2f}h")
    print(f"  Résultats : {OUTPUT_CSV}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
