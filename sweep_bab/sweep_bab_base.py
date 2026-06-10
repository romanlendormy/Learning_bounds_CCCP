#!/usr/bin/env python3
"""
sweep_bab_base.py — Sweep Gecode B&B baseline (CP pur, sans borne Lagrangienne).

Mêmes instances Pisinger que sweep_bab.py.
Champs sans sens pour le mode base : NaN dans le CSV.

Usage :
    python sweep_bab_base.py
"""

import csv
import os
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

# ── Paramètres sweep (identiques à sweep_bab.py) ──────────────────────────────
N_VALS       = [5,10,20]      # nombre d'items. 30
M_VALS       = [2,5,10,15]      # nombre de contraintes
S_VALS       = [50, 100, 250, 500]  # nombre de scénarios (N)

EPSILON     = 0.10
TIME_LIMIT  = -1
REPLICATION = 0

ALPHA       = 0.5
CORRELATION = "weakly_correlated"
SEED        = 42

OUTPUT_CSV  = ROOT / "results" / "sweep_bab_base.csv"
BAB_BIN     = ROOT / "solver_bab" / "build" / "bab_mkp_gecode_base"
DATASET_DIR = ROOT / "data" / "knapsack" / "datasets-pisinger"

NAN = float("nan")

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


# ── Dataset ────────────────────────────────────────────────────────────────────

def _inst_name(n, m):
    alpha_pct = int(round(ALPHA * 100))
    corr_abbr = CORRELATION[:2]
    return f"pisinger-n{n}-m{m}-a{alpha_pct}-{corr_abbr}-r{SEED}"


def dataset_path(n, m, N):
    return DATASET_DIR / f"ccmk-song-{_inst_name(n, m)}-S{N}.txt"


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_bab(n, m, N):
    ds = dataset_path(n, m, N)
    cmd = [
        str(BAB_BIN), str(ds),
        str(EPSILON), str(TIME_LIMIT), str(REPLICATION),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Erreur {proc.returncode}\n{proc.stderr.strip()}")
    return proc.stdout, t_wall


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_result(out):
    def pf(pat):
        m = re.search(pat, out)
        return float(m.group(1)) if m else NAN

    def pi(pat):
        m = re.search(pat, out)
        return int(m.group(1)) if m else -1

    st = re.search(r"Statut\s*:\s*(\S+)", out)
    return dict(
        statut    = st.group(1) if st else "?",
        obj       = pf(r"Valeur optimale\s*:\s*([\d.]+)"),
        greedy    = pf(r"Incumbent initial.*?:\s*([\d.]+)"),
        nodes     = pi(r"N.+uds Gecode\s*:\s*(\d+)"),
        naive     = pi(r"N.+uds Gecode\s*:\s*\d+ / (\d+)"),
        depth     = pi(r"Profondeur max\s*:\s*(\d+)"),
        depth_max = pi(r"Profondeur max\s*:\s*\d+ / (\d+)"),
        fails     = pi(r"Échecs totaux\s*:\s*(\d+)"),
        inc_imp   = pi(r"Améliorations incumbent\s*:\s*(\d+)"),
        t_solve   = pf(r"Temps total\s*:\s*([\d.]+)"),
        ms_node   = pf(r"Temps moyen par n.+ud\s*:\s*([\d.]+)"),
    )


def build_row(r, n, m, N, t_wall):
    nodes, naive = r["nodes"], r["naive"]
    obj = r["obj"]
    greedy = r["greedy"]
    return {
        "mode"           : "Gecode_base",
        "n"              : n,
        "m"              : m,
        "N"              : N,
        "rep"            : REPLICATION,
        "T_root"         : NAN,
        "T_node"         : NAN,
        "patience"       : NAN,
        "patience_tol"   : NAN,
        "statut"         : r["statut"],
        "obj"            : obj,
        "greedy"         : greedy,
        "best_ub"        : NAN,
        "gap_greedy_pct" : round(100.0 * (obj - greedy) / obj, 3) if obj > 0 else NAN,
        "gap_dual_pct"   : NAN,
        "nodes"          : nodes,
        "naive_nodes"    : naive,
        "pct_naive"      : round(100.0 * nodes / naive, 2) if naive > 0 else NAN,
        "depth"          : r["depth"],
        "depth_max"      : r["depth_max"],
        "fails"          : r["fails"],
        "fails_bound"    : NAN,
        "fails_cp"       : NAN,
        "pct_fails_bound": NAN,
        "inc_imp"        : r["inc_imp"],
        "more_calls"     : NAN,
        "ub_tight"       : NAN,
        "pct_ub_tight"   : NAN,
        "t_solve"        : r["t_solve"],
        "t_root_bound"   : NAN,
        "t_bound_total"  : NAN,
        "t_bab_pure"     : NAN,
        "ms_node"        : r["ms_node"],
        "t_wall"         : round(t_wall, 3),
    }


# ── Reprise ────────────────────────────────────────────────────────────────────

def load_done(csv_path):
    done = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    done.add((int(row["n"]), int(row["m"]), int(row["N"])))
                except (KeyError, ValueError):
                    pass
    return done


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not BAB_BIN.exists():
        print(f"ERREUR : binaire introuvable : {BAB_BIN}")
        print("Compiler avec : bash compile_cc.sh")
        sys.exit(1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(OUTPUT_CSV)
    is_new = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    combos = list(product(N_VALS, M_VALS, S_VALS))
    total  = len(combos)

    print(f"\n{'═'*55}")
    print(f"  SWEEP BAB BASE  —  {total} runs  ({len(done)} déjà faits)")
    print(f"{'═'*55}\n")

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()

        for idx, (n, m, N) in enumerate(combos, 1):
            key = (n, m, N)
            if key in done:
                print(f"[{idx:>3}/{total}] SKIP  n={n} m={m} N={N}", flush=True)
                continue

            ds = dataset_path(n, m, N)
            if not ds.exists():
                print(f"[{idx:>3}/{total}] MANQUANT  {ds.name}", flush=True)
                continue

            print(f"[{idx:>3}/{total}] Gecode_base  n={n} m={m} N={N}  ...",
                  end="  ", flush=True)

            t0 = time.perf_counter()
            try:
                out, t_wall = run_bab(n, m, N)
                r   = parse_result(out)
                row = build_row(r, n, m, N, t_wall)
                writer.writerow(row)
                f.flush()
                print(f"{r['statut']:<8}  obj={r['obj']:10.1f}  "
                      f"nodes={r['nodes']:>8}  t={r['t_solve']:8.3f}s")
            except Exception as e:
                t_elapsed = time.perf_counter() - t0
                print(f"ERREUR — {e}")
                err_row = {k: "" for k in CSV_FIELDS}
                err_row.update(mode="Gecode_base", n=n, m=m, N=N,
                               rep=REPLICATION, statut="ERROR",
                               t_wall=round(t_elapsed, 3))
                writer.writerow(err_row)
                f.flush()

    print(f"\n{'═'*55}")
    print(f"  Résultats : {OUTPUT_CSV}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    main()
