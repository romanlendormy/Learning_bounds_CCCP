#!/usr/bin/env python3
"""
table_synth.py — Tableau base / V1LD / V1CLD sur des instances synthétiques
(Song-Pisinger, weakly_correlated), au format du tableau de comparaison
utilisé pour les instances OR-Library (mkp1_4_S100, etc.).

Réutilise les binaires et le format de sortie de sweep_bab.py /
sweep_bab_base.py (mêmes régex de parsing), mais pour une liste ad hoc
d'instances au lieu d'un balayage grille.

Usage :
    python table_synth.py
    python table_synth.py --time_limit 300
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

BAB_BIN      = ROOT / "solver_bab" / "build" / "bab_mkp_gecode"
BAB_BASE_BIN = ROOT / "solver_bab" / "build" / "bab_mkp_gecode_base"
DATASET_DIR  = ROOT / "data" / "knapsack" / "datasets-pisinger"

EPSILON      = 0.10
T_ROOT       = 200   # itérations sous-gradient, cold start (racine)
T_NODE       = 30    # itérations sous-gradient, warm-start (nœuds suivants)
REPLICATION  = 0

ALPHA        = 0.5
CORRELATION  = "weakly_correlated"
SEED         = 42

# (label affiché, n, m, N)
INSTANCES = [
    ("synth_n20_m5_N50", 20, 5, 50),
    ("synth_n30_m5_N50", 30, 5, 50),
]


def _inst_name(n, m):
    alpha_pct = int(round(ALPHA * 100))
    corr_abbr = CORRELATION[:2]
    return f"pisinger-n{n}-m{m}-a{alpha_pct}-{corr_abbr}-r{SEED}"


def dataset_path(n, m, N):
    return DATASET_DIR / f"ccmk-song-{_inst_name(n, m)}-S{N}.txt"


# ── Runners ──────────────────────────────────────────────────────────────────

def _subprocess_timeout(time_limit):
    # Marge de sécurité Python au-delà du time_limit du solveur (qui gère son
    # propre cutoff en interne). time_limit <= 0 = illimité côté solveur ->
    # on garde un plafond dur pour ne jamais bloquer indéfiniment.
    return (time_limit + 60) if time_limit > 0 else 1800


def run_base(ds, time_limit):
    cmd = [str(BAB_BASE_BIN), str(ds), str(EPSILON), str(time_limit), str(REPLICATION)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=_subprocess_timeout(time_limit))
    t_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Erreur {proc.returncode}\n{proc.stderr.strip()}")
    return proc.stdout, t_wall


def run_bab(ds, mode_str, time_limit):
    cmd = [
        str(BAB_BIN), str(ds),
        str(EPSILON), str(T_NODE), str(time_limit),
        str(REPLICATION), str(T_ROOT), mode_str,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=_subprocess_timeout(time_limit))
    t_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Erreur {proc.returncode}\n{proc.stderr.strip()}")
    return proc.stdout, t_wall


# ── Parsing ──────────────────────────────────────────────────────────────────

def _pf(out, pat, default=float("nan")):
    m = re.search(pat, out)
    return float(m.group(1)) if m else default


def _pi(out, pat, default=-1):
    m = re.search(pat, out)
    return int(m.group(1)) if m else default


def parse_base(out):
    st = re.search(r"Statut\s*:\s*(\S+)", out)
    return dict(
        statut = st.group(1) if st else "?",
        t_solve = _pf(out, r"Temps total\s*:\s*([\d.]+)"),
        nodes   = _pi(out, r"N.+uds Gecode\s*:\s*(\d+)"),
    )


def parse_bab(out):
    st = re.search(r"Statut\s*:\s*(\S+)", out)
    fails = _pi(out, r"Échecs totaux\s*:\s*(\d+)")
    more_calls = _pi(out, r"Appels more\(\).*?:\s*(\d+)")
    return dict(
        statut      = st.group(1) if st else "?",
        t_solve     = _pf(out, r"Temps total\s*:\s*([\d.]+)"),
        t_bab_pure  = _pf(out, r"Temps BAB pur\s*:\s*([\d.]+)"),
        nodes       = _pi(out, r"N.+uds Gecode\s*:\s*(\d+)"),
        fails       = fails,
        fails_bound = _pi(out, r"dont borne.*?:\s*(\d+)"),
        more_calls  = more_calls,
        ub_tight    = _pi(out, r"UB V1LD utile.*?:\s*(\d+)"),
        inc_imp     = _pi(out, r"Am.lioration.+incumbent\s*:\s*(\d+)"),
    )


def safe_pct(a, b):
    return round(100.0 * a / b, 1) if b and b > 0 else None


# ── Formatage tableau ──────────────────────────────────────────────────────────

def fmt_time(t_solve, t_wall, timed_out):
    t = t_solve if t_solve == t_solve else t_wall  # NaN check
    suffix = "+" if timed_out else ""
    return f"{t:.3f}{suffix}"


def fmt_bab_pure(t_bab_pure, timed_out):
    if t_bab_pure != t_bab_pure:  # NaN check
        return "?"
    suffix = "+" if timed_out else ""
    return f"{t_bab_pure:.3f}{suffix}"


def fmt_cell_pct(count, pct):
    if pct is None:
        return "– (–%)"
    return f"{count} ({pct}%)"


def build_table(rows):
    header = ["Jeu / Méthode", "Statut", "Temps total (s)", "Temps BAB pur (s)",
               "Nb. nœuds",
               "Élagages tot. (dont borne %)", "Appels à more (dont UB utile %)",
               "Amél. incumbent"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def parse_instances_arg(specs):
    out = []
    for spec in specs:
        n, m, N = (int(x) for x in spec.split(","))
        out.append((f"synth_n{n}_m{m}_N{N}", n, m, N))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_limit", type=float, default=120.0,
                     help="Budget de temps par run (s). -1 = illimité pour 'base'.")
    ap.add_argument("--instances", nargs="+", default=None,
                     help="Liste d'instances 'n,m,N' (ex: 10,5,50 10,5,100). "
                          "Défaut : INSTANCES codé en dur.")
    ap.add_argument("--out_md", default=str(ROOT / "results" / "table_synth.md"),
                     help="Fichier markdown de sortie (résultat persistant).")
    args = ap.parse_args()

    instances = parse_instances_arg(args.instances) if args.instances else INSTANCES

    rows = []
    for label, n, m, N in instances:
        ds = dataset_path(n, m, N)
        if not ds.exists():
            print(f"[SKIP] dataset manquant : {ds}")
            continue

        rows.append([f"**{label}** (n={n}, m={m}, N={N})", "", "", "", "", "", "", ""])

        # base — pas de borne calculée : le temps total EST le temps BAB pur
        print(f"[{label}] base ...", flush=True)
        try:
            out, t_wall = run_base(ds, args.time_limit)
            r = parse_base(out)
            timed_out = r["statut"] not in ("OPTIMAL",)
            t_str = fmt_time(r["t_solve"], t_wall, timed_out)
            rows.append([
                "base", r["statut"], t_str, t_str,
                str(r["nodes"]), "– (–%)", "– (–%)", "–",
            ])
            print(f"  -> {r['statut']}  t={r['t_solve']:.3f}s  nodes={r['nodes']}")
        except subprocess.TimeoutExpired:
            rows.append(["base", "TIMEOUT", f">{args.time_limit:.0f}+", f">{args.time_limit:.0f}+",
                         "?", "– (–%)", "– (–%)", "–"])
            print(f"  -> TIMEOUT (> {args.time_limit}s)")

        # V1LD / V1CLD
        for mode_str, mode_label in [("v1ld", "V1LD"), ("v1cld", "V1CLD")]:
            print(f"[{label}] {mode_label} ...", flush=True)
            try:
                out, t_wall = run_bab(ds, mode_str, args.time_limit)
                r = parse_bab(out)
                timed_out = r["statut"] not in ("OPTIMAL",)
                rows.append([
                    mode_label, r["statut"],
                    fmt_time(r["t_solve"], t_wall, timed_out),
                    fmt_bab_pure(r["t_bab_pure"], timed_out),
                    str(r["nodes"]),
                    fmt_cell_pct(r["fails"], safe_pct(r["fails_bound"], r["fails"])),
                    fmt_cell_pct(r["more_calls"], safe_pct(r["ub_tight"], r["more_calls"])),
                    str(r["inc_imp"]),
                ])
                print(f"  -> {r['statut']}  t={r['t_solve']:.3f}s  t_bab_pur={r['t_bab_pure']:.3f}s  nodes={r['nodes']}")
            except subprocess.TimeoutExpired:
                rows.append([mode_label, "TIMEOUT", f">{args.time_limit:.0f}+", "?",
                             "?", "– (–%)", "– (–%)", "–"])
                print(f"  -> TIMEOUT (> {args.time_limit}s)")

    table_md = build_table(rows)
    print()
    print(table_md)

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# table_synth — base / V1LD / V1CLD (ε={EPSILON}, T_root={T_ROOT}, "
        f"T_node={T_NODE}, time_limit={args.time_limit}s)\n\n"
        "Instances Song-Pisinger (weakly_correlated, α=0.5, R=1000, "
        f"seed={SEED}), générées via data/knapsack/create_database_knapsack_pisinger.py.\n\n"
    )
    out_path.write_text(header + table_md + "\n")
    print(f"\n[écrit] {out_path}")


if __name__ == "__main__":
    main()
