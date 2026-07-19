#!/usr/bin/env python3
"""
table_mkp_s10.py — Tableau base / V1LD / V1CLD (comme table_synth.py) sur des
instances tirées de la collecte data/train_base_s10/ (n=10/20, N=20/50/100).

Contrairement à table_synth.py (instances Song-Pisinger générées à la volée),
les instances viennent ici du format "collecte" (instances_n{n}_m{m}_N{N}.txt,
cf. solver_bab/mkp/gecode_bab_collect.hpp) : une ligne =
  inst_id, m, n, N, epsilon, opt, c[n], b[N*m], A[N*m*n]

On convertit la ligne choisie (--inst_index, défaut 0 = première du fichier)
au format "song" par scénario attendu par bab_mkp_gecode(_base) (une ligne par
scénario), dans un fichier temporaire, puis on réutilise les runners/parsers
de table_synth.py (mêmes régex de sortie des binaires).

Usage :
    python sweep_bab/table_mkp_s10.py
    python sweep_bab/table_mkp_s10.py --time_limit 300 --pairs 10,20 20,50
"""

import argparse
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

BAB_BIN      = ROOT / "solver_bab" / "build" / "bab_mkp_gecode"
BAB_BASE_BIN = ROOT / "solver_bab" / "build" / "bab_mkp_gecode_base"
DATA_DIR     = ROOT / "data" / "train_base_s10"

EPSILON      = 0.10
T_ROOT       = 200   # itérations sous-gradient, cold start (racine)
T_NODE       = 30    # itérations sous-gradient, warm-start (nœuds suivants)
REPLICATION  = 0

M_FIXED = 5   # seul m collecté dans train_base_s10

# (n, N) à couvrir par défaut
PAIRS = [
    (10, 20), (10, 50), (10, 100),
    (20, 20), (20, 50), (20, 100),
]


# ── Chargement instance depuis le format "collecte" + conversion "song" ──────

def collect_file(n, m, N):
    return DATA_DIR / f"instances_n{n}_m{m}_N{N}.txt"


def load_instance_from_collect(path, index=0):
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == index:
                vals = [float(x) for x in line.split(",")]
                break
        else:
            raise ValueError(f"Index {index} introuvable dans {path}")

    inst_id = int(vals[0]); m = int(vals[1]); n = int(vals[2]); N = int(vals[3])
    epsilon = vals[4]; opt = vals[5]

    off = 6
    c = vals[off:off + n]; off += n
    b = vals[off:off + N * m]; off += N * m
    A = vals[off:off + N * m * n]

    return dict(inst_id=inst_id, n=n, m=m, N=N, epsilon=epsilon, opt=opt, c=c, b=b, A=A)


def write_song_tempfile(inst):
    """Convertit une instance (format collecte, compactée) en fichier "song"
    une-ligne-par-scénario, tel qu'attendu par load_song_local() dans
    solver_bab/mkp/main_gecode.cpp / main_gecode_base.cpp."""
    n, m, N = inst["n"], inst["m"], inst["N"]
    c, b, A = inst["c"], inst["b"], inst["A"]

    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="ccmk_s10_")
    for i in range(N):
        A_i = A[i * m * n:(i + 1) * m * n]   # déjà en ordre (j,k) row-major
        b_i = b[i * m:(i + 1) * m]
        header = [0, i, m, n, 0]
        row = header + A_i + b_i + c
        tf.write(",".join(str(v) for v in row) + "\n")
    tf.close()
    return tf.name


# ── Runners (identiques à table_synth.py) ───────────────────────────────────

def _subprocess_timeout(time_limit):
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


# ── Parsing (étend table_synth.py : ajoute fails pour base, temps borne total) ─

def _pf(out, pat, default=float("nan")):
    m = re.search(pat, out)
    return float(m.group(1)) if m else default


def _pi(out, pat, default=-1):
    m = re.search(pat, out)
    return int(m.group(1)) if m else default


def parse_base(out):
    st = re.search(r"Statut\s*:\s*(\S+)", out)
    return dict(
        statut  = st.group(1) if st else "?",
        t_solve = _pf(out, r"Temps total\s*:\s*([\d.]+)"),
        nodes   = _pi(out, r"N.+uds Gecode\s*:\s*(\d+)"),
        fails   = _pi(out, r"Échecs totaux\s*:\s*(\d+)"),
    )


def parse_bab(out):
    st = re.search(r"Statut\s*:\s*(\S+)", out)
    fails = _pi(out, r"Échecs totaux\s*:\s*(\d+)")
    return dict(
        statut        = st.group(1) if st else "?",
        t_solve       = _pf(out, r"Temps total\s*:\s*([\d.]+)"),
        t_bound_total = _pf(out, r"Temps borne total\s*:\s*([\d.]+)"),
        t_bab_pure    = _pf(out, r"Temps BAB pur\s*:\s*([\d.]+)"),
        nodes         = _pi(out, r"N.+uds Gecode\s*:\s*(\d+)"),
        fails         = fails,
        fails_bound   = _pi(out, r"dont borne.*?:\s*(\d+)"),
    )


def safe_pct(a, b):
    return round(100.0 * a / b, 1) if b and b > 0 else None


# ── Formatage tableau (texte aligné, pas de pipes markdown) ─────────────────

COL_METHODE = 20
COL_STATUT  = 10
COL_NUM     = 14
COL_FAILS   = 20


def fmt_num(x, suffix=""):
    if x != x:  # NaN
        return "-"
    return f"{x:.3f}{suffix}"


def fmt_fails(fails, fails_bound):
    if fails is None or fails < 0:
        return "-"
    if fails_bound is None or fails_bound < 0:
        return f"{fails}"
    pct = safe_pct(fails_bound, fails)
    pct_str = f"{pct}%" if pct is not None else "-"
    return f"{fails} ({fails_bound} dont borne, {pct_str})"


def header_line():
    return (f"{'Méthode':<{COL_METHODE}}{'Statut':<{COL_STATUT}}"
            f"{'T total(s)':>{COL_NUM}}{'T borne(s)':>{COL_NUM}}{'T BAB pur(s)':>{COL_NUM}}"
            f"{'Nb nœuds':>{COL_NUM}}  {'Fails (dont borne)':<{COL_FAILS}}")


def row_line(methode, statut, t_total, t_borne, t_bab_pur, nodes, fails, fails_bound, timed_out):
    suffix = "+" if timed_out else ""
    return (f"{methode:<{COL_METHODE}}{statut:<{COL_STATUT}}"
            f"{fmt_num(t_total, suffix):>{COL_NUM}}{fmt_num(t_borne):>{COL_NUM}}"
            f"{fmt_num(t_bab_pur, suffix):>{COL_NUM}}{nodes:>{COL_NUM}}  "
            f"{fmt_fails(fails, fails_bound):<{COL_FAILS}}")


def parse_pairs_arg(specs):
    out = []
    for spec in specs:
        n, N = (int(x) for x in spec.split(","))
        out.append((n, N))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_limit", type=float, default=120.0,
                     help="Budget de temps par run (s).")
    ap.add_argument("--pairs", nargs="+", default=None,
                     help="Liste de paires 'n,N' (ex: 10,20 10,50 20,100). "
                          "Défaut : PAIRS codé en dur (n=10/20, N=20/50/100).")
    ap.add_argument("--inst_index", type=int, default=0,
                     help="Index (ligne) de l'instance à charger dans le fichier collecte.")
    ap.add_argument("--out_txt", default=str(ROOT / "results" / "table_mkp_s10.txt"),
                     help="Fichier texte de sortie (résultat persistant).")
    args = ap.parse_args()

    pairs = parse_pairs_arg(args.pairs) if args.pairs else PAIRS

    blocks = []
    for n, N in pairs:
        m = M_FIXED
        path = collect_file(n, m, N)
        label = f"n={n}  m={m}  N={N}"

        if not path.exists():
            print(f"[SKIP] fichier manquant : {path}")
            blocks.append((label, None, f"[SKIP] fichier manquant : {path.name}"))
            continue

        inst = load_instance_from_collect(path, args.inst_index)
        ds = write_song_tempfile(inst)
        print(f"[{label}] inst_id={inst['inst_id']}  opt={inst['opt']:.3f}  "
              f"(fichier {path.name}, ligne {args.inst_index})")

        rows = []
        try:
            # base — pas de borne calculée : temps total == temps BAB pur
            print(f"  base ...", flush=True)
            try:
                out, t_wall = run_base(ds, args.time_limit)
                r = parse_base(out)
                timed_out = r["statut"] not in ("OPTIMAL",)
                t = r["t_solve"] if r["t_solve"] == r["t_solve"] else t_wall
                rows.append(("Gecode base", r["statut"], t, float("nan"), t,
                             r["nodes"], r["fails"], None, timed_out))
                print(f"    -> {r['statut']}  t={t:.3f}s  nodes={r['nodes']}  fails={r['fails']}")
            except subprocess.TimeoutExpired:
                rows.append(("Gecode base", "TIMEOUT", args.time_limit, float("nan"),
                             args.time_limit, -1, -1, None, True))
                print(f"    -> TIMEOUT (> {args.time_limit}s)")

            for mode_str, mode_label in [("ref", "B&B REF (sans borne)"),
                                          ("v1ld", "B&B V1LD"), ("v1cld", "B&B V1CLD")]:
                print(f"  {mode_label} ...", flush=True)
                try:
                    out, t_wall = run_bab(ds, mode_str, args.time_limit)
                    r = parse_bab(out)
                    timed_out = r["statut"] not in ("OPTIMAL",)
                    t_total = r["t_solve"] if r["t_solve"] == r["t_solve"] else t_wall
                    rows.append((mode_label, r["statut"], t_total, r["t_bound_total"],
                                 r["t_bab_pure"], r["nodes"], r["fails"], r["fails_bound"],
                                 timed_out))
                    print(f"    -> {r['statut']}  t={t_total:.3f}s  t_borne={r['t_bound_total']:.3f}s  "
                          f"t_bab_pur={r['t_bab_pure']:.3f}s  nodes={r['nodes']}")
                except subprocess.TimeoutExpired:
                    rows.append((mode_label, "TIMEOUT", args.time_limit, float("nan"),
                                 float("nan"), -1, -1, None, True))
                    print(f"    -> TIMEOUT (> {args.time_limit}s)")
        finally:
            os.unlink(ds)

        extra = f"  (inst_id={inst['inst_id']}, opt={inst['opt']:.3f})"
        blocks.append((label + extra, rows, None))

    # ── Construction texte final ─────────────────────────────────────────────
    lines = [
        "TABLEAU MKP — base / B&B REF / B&B V1LD / B&B V1CLD sur instances data/train_base_s10",
        f"(eps={EPSILON}, T_root={T_ROOT}, T_node={T_NODE}, time_limit={args.time_limit}s, "
        f"inst_index={args.inst_index})",
        "REF = squelette V1LD nu (même heuristique ratio, même incumbent glouton, même",
        "moteur DFS que V1LD/V1CLD) mais sans aucun calcul de borne — sert de référence",
        "à heuristique/moteur fixés, pour isoler l'effet de la borne seule (REF vs V1LD/",
        "V1CLD) de celui de l'heuristique+incumbent+moteur (Gecode base vs REF).",
        "",
    ]
    for label, rows, skip_msg in blocks:
        lines.append("=" * 100)
        lines.append(label)
        lines.append("=" * 100)
        if skip_msg:
            lines.append(skip_msg)
            lines.append("")
            continue
        lines.append(header_line())
        lines.append("-" * 100)
        for row in rows:
            lines.append(row_line(*row))
        lines.append("")

    table_txt = "\n".join(lines)
    print()
    print(table_txt)

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table_txt + "\n")
    print(f"\n[écrit] {out_path}")


if __name__ == "__main__":
    main()
