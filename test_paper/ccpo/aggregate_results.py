#!/usr/bin/env python3
"""
aggregate_results.py — CCPO : réunit les CSV par run (results/ref/, results/ld0/, results/ld/, écrits par parse_result.py) en
un seul CSV avec en-tête, calcule gap_root_pct hors ligne, et imprime la comparaison REF / LD0 / LD par cellule (n, N).

Usage (depuis la racine du dépôt) :
  python3 test_paper/ccpo/aggregate_results.py [dossier_resultats] [--out results.csv] [--stat median|mean|geomean]
  (défaut : test_paper/ccpo/results ; tous les *.csv des sous-dossiers sont lus)

- Optimum d'une instance : objectif d'un run OPTIMAL de cette instance (les trois bras donnent le même).
- gap_root_pct = 100 · (borne du premier niveau − optimum) / optimum (LD0 et LD ; REF n'a pas de borne lagrangienne).
- Tableau : statistique (médiane par défaut) sur les runs OPTIMAL de la cellule du temps de résolution (s), du nombre de nœuds
  et du gap ; (s/t) = résolues / total. Colonnes REF/LD et LD0/LD : statistique, sur les instances résolues par les deux bras,
  du rapport des temps par instance (> 1 : LD plus rapide) ; pour LD0/LD, nombre d'instances où LD est plus rapide.
"""
import argparse
import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_result import FIELDS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("results_dir", nargs="?", default=os.path.join(HERE, "results"))
ap.add_argument("--out", default="results.csv")
ap.add_argument("--stat", default="median", choices=["median", "mean", "geomean"])
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(a.results_dir, "**", "*.csv"), recursive=True)):
    with open(f) as fh:
        for r in csv.reader(fh):
            if r and r[0] != "n":
                rows.append(dict(zip(FIELDS, r + [""] * (len(FIELDS) - len(r)))))
fl = lambda x: float(x) if x not in ("", None) else float("nan")

opt = {}
for r in rows:
    if r["status"] == "OPTIMAL":
        opt.setdefault((r["n"], r["N"], r["idx"]), fl(r["objective"]))
for r in rows:
    bound = fl(r["first_level_bound_value"]) if r["mode"] in ("gnn", "gnn_zero") else float("nan")
    o = opt.get((r["n"], r["N"], r["idx"]), float("nan"))
    r["gap_root_pct"] = "" if (math.isnan(bound) or math.isnan(o) or o <= 0) else repr(100.0 * (bound - o) / o)
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(FIELDS)
    for r in rows:
        w.writerow([r[k] for k in FIELDS])
print(f"{len(rows)} runs -> {a.out}")


def stat(v):
    v = [x for x in v if not math.isnan(x)]
    if not v:
        return float("nan")
    if a.stat == "median":
        return statistics.median(v)
    if a.stat == "mean":
        return statistics.fmean(v)
    return math.exp(sum(math.log(max(x, 1e-12)) for x in v) / len(v))


def fmt_t(x):
    return "—" if math.isnan(x) else (f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 1000 else f"{x / 1000:.2f}K"))


def fmt_n(x):
    return "—" if math.isnan(x) else (f"{x:.0f}" if x < 1000 else (f"{x / 1000:.2f}K" if x < 1e6 else f"{x / 1e6:.2f}M"))


cells = defaultdict(lambda: defaultdict(dict))   # (n, N) -> mode -> idx -> run
for r in rows:
    cells[(int(r["n"]), int(r["N"]))][r["mode"]][r["idx"]] = r
arms = (("feas", "REF"), ("gnn_zero", "LD0"), ("gnn", "LD"))
print(f"\n{a.stat} sur les runs OPTIMAL ; (s/t) = résolues/total ; gap = gap du premier niveau (%) vs optimum")
print(f"{'n':>3} {'N':>5} | " + " | ".join(f"{h} t(s) nœuds" + (" gap" if m != "feas" else "") + " (s/t)" for m, h in arms)
      + " | REF/LD | LD0/LD  LD plus rapide")
for (n, N) in sorted(cells):
    out = []
    for m, _ in arms:
        rs = list(cells[(n, N)].get(m, {}).values())
        sol = [r for r in rs if r["status"] == "OPTIMAL"]
        if not sol:
            out.append(f"— ({len(sol)}/{len(rs)})")
            continue
        t, nd = stat([fl(r["solve_time_s"]) for r in sol]), stat([fl(r["nodes"]) for r in sol])
        gap = "" if m == "feas" else f" {stat([fl(r['gap_root_pct']) for r in sol]):.1f}"
        out.append(f"{fmt_t(t)} {fmt_n(nd)}{gap} ({len(sol)}/{len(rs)})")
    ld = cells[(n, N)].get("gnn", {})
    cmp_ = []
    for m in ("feas", "gnn_zero"):
        other = cells[(n, N)].get(m, {})
        both = [i for i in ld if i in other and ld[i]["status"] == other[i]["status"] == "OPTIMAL"]
        ratios = [fl(other[i]["solve_time_s"]) / fl(ld[i]["solve_time_s"]) for i in both]
        cmp_.append(("—" if not both else f"{stat(ratios):.2f}") + ("" if m == "feas" or not both else f"  {sum(x > 1 for x in ratios)}/{len(both)}"))
    print(f"{n:>3} {N:>5} | " + " | ".join(out) + " | " + " | ".join(cmp_))
