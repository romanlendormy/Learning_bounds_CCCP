#!/usr/bin/env python3
"""
extract_time_vs_tstar.py — MKP : pour chaque (formulation, n, N), nombre d'itérations de sous-gradient t* nécessaire
pour battre la borne GNN root-only, et comparaison du temps d'inférence GNN au temps d'exécution de ces t* itérations.

Sources :
  - t*, censure, t_max : results_gnn_vs_subgrad_detail.csv (parse_gnn_vs_subgrad.py).
  - temps GNN : logs gnn_root_<tag>.log, ligne de profondeur 1 du tableau « utilité de la borne par profondeur » (ms/nœud) =
    durée d'UN appel de borne GNN à la racine (construction du graphe + forward + évaluation de L(λ)), pré-chauffage TorchScript
    exclu. (« Temps borne root » n'est pas exploitable dans ces logs : imprimé en std::fixed à 1 décimale -> « 0.0 s ».)
    Le forward seul est aussi relevé (« Inférence pure (forward) », cumulé sur les 2 appels de profondeur 1 -> /2).
  - temps sous-gradient : logs subgrad_<tag>.stdout.log, « Temps borne root » = durée du cold start racine (dernier enfant direct,
    même bloc que la trace retenue pour t*) :
      * n=20..30 (run_gnn_vs_subgrad_multi_n.sh, arrêt anticipé GNN_COMPARE_BOUND) : exactement t* itérations
        (t_max = t* si résolu ; si censuré, t_max = 300 itérations -> temps minorant) ;
      * n=15 (run_gnn_vs_subgrad_n15.sh, trace complète jusqu'à convergence, t_max itérations) : temps(t*) = temps racine * t*/t_max
        (coût par itération supposé constant).
  Dossiers de logs : $SCRATCH/test_paper_mkp/gnn_vs_subgrad_{n15,multin}/logs (OUT_DIR des scripts de lancement),
  modifiables par LOGS_N15 / LOGS_MULTIN. Les logs de la campagne de l'article portent les anciens noms de formulation
  (v1ld = sld, v1cld = dld) : les deux sont acceptés.

Sorties : latex_data/time_vs_tstar_detail.csv (par instance) et latex_data/time_vs_tstar_<form>.csv (médianes par (n, N), pgfplots).
Médianes avec censure à droite : un censuré compte comme +inf ; si la médiane tombe sur un censuré, elle est rapportée comme minorant
(colonne *_lb = 1). Usage : python3 extract_time_vs_tstar.py
"""
import csv
import math
import os
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
DETAIL = HERE / "results_gnn_vs_subgrad_detail.csv"
_OUT = Path(os.environ.get("SCRATCH", "runs")) / "test_paper_mkp"
LOGDIRS = {15: Path(os.environ.get("LOGS_N15", _OUT / "gnn_vs_subgrad_n15" / "logs"))}
LOGDIRS.update({n: Path(os.environ.get("LOGS_MULTIN", _OUT / "gnn_vs_subgrad_multin" / "logs")) for n in (20, 25, 30)})
OLD = {"sld": "v1ld", "dld": "v1cld"}

DEPTH1_RE = re.compile(r"^\s+1\s+\d+\s+\d+\s+\d+\s+\S+%\s+\S+%\s+([0-9.]+)\s", re.M)
FORWARD_RE = re.compile(r"Inférence pure \(forward\)\s*:\s*([0-9.eE+-]+) s")
ROOT_T_RE = re.compile(r"Temps borne root\s*:\s*([0-9.eE+-]+) s")


def find(path_glob_dir, names):
    for nm in names:
        p = path_glob_dir / nm
        if p.exists():
            return p
    return None


def median_censored(vals):
    """vals : liste de (valeur, censuré). Renvoie (médiane, minorant?)."""
    s = sorted(vals, key=lambda x: (x[1], x[0]))   # censurés (+inf) en fin
    k = len(s)
    lo, hi = s[(k - 1) // 2], s[k // 2]
    if hi[1]:   # la médiane touche un censuré -> minorant (valeur finie du censuré = itérations/temps déjà exécutés)
        return (statistics.median([lo[0], hi[0]]) if not lo[1] else lo[0]), 1
    return statistics.median([lo[0], hi[0]]), 0


rows = []
with open(DETAIL) as fh:
    for r in csv.DictReader(fh):
        n, N, seed, form = int(r["n"]), int(r["N"]), int(r["seed"]), r["formulation"]
        d = LOGDIRS[n]
        tag_new, tag_old = f"n{n}_N{N}_seed{seed}_{form}", f"n{n}_N{N}_seed{seed}_{OLD[form]}"
        g = find(d, [f"gnn_root_{tag_new}.log", f"gnn_root_{tag_old}.log"])
        s = find(d, [f"subgrad_{tag_new}.stdout.log", f"subgrad_{tag_old}.stdout.log", f"subgrad_{tag_new}.log", f"subgrad_{tag_old}.log"])
        if g is None or s is None:
            print(f"[manquant] {tag_new} gnn={g} subgrad={s}")
            continue
        gt, st_ = g.read_text(errors="replace"), s.read_text(errors="replace")
        m1, mf, mr = DEPTH1_RE.search(gt), FORWARD_RE.search(gt), ROOT_T_RE.search(st_)
        if not (m1 and mf and mr):
            print(f"[illisible] {tag_new} depth1={bool(m1)} forward={bool(mf)} root={bool(mr)}")
            continue
        censored = r["censored"] == "True"
        t_max = int(r["t_max"])
        t_star = t_max if censored else int(r["t_star"])
        t_root = float(mr.group(1))
        t_sub = t_root if n != 15 else t_root * t_star / t_max
        rows.append(dict(n=n, formulation=form, N=N, seed=seed, t_star=t_star, censored=int(censored), t_max=t_max,
                         gnn_time_s=float(m1.group(1)) / 1e3, gnn_forward_s=float(mf.group(1)) / 2,
                         subgrad_root_time_s=t_root, subgrad_time_tstar_s=t_sub))

out = HERE / "latex_data"
with open(out / "time_vs_tstar_detail.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

for form in ("sld", "dld"):
    agg = []
    for n in (15, 20, 25, 30):
        for N in (100, 500, 1000):
            rs = [r for r in rows if r["formulation"] == form and r["n"] == n and r["N"] == N]
            if not rs:
                continue
            ts, ts_lb = median_censored([(r["t_star"], r["censored"]) for r in rs])
            tt, tt_lb = median_censored([(r["subgrad_time_tstar_s"], r["censored"]) for r in rs])
            gnn = statistics.median(r["gnn_time_s"] for r in rs)
            agg.append(dict(n=n, N=N, t_star_median=ts, t_star_lb=ts_lb, n_resolved=sum(1 - r["censored"] for r in rs), n_total=len(rs),
                            subgrad_time_median_s=tt, subgrad_time_lb=tt_lb, gnn_time_median_s=gnn,
                            gnn_forward_median_s=statistics.median(r["gnn_forward_s"] for r in rs), speedup=tt / gnn))
    with open(out / f"time_vs_tstar_{form}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0])); w.writeheader(); w.writerows(agg)
    print(f"\n== {form.upper()}")
    print(f"{'n':>3} {'N':>5} {'t*':>8} {'res':>6} {'t_sub(s)':>10} {'t_gnn(s)':>9} {'fwd(s)':>8} {'ratio':>7}")
    for a in agg:
        print(f"{a['n']:>3} {a['N']:>5} {('≥' if a['t_star_lb'] else '') + format(a['t_star_median'], 'g'):>8} "
              f"{a['n_resolved']:>2}/{a['n_total']:<3} {('≥' if a['subgrad_time_lb'] else '') + format(a['subgrad_time_median_s'], '.3g'):>10} "
              f"{a['gnn_time_median_s']:>9.3g} {a['gnn_forward_median_s']:>8.3g} {a['speedup']:>7.1f}")
print(f"\n{len(rows)} instances -> {out}/time_vs_tstar_*.csv")
