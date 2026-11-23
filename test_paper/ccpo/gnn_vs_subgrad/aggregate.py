#!/usr/bin/env python3
"""
aggregate.py — CCPO, comparaison t* (GNN un-coup vs sous-gradient à la racine) : joint les CSV par instance des deux phases
(run_gnn.sh -> results_gnn/, run_subgrad_nodepack.sh -> results_subgrad/) et produit, au format de la figure MKP
(test_paper/mkp/gnn_vs_subgrad/fig_tstar_sld.tex) :
  - results.csv (toutes les instances, à côté du dossier subgrad) et latex_data/time_vs_tstar_ccpo.csv (médianes par (n, N)) ;
  - fig_tstar_ccpo.tex (environnement figure pgfplots, données intégrées, à \\input dans l'article).

Temps d'un appel GNN = graphe + forward (phase 1, GPU ou CPU selon DEVICE) + évaluation de L(λ_GNN) mesurée en phase 2 sur le MÊME
CPU que le sous-gradient (gnn_eval_cpu_s) : l'évaluation domine ce coût, et ce choix évite de comparer deux processeurs différents.
Le temps total mesuré en phase 1 (gnn_time_s, évaluation sur le CPU du nœud GPU) est conservé dans results.csv.

Pour les censurés, last_impr_iter (dernière amélioration stricte de la meilleure borne) et gap_to_gnn_pct (écart restant à la
borne GNN) disent si le sous-gradient stagnait ou progressait encore à la coupure ; ils sont listés à part.

Médianes avec censure à droite (t* non atteint en T_max itérations ou TIME_LIM secondes) : un censuré compte comme +inf ;
si la médiane tombe sur un censuré, elle est rapportée comme MINORANT (valeur = itérations / temps effectivement exécutés,
colonne *_lb = 1, marqueur creux dans la figure). Aucune dépendance hors stdlib.

Usage : python3 aggregate.py $SCRATCH/test_paper_ccpo_tstar/results_gnn $SCRATCH/test_paper_ccpo_tstar/results_subgrad [--fig fig_tstar_ccpo.tex]
"""
import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("gnn_dir")
ap.add_argument("subgrad_dir")
ap.add_argument("--fig", default="fig_tstar_ccpo.tex")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))


def read_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                out[(r["n"], r["N"], r["idx"])] = r
    return out


gnn, sub = read_dir(a.gnn_dir), read_dir(a.subgrad_dir)
missing = sorted(set(gnn) ^ set(sub))
if missing:
    print(f"[attention] {len(missing)} instances présentes dans une seule phase (ignorées) : "
          + ", ".join(f"n{k[0]}_N{k[1]}_i{k[2]}" for k in missing[:10]) + (" ..." if len(missing) > 10 else ""))
rows = []
for key in sorted(set(gnn) & set(sub), key=lambda k: tuple(map(int, k))):
    g, s_ = gnn[key], sub[key]
    if abs(float(g["gnn_bound"]) - float(s_["gnn_bound"])) > 1e-6 * max(1.0, abs(float(g["gnn_bound"]))):
        print(f"[attention] n{key[0]}_N{key[1]}_i{key[2]} : borne GNN phase 1 {g['gnn_bound']} != ré-évaluée {s_['gnn_bound']}")
    r = {**{k: g[k] for k in ("idx", "seed", "n", "N", "n_free", "gnn_bound", "gnn_graph_s", "gnn_forward_s")},
         "gnn_eval_cpu_s": s_["gnn_eval_cpu_s"],
         "gnn_call_s": repr(float(g["gnn_graph_s"]) + float(g["gnn_forward_s"]) + float(s_["gnn_eval_cpu_s"])),
         "gnn_time_phase1_s": g["gnn_time_s"], "gnn_device": g["device"], "gnn_host": g["host"],
         **{k: s_[k] for k in ("subgrad_first", "subgrad_best", "t_star", "t_star_time_s", "iters_run", "subgrad_time_s",
                               "stop_reason", "conv_iter", "last_impr_iter", "last_impr_time_s", "omp", "host")},
         "gap_to_gnn_pct": repr(100.0 * (float(s_["subgrad_best"]) - float(g["gnn_bound"])) / float(g["gnn_bound"]))}
    rows.append(r)
if not rows:
    raise SystemExit("aucune instance commune aux deux phases")
with open(os.path.join(os.path.dirname(os.path.abspath(a.subgrad_dir)), "results.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def median_censored(vals):
    """vals : liste de (valeur, censuré) -> (médiane, minorant 0/1)."""
    s = sorted(vals, key=lambda x: (x[1], x[0]))
    k = len(s)
    lo, hi = s[(k - 1) // 2], s[k // 2]
    if hi[1]:
        return (lo[0] if lo[1] else statistics.median([lo[0], hi[0]])), 1
    return statistics.median([lo[0], hi[0]]), 0


cells = defaultdict(list)
for r in rows:
    cells[(int(r["n"]), int(r["N"]))].append(r)
agg = []
for (n, N), rs in sorted(cells.items()):
    cens = [r["stop_reason"] != "reached" for r in rs]
    ts, ts_lb = median_censored([(float(r["t_star"]) if not c else float(r["iters_run"]), c) for r, c in zip(rs, cens)])
    tt, tt_lb = median_censored([(float(r["t_star_time_s"]) if not c else float(r["subgrad_time_s"]), c) for r, c in zip(rs, cens)])
    gnn = statistics.median(float(r["gnn_call_s"]) for r in rs)
    agg.append(dict(n=n, N=N, t_star_median=ts, t_star_lb=ts_lb, n_resolved=cens.count(False), n_total=len(rs),
                    subgrad_time_median_s=tt, subgrad_time_lb=tt_lb, gnn_time_median_s=gnn,
                    gnn_forward_median_s=statistics.median(float(r["gnn_forward_s"]) for r in rs),
                    gnn_eval_median_s=statistics.median(float(r["gnn_eval_cpu_s"]) for r in rs),
                    n_free_median=statistics.median(int(r["n_free"]) for r in rs), speedup=tt / gnn))
os.makedirs(os.path.join(here, "latex_data"), exist_ok=True)
with open(os.path.join(here, "latex_data", "time_vs_tstar_ccpo.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(agg[0])); w.writeheader(); w.writerows(agg)

cens_rows = [r for r in rows if r["stop_reason"] != "reached"]
if cens_rows:
    print(f"\n{len(cens_rows)} instances censurées (t* non atteint) : dernière amélioration de la borne du sous-gradient, écart restant à la borne GNN")
    print(f"{'n':>3} {'N':>5} {'idx':>5} {'arrêt':>11} {'itér.':>6} {'dern. amél.':>11} {'écart (%)':>9}")
    for r in sorted(cens_rows, key=lambda r: (int(r["n"]), int(r["N"]), int(r["idx"]))):
        print(f"{r['n']:>3} {r['N']:>5} {r['idx']:>5} {r['stop_reason']:>11} {r['iters_run']:>6} {r['last_impr_iter']:>11} {float(r['gap_to_gnn_pct']):>9.3f}")
    print()
print(f"{len(rows)} instances ; médianes (≥ = minorant, censure)")
print(f"{'n':>3} {'N':>5} {'t*':>8} {'res':>6} {'t_sub(s)':>10} {'t_gnn(s)':>9} {'fwd(s)':>8} {'eval(s)':>8} {'ratio':>8}")
for g in agg:
    print(f"{g['n']:>3} {g['N']:>5} {('≥' if g['t_star_lb'] else '') + format(g['t_star_median'], 'g'):>8} "
          f"{g['n_resolved']:>2}/{g['n_total']:<3} {('≥' if g['subgrad_time_lb'] else '') + format(g['subgrad_time_median_s'], '.3g'):>10} "
          f"{g['gnn_time_median_s']:>9.3g} {g['gnn_forward_median_s']:>8.3g} {g['gnn_eval_median_s']:>8.3g} {g['speedup']:>8.1f}")

# ---- figure (même mise en forme que fig_tstar_sld.tex côté MKP) ----
ns = sorted({g["n"] for g in agg})
Ns = sorted({g["N"] for g in agg})
cols = {100: "6DA7EC", 500: "2A78D6", 1000: "104281"}
ymax_t = max(g["t_star_median"] for g in agg) * 1.12
tmin = min(min(g["gnn_time_median_s"], g["subgrad_time_median_s"]) for g in agg)
tmax = max(g["subgrad_time_median_s"] for g in agg)
data = "n  N     tstar  tstarlb  tsub      tsublb  tgnn\n" + "\n".join(
    f"{g['n']:<2} {g['N']:<5} {g['t_star_median']:<6g} {g['t_star_lb']:<8} {g['subgrad_time_median_s']:<9.4g} {g['subgrad_time_lb']:<7} {g['gnn_time_median_s']:.4g}"
    for g in agg)


def plots(y, style, lbcol=None):
    out = []
    for N in Ns:
        out.append(f"    \\addplot+[{style}, color=cc{N}, mark options={{fill=cc{N}}}, x filter/.expression={{\\thisrow{{N}}=={N} ? x : nan}}] "
                   f"table[x=n, y={y}] {{\\tstarccpo}};")
        if lbcol:
            out.append(f"    \\addplot+[only marks, mark=o, mark size=3pt, line width=0.8pt, color=cc{N}, mark options={{fill=white}}, forget plot, "
                       f"x filter/.expression={{(\\thisrow{{N}}=={N} && \\thisrow{{{lbcol}}}==1) ? x : nan}}] table[x=n, y={y}] {{\\tstarccpo}};")
    return "\n".join(out)


n_cens = len(cens_rows)
T_MAX, TIME_LIM = "1\\,000", "3\\,600"
has_lb = any(g["t_star_lb"] or g["subgrad_time_lb"] for g in agg)
legend_N = ", ".join(f"$N={N}$" for N in Ns)
fig = rf"""%% fig_tstar_ccpo.tex -- CCPO: subgradient iterations t* needed to beat the GNN root bound, and time of these t* iterations
%% vs. one GNN bound call. Medians over the instances of each (n, N); generated by aggregate.py from {os.path.abspath(a.gnn_dir)} and {os.path.abspath(a.subgrad_dir)}.
%% Preamble: \usepackage{{pgfplots}} \usepgfplotslibrary{{groupplots}} \pgfplotsset{{compat=1.16}}. Usage: \input{{fig_tstar_ccpo}}
\pgfplotstableread{{
{data}
}}\tstarccpo
\tikzset{{tstarSG/.style={{line width=1pt, mark=*, mark size=1.6pt}},
         tstarGNN/.style={{line width=1pt, dashed, mark=square*, mark size=1.4pt}}}}
""" + "\n".join(f"\\definecolor{{cc{N}}}{{HTML}}{{{cols.get(N, '2A78D6')}}}" for N in Ns) + rf"""
\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group size=2 by 1, horizontal sep=1.45cm}},
    width=0.5\linewidth, height=4.4cm,
    every axis/.append style={{font=\footnotesize}}, xlabel={{$n$}}, xtick={{{",".join(map(str, ns))}}},
    xmin={ns[0] - 0.6}, xmax={ns[-1] + 0.6},
    grid=major, grid style={{gray!20}}, axis line style={{gray!60}}, tick style={{gray!60}},
    unbounded coords=discard,
  ]
  \nextgroupplot[ylabel={{Iterations $t^\star$}}, ymin=0, ymax={ymax_t:.0f},
    legend to name=legtstarccpoN, legend columns={len(Ns)}, legend style={{draw=none, font=\footnotesize, column sep=6pt}}]
{plots("tstar", "tstarSG", "tstarlb")}
    \legend{{{legend_N}}}
  \nextgroupplot[ymode=log, ylabel={{Time (s)}}, ymin={tmin / 2:.2g}, ymax={tmax * 2:.2g},
    legend to name=legtstarccpoM, legend columns=2, legend style={{draw=none, font=\footnotesize, column sep=6pt}}]
    \addlegendimage{{tstarSG, color=black, mark options={{fill=black}}}}
    \addlegendimage{{tstarGNN, color=black, mark options={{fill=black}}}}
{plots("tsub", "tstarSG", "tsublb")}
{plots("tgnn", "tstarGNN")}
    \legend{{Subgradient ($t^\star$ iterations), GNN (one call)}}
\end{{groupplot}}
\end{{tikzpicture}}\\[2pt]
\pgfplotslegendfromname{{legtstarccpoN}}\\
\pgfplotslegendfromname{{legtstarccpoM}}
\caption{{CCPO. Left: number of subgradient iterations $t^\star$ (cold start at the root) needed to reach the bound inferred by the GNN in a single forward pass. Right: wall-clock time of these $t^\star$ iterations (solid) versus one GNN bound call (dashed; graph construction and forward pass on one H100 GPU, evaluation of $L(\lambda)$ on the same 24 CPU cores as the subgradient, TorchScript warm-up excluded). Medians over the 10 instances of each $(n, N)$. The subgradient is capped at {T_MAX}\,iterations or {TIME_LIM}\,s per instance; {n_cens} instances do not reach the GNN bound within this budget and count as $+\infty$ in the medians.""" + (
    r" Hollow markers: lower bounds (the subgradient did not reach the GNN bound within the iteration or time budget on at least half of the instances)." if has_lb else "") + r"""}
\label{fig:tstar_ccpo}
\end{figure}
"""
with open(os.path.join(here, a.fig), "w") as fh:
    fh.write(fig)
print(f"-> latex_data/time_vs_tstar_ccpo.csv, {a.fig}")
