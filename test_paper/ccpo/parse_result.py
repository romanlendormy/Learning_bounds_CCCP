#!/usr/bin/env python3
"""
parse_result.py — CCPO : extrait les métriques d'UN run de bab_ccpo depuis son log (stdout+stderr) en une ligne CSV,
au même schéma que CCMKP (test_paper/mkp/parse_result.py). Modes : gnn (LD), gnn_zero (LD0) ; ref | feas | sld aussi lus.

Usage :
  python3 parse_result.py <mode> <n> <N> <idx> <seed> <logfile> > run.csv        (une ligne CSV, sans en-tête)
  python3 parse_result.py --header                                              (l'en-tête)

Champs communs avec CCMKP : status, objective, greedy_value, nodes, fails, incumbent_improvements, solve_time_s, t_bab_pure_s,
  t_bound_total_s, t_gnn_graph_build_s, t_gnn_forward_s, first_level_bound_value, gap_root_pct, n_bucket_flushes,
  mean_bucket_fill_at_flush, max_pending_live, gpu_model, K.
  t_bound_compute_s = évaluation de L(λ) (θ, ζ, sac-à-dos) ; naive_nodes : sans objet en CCPO (vide).
Spécifiques CCPO : k, Q, idx, fails_from_bound, max_depth, t_gnn_post_s, members (nombre d'évaluations de L), nodes_bypassed_leaf,
  infeas_inherited, compteurs de l'hybride SCIP (scip_*), time_limit_s, timed_out, omp_threads, torch_threads, device, model,
  root_bound_value_sld.
Borne du premier niveau (gnn, gnn_zero) : max sur les enfants directs de la racine de (coût fixé + L(λ)) ; mode sld : borne racine
du sous-gradient. gap_root_pct est calculé hors ligne par aggregate_results.py (borne du premier niveau vs optimum prouvé).
Statut KILLED_WALL : run arrêté par le timeout externe (code 124/137, cf. run_ld_array.sh).
"""
import math
import os
import re
import sys

FIELDS = [
    "n", "N", "k", "Q", "idx", "seed", "mode", "status", "objective", "greedy_value",
    "nodes", "naive_nodes", "fails", "fails_from_bound", "max_depth", "incumbent_improvements",
    "solve_time_s", "t_bab_pure_s",
    "t_bound_total_s", "t_gnn_graph_build_s", "t_gnn_forward_s", "t_gnn_post_s", "t_bound_compute_s",
    "first_level_bound_value", "first_level_n", "root_bound_value_sld", "gap_root_pct",
    "n_bucket_flushes", "mean_bucket_fill_at_flush", "max_pending_live", "nodes_bypassed_leaf", "members", "infeas_inherited",
    "scip_mode", "scip_budget", "scip_nf0", "scip_calls", "scip_aborted", "scip_ok", "scip_fail", "scip_noinc", "t_scip_s",
    "time_limit_s", "timed_out", "K", "omp_threads", "torch_threads", "device", "gpu_model", "model",
]
NAN = float("nan")


def _f(x, default=""):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def parse(mode, n, N, idx, seed, text):
    row = {k: "" for k in FIELDS}
    row.update(n=n, N=N, idx=idx, seed=seed, mode=mode)
    line = None
    for l in text.splitlines():
        p = l.split(",")
        if len(p) >= 19 and p[0].startswith("/") and p[5] == mode:
            line = p                                          # dernier résultat du log
    if line is None:
        rc = os.environ.get("RUN_RC", "")
        row["status"] = "KILLED_WALL" if rc in ("124", "137") else "NO_RESULT"   # 124/137 : arrêt par timeout externe (limite dépassée)
        row["timed_out"] = 1 if rc in ("124", "137") else ""
        row["solve_time_s"] = _f(os.environ.get("RUN_ELAPSED", ""))
        return row
    p = line
    kv = {}
    for tok in p[19:]:
        if "=" in tok:
            k, v = tok.split("=", 1); kv[k] = v
    row.update(k=p[3], Q=p[4], status=p[6], nodes=p[7], fails=p[8], fails_from_bound=p[9], max_depth=p[10],
               solve_time_s=_f(p[11]), objective=_f(p[12]), greedy_value=_f(p[13]))
    row["incumbent_improvements"] = kv.get("inc_impr", "")
    row["time_limit_s"] = kv.get("tlim", "")
    row["timed_out"] = 1 if p[6] in ("TIMEOUT", "FEASIBLE") else 0     # non optimal et non infaisable : limite de temps atteinte
    if mode == "sld":
        row["root_bound_value_sld"] = _f(p[17])
        row["t_bound_total_s"] = _f(p[18])
    if mode in ("gnn", "gnn_zero"):
        t_eval, t_backend = _f(kv.get("t_eval"), 0.0), _f(kv.get("t_backend"), 0.0)
        row["t_bound_compute_s"] = t_eval
        row["t_bound_total_s"] = t_backend + t_eval             # forward + graphe + remise en forme + évaluation de L (MKP : temps borne)
        row["t_bab_pure_s"] = row["solve_time_s"] - row["t_bound_total_s"] if row["solve_time_s"] != "" else ""
        row["t_gnn_graph_build_s"] = _f(kv.get("t_graph")); row["t_gnn_forward_s"] = _f(kv.get("t_forward")); row["t_gnn_post_s"] = _f(kv.get("t_post"))
        row["first_level_bound_value"] = _f(kv.get("first_level_bound"), NAN); row["first_level_n"] = kv.get("first_level_n", "")
        row["n_bucket_flushes"] = kv.get("flushes", ""); row["mean_bucket_fill_at_flush"] = kv.get("fill", "")
        row["max_pending_live"] = kv.get("max_pending", ""); row["nodes_bypassed_leaf"] = kv.get("leaf", "")
        row["members"] = kv.get("members", ""); row["infeas_inherited"] = kv.get("infeas_inh", "")
        for k_out, k_in in (("scip_mode", "scip_mode"), ("scip_budget", "scip_budget"), ("scip_nf0", "scip_nf0"), ("scip_calls", "scip_calls"),
                            ("scip_aborted", "scip_aborted"), ("scip_ok", "scip_ok"), ("scip_fail", "scip_fail"), ("scip_noinc", "scip_noinc"),
                            ("t_scip_s", "t_scip"), ("K", "K"), ("omp_threads", "omp"), ("torch_threads", "torch_threads"),
                            ("device", "device"), ("model", "model")):
            row[k_out] = kv.get(k_in, "")
    else:
        row["t_bab_pure_s"] = row["solve_time_s"]
    m = re.search(r"GPU \d+:\s*([^(\n]+?)\s*\(", text)         # modèle de GPU alloué (renseigné pour toutes les méthodes : matériel homogène)
    row["gpu_model"] = m.group(1).strip() if m else ""
    return row


def to_csv_row(row):
    def fmt(v):
        return "" if v == "" else (repr(v) if isinstance(v, float) else str(v))
    return ",".join(fmt(row.get(k, "")) for k in FIELDS)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--header":
        print(",".join(FIELDS)); return
    if len(sys.argv) != 7:
        print(__doc__); sys.exit(2)
    mode, n, N, idx, seed, log = sys.argv[1:7]
    print(to_csv_row(parse(mode, int(n), int(N), int(idx), int(seed), open(log, errors="replace").read())))


if __name__ == "__main__":
    main()
