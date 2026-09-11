#!/usr/bin/env python3
"""
parse_gnn_vs_subgrad.py — Post-traitement de run_gnn_vs_subgrad_n15.sh.

Pour chaque instance (n=15, N in {100,500,1000}, seed in 43..52, formulation
in {sld, dld}) :
  - lit la borne GNN root-only (logs/gnn_root_<tag>.log, ligne "Borne UB à la
    racine").
  - lit la trace de convergence du sous-gradient (logs/subgrad_<tag>.stderr.log,
    lignes SUBGRAD_TRACE,<mode>,<t>,<best_upper_at_t>) — DEUX blocs sont
    présents (un par enfant direct de la racine littérale, cf. commentaire
    dans gecode_bab.hpp sur lambda_warmed hérité par clonage) ; on ne garde
    que le DERNIER bloc, qui est celui dont la valeur finale correspond à
    root_bound_value affiché par le solveur (le second enfant traité écrase
    root_bound_value du premier — cf. mémoire/sanity check du 2026-08-22).
  - calcule t* = min{t : best_upper(t) <= borne_GNN_root} (borne = upper
    bound, plus petit = plus serré). Si jamais atteint dans la trace,
    t* = None (censuré à droite) et on rapporte >T_max plutôt que d'extrapoler.

Aucune dépendance non-stdlib (exécutable directement sur le login node
Trillium, cf. mémoire project_ccmkp_trillium_migration).

Usage :
  python3 parse_gnn_vs_subgrad.py [scratch_dir] [n_items] [--csv PATH]
                                   [--detail-csv PATH] [--ref-dir DIR]
    (scratch_dir défaut : /scratch/romanlen/gnn_vs_subgrad_n15, n_items défaut : 15)
    --csv PATH : ajoute (append) le résumé agrégé (une ligne par
    formulation x N) à PATH, colonnes n,formulation,N,t_star_median,
    t_star_mean,n_resolved,n_total,gap_gnn_pct_mean,gap_subgrad_pct_mean —
    pense au diagramme LaTeX/pgfplots. Le fichier est créé avec son en-tête
    s'il n'existe pas encore ; sinon les nouvelles lignes sont ajoutées à la
    suite (permet de lancer ce script une fois par n et d'accumuler dans le
    même CSV).
    --detail-csv PATH : ajoute (append) le détail PAR INSTANCE (une ligne par
    seed), colonnes n,formulation,N,seed,optimal,gnn_bound,gap_gnn_pct,
    subgrad_bound,gap_subgrad_pct,t_star,t_max,censored.
    --ref-dir DIR : répertoire contenant les résultats REF (défaut
    test_paper/mkp/trillium/sld/results, Trillium — 105/120 instances, cf.
    commit "Verse les resultats MKP SLD vs REF de la campagne Trillium")
    utilisés comme source de la valeur
    optimale certifiée (ref_n<n>_N<N>_seed<seed>.csv, colonne "objective",
    UNIQUEMENT si status=="OPTIMAL" — sinon pas de gap calculable pour cette
    instance, rapporté comme tel plutôt que d'utiliser un incumbent non prouvé).
"""
import csv as csv_module
import re
import sys
import statistics
from pathlib import Path

N_VALUES = [100, 500, 1000]
SEEDS = list(range(43, 53))
FORMULATIONS = ["sld", "dld"]

ROOT_BOUND_RE = re.compile(r"Borne UB à la racine\s*:\s*([\-0-9.]+)")
OPTIMAL_RE = re.compile(r"Valeur optimale\s*:\s*([\-0-9.]+)")
TRACE_RE = re.compile(r"^SUBGRAD_TRACE,(\w+),(\d+),([\-0-9.]+)$")


def read_root_bound(log_path: Path):
    if not log_path.exists():
        return None, None
    text = log_path.read_text()
    m_bound = ROOT_BOUND_RE.search(text)
    m_opt = OPTIMAL_RE.search(text)
    bound = float(m_bound.group(1)) if m_bound else None
    opt = float(m_opt.group(1)) if m_opt else None
    return bound, opt


def read_last_trace_block(stderr_path: Path):
    """Retourne la liste [(t, value), ...] du DERNIER bloc (t repart à 1)."""
    if not stderr_path.exists():
        return []
    blocks = []
    current = []
    for line in stderr_path.read_text().splitlines():
        m = TRACE_RE.match(line)
        if not m:
            continue
        t = int(m.group(2))
        v = float(m.group(3))
        if t == 1 and current:
            blocks.append(current)
            current = []
        current.append((t, v))
    if current:
        blocks.append(current)
    return blocks[-1] if blocks else []


def find_t_star(trace, gnn_bound):
    """t* = min{t : trace_value(t) <= gnn_bound}. None si jamais atteint."""
    for t, v in trace:
        if v <= gnn_bound:
            return t
    return None


def load_ref_optimals(ref_dir: Path):
    """{(n,N,seed): objective} pour les instances REF avec status==OPTIMAL
    uniquement (cf. doc module : pas d'incumbent non prouvé comme substitut)."""
    optimals = {}
    if not ref_dir.exists():
        return optimals
    for path in ref_dir.glob("ref_n*_N*_seed*.csv"):
        m = re.match(r"ref_n(\d+)_N(\d+)_seed(\d+)\.csv", path.name)
        if not m:
            continue
        n, N, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        fields = path.read_text().strip().split(",")
        status, objective = fields[5], fields[6]
        if status == "OPTIMAL":
            optimals[(n, N, seed)] = float(objective)
    return optimals


def gap_pct(bound, optimal):
    if bound is None or optimal is None or optimal == 0:
        return None
    return 100.0 * (bound - optimal) / optimal


def _opt_arg(name, default=None):
    if f"--{name}" in sys.argv:
        return sys.argv[sys.argv.index(f"--{name}") + 1]
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    csv_path = _opt_arg("csv")
    detail_csv_path = _opt_arg("detail-csv")
    default_ref_dir = Path(__file__).resolve().parents[1] / "sld" / "results"
    ref_dir = Path(_opt_arg("ref-dir", str(default_ref_dir)))

    scratch_dir = Path(positional[0] if len(positional) > 0 else "/scratch/romanlen/gnn_vs_subgrad_n15")
    n_items = int(positional[1]) if len(positional) > 1 else 15
    logs_dir = scratch_dir / "logs"
    ref_optimals = load_ref_optimals(ref_dir)

    rows = []
    for formulation in FORMULATIONS:
        for N in N_VALUES:
            for seed in SEEDS:
                tag = f"n{n_items}_N{N}_seed{seed}_{formulation}"
                gnn_log = logs_dir / f"gnn_root_{tag}.log"
                subgrad_stdout = logs_dir / f"subgrad_{tag}.stdout.log"
                subgrad_stderr = logs_dir / f"subgrad_{tag}.stderr.log"

                gnn_bound, gnn_opt = read_root_bound(gnn_log)
                subgrad_root_bound, subgrad_opt = read_root_bound(subgrad_stdout)
                trace = read_last_trace_block(subgrad_stderr)

                optimal = ref_optimals.get((n_items, N, seed))

                if gnn_bound is None or not trace:
                    rows.append(dict(formulation=formulation, N=N, seed=seed,
                                      status="MISSING", gnn_bound=gnn_bound,
                                      subgrad_root_bound=subgrad_root_bound,
                                      t_star=None, t_max=len(trace),
                                      optimal=optimal))
                    continue

                # Sanity : la dernière valeur de la trace doit correspondre à
                # root_bound_value affiché par le solveur (même quantité).
                trace_final = trace[-1][1]
                consistent = (subgrad_root_bound is not None and
                              abs(trace_final - subgrad_root_bound) < 1e-2)

                t_star = find_t_star(trace, gnn_bound)
                rows.append(dict(
                    formulation=formulation, N=N, seed=seed, status="OK",
                    gnn_bound=gnn_bound, subgrad_root_bound=subgrad_root_bound,
                    trace_final=trace_final, consistent=consistent,
                    t_star=t_star, t_max=len(trace),
                    optimal=optimal,
                    gap_gnn_pct=gap_pct(gnn_bound, optimal),
                    gap_subgrad_pct=gap_pct(trace_final, optimal),
                ))

    # ── Détail par instance ────────────────────────────────────────────────
    print(f"{'formulation':10} {'N':6} {'seed':5} {'optimal':>10} {'GNN_root':>10} "
          f"{'gap_GNN%':>9} {'subgrad_final':>13} {'gap_sg%':>8} "
          f"{'t_max':>6} {'t*':>8} {'cohérent':>9}")
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['formulation']:10} {r['N']:<6} {r['seed']:<5} "
                  f"{'MANQUANT':>10}")
            continue
        t_star_str = str(r["t_star"]) if r["t_star"] is not None else f">{r['t_max']}"
        opt_str = f"{r['optimal']:.4f}" if r["optimal"] is not None else "N/A"
        gap_gnn_str = f"{r['gap_gnn_pct']:.2f}" if r["gap_gnn_pct"] is not None else "N/A"
        gap_sg_str = f"{r['gap_subgrad_pct']:.2f}" if r["gap_subgrad_pct"] is not None else "N/A"
        print(f"{r['formulation']:10} {r['N']:<6} {r['seed']:<5} {opt_str:>10} "
              f"{r['gnn_bound']:>10.4f} {gap_gnn_str:>9} "
              f"{r['trace_final']:>13.4f} {gap_sg_str:>8} "
              f"{r['t_max']:>6} {t_star_str:>8} {str(r['consistent']):>9}")

    # ── Agrégation par (formulation, N) ─────────────────────────────────────
    print("\n=== Agrégation (médiane/moyenne de t*, sur les instances non censurées) ===")
    csv_rows = []
    for formulation in FORMULATIONS:
        for N in N_VALUES:
            subset = [r for r in rows if r["formulation"] == formulation and r["N"] == N
                      and r["status"] == "OK"]
            resolved = [r["t_star"] for r in subset if r["t_star"] is not None]
            censored = len(subset) - len(resolved)
            if resolved:
                med = statistics.median(resolved)
                mean = statistics.mean(resolved)
                print(f"  {formulation:6} N={N:<5} : t* médiane={med:.0f}  "
                      f"moyenne={mean:.1f}  (n={len(resolved)}/{len(subset)}, "
                      f"{censored} censurée(s) >T_max)")
            else:
                med = mean = None
                print(f"  {formulation:6} N={N:<5} : aucune instance résolue "
                      f"(0/{len(subset)}) — GNN bat le sous-gradient même au plafond T_max")
            gaps_gnn = [r["gap_gnn_pct"] for r in subset if r.get("gap_gnn_pct") is not None]
            gaps_sg = [r["gap_subgrad_pct"] for r in subset if r.get("gap_subgrad_pct") is not None]
            csv_rows.append(dict(
                n=n_items, formulation=formulation, N=N,
                t_star_median=med, t_star_mean=mean,
                n_resolved=len(resolved), n_total=len(subset),
                gap_gnn_pct_mean=statistics.mean(gaps_gnn) if gaps_gnn else None,
                gap_subgrad_pct_mean=statistics.mean(gaps_sg) if gaps_sg else None,
                n_with_optimal=len(gaps_gnn),
            ))

    incoherent = [r for r in rows if r["status"] == "OK" and not r["consistent"]]
    if incoherent:
        print(f"\n[ATTENTION] {len(incoherent)} instance(s) où la trace finale "
              f"ne correspond pas à root_bound_value affiché — vérifier "
              f"(cf. commentaire sur les 2 blocs par instance).")

    if csv_path:
        out_path = Path(csv_path)
        write_header = not out_path.exists()
        fieldnames = ["n", "formulation", "N", "t_star_median", "t_star_mean",
                      "n_resolved", "n_total", "gap_gnn_pct_mean",
                      "gap_subgrad_pct_mean", "n_with_optimal"]
        with out_path.open("a", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[CSV] {len(csv_rows)} ligne(s) ajoutée(s) à {out_path}"
              f"{' (créé)' if write_header else ' (append)'}")

    if detail_csv_path:
        out_path = Path(detail_csv_path)
        write_header = not out_path.exists()
        fieldnames = ["n", "formulation", "N", "seed", "optimal", "gnn_bound",
                      "gap_gnn_pct", "subgrad_bound", "gap_subgrad_pct",
                      "t_star", "t_max", "censored"]
        detail_rows = []
        for r in rows:
            if r["status"] != "OK":
                detail_rows.append(dict(
                    n=n_items, formulation=r["formulation"], N=r["N"], seed=r["seed"],
                    optimal=r.get("optimal"), gnn_bound=r.get("gnn_bound"),
                    gap_gnn_pct=None, subgrad_bound=None, gap_subgrad_pct=None,
                    t_star=None, t_max=r.get("t_max"), censored="MISSING",
                ))
                continue
            detail_rows.append(dict(
                n=n_items, formulation=r["formulation"], N=r["N"], seed=r["seed"],
                optimal=r["optimal"], gnn_bound=r["gnn_bound"],
                gap_gnn_pct=r["gap_gnn_pct"], subgrad_bound=r["trace_final"],
                gap_subgrad_pct=r["gap_subgrad_pct"],
                t_star=r["t_star"], t_max=r["t_max"],
                censored=(r["t_star"] is None),
            ))
        with out_path.open("a", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(detail_rows)
        print(f"[CSV détail] {len(detail_rows)} ligne(s) ajoutée(s) à {out_path}"
              f"{' (créé)' if write_header else ' (append)'}")


if __name__ == "__main__":
    main()
