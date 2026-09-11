#!/usr/bin/env python3
"""
parse_gnn_vs_subgrad_ssp.py — Post-traitement de run_gnn_vs_subgrad_ssp.sh.

Portage direct de parse_gnn_vs_subgrad.py (CCMKP) vers CCSSP -- DLD
uniquement (pas de formulation sld côté SSP pour cette expérience), tailles
(E,T) in {(4,8),(6,10),(8,12)}, N in {100,500}, rep in 0..9.

Pour chaque instance :
  - lit la borne GNN root-only (logs/gnn_vs_subgrad_root_<tag>.log, clé
    root_bound_value=... -- imprimée par test_gnn_dld_bab.cpp).
  - lit la trace de convergence du sous-gradient DLD
    (logs/gnn_vs_subgrad_dld_<tag>.stderr.log, lignes
    SUBGRAD_TRACE,dld,<t>,<best_upper_convention_cout_reel>) -- PLUSIEURS
    blocs sont présents (un par branche explorée à la racine littérale, cf.
    commentaire dans gecode_bab_ssp.hpp sur lambda_warmed hérité par
    clonage) ; on ne garde que le DERNIER bloc, dont la valeur finale
    correspond à root_bound_value affiché par le solveur.
  - calcule t* = min{t : best_upper(t) >= borne_GNN_root} (convention coût
    réel -- borne inférieure, PLUS GRAND = PLUS SERRÉ, contrairement à MKP où
    la borne est une borne supérieure). Si jamais atteint dans la trace,
    t* = None (censuré à droite) et on rapporte >T_max plutôt que d'extrapoler.

Aucune dépendance non-stdlib.

Usage :
  python3 parse_gnn_vs_subgrad_ssp.py [logs_dir] [--csv PATH] [--detail-csv PATH]
                                       [--ref-csv PATH]
    (logs_dir défaut : test_paper/ssp/rorqual/gnn_vs_subgrad/logs)
    --csv PATH : ajoute (append) le résumé agrégé (une ligne par (E,T,N)),
    colonnes E,T,N,t_star_median,t_star_mean,n_resolved,n_total,
    gap_gnn_pct_mean,gap_subgrad_pct_mean.
    --detail-csv PATH : ajoute (append) le détail PAR INSTANCE, colonnes
    E,T,N,rep,optimal,gnn_bound,gap_gnn_pct,subgrad_bound,gap_subgrad_pct,
    t_star,t_max,censored.
    --ref-csv PATH : fichier agrégé contenant les résultats REF déjà produits
    par la campagne trillium_ssp_ref (défaut test_paper/ssp/rorqual/results_ssp.csv,
    colonnes E,T,N,rep,mode,status,objective,...) -- utilisé UNIQUEMENT si
    mode=="ref" et status=="OPTIMAL" pour la valeur optimale certifiée (pas
    d'incumbent non prouvé comme substitut, même garde-fou que côté MKP).
"""
import csv as csv_module
import re
import sys
import statistics
from pathlib import Path

SIZES = [(4, 8), (6, 10), (8, 12)]
N_VALUES = [100, 500]
REPS = list(range(10))

ROOT_BOUND_RE = re.compile(r"root_bound_value=(-?[0-9.]+)")
TRACE_RE = re.compile(r"^SUBGRAD_TRACE,(\w+),(\d+),([\-0-9.]+)$")


def read_root_bound(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text()
    m = ROOT_BOUND_RE.search(text)
    return float(m.group(1)) if m else None


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
    """t* = min{t : trace_value(t) >= gnn_bound} (borne INFÉRIEURE, convention
    coût réel -- PLUS GRAND = PLUS SERRÉ, contrairement à MKP). None si jamais
    atteint."""
    for t, v in trace:
        if v >= gnn_bound:
            return t
    return None


def load_ref_optimals(ref_csv: Path):
    """{(E,T,N,rep): objective} pour les lignes mode=="ref" avec
    status=="OPTIMAL" uniquement (cf. doc module : pas d'incumbent non prouvé
    comme substitut)."""
    optimals = {}
    if not ref_csv.exists():
        return optimals
    with ref_csv.open() as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            if row.get("mode") != "ref" or row.get("status") != "OPTIMAL":
                continue
            try:
                key = (int(row["E"]), int(row["T"]), int(row["N"]), int(row["rep"]))
                optimals[key] = float(row["objective"])
            except (KeyError, ValueError):
                continue
    return optimals


def gap_pct(bound, optimal):
    if bound is None or optimal is None or optimal == 0:
        return None
    return 100.0 * (optimal - bound) / optimal


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
    script_dir = Path(__file__).resolve().parent
    ref_csv = Path(_opt_arg("ref-csv", str(script_dir.parent / "results_ssp.csv")))

    logs_dir = Path(positional[0] if positional else str(script_dir / "logs"))
    ref_optimals = load_ref_optimals(ref_csv)

    rows = []
    for E, T in SIZES:
        for N in N_VALUES:
            for rep in REPS:
                tag = f"E{E}_T{T}_N{N}_rep{rep}"
                gnn_log = logs_dir / f"gnn_vs_subgrad_root_{tag}.log"
                subgrad_stdout = logs_dir / f"gnn_vs_subgrad_dld_{tag}.stdout.log"
                subgrad_stderr = logs_dir / f"gnn_vs_subgrad_dld_{tag}.stderr.log"

                gnn_bound = read_root_bound(gnn_log)
                subgrad_root_bound = read_root_bound(subgrad_stdout)
                trace = read_last_trace_block(subgrad_stderr)

                optimal = ref_optimals.get((E, T, N, rep))

                if gnn_bound is None or not trace:
                    rows.append(dict(E=E, T=T, N=N, rep=rep,
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
                    E=E, T=T, N=N, rep=rep, status="OK",
                    gnn_bound=gnn_bound, subgrad_root_bound=subgrad_root_bound,
                    trace_final=trace_final, consistent=consistent,
                    t_star=t_star, t_max=len(trace),
                    optimal=optimal,
                    gap_gnn_pct=gap_pct(gnn_bound, optimal),
                    gap_subgrad_pct=gap_pct(trace_final, optimal),
                ))

    # ── Détail par instance ────────────────────────────────────────────────
    print(f"{'E':3} {'T':3} {'N':6} {'rep':4} {'optimal':>10} {'GNN_root':>10} "
          f"{'gap_GNN%':>9} {'subgrad_final':>13} {'gap_sg%':>8} "
          f"{'t_max':>6} {'t*':>8} {'cohérent':>9}")
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['E']:<3} {r['T']:<3} {r['N']:<6} {r['rep']:<4} "
                  f"{'MANQUANT':>10}")
            continue
        t_star_str = str(r["t_star"]) if r["t_star"] is not None else f">{r['t_max']}"
        opt_str = f"{r['optimal']:.4f}" if r["optimal"] is not None else "N/A"
        gap_gnn_str = f"{r['gap_gnn_pct']:.2f}" if r["gap_gnn_pct"] is not None else "N/A"
        gap_sg_str = f"{r['gap_subgrad_pct']:.2f}" if r["gap_subgrad_pct"] is not None else "N/A"
        print(f"{r['E']:<3} {r['T']:<3} {r['N']:<6} {r['rep']:<4} {opt_str:>10} "
              f"{r['gnn_bound']:>10.4f} {gap_gnn_str:>9} "
              f"{r['trace_final']:>13.4f} {gap_sg_str:>8} "
              f"{r['t_max']:>6} {t_star_str:>8} {str(r['consistent']):>9}")

    # ── Agrégation par (E,T,N) ───────────────────────────────────────────────
    print("\n=== Agrégation (médiane/moyenne de t*, sur les instances non censurées) ===")
    csv_rows = []
    for E, T in SIZES:
        for N in N_VALUES:
            subset = [r for r in rows if r["E"] == E and r["T"] == T and r["N"] == N
                      and r["status"] == "OK"]
            resolved = [r["t_star"] for r in subset if r["t_star"] is not None]
            censored = len(subset) - len(resolved)
            if resolved:
                med = statistics.median(resolved)
                mean = statistics.mean(resolved)
                print(f"  E={E} T={T} N={N:<5} : t* médiane={med:.0f}  "
                      f"moyenne={mean:.1f}  (n={len(resolved)}/{len(subset)}, "
                      f"{censored} censurée(s) >T_max)")
            else:
                med = mean = None
                print(f"  E={E} T={T} N={N:<5} : aucune instance résolue "
                      f"(0/{len(subset)}) — GNN bat le sous-gradient même au plafond T_max")
            gaps_gnn = [r["gap_gnn_pct"] for r in subset if r.get("gap_gnn_pct") is not None]
            gaps_sg = [r["gap_subgrad_pct"] for r in subset if r.get("gap_subgrad_pct") is not None]
            csv_rows.append(dict(
                E=E, T=T, N=N,
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
              f"(cf. commentaire sur les blocs multiples par instance).")

    if csv_path:
        out_path = Path(csv_path)
        write_header = not out_path.exists()
        fieldnames = ["E", "T", "N", "t_star_median", "t_star_mean",
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
        fieldnames = ["E", "T", "N", "rep", "optimal", "gnn_bound",
                      "gap_gnn_pct", "subgrad_bound", "gap_subgrad_pct",
                      "t_star", "t_max", "censored"]
        detail_rows = []
        for r in rows:
            if r["status"] != "OK":
                detail_rows.append(dict(
                    E=r["E"], T=r["T"], N=r["N"], rep=r["rep"],
                    optimal=r.get("optimal"), gnn_bound=r.get("gnn_bound"),
                    gap_gnn_pct=None, subgrad_bound=None, gap_subgrad_pct=None,
                    t_star=None, t_max=r.get("t_max"), censored="MISSING",
                ))
                continue
            detail_rows.append(dict(
                E=r["E"], T=r["T"], N=r["N"], rep=r["rep"],
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
