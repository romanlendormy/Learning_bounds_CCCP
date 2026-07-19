"""
sweep_bound_efficacy.py — Sweep (E,T,N) pour mesurer l'efficacité de la borne
v1CLD sur SSP : compare B&B base_ref (même B&B custom, sans borne) vs B&B
v1CLD (même B&B, avec borne glouton) sur des instances calibrées par le
générateur adaptatif (generator_adaptive.generate_episode_fixed_E).

Grille par défaut restreinte aux 3 paires (E,T) qui atteignent effectivement
la fenêtre de difficulté cible [target_lo, target_hi] nœuds sous Gecode base —
cf. étude de calibration : sur les 9 paires (E,T) de la grille initialement
envisagée (E∈{2,3,4}, T∈{5,8,12}), 6 sont structurellement trop petites pour
sortir du régime trivial (E=2-4 avec T≤8 bascule directement de "quelques
nœuds, OPTIMAL" à INFEASIBLE dès qu'on augmente la demande, sans palier dur
intermédiaire, même à N=50). Seules (E=3,T=12), (E=4,T=8), (E=4,T=12)
calibrent dans [1e4, 1e6] nœuds.

Pour chaque instance : génère (calibration par bissection), exporte en texte
(export_to_txt.write_instance), lance bench_ssp_bab_file avec v1LD désactivé
(sentinelle -1, coûteux et hors scope ici), parse les lignes "CSV_ROW,..." de
stdout, écrit incrémentalement dans un CSV de sortie (flush après chaque
instance — robustesse si le job est interrompu).

Usage :
    python -m ssp_stochastic.sweep_bound_efficacy \
        --out ssp_stochastic/results/sweep_bound_efficacy.csv

    python -m ssp_stochastic.sweep_bound_efficacy \
        --combos 3:12 4:8 --N-values 20 --n-reps 1   # scope minimal (smoke test)
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "solver_generic" / "ssp"))

from ssp_stochastic.generator_adaptive import generate_episode_fixed_E  # noqa: E402
from export_to_txt import write_instance  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_COMBOS = [(3, 12), (4, 8), (4, 12)]  # (E, T) calibrables dans [1e4, 1e6] nœuds

CSV_FIELDS = [
    "combo_E", "combo_T", "combo_N", "eps", "rep", "ep_id",
    "dir", "inst_E", "inst_T", "inst_N", "inst_eps",
    "variant", "status", "optimal", "feasible", "nodes", "fails",
    "t_total", "t_bound", "t_pure", "tightenings", "fails_bound", "objective",
    "gap_root", "calib_scale", "calib_iters",
]


def parse_combo(s: str) -> tuple[int, int]:
    # Séparateur ':' (pas ',') — une valeur --export=ALL,COMBOS="3,12" serait
    # tronquée par sbatch, qui utilise lui-même la virgule comme séparateur
    # de sa propre liste VAR=valeur.
    e, t = s.split(":")
    return int(e), int(t)


def parse_csv_rows(stdout: str) -> list[list[str]]:
    return [[x.strip() for x in line.split(",")]
            for line in stdout.splitlines() if line.startswith("CSV_ROW,")]


def run_one_instance(E: int, T: int, N: int, eps: float, rep: int, ep_id: int,
                     probe_bin: Path, bench_bin: Path, T_node: int, time_lim: float,
                     probe_time_lim: float, target_lo: float, target_hi: float,
                     master_seed: int, work_root: Path, keep_instances: bool) -> list[dict]:
    try:
        ep = generate_episode_fixed_E(
            ep_id=ep_id, E=E, T=T, N=N, epsilon=eps, probe_bin=probe_bin,
            master_seed=master_seed, target_lo=target_lo, target_hi=target_hi,
            probe_time_lim=probe_time_lim,
        )
    except RuntimeError as e:
        log.error("E=%d T=%d N=%d rep=%d : calibration échouée (%s)", E, T, N, rep, e)
        return [dict.fromkeys(CSV_FIELDS, "") | dict(
            combo_E=E, combo_T=T, combo_N=N, eps=eps, rep=rep, ep_id=ep_id,
            status="CALIB_FAILED",
        )]

    inst_dir = work_root / f"E{E}_T{T}_N{N}_rep{rep}"
    write_instance(ep, inst_dir)

    proc = subprocess.run(
        [str(bench_bin), str(inst_dir), str(T_node), str(time_lim), "-1"],
        capture_output=True, text=True,
    )
    print(proc.stdout, end="", flush=True)
    if proc.returncode not in (0, 1):
        log.error("E=%d T=%d N=%d rep=%d : bench_bin a échoué (rc=%d)\n%s",
                  E, T, N, rep, proc.returncode, proc.stderr[-2000:])

    rows = []
    for fields in parse_csv_rows(proc.stdout):
        # CSV_ROW,dir,E,T,N,eps,variant,status,optimal,feasible,nodes,fails,
        #         t_total,t_bound,t_pure,tightenings,fails_bound,objective,gap_root_pct
        # (gap_root_pct ajouté depuis à bench_ssp_bab_file.cpp — le parseur ici
        # était resté sur l'ancien format à 18 champs, cf. discussion de session.)
        (_, d, ie, it, iN, ieps, variant, status, optimal, feasible, nodes, fails,
         t_total, t_bound, t_pure, tightenings, fails_bound, objective,
         gap_root) = fields
        rows.append(dict(
            combo_E=E, combo_T=T, combo_N=N, eps=eps, rep=rep, ep_id=ep_id,
            dir=d, inst_E=ie, inst_T=it, inst_N=iN, inst_eps=ieps,
            variant=variant, status=status, optimal=optimal, feasible=feasible,
            nodes=nodes, fails=fails, t_total=t_total, t_bound=t_bound, t_pure=t_pure,
            tightenings=tightenings, fails_bound=fails_bound, objective=objective,
            gap_root=gap_root,
            calib_scale=ep["meta"]["scale"], calib_iters=ep["meta"]["calib_n_iters"],
        ))

    if not keep_instances:
        shutil.rmtree(inst_dir, ignore_errors=True)
    return rows


def print_summary(csv_path: Path) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    combos = sorted({(r["combo_E"], r["combo_T"], r["combo_N"]) for r in rows},
                    key=lambda c: (int(c[0]), int(c[1]), int(c[2])))
    print(f"\n{'E':>3} {'T':>3} {'N':>3} {'n_ok':>5} "
          f"{'nodes_base_ref(med)':>20} {'nodes_v1cld(med)':>17} {'reduction_x':>12} "
          f"{'t_pure/t_total v1cld(med)':>26}")
    for (E, T, N) in combos:
        sub = [r for r in rows if (r["combo_E"], r["combo_T"], r["combo_N"]) == (E, T, N)]
        base = [float(r["nodes"]) for r in sub if r["variant"] == "base_ref" and r["nodes"]]
        cld = [float(r["nodes"]) for r in sub if r["variant"] == "v1cld" and r["nodes"]]
        cld_ratio = [float(r["t_pure"]) / float(r["t_total"]) for r in sub
                    if r["variant"] == "v1cld" and float(r.get("t_total") or 0) > 0]
        med_base = statistics.median(base) if base else float("nan")
        med_cld = statistics.median(cld) if cld else float("nan")
        reduction = med_base / med_cld if cld and med_cld > 0 else float("nan")
        med_ratio = statistics.median(cld_ratio) if cld_ratio else float("nan")
        print(f"{E:>3} {T:>3} {N:>3} {len(base):>5} "
              f"{med_base:>20.0f} {med_cld:>17.0f} {reduction:>12.2f} {med_ratio:>26.4f}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--combos", nargs="+", type=parse_combo, default=DEFAULT_COMBOS,
                  help="Paires 'E:T' calibrables (défaut : celles validées empiriquement)")
    p.add_argument("--N-values", nargs="+", type=int, default=[20, 50])
    p.add_argument("--eps", type=float, default=0.10)
    p.add_argument("--n-reps", type=int, default=3)
    p.add_argument("--T-node", type=int, default=30)
    p.add_argument("--time-lim", type=float, default=60.0,
                  help="Limite de temps par méthode (base_ref, v1CLD), secondes")
    p.add_argument("--probe-time-lim", type=float, default=5.0)
    p.add_argument("--target-lo", type=float, default=1e4)
    p.add_argument("--target-hi", type=float, default=1e6)
    p.add_argument("--master-seed", type=int, default=42)
    p.add_argument("--bench-bin", default=str(REPO_ROOT / "solver_bab" / "build" / "bench_ssp_bab_file"))
    p.add_argument("--probe-bin", default=str(REPO_ROOT / "solver_bab" / "build" / "ssp_probe"))
    p.add_argument("--out", default=str(REPO_ROOT / "ssp_stochastic" / "results" / "sweep_bound_efficacy.csv"))
    p.add_argument("--work-dir", default=None,
                  help="Répertoire de travail pour les instances texte (défaut : tempdir)")
    p.add_argument("--keep-instances", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    bench_bin = Path(args.bench_bin)
    probe_bin = Path(args.probe_bin)
    if not bench_bin.exists():
        sys.exit(f"binaire introuvable : {bench_bin}")
    if not probe_bin.exists():
        sys.exit(f"binaire introuvable : {probe_bin}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combos_full = [(E, T, N) for (E, T) in args.combos for N in args.N_values]
    total = len(combos_full) * args.n_reps
    print(f"=== sweep_bound_efficacy : {len(combos_full)} combos (E,T,N) x "
          f"{args.n_reps} reps = {total} instances ===", flush=True)
    print(f"    combos (E,T) : {args.combos}   N : {args.N_values}   eps={args.eps}", flush=True)
    print(f"    time_lim/méthode={args.time_lim}s  T_node={args.T_node}  "
          f"target=[{args.target_lo:.0f},{args.target_hi:.0f}] nœuds", flush=True)

    work_ctx = tempfile.TemporaryDirectory(prefix="ssp_sweep_") if args.work_dir is None else None
    work_root = Path(args.work_dir) if args.work_dir else Path(work_ctx.name)
    work_root.mkdir(parents=True, exist_ok=True)

    ep_id = 0
    n_done = 0
    try:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            f.flush()
            for (E, T, N) in combos_full:
                for rep in range(args.n_reps):
                    n_done += 1
                    print(f"\n--- [{n_done}/{total}] E={E} T={T} N={N} rep={rep} "
                          f"(ep_id={ep_id}) ---", flush=True)
                    rows = run_one_instance(
                        E, T, N, args.eps, rep, ep_id, probe_bin, bench_bin,
                        args.T_node, args.time_lim, args.probe_time_lim,
                        args.target_lo, args.target_hi, args.master_seed,
                        work_root, args.keep_instances,
                    )
                    for r in rows:
                        writer.writerow(r)
                    f.flush()
                    ep_id += 1
    finally:
        if work_ctx is not None:
            work_ctx.cleanup()

    print(f"\n-> {out_path}  ({n_done} instances)")
    print_summary(out_path)


if __name__ == "__main__":
    main()
