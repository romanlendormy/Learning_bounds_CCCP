"""
sweep_kp_gecode_grid.py — Sweep REF vs V1LD (bab_kp_gecode) sur une grille n×N.

Objectif : voir si/où la borne V1LD (KPProblem, B&B 1D exact) finit par payer
son coût par nœud face à REF (squelette nu, aucune borne) quand n ET N
grandissent — cf. session : à n=15/N=100, REF bat encore V1LD en temps réel
(0.19s/2526 nœuds vs 3.73s/1466 nœuds) malgré moins de nœuds pour V1LD.

Grille croisée : n × N (pas de couplage), weakly_correlated (régime validé
lors du calibrage précédent — strongly_correlated montrait des INFEASIBLE
suspects à petit n, non résolu, cf. mémoire de session).

Usage :
    python sweep_kp_gecode_grid.py
Variables d'environnement (surchargeables) :
    N_LIST, n_LIST, SEEDS, TIME_LIM_S, MAX_WORKERS (cf. valeurs par défaut ci-dessous)
"""

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent))
from collect_kp import generate_song_tempfile

REPO = Path(__file__).parent.parent
BINARY = REPO / "solver_bab" / "build" / "bab_kp_gecode"

N_LIST      = [int(x) for x in os.environ.get("N_LIST", "10,15,20,25,30").split(",")]
CAP_LIST    = [int(x) for x in os.environ.get("CAP_LIST", "50,100,250,500").split(",")]
SEEDS       = [int(x) for x in os.environ.get("SEEDS", "1,2").split(",")]
MODES       = ["ref", "v1ld"]
CORRELATION = os.environ.get("CORRELATION", "weakly_correlated")
EPSILON     = float(os.environ.get("EPSILON", "0.1"))
ALPHA       = float(os.environ.get("ALPHA", "0.5"))
R           = int(os.environ.get("R", "1000"))
T_ROOT      = int(os.environ.get("T_ROOT", "200"))
T_NODE      = int(os.environ.get("T_NODE", "30"))
TIME_LIM_S  = int(os.environ.get("TIME_LIM_S", "120"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))

RE_STATUS = re.compile(r"Statut\s+:\s+(\S+)")
RE_OBJ    = re.compile(r"Valeur optimale\s+:\s+([\-0-9.]+|nan|-nan)")
RE_NODES  = re.compile(r"Noeuds Gecode\s+:\s+(\d+)")
RE_TIME   = re.compile(r"Temps total\s+:\s+([0-9.]+) s")

print_lock = Lock()


def run_one(n, N, seed, mode):
    tmpfile = generate_song_tempfile(
        n, 1, N, seed, alpha=ALPHA, R=R, correlation=CORRELATION
    )
    try:
        cmd = [str(BINARY), tmpfile, str(EPSILON), str(T_NODE), str(TIME_LIM_S),
               "0", str(T_ROOT), mode]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=TIME_LIM_S + 15)
            out = res.stdout
            m_status = RE_STATUS.search(out)
            m_obj    = RE_OBJ.search(out)
            m_nodes  = RE_NODES.search(out)
            m_time   = RE_TIME.search(out)
            row = {
                "n": n, "N": N, "seed": seed, "mode": mode,
                "status": m_status.group(1) if m_status else "PARSE_ERR",
                "obj":    m_obj.group(1)    if m_obj    else "",
                "nodes":  m_nodes.group(1)  if m_nodes  else "",
                "time_s": m_time.group(1)   if m_time   else "",
                "rc": res.returncode,
                "stderr": res.stderr[:200] if res.returncode != 0 else "",
            }
        except subprocess.TimeoutExpired:
            row = {"n": n, "N": N, "seed": seed, "mode": mode,
                   "status": "PROCESS_TIMEOUT", "obj": "", "nodes": "",
                   "time_s": "", "rc": -1, "stderr": ""}
    finally:
        os.unlink(tmpfile)
    return row


def main():
    if not BINARY.is_file():
        print(f"Erreur : binaire introuvable : {BINARY}", file=sys.stderr)
        sys.exit(1)

    tasks = [(n, N, seed, mode)
             for n in N_LIST for N in CAP_LIST for seed in SEEDS for mode in MODES]

    print(f"Grille : n={N_LIST}  N={CAP_LIST}  seeds={SEEDS}  modes={MODES}")
    print(f"corr={CORRELATION}  eps={EPSILON}  T_root={T_ROOT}  T_node={T_NODE}  "
          f"time_lim={TIME_LIM_S}s  max_workers={MAX_WORKERS}")
    print(f"Total runs : {len(tasks)}")
    print("=" * 100)
    header = f"{'n':4s} {'N':5s} {'seed':5s} {'mode':6s} {'status':16s} {'obj':>12s} {'nodes':>8s} {'t(s)':>8s}"
    print(header)

    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one, n, N, seed, mode): (n, N, seed, mode)
                   for (n, N, seed, mode) in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            with print_lock:
                line = (f"{row['n']:<4d} {row['N']:<5d} {row['seed']:<5d} "
                         f"{row['mode']:<6s} {row['status']:16s} "
                         f"{row['obj']:>12s} {row['nodes']:>8s} {row['time_s']:>8s}")
                if row["rc"] not in (0, None) and row["status"] not in (
                    "OPTIMAL", "FEASIBLE", "INFEASIBLE", "TIMEOUT"):
                    line += f"  rc={row['rc']} stderr={row['stderr']}"
                print(line, flush=True)

    csv_path = REPO / "collect" / "sweep_kp_gecode_grid_results.csv"
    with open(csv_path, "w") as f:
        f.write("n,N,seed,mode,status,obj,nodes,time_s\n")
        for row in sorted(rows, key=lambda r: (r["n"], r["N"], r["seed"], r["mode"])):
            f.write(f"{row['n']},{row['N']},{row['seed']},{row['mode']},"
                    f"{row['status']},{row['obj']},{row['nodes']},{row['time_s']}\n")
    print(f"\nRésultats écrits dans {csv_path}")


if __name__ == "__main__":
    main()
