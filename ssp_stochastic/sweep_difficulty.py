"""
sweep_difficulty.py — Balayage de tightness_target (et éventuellement d'autres
axes) pour trouver la région de paramètres qui donne un bon compromis :
  - peu d'instances INFEASIBLE (il faut un dataset exploitable pour le GNN)
  - peu d'instances triviales (sinon la borne Lagrangienne n'a aucun intérêt)
  - une majorité d'instances "dures mais résolues" dans un budget raisonnable

Pour chaque valeur de tightness_target : génère n_reps épisodes via
ssp_stochastic.generator, les exporte en texte, et sonde chacun avec
ssp_probe (Gecode base seul — pas v1CLD/v1LD, trop coûteux pour un balayage).

Usage :
    python sweep_difficulty.py --out sweep_results.csv
    python sweep_difficulty.py --tightness 0.9 1.0 1.1 1.2 1.4 1.6 --n-reps 15
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ssp_stochastic.generator import generate_dataset, DEFAULT_CONFIG
from ssp_stochastic.writers import open_dataset, read_episode, list_episodes

sys.path.insert(0, str(REPO_ROOT / "solver_generic" / "ssp"))
from export_to_txt import write_instance

PROBE_BIN = REPO_ROOT / "solver_bab" / "build" / "ssp_probe"

# Seuils de classification (Gecode base, sonde rapide)
TRIVIAL_NODE_THRESHOLD = 2000     # OPTIMAL en dessous de ce nb de nœuds -> trivial
TRIVIAL_TIME_THRESHOLD = 0.3      # ou en dessous de ce temps (s)


def classify(status: str, nodes: int, time_s: float) -> str:
    if status == "INFEASIBLE":
        return "infeasible"
    if status == "TIMEOUT":
        return "timeout"
    if status == "OPTIMAL" and (nodes < TRIVIAL_NODE_THRESHOLD or time_s < TRIVIAL_TIME_THRESHOLD):
        return "trivial"
    if status == "OPTIMAL":
        return "hard_solved"
    return "other"  # FEASIBLE non prouvé optimal, etc.


def run_one_tightness(tightness: float, n_reps: int, T: int, N: int, eps: float,
                      probe_time_lim: float, master_seed: int, tmp_root: Path) -> list[dict]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(n_episodes=n_reps, T=T, N=N, epsilon=eps,
              tightness_target=tightness, master_seed=master_seed)

    work_dir = tmp_root / f"tgt_{tightness:.2f}"
    generate_dataset(cfg, work_dir)

    h5_path = work_dir / "dataset.h5"
    rows = []
    with open_dataset(h5_path, mode="r") as h5f:
        ep_ids = list_episodes(h5f)
        for ep_id in ep_ids:
            ep = read_episode(h5f, ep_id)
            inst_dir = work_dir / f"ep_{ep_id:05d}"
            write_instance(ep, inst_dir)

            proc = subprocess.run(
                [str(PROBE_BIN), str(inst_dir), str(probe_time_lim)],
                capture_output=True, text=True, timeout=probe_time_lim + 15,
            )
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            if not line:
                rows.append(dict(tightness=tightness, ep_id=ep_id, E=ep["E"],
                                 dfa_states=ep["dfa"].num_states, status="ERROR",
                                 nodes=-1, time_s=-1.0, category="error"))
                continue
            parts = line.split(",")
            _, E, T_, N_, eps_, Q, n_trans, status, nodes, fails, time_s, obj = parts
            cat = classify(status, int(nodes), float(time_s))
            rows.append(dict(
                tightness=tightness, ep_id=ep_id, E=int(E), dfa_states=ep["dfa"].num_states,
                n_trans=int(n_trans), status=status, nodes=int(nodes), fails=int(fails),
                time_s=float(time_s), objective=obj, category=cat,
            ))
    shutil.rmtree(work_dir, ignore_errors=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tightness", type=float, nargs="+",
                        default=[0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8])
    parser.add_argument("--n-reps", type=int, default=12)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--eps", type=float, default=0.10)
    parser.add_argument("--probe-time-lim", type=float, default=5.0)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--out", default="sweep_results.csv")
    args = parser.parse_args()

    if not PROBE_BIN.exists():
        sys.exit(f"ssp_probe introuvable : {PROBE_BIN} (compiler solver_bab/build d'abord)")

    all_rows = []
    with tempfile.TemporaryDirectory(prefix="ssp_sweep_") as tmp:
        tmp_root = Path(tmp)
        for tgt in args.tightness:
            print(f"--- tightness_target={tgt:.2f} ({args.n_reps} épisodes) ---", flush=True)
            rows = run_one_tightness(tgt, args.n_reps, args.T, args.N, args.eps,
                                     args.probe_time_lim, args.master_seed, tmp_root)
            all_rows.extend(rows)
            for r in rows:
                print(f"  ep{r['ep_id']:02d} E={r.get('E','?'):>3} "
                      f"states={r.get('dfa_states','?'):>3} "
                      f"{r['status']:<10} nodes={r.get('nodes','?'):>8} "
                      f"t={r.get('time_s',-1):>6.2f}s  -> {r['category']}", flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n-> {args.out}  ({len(all_rows)} lignes)")

    # ── Résumé par tightness_target ───────────────────────────────────────────
    print(f"\n{'tightness':>10} {'infeasible':>11} {'trivial':>9} {'hard_solved':>12} "
          f"{'timeout':>9} {'other':>7}  {'med_nodes(solved)':>18}")
    for tgt in args.tightness:
        sub = [r for r in all_rows if r["tightness"] == tgt]
        n = len(sub)
        if n == 0:
            continue
        counts = {c: sum(1 for r in sub if r["category"] == c)
                 for c in ["infeasible", "trivial", "hard_solved", "timeout", "error", "other"]}
        solved_nodes = sorted(r["nodes"] for r in sub if r["category"] == "hard_solved")
        med = solved_nodes[len(solved_nodes)//2] if solved_nodes else -1
        print(f"{tgt:>10.2f} {100*counts['infeasible']/n:>10.0f}% {100*counts['trivial']/n:>8.0f}% "
              f"{100*counts['hard_solved']/n:>11.0f}% {100*counts['timeout']/n:>8.0f}% "
              f"{100*counts['other']/n:>6.0f}%  {med:>18}")


if __name__ == "__main__":
    main()
