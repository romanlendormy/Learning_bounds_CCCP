"""
build_extra_feasible_nodes.py — complète le jeu d'entraînement CCPO à ~10k nœuds FAISABLES.

Contexte : la sélection historique (target_graphs 10000, seed 42, split 0.8) donne 8008 nœuds train dont 1431 au
sous-arbre prouvé infaisable (infeasible_nodes/node_labels.jsonl), exclus de la perte -> 6577 vus par epoch.
Ici on tire, dans les pools BRUTS (collect_campaign_v2/nodes_n*_N*.txt), des nœuds supplémentaires jusqu'à atteindre
--target_train_per_cell nœuds faisables train par cellule, SANS toucher la VAL :
  - instances de la VAL (split seed 42) interdites ; instances non encore utilisées par le train en priorité,
    round-robin (1 nœud par instance puis on recommence), comme stratified_select ;
  - quota de racines = root_frac du complément ; nœuds déjà présents (ligne identique) dans le fichier curaté exclus ;
  - critère de la curation historique (curate_feasible_examples.is_feasible : feuilles rejetées, >= min_feas
    scénarios avec ζ faisable à λ = 0), PUIS faisabilité EXACTE du sous-arbre (même critère que find_infeasible_nodes.py) : force brute sur les variables
    libres, une affectation au moins satisfait >= min_feas scénarios sur la solution complète (tous les scénarios,
    indépendant du presolve) ; "unknown" (énumération > cap) rejeté.
Instances : drapeaux du presolve exact (collect_campaign_v2_presolve_exact), mêmes ids que le pool brut.
Sorties (dossier --out_dir NEUF) : nodes_n*_N*_extra.txt (lignes brutes du pool, format inchangé), extra_labels.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gnn" / "ccpo"))
import train_ccpo_gnn as T  # noqa: E402
from ccpo_dataset import load_nodes_ccpo  # noqa: E402,F401
from ccpo_bruteforce import gen_free  # noqa: E402
from ccpo_dataset import CCPONodeRecord  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from curate_feasible_examples import is_feasible as curation_ok  # noqa: E402  (critère de la curation historique)

CELLS = None
CAP = 5_000_000


def check(t):
    """(n, N, line) -> (line, statut, n_free, n_enum)."""
    n, N, line = t
    p = line.strip().split(",")
    inst = CELLS[(n, N)].instances[int(p[0])]
    depth = int(p[1])
    q = np.array([int(x) for x in p[3:]], dtype=np.int64)
    free = np.where(q < 0)[0]
    # 1) critère de la curation historique (feuilles rejetées ; >= min_feas scénarios avec ζ faisable à λ = 0)
    if not curation_ok(inst, CCPONodeRecord(int(p[0]), depth, float(p[2]), q)):
        return line, "rejet_curation", int(len(free)), 0
    # 2) faisabilité exacte du sous-arbre (find_infeasible_nodes.py)
    if depth == 0:
        return line, "feasible", int(len(free)), 0
    v_fixed = np.where(q > 0, q, 0).astype(np.float64)
    Q_res = inst.Q - int(v_fixed.sum()); k_res = inst.k - int((q > 0).sum())
    X = gen_free(len(free), Q_res, k_res, inst.L, inst.U, CAP)
    if X is None:
        return line, "unknown", int(len(free)), -1
    if len(X) == 0:
        return line, "infeasible", int(len(free)), 0
    full = np.tile(v_fixed, (len(X), 1)); full[:, free] = X
    cnt = np.zeros(len(X), dtype=np.int64)
    for j in range(inst.N):
        cnt += np.einsum("mi,ij,mj->m", full, inst.Sigma_dense[j], full) <= inst.B + 1e-9
    return line, ("feasible" if (cnt >= inst.min_feas).any() else "infeasible"), int(len(free)), int(len(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="jeu curaté à drapeaux exacts (recompute_presolve_flags.py)")
    ap.add_argument("--pool_dir", required=True, help="collecte brute (collect_ccpo_samples.py)")
    ap.add_argument("--labels", required=True, help="étiquettes de find_infeasible_nodes.py (node_labels.jsonl)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_train_per_cell", type=int, default=1112)
    ap.add_argument("--root_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draw_seed", type=int, default=2026)
    ap.add_argument("--procs", type=int, default=96)
    ap.add_argument("--smoke", type=int, default=0, help="test : plafonne le besoin par cellule et ne traite que n=10 N=100")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=False)          # dossier neuf obligatoire

    global CELLS
    CELLS = T.load_all(Path(a.data_dir), node_suffix="_feasible")
    sel = T.stratified_select(CELLS, 10000, a.root_frac, random.Random(a.seed))
    train, val = T.split_by_instance(sel, 0.8, a.seed)
    inf = {(r["n"], r["N"], r["inst"], r["idx"]) for r in map(json.loads, open(a.labels)) if r["status"] == "infeasible"}
    val_inst = {(e.n, e.N, e.instance_id) for e in val}
    train_inst = {(e.n, e.N, e.instance_id) for e in train}
    feas_train = Counter((e.n, e.N) for e in train if (e.n, e.N, e.instance_id, e.node_idx) not in inf)
    rng = random.Random(a.draw_seed)
    summary = {}

    lab_f = open(out / "extra_labels.jsonl", "x")
    with Pool(a.procs) as pool:                       # fork : CELLS hérité
        for (n, N) in sorted(CELLS):
            if a.smoke and (n, N) != (10, 100):
                continue
            need = a.target_train_per_cell - feas_train[(n, N)]
            if a.smoke:
                need = min(need, a.smoke)
            need_root = round(need * a.root_frac)
            base_lines = set(open(Path(a.data_dir) / f"nodes_n{n}_N{N}_feasible.txt").read().splitlines())
            roots, nonroots = [], {}
            with open(Path(a.pool_dir) / f"nodes_n{n}_N{N}.txt") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line or line in base_lines:
                        continue
                    i, d = line.split(",", 2)[:2]
                    i = int(i)
                    if (n, N, i) in val_inst or i not in CELLS[(n, N)].instances:
                        continue
                    if d == "0":
                        roots.append(line)
                    else:
                        nonroots.setdefault(i, []).append(line)
            rng.shuffle(roots)
            # racines : instances non utilisées par le train d'abord
            roots.sort(key=lambda l: (n, N, int(l.split(",", 1)[0])) in train_inst)
            order = sorted(nonroots, key=lambda i: ((n, N, i) in train_inst, rng.random()))
            for i in order:
                rng.shuffle(nonroots[i])

            def rr():                                 # round-robin 1 nœud / instance
                k = 0
                while True:
                    any_left = False
                    for i in order:
                        if k < len(nonroots[i]):
                            any_left = True
                            yield nonroots[i][k]
                    if not any_left:
                        return
                    k += 1

            kept, stats = [], Counter()
            for quota, stream in ((need_root, iter(roots)), (need - need_root, rr())):
                got = 0
                while got < quota:
                    batch = [next(stream, None) for _ in range(4 * a.procs)]
                    batch = [b for b in batch if b is not None]
                    if not batch:
                        break
                    for line, st, nf, ne in pool.imap(check, [(n, N, b) for b in batch], chunksize=2):
                        stats[st] += 1
                        if st == "feasible" and got < quota:
                            kept.append(line); got += 1
                            p = line.split(",", 2)
                            lab_f.write(json.dumps(dict(n=n, N=N, inst=int(p[0]), depth=int(p[1]), n_free=nf,
                                                        status=st, n_enum=ne)) + "\n")
            with open(out / f"nodes_n{n}_N{N}_extra.txt", "x") as f:
                f.write("".join(l + "\n" for l in kept))
            summary[f"n{n}_N{N}"] = dict(feas_train_base=feas_train[(n, N)], need=need, kept=len(kept),
                                         tested=dict(stats), new_instances=len({l.split(',', 1)[0] for l in kept
                                                                                if (n, N, int(l.split(',', 1)[0])) not in train_inst}))
            print(f"[n={n} N={N}] base faisable {feas_train[(n, N)]} besoin {need} gardés {len(kept)} testés {dict(stats)}",
                  flush=True)
    lab_f.close()
    json.dump(summary, open(out / "summary.json", "x"), indent=1)
    print("TOTAL gardés", sum(v["kept"] for v in summary.values()))


if __name__ == "__main__":
    main()
