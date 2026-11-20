"""
verify_presolve_flags.py — contrôle de la copie produite par recompute_presolve_flags.py.
  1. instance par instance : tous les champs identiques à la source SAUF `infeasible` ;
  2. (optionnel, --scan) totaux de drapeaux cohérents avec un scan indépendant (JSONL : file, new) ;
  3. aucune instance devenue globalement infaisable (n_infeasible > N - min_feas) ;
  4. chaque nœud curaté (nodes_*_feasible.txt) reste faisable sous les nouveaux drapeaux (évaluateur C++, λ = 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gnn" / "ccpo")); sys.path.insert(0, str(REPO / "collect" / "ccpo"))
from ccpo_dataset import load_instances_ccpo, load_nodes_ccpo  # noqa: E402
from curate_feasible_examples import is_feasible  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--scan", default=None)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    scan = {json.loads(l)["file"]: json.loads(l) for l in open(a.scan)} if a.scan else None
    ok = True
    for f in sorted(src.glob("instances_n*_N*.txt")):
        A = load_instances_ccpo(str(f)); Bn = load_instances_ccpo(str(dst / f.name))
        assert A.keys() == Bn.keys(), f.name
        n_diff_fields = n_glob_inf = new_flags = 0
        for i, x in A.items():
            y = Bn[i]
            same = (x.n, x.N, x.k, x.Q, x.L, x.U, x.B, x.min_feas, x.opt) == (y.n, y.N, y.k, y.Q, y.L, y.U, y.B, y.min_feas, y.opt) \
                and np.array_equal(x.c_int, y.c_int) and np.array_equal(x.Sigma_dense, y.Sigma_dense)
            n_diff_fields += int(not same)
            n_glob_inf += int(y.infeasible.sum() > y.N - y.min_feas)
            new_flags += int(y.infeasible.sum())
        s = scan.get(f.name, {}) if scan is not None else {}
        agree = scan is None or s.get("new") == new_flags
        nodes = load_nodes_ccpo(str(dst / f.name.replace("instances", "nodes").replace(".txt", "_feasible.txt")))
        n_inf_nodes = sum(0 if is_feasible(Bn[r.instance_id], r) else 1 for r in nodes)
        print(f"{f.name}: champs ≠ hors drapeaux = {n_diff_fields} | drapeaux exacts {new_flags} (scan {s.get('new', '—')}, "
              f"{'accord' if agree else 'DÉSACCORD'}) | globalement infaisables = {n_glob_inf} | "
              f"nœuds curatés devenus infaisables = {n_inf_nodes}/{len(nodes)}", flush=True)
        ok &= n_diff_fields == 0 and agree and n_glob_inf == 0 and n_inf_nodes == 0
    print("STATUT VERIF", "OK" if ok else "ÉCHEC")


if __name__ == "__main__":
    main()
