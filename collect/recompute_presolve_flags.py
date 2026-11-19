"""
recompute_presolve_flags.py — copie d'un jeu de données CCPO avec les drapeaux d'infaisabilité de scénario recalculés
par le presolve EXACT certifié.

Les drapeaux de collect_campaign_v2 viennent de l'ancien presolve (presolve_min_variance_box), qui écarte à tort des
scénarios satisfaisables. On écrit un NOUVEAU dossier (jamais d'écrasement) :
  - instances_n*_N*.txt : chaque ligne identique AU CARACTÈRE PRÈS sauf les N jetons de drapeaux, recalculés par le
    code C++ de production (ccpo_presolve_exact::box_min_certified via solver_bab/ccpo/presolve_exact.so, règle
    lower > B·(1+1e-9), identique à presolve_infeasible_scenarios) ;
  - nodes_n*_N*_feasible.txt : copiés tels quels (nœuds curatés ; restent faisables, vérifié par verify_presolve_flags.py) ;
  - flag_stats.jsonl : par fichier, drapeaux anciens / nouveaux / faux positifs / instances touchées.
Le champ `opt` des instances est conservé tel quel (optimum du modèle legacy ; non utilisé par l'entraînement).
Les pools complets nodes_n*_N*.txt (1,2 Go) ne sont pas copiés (inutiles à l'entraînement, qui lit *_feasible).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "gnn_ccpo"))
from ccpo_libs import lib_path  # noqa: E402

SO = lib_path("presolve_exact")
REL_TOL = 1e-9  # = ccpo_presolve_exact::kRelTol
_lib = None


def _cpp():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(SO))
        P = ctypes.POINTER(ctypes.c_double)
        _lib.ccpo_box_min_certified_c.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, P, P]
    return _lib


def process_line(line: str):
    """-> (nouvelle ligne, n_old, n_new, n_fp, n_new_not_old, instance_id)."""
    body = line.rstrip("\n")
    t = body.split(",")
    iid, n, N = int(t[0]), int(t[1]), int(t[2])
    Q, U, B = int(t[4]), int(t[6]), float(t[7])
    f0 = 10 + n
    old = np.array([int(x) for x in t[f0:f0 + N]], dtype=bool)
    p = f0 + N
    nt = n * (n + 1) // 2
    lib = _cpp(); P = ctypes.POINTER(ctypes.c_double)
    out = np.zeros(4)
    new = np.zeros(N, dtype=bool)
    for j in range(N):
        S = np.zeros((n, n))
        for _ in range(nt):
            a, b, v = int(t[p]), int(t[p + 1]), float(t[p + 2]); p += 3
            S[a, b] = S[b, a] = v
        S = np.ascontiguousarray(S)
        lib.ccpo_box_min_certified_c(n, float(Q), float(U), S.ctypes.data_as(P), out.ctypes.data_as(P))
        new[j] = out[0] > B * (1.0 + REL_TOL)
    assert p == len(t), (iid, p, len(t))
    t2 = t[:f0] + [str(int(x)) for x in new] + t[f0 + N:]
    # contrôle de format : avec les anciens drapeaux, la reconstruction redonne la ligne d'origine
    assert ",".join(t[:f0] + [str(int(x)) for x in old] + t[f0 + N:]) == body, iid
    return (",".join(t2) + ("\n" if line.endswith("\n") else ""), int(old.sum()), int(new.sum()),
            int((old & ~new).sum()), int((new & ~old).sum()), iid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=False)  # refus si le dossier existe : aucun écrasement possible
    with Pool(a.procs) as pool, open(dst / "flag_stats.jsonl", "w") as fs:
        for f in sorted(src.glob("instances_n*_N*.txt")):
            tot = dict(file=f.name, n_inst=0, old=0, new=0, fp=0, new_not_old=0, touched=0, touched_ids=[])
            with open(f) as fin, open(dst / f.name, "x") as fout:
                lines = (l for l in fin if l.strip())
                for nl, o, nw, fp, nno, iid in pool.imap(process_line, lines, chunksize=16):
                    fout.write(nl)
                    tot["n_inst"] += 1; tot["old"] += o; tot["new"] += nw; tot["fp"] += fp; tot["new_not_old"] += nno
                    if fp or nno:
                        tot["touched"] += 1; tot["touched_ids"].append(iid)
            fs.write(json.dumps(tot) + "\n"); fs.flush()
            print(f"{f.name}: {tot['n_inst']} inst | drapeaux ancien {tot['old']} -> exact {tot['new']} | "
                  f"faux positifs retirés {tot['fp']} | nouveaux {tot['new_not_old']} | instances touchées {tot['touched']}",
                  flush=True)
    for f in sorted(src.glob("nodes_n*_N*_feasible.txt")):
        shutil.copy2(f, dst / f.name)
    print("copiés :", len(list(dst.glob("nodes_*_feasible.txt"))), "fichiers de nœuds curatés")


if __name__ == "__main__":
    main()
