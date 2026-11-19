"""gen_and_select.py — génération de la campagne de TEST du papier CCPO : pour chaque cellule (n, N), 10 instances FAISABLES.

Protocole (cf. test_paper/ccpo/README.md) :
  * candidats : seeds i = 100, 110, 120, ... (pas de 10 : le générateur utilise seed et seed+1 ; un pas de 1 ferait réutiliser le flux
    aléatoire des scénarios de l'instance i comme flux "mère" de l'instance i+1) ; seed = 7_000_000_000 + 100_000*n + 1_000*N + i,
    disjoint des seeds d'entraînement (>= 7e9 réservé aux tests) et des instances exploratoires déjà vues (i < 100) ;
  * certificat de faisabilité : un run de REF de T_SCREEN secondes qui trouve un incumbent (statut OPTIMAL/FEASIBLE) OU un glouton faisable ;
  * rejet : INFEASIBLE (prouvé), indéterminé (TIMEOUT sans incumbent ni glouton), non discrétisable ;
  * on garde les 10 premiers candidats acceptés (par indice) de chaque cellule ; biais connu : la sélection favorise les instances où REF/glouton
    trouvent une solution faisable en T_SCREEN s (instances indéterminées écartées, comptées à part).
Usage : gen_and_select.py <repo> <out_dir> <build_bin> <python_venv> [--ns 8,10,12,15] [--Ns 100,500,1000] [--maxc 40] [--keep 10] [--t_screen 120] [--P 100]"""
import argparse
import csv
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("repo"); ap.add_argument("out"); ap.add_argument("binary"); ap.add_argument("python")
ap.add_argument("--ns", default="8,10,12,15"); ap.add_argument("--Ns", default="100,500,1000")
ap.add_argument("--maxc", type=int, default=40); ap.add_argument("--keep", type=int, default=10)
ap.add_argument("--t_screen", type=float, default=120.0); ap.add_argument("--P", type=int, default=100)
a = ap.parse_args()
OUT = Path(a.out); REPO = Path(a.repo)
NS = [int(x) for x in a.ns.split(",")]; NNS = [int(x) for x in a.Ns.split(",")]
tasks = [(n, N, 100 + 10 * k) for n in NS for N in NNS for k in range(a.maxc)]
T0 = time.time()


def seed_of(n, N, i):
    return 7_000_000_000 + 100_000 * n + 1_000 * N + i


def work(t):
    n, N, i = t
    ep = OUT / "candidates" / f"n{n}_N{N}" / f"i{i}"
    row = dict(n=n, N=N, idx=i, seed=seed_of(n, N, i), status="", nodes="", time="", objective="", greedy="", verdict="")
    if not (ep / "ep_000000").exists():
        r = subprocess.run([a.python, str(REPO / "collect/gen_bab_test_episode.py"), str(n), str(N), str(i), str(ep)],
                           capture_output=True, text=True, cwd=REPO)
        if r.returncode != 0:
            row["status"] = "NON_DISCRETISABLE"; row["verdict"] = "reject:non_discretisable"
            return row
    try:
        r = subprocess.run([a.binary, str(ep / "ep_000000"), str(a.t_screen), "ref"], capture_output=True, text=True,
                           timeout=a.t_screen + 600)
        p = r.stdout.strip().splitlines()[-1].split(",")
        row.update(status=p[6], nodes=p[7], time=p[11], objective=p[12], greedy=p[13])
    except Exception as e:  # noqa: BLE001
        row["status"] = "ERROR"; row["verdict"] = f"reject:error:{type(e).__name__}"
        return row
    greedy_ok = row["greedy"] not in ("nan", "")
    if row["status"] in ("OPTIMAL", "FEASIBLE"):
        row["verdict"] = "accept:incumbent_REF"
    elif greedy_ok:
        row["verdict"] = "accept:greedy"
    elif row["status"] == "INFEASIBLE":
        row["verdict"] = "reject:infeasible"
    else:
        row["verdict"] = "reject:undetermined"
    return row


print(f"{len(tasks)} candidats, {a.P} en parallele, REF {a.t_screen:.0f}s chacun", flush=True)
rows = []
with ThreadPoolExecutor(max_workers=a.P) as ex:
    for k, row in enumerate(ex.map(work, tasks)):
        rows.append(row)
        if (k + 1) % 20 == 0:
            print(f"[{time.time() - T0:6.0f}s] {k + 1}/{len(tasks)}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "manifest_candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

sel_lines, summary = [], []
for n in NS:
    for N in NNS:
        cell = sorted([r for r in rows if r["n"] == n and r["N"] == N], key=lambda r: r["idx"])
        acc = [r for r in cell if r["verdict"].startswith("accept")][: a.keep]
        for r in acc:
            src = OUT / "candidates" / f"n{n}_N{N}" / f"i{r['idx']}"
            dst = OUT / "instances" / f"n{n}_N{N}" / f"i{r['idx']}"
            if dst.exists(): shutil.rmtree(dst)
            shutil.copytree(src, dst)
            sel_lines.append(f"{n} {N} {r['idx']} {r['seed']} {r['verdict']}")
        cnt = lambda v: sum(1 for r in cell if r["verdict"] == v)
        summary.append((n, N, len(acc), len(cell), cnt("reject:infeasible"), cnt("reject:undetermined"), cnt("reject:non_discretisable"),
                        sum(1 for r in cell if r["verdict"] == "accept:greedy"), cell[len(cell) - 1]["idx"] if cell else None))
(OUT / "instances_list.txt").write_text("\n".join(sel_lines) + "\n")
print("\ncellule | retenues/10 | candidats | rejetees: infaisables, indeterminees, non discretisables | dont certifiees par le glouton seul")
for s in summary:
    print(f"n={s[0]:2d} N={s[1]:4d} | {s[2]:2d}/{a.keep} | {s[3]:3d} | {s[4]:3d}, {s[5]:3d}, {s[6]:3d} | {s[7]:3d}" + ("   <-- INCOMPLET : relancer avec --maxc plus grand" if s[2] < a.keep else ""))
print(f"\nduree {time.time() - T0:.0f}s ; liste : {OUT / 'instances_list.txt'}")
