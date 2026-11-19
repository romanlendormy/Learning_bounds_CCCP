"""stage2_certify.py — ÉTAPE 2 de la certification de faisabilité : reprend les candidats 'reject:undetermined' de l'étape 1 (REF 120 s sans incumbent ni
glouton) et leur donne une limite LONGUE avec REF et FEAS en parallèle (deux processus mono-thread par candidat). Un candidat est accepté si l'un des deux
trouve un incumbent (statut OPTIMAL/FEASIBLE) ; rejeté 'infeasible' si l'un des deux PROUVE l'infaisabilité ; sinon il reste indéterminé.
Met à jour manifest_candidates.csv (colonnes ajoutées : stage2_ref, stage2_feas, stage2_time_s), refait la sélection (10 premiers acceptés par indice,
étape 1 + étape 2 confondues -> pas de biais de difficulté) et réécrit instances_list.txt + instances/.
Usage : stage2_certify.py <out_dir> <binary> [--t2 1800] [--P 90] [--keep 10]"""
import argparse
import csv
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("binary")
ap.add_argument("--t2", type=float, default=1800.0); ap.add_argument("--P", type=int, default=90); ap.add_argument("--keep", type=int, default=10)
a = ap.parse_args()
OUT = Path(a.out)
rows = list(csv.DictReader(open(OUT / "manifest_candidates.csv")))
for r in rows:
    r.setdefault("stage2_ref", ""); r.setdefault("stage2_feas", ""); r.setdefault("stage2_time_s", "")
und = [r for r in rows if r["verdict"] == "reject:undetermined"]
print(f"{len(und)} candidats indetermines a recertifier (REF et FEAS, {a.t2:.0f}s chacun), {a.P} processus en parallele", flush=True)
T0 = time.time()


def run_one(job):
    r, mode = job
    ep = OUT / "candidates" / f"n{r['n']}_N{r['N']}" / f"i{r['idx']}" / "ep_000000"
    t = time.time()
    try:
        p = subprocess.run([a.binary, str(ep), str(a.t2), mode], capture_output=True, text=True, timeout=a.t2 + 900).stdout.strip().splitlines()[-1].split(",")
        return dict(status=p[6], nodes=p[7], time=p[11], objective=p[12], greedy=p[13])
    except Exception as e:  # noqa: BLE001
        return dict(status="ERROR:" + type(e).__name__, nodes="", time=f"{time.time() - t:.0f}", objective="", greedy="")


jobs = [(r, m) for r in und for m in ("ref", "feas")]
res = {}
with ThreadPoolExecutor(max_workers=a.P) as ex:
    for k, (job, out) in enumerate(zip(jobs, ex.map(run_one, jobs))):
        res[(id(job[0]), job[1])] = out
        if (k + 1) % 10 == 0:
            print(f"[{time.time() - T0:6.0f}s] {k + 1}/{len(jobs)}", flush=True)

for r in und:
    rf, ff = res[(id(r), "ref")], res[(id(r), "feas")]
    r["stage2_ref"] = f"{rf['status']}:{rf['objective']}"; r["stage2_feas"] = f"{ff['status']}:{ff['objective']}"
    r["stage2_time_s"] = f"{max(float(rf['time'] or 0), float(ff['time'] or 0)):.0f}"
    if rf["status"] in ("OPTIMAL", "FEASIBLE"):
        r["verdict"] = "accept:REF_long"; r.update(status=rf["status"], objective=rf["objective"])
    elif ff["status"] in ("OPTIMAL", "FEASIBLE"):
        r["verdict"] = "accept:FEAS_long"; r.update(status=ff["status"], objective=ff["objective"])
    elif rf["status"] == "INFEASIBLE" or ff["status"] == "INFEASIBLE":
        r["verdict"] = "reject:infeasible"
fn = list(rows[0].keys())
with open(OUT / "manifest_candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(rows)

NS = sorted({int(r["n"]) for r in rows}); NNS = sorted({int(r["N"]) for r in rows})
sel_lines, summary = [], []
for n in NS:
    for N in NNS:
        cell = sorted([r for r in rows if int(r["n"]) == n and int(r["N"]) == N], key=lambda r: int(r["idx"]))
        acc = [r for r in cell if r["verdict"].startswith("accept")][: a.keep]
        dst_root = OUT / "instances" / f"n{n}_N{N}"
        if dst_root.exists(): shutil.rmtree(dst_root)
        for r in acc:
            shutil.copytree(OUT / "candidates" / f"n{n}_N{N}" / f"i{r['idx']}", dst_root / f"i{r['idx']}")
            sel_lines.append(f"{n} {N} {r['idx']} {r['seed']} {r['verdict']}")
        cnt = lambda v: sum(1 for r in cell if r["verdict"] == v)
        summary.append((n, N, len(acc), len(cell), cnt("reject:infeasible"), cnt("reject:undetermined"), cnt("reject:non_discretisable"),
                        cnt("accept:greedy"), cnt("accept:incumbent_REF"), cnt("accept:REF_long"), cnt("accept:FEAS_long")))
(OUT / "instances_list.txt").write_text("\n".join(sel_lines) + "\n")
print("\ncellule | retenues/10 | candidats | rejetees: infaisables, indeterminees, non discretisables | acceptees par : glouton, REF 120s, REF long, FEAS long")
for s in summary:
    print(f"n={s[0]:2d} N={s[1]:4d} | {s[2]:2d}/{a.keep} | {s[3]:3d} | {s[4]:3d}, {s[5]:3d}, {s[6]:3d} | {s[7]:3d}, {s[8]:3d}, {s[9]:3d}, {s[10]:3d}" + ("   <-- INCOMPLET" if s[2] < a.keep else ""))
print(f"\nduree etape 2 : {time.time() - T0:.0f}s")
