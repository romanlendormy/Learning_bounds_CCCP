#!/usr/bin/env python3
"""
sweep_bab_bucketed_risk.py — Mesure le risque algorithmique PUR du bucketing
de nœuds B&B (gecode_bab_bucketed.hpp/cpp) : grouper K nœuds par n_g avant de
les juger contre l'incumbent, à borne INCHANGÉE (V1LD/REF), fait-il exploser
le nombre de nœuds nécessaires à la preuve d'optimalité ?

Compare, pour chaque instance et chaque K ∈ K_VALS :
  - bab_mkp_gecode_bucketed <...> K   (K=1 doit reproduire EXACTEMENT
    bab_mkp_gecode — test de non-régression déjà validé manuellement)

Métrique de décision : nodes(K) / nodes(K=1) — si ça reste proche de 1 même
à K grand, le risque algorithmique est négligeable (cf. plan, Étape 1).

Usage :
    python sweep_bab/sweep_bab_bucketed_risk.py
"""

import csv
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

BUCKETED_BIN = ROOT / "solver_bab" / "build" / "bab_mkp_gecode_bucketed"
DATASET_DIR  = ROOT / "data" / "knapsack" / "datasets-pisinger"
OUTPUT_CSV   = ROOT / "results" / "sweep_bucketed_risk.csv"

EPSILON = 0.10
T_ROOT  = 200
T_NODE  = 30
K_VALS  = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# ── Instances ────────────────────────────────────────────────────────────────
# V1LD (borne exacte, CPU) : restreint à n=10 — n=20 est déjà ~200x plus lent
# par nœud (ζ exact B&B sur N scénarios), impraticable pour un sweep multi-K.
# REF (aucune borne) : n=20, arbre naturellement bien plus gros (pas de
# pruning par borne), résolu en quelques secondes malgré tout — bon stress
# test du bucketing sur un GRAND nombre de nœuds.
INSTANCES = [
    # (mode, dataset_filename, rep, time_lim)
    ("v1ld", "ccmk-song-pisinger-n10-m5-a50-we-r42-S20.txt",    0, -1),
    ("v1ld", "ccmk-song-pisinger-n10-m5-a50-we-r42-S50.txt",    0, -1),
    ("v1ld", "ccmk-song-pisinger-n10-m5-a50-we-r42-S100.txt",   0, -1),
    ("v1ld", "ccmk-song-pisinger-n10-m5-a50-we-r50001-S50.txt", 0, -1),
    ("ref",  "ccmk-song-pisinger-n20-m5-a50-we-r42-S20.txt",    0, -1),
    ("ref",  "ccmk-song-pisinger-n20-m5-a50-we-r42-S50.txt",    0, -1),
    ("ref",  "ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt",   0, 30),
]

CSV_FIELDS = [
    "mode", "dataset", "rep", "K",
    "statut", "obj", "nodes", "fails", "inc_imp", "more_calls",
    "n_bucket_flushes", "mean_bucket_fill", "max_pending_live",
    "t_solve", "t_wall", "nodes_ratio_vs_K1",
]


def run_bucketed(dataset_path, mode, rep, time_lim, K):
    cmd = [
        str(BUCKETED_BIN), str(dataset_path),
        str(EPSILON), str(T_NODE), str(time_lim),
        str(rep), str(T_ROOT), mode, str(K),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    t_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Erreur {proc.returncode}\n{proc.stderr.strip()}")
    return proc.stdout, t_wall


def parse_result(out):
    def pf(pat, default=float("nan")):
        m = re.search(pat, out)
        return float(m.group(1)) if m else default

    def pi(pat, default=-1):
        m = re.search(pat, out)
        return int(m.group(1)) if m else default

    st = re.search(r"Statut\s*:\s*(\S+)", out)
    return dict(
        statut           = st.group(1) if st else "?",
        obj              = pf(r"Valeur optimale\s*:\s*([\d.\-]+)"),
        nodes            = pi(r"N.+uds\s*:\s*(\d+)"),
        fails            = pi(r"Échecs totaux\s*:\s*(\d+)"),
        inc_imp          = pi(r"Améliorations incumbent\s*:\s*(\d+)"),
        more_calls       = pi(r"Appels more\(\)\s*:\s*(\d+)"),
        n_bucket_flushes = pi(r"Flushes de bucket\s*:\s*(\d+)"),
        mean_bucket_fill = pf(r"Remplissage moyen au flush\s*:\s*([\d.]+)"),
        max_pending_live = pi(r"Pic de n.+uds en attente\s*:\s*(\d+)"),
        t_solve          = pf(r"Temps total\s*:\s*([\d.]+)"),
    )


def main():
    if not BUCKETED_BIN.exists():
        raise FileNotFoundError(
            f"Binaire introuvable : {BUCKETED_BIN} — compiler d'abord "
            f"(cmake --build solver_bab/build --target bab_mkp_gecode_bucketed)"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    total = len(INSTANCES) * len(K_VALS)
    run_idx = 0

    print(f"\n{'═'*70}")
    print(f"  SWEEP BUCKETED RISK — {total} runs ({len(INSTANCES)} instances × {len(K_VALS)} K)")
    print(f"{'═'*70}\n")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for mode, ds_name, rep, time_lim in INSTANCES:
            ds_path = DATASET_DIR / ds_name
            if not ds_path.exists():
                print(f"  MANQUANT : {ds_path}")
                continue

            nodes_k1 = None
            for K in K_VALS:
                run_idx += 1
                print(f"[{run_idx:>3}/{total}] {mode:<5} {ds_name:<48} rep={rep} K={K:<4} ...",
                      end="  ", flush=True)
                try:
                    out, t_wall = run_bucketed(ds_path, mode, rep, time_lim, K)
                    r = parse_result(out)
                    if K == 1:
                        nodes_k1 = r["nodes"]
                    ratio = (r["nodes"] / nodes_k1) if nodes_k1 else float("nan")
                    row = {
                        "mode": mode, "dataset": ds_name, "rep": rep, "K": K,
                        "statut": r["statut"], "obj": r["obj"], "nodes": r["nodes"],
                        "fails": r["fails"], "inc_imp": r["inc_imp"],
                        "more_calls": r["more_calls"],
                        "n_bucket_flushes": r["n_bucket_flushes"],
                        "mean_bucket_fill": r["mean_bucket_fill"],
                        "max_pending_live": r["max_pending_live"],
                        "t_solve": r["t_solve"], "t_wall": round(t_wall, 3),
                        "nodes_ratio_vs_K1": round(ratio, 4) if ratio == ratio else "",
                    }
                    writer.writerow(row)
                    f.flush()
                    print(f"{r['statut']:<10} obj={r['obj']:>10.2f}  nodes={r['nodes']:>8}  "
                          f"ratio={ratio:.3f}  t={t_wall:.2f}s")
                except Exception as e:
                    print(f"ERREUR — {e}")

    print(f"\n{'═'*70}")
    print(f"  Résultats : {OUTPUT_CSV}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
