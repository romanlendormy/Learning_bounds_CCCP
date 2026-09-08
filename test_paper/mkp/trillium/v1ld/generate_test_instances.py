#!/usr/bin/env python3
"""
Génère les instances CC-MKP de test (held-out) pour le rapport de recherche.

Configs : n in {15,20,25,30} x N in {100,500,1000}, m=5.
epsilon=0.1 n'intervient pas ici (paramètre de résolution, pas de génération) —
à passer au solveur au moment du benchmark.

Seeds 43..52 (10 instances par config) — décision explicite de l'utilisateur
du 2026-08-13, malgré un risque de contamination train/test connu pour 3/12
configs (n20-N100 : seeds 0-5171 vues à l'entraînement ; n20-N500 : 0-1041 ;
n25-N100 : 0-1333 — cf. data/train_base_s10/seeds_done_n{n}_m5_N{N}.txt).
Voir aussi compare_holdout_multiinstance.sh, qui documente le même piège pour
seed=42 sur n=10/20 et l'a corrigé en passant à seed>=50001.

Usage:
  python3 test_paper/mkp/trillium/v1ld/generate_test_instances.py
"""
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "collect"))
from collect_large_base_samples import generate_song_tempfile  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
N_VALUES = [15, 20, 25, 30]
N_SCEN_VALUES = [100, 500, 1000]
M = 5
SEEDS = list(range(43, 53))  # 10 seeds : 43..52
ALPHA = 0.5
R = 1000
CORRELATION = "weakly_correlated"


def main():
    n_written = 0
    for n in N_VALUES:
        for N_scen in N_SCEN_VALUES:
            for seed in SEEDS:
                tmp_path = generate_song_tempfile(
                    n, M, N_scen, seed, alpha=ALPHA, R=R, correlation=CORRELATION,
                )
                dest = OUT_DIR / f"instance_n{n}_m{M}_N{N_scen}_seed{seed}.txt"
                shutil.move(tmp_path, dest)
                n_written += 1
    print(f"{n_written} instances écrites dans {OUT_DIR}")


if __name__ == "__main__":
    main()
