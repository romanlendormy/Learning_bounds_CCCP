"""
gen_bench_instance.py — Génère une instance CC-MKP unique (même distribution
que la collecte collect_s10, cf. collect_large_base_samples.generate_song_tempfile)
et imprime son chemin sur stdout.

Sert de source d'instance pour les benchmarks compare_ref_gnn_gen.sh et
compare_ref_v1ld_v1cld_gnn_gen.sh, qui utilisent une instance générée à la
volée plutôt qu'un fichier figé du jeu Song et al.

Usage :
  python3 collect/gen_bench_instance.py --n 20 --m 5 --N 100 --seed 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_large_base_samples import generate_song_tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--R", type=int, default=1000)
    parser.add_argument("--correlation", type=str, default="weakly_correlated",
                        choices=["uncorrelated", "weakly_correlated",
                                 "strongly_correlated"])
    args = parser.parse_args()

    path = generate_song_tempfile(
        args.n, args.m, args.N, args.seed,
        alpha=args.alpha, R=args.R, correlation=args.correlation,
    )
    print(path)


if __name__ == "__main__":
    main()
