#!/usr/bin/env python3
"""
gen_ssp_instance_custom.py — Génère UNE instance SSP-CCP (E,T,N fixés), avec
fenêtre de difficulté REF (nombre de nœuds Gecode base) personnalisable, et
l'écrit au format 4-fichiers attendu par load_ssp_instance()
(solver_generic/ssp/ssp_loader.hpp) -- directement utilisable comme <test_dir>
pour test_gnn_v1cld_bab / test_gnn_v1cld_bab_batched.

Contrairement à collect/collect_large_base_samples_ssp.py (dataset
d'entraînement, échantillonnage 10% des nœuds B&B), ce script produit UNE
instance pour un test ponctuel -- ajouté 2026-08-20 pour tester la
généralisation en N (E4_T12_N100/N500, au-delà du plafond d'entraînement du
modèle N∈{20,50}) avec REF forcé à galérer (fenêtre cible relevée par rapport
au défaut [1e4,1e6], qui laissait passer des instances résolues en <0.2s).

Usage :
  python gen_ssp_instance_custom.py --E 4 --T 12 --N 100 \\
      --target_lo 1e6 --target_hi 1e8 --probe_bin solver_bab/build/ssp_probe \\
      --out_dir gnn_ssp/test_data/ssp_native_test_E4T12N100_hard
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "solver_generic" / "ssp"))

from ssp_stochastic.generator_adaptive import generate_episode_fixed_E  # noqa: E402
from export_to_txt import write_instance  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--E", type=int, required=True)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--ep_id", type=int, default=0)
    p.add_argument("--master_seed", type=int, default=42)
    p.add_argument("--target_lo", type=float, default=1e4)
    p.add_argument("--target_hi", type=float, default=1e6)
    p.add_argument("--probe_time_lim", type=float, default=5.0)
    p.add_argument("--max_bisect_iters", type=int, default=8)
    p.add_argument("--max_automaton_retries", type=int, default=15)
    p.add_argument("--probe_bin", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()

    ep = generate_episode_fixed_E(
        ep_id=args.ep_id, E=args.E, T=args.T, N=args.N, epsilon=args.epsilon,
        probe_bin=args.probe_bin, master_seed=args.master_seed,
        target_lo=args.target_lo, target_hi=args.target_hi,
        probe_time_lim=args.probe_time_lim,
        max_bisect_iters=args.max_bisect_iters,
        max_automaton_retries=args.max_automaton_retries,
    )

    write_instance(ep, args.out_dir)

    print(f"[gen] E={ep['E']} T={ep['T']} N={ep['N']} epsilon={ep['epsilon']}")
    print(f"[gen] scale={ep['meta']['scale']:.4f}  calib_n_iters={ep['meta']['calib_n_iters']}  "
          f"automaton_attempt={ep['meta']['automaton_attempt']}")
    print("[gen] calib_history (bissection, scale -> n_nœuds sous ssp_probe) :")
    for h in ep["meta"]["calib_history"]:
        print(f"       {h}")
    print(f"[gen] instance écrite dans {args.out_dir}")


if __name__ == "__main__":
    main()
