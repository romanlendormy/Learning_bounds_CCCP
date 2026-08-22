#!/usr/bin/env python3
"""
gen_ssp_instance_fixed_automaton.py — Réutilise l'automate ET le salaire d'une
instance SSP-CCP déjà générée (ex: ssp_native_test_E10T8N100_generous, celle
qui s'est révélée dure pour REF le 2026-08-20), pour tester à E,N variables
(T fixe -- l'automate a été validé "non vide" pour ce T précis, changer T
demanderait de re-vérifier is_nonempty_at_length) si LA DIFFICULTÉ SUIT
L'AUTOMATE ou si c'est autre chose (l'interaction demande/échelle/N).

Seule la demande est redessinée (nécessairement, sa taille dépend de N) puis
recalibrée par bissection comme d'habitude -- automate et salaire sont
IDENTIQUES à la source, pas re-tirés.

rho/sigma_dem ne sont pas récupérables depuis les fichiers texte déjà écrits
(pas sauvegardés dans le format 4-fichiers) -- on utilise le milieu des
plages par défaut du générateur plutôt que les valeurs exactes d'origine
(confond légèrement la forme de la demande, mais l'automate -- ce qu'on
teste ici -- est lui bien identique).

Usage :
  python gen_ssp_instance_fixed_automaton.py \\
      --source_dir gnn_ssp/test_data/ssp_native_test_E10T8N100_generous \\
      --E 4 --N 100 --epsilon 0.1 \\
      --target_lo 1e5 --target_hi 1e7 \\
      --probe_bin solver_bab/build_gpu_ssp/ssp_probe \\
      --out_dir gnn_ssp/test_data/ssp_native_test_E4T8N100_fixedaut
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "solver_generic" / "ssp"))

from ssp_stochastic.automaton import DFA, ALPHABET  # noqa: E402
from ssp_stochastic.demand import make_bimodal_nominal, generate_demand_scenarios  # noqa: E402
from ssp_stochastic.calibrate_adaptive import calibrate_demand_scale  # noqa: E402
from export_to_txt import write_instance  # noqa: E402

INT_TO_SYM = {i: s for i, s in enumerate(ALPHABET)}


def load_dfa_and_salary(source_dir: Path):
    aut_lines = (source_dir / "automaton.txt").read_text().splitlines()
    num_states = int(aut_lines[0].split()[1])
    initial = int(aut_lines[1].split()[1])
    n_accepting = int(aut_lines[2].split()[1])
    accepting = frozenset(int(x) for x in aut_lines[3].split()) if n_accepting > 0 else frozenset()
    n_trans = int(aut_lines[4].split()[1])
    delta = {}
    for line in aut_lines[5:5 + n_trans]:
        src, sym_int, dst = (int(x) for x in line.split())
        delta[(src, INT_TO_SYM[sym_int])] = dst
    dfa = DFA(num_states=num_states, delta=delta, initial=initial, accepting=accepting)

    cost_lines = (source_dir / "cost.txt").read_text().splitlines()
    salary = np.array([float(x) for x in cost_lines[1].split()[1:]])

    inst_lines = (source_dir / "instance.txt").read_text().splitlines()
    T_source = int(inst_lines[1].split()[1])

    return dfa, salary, T_source


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_dir", type=Path, required=True)
    p.add_argument("--E", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42, help="graine pour le tirage de la demande (pas l'automate, réutilisé)")
    p.add_argument("--target_lo", type=float, default=1e5)
    p.add_argument("--target_hi", type=float, default=1e7)
    p.add_argument("--probe_time_lim", type=float, default=5.0)
    p.add_argument("--max_bisect_iters", type=int, default=12)
    p.add_argument("--probe_bin", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()

    dfa, salary, T = load_dfa_and_salary(args.source_dir)
    print(f"[gen_fixed] automate réutilisé depuis {args.source_dir} "
          f"(num_states={dfa.num_states}, T={T} -- inchangé)")

    rng = np.random.default_rng(args.seed)
    d_bar = make_bimodal_nominal(T, rng=rng)
    # Milieu des plages par défaut de generate_episode_fixed_E (rho_range=(0.3,0.8),
    # sigma_dem_range=(0.1,0.3)) -- rho/sigma_dem exacts de la source non
    # récupérables depuis le format 4-fichiers déjà écrit.
    rho, sigma_dem = 0.55, 0.20
    demand_raw = generate_demand_scenarios(d_bar, rho, sigma_dem, args.N, rng)

    calib = calibrate_demand_scale(
        dfa, salary, demand_raw, args.E, args.epsilon, args.probe_bin,
        target_lo=args.target_lo, target_hi=args.target_hi,
        probe_time_lim=args.probe_time_lim, max_iters=args.max_bisect_iters,
    )

    print(f"[gen_fixed] scale={calib['scale']:.4f}  status={calib['status']}  n_iters={calib['n_iters']}")
    print("[gen_fixed] calib_history :")
    for h in calib["history"]:
        print(f"       {h}")

    if calib["status"] != "target":
        print(f"[gen_fixed] AUCUNE fenêtre cible atteinte avec cet automate à E={args.E} N={args.N} "
              f"(dernier essai gardé quand même, à examiner)")

    ep = dict(dfa=dfa, E=args.E, T=T, N=args.N, epsilon=args.epsilon,
             demand=calib["demand"], salary=salary)
    write_instance(ep, args.out_dir)
    print(f"[gen_fixed] instance écrite dans {args.out_dir}")


if __name__ == "__main__":
    main()
