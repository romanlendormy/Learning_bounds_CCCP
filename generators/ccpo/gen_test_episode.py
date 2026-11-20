"""gen_test_episode.py — épisode de TEST pour la comparaison B&B (REF / GNN / SLD), mêmes paramètres que
les données d'entraînement (Q=round(2.5n), k=max(3,n//2), L=2, U=8, gamma=1.25, eps=0.1,
cf. collect/ccpo/collect_ccpo_samples.py) mais seeds DISJOINTS de ceux de la collecte (>= 7e9).
Usage : python3 generators/ccpo/gen_test_episode.py n N i out_dir   (i = indice d'instance)
Écrit out_dir/ep_000000 ; code retour 0 si écrit, 3 si l'instance est non discrétisable (à ignorer)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from generators.ccpo.instance import draw_mother_instance
from generators.ccpo.scenarios import NOISE_PARAMS_DEFAULT as NOISE_PARAMS, generate_scenarios
from generators.ccpo.discretize import discretize_instance
from generators.ccpo.writers import write_episode

n, N, i, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
Q, k, L, U, gamma, eps = round(2.5 * n), max(3, n // 2), 2, 8, 1.25, 0.1
seed = 7_000_000_000 + 100_000 * n + 1_000 * N + i
mother = draw_mother_instance(np.random.default_rng(seed), n=n)
sc = generate_scenarios(np.random.default_rng(seed + 1), mother, N=N)
try:
    di = discretize_instance(mother, sc, Q=Q, k=k, L=L, U=U, gamma=gamma)
except Exception as e:  # noqa: BLE001
    print(f"SKIP n={n} N={N} i={i} seed={seed} : {type(e).__name__}: {e}")
    sys.exit(3)
write_episode(out, 0, mother=mother, discretized=di, epsilon=eps, seed_master=seed, n_episodes=1,
              noise_params=NOISE_PARAMS)
print(f"OK n={n} N={N} i={i} seed={seed} Q={Q} k={k} -> {out}/ep_000000")
