"""
seeding.py — Dérivation hiérarchique des graines (cf. README.md, « Reproducibility »).

seed_master
  └── SeedSequence(seed_master).spawn(n_episodes)[ep_id]
        ├── spawn(2)[0] → rng_nominal   (instance mère, cf. instance.py)
        └── spawn(2)[1] → rng_scenarios (perturbation en scénarios, cf. scenarios.py)

Aucun appel à numpy.random.seed global : chaque rng est un np.random.Generator local,
dérivé indépendamment. Changer N (nombre de scénarios à générer plus tard) ne modifie
jamais le flux qui a produit l'instance mère d'un épisode donné, et inversement.
"""

from __future__ import annotations

import numpy as np


def episode_rngs(
    seed_master: int, ep_id: int, n_episodes: int
) -> tuple[np.random.Generator, np.random.Generator]:
    """Dérive (rng_nominal, rng_scenarios) pour l'épisode `ep_id` parmi `n_episodes`."""
    if not (0 <= ep_id < n_episodes):
        raise ValueError(f"ep_id={ep_id} hors de [0, {n_episodes})")
    ep_seq = np.random.SeedSequence(seed_master).spawn(n_episodes)[ep_id]
    seq_nominal, seq_scenarios = ep_seq.spawn(2)
    return np.random.default_rng(seq_nominal), np.random.default_rng(seq_scenarios)
