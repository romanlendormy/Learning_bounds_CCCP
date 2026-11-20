"""
discretize.py — Discrétisation entière de l'instance (mère + scénarios).

Convertit l'instance mère + les scénarios (facteurs flottants) en entiers, prêts à être
consommés tels quels par le solveur (Gecode) — aucun arrondi n'a lieu au chargement,
cf. README.md.

Arrondi **vers le bas systématique** (floor) pour Sigma_int : `floor(x) <= x` quel que
soit le signe de `x`, donc `q'Sigma_int q <= q'Sigma q` pour tout `q >= 0` (quantités de
portefeuille) — l'ensemble réalisable de la contrainte de risque est élargi
(relaxation), jamais rétréci.

`psi_int` est stocké séparément pour préserver la décomposition rang-1 exacte en
entiers :

    rank1_diag_int = floor(S * sigma_f_j**2 * beta_j**2)
    psi_int[j]     = Sigma_int[j].diagonal() - rank1_diag_int

`Sigma_bar` (pour `B`) est calculée sur l'instance MERE uniquement (jamais par
scénario) : le budget de risque `B` doit être commun à tous les scénarios.

Checks bloquants (exception explicite en cas de violation) :

    1. PSD sur les flottants, avant tout arrondi.
    2. psi > 0 et psi_int[j] > 0 pour tout j.
    3. Symétrie de Sigma_int[j], diagonale strictement positive.
    4. Faisabilité structurelle k*L <= Q <= k*U.
    5. Non-débordement : max|Sigma_int| * Q**2 < 2**30.

Toute violation lève une exception explicite (échec bruyant, pas de warning).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generators.ccpo.instance import MotherInstance
from generators.ccpo.scenarios import ScenarioSet

S_DEFAULT = 1_000_000
S_C_DEFAULT = 1_000_000
GAMMA_DEFAULT = 1.25
INT64_SAFE_BOUND = 2**30


def check_structural_feasibility(Q: int, k: int, L: int, U: int) -> None:
    """Check 4 : k*L <= Q <= k*U. Échec bruyant (ValueError), pas de warning."""
    if not (k * L <= Q <= k * U):
        raise ValueError(
            f"instance infaisable structurellement : k*L={k * L} <= Q={Q} <= k*U={k * U} "
            f"violé (k={k}, L={L}, U={U})"
        )


@dataclass
class DiscretizedInstance:
    """Instance CCPO entière, prête pour l'écriture texte (cf. writers.py)."""

    n: int
    N: int
    Q: int
    k: int
    L: int
    U: int
    c_int: np.ndarray  # (n,) int64
    B: int
    Sigma_int: np.ndarray  # (N, n, n) int64
    psi_int: np.ndarray  # (N, n) int64
    S: int
    S_c: int
    gamma: float
    sigma_bar: float  # Sigma_bar de l'instance mère (flottant, avant discrétisation)


def discretize_instance(
    mother: MotherInstance,
    scenarios: ScenarioSet,
    Q: int,
    k: int,
    L: int,
    U: int,
    S: int = S_DEFAULT,
    S_c: int = S_C_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
) -> DiscretizedInstance:
    """Discrétise (mother, scenarios) en entiers, avec les checks bloquants 1-5."""
    n = mother.n
    N = scenarios.N

    # Check 4 -- fail-fast, indépendant des matrices.
    check_structural_feasibility(Q, k, L, U)

    # Check 1 -- PSD sur les flottants, avant tout arrondi.
    min_eigval = min(np.linalg.eigvalsh(scenarios.Sigma[j]).min() for j in range(N))
    if min_eigval <= -1e-12:
        raise RuntimeError(
            f"discretize_instance: Sigma_j non PSD détecté (min eigval={min_eigval}) "
            "avant discrétisation -- check 1 bloquant."
        )

    # Check 2 (partie flottante) -- garanti par construction en amont (instance.py,
    # scenarios.py) ; revérifié ici par défense en profondeur.
    if np.any(mother.psi <= 0) or np.any(scenarios.psi <= 0):
        raise RuntimeError("discretize_instance: psi <= 0 détecté avant discrétisation.")

    Sigma_bar = mother.Sigma_bar  # calculée sur la MERE uniquement

    c_int = np.floor(S_c * mother.c).astype(np.int64)
    B = int(np.floor(S * gamma * Sigma_bar)) * Q**2

    Sigma_int = np.floor(S * scenarios.Sigma).astype(np.int64)

    # Check 5 -- non-débordement, avant tout usage aval. Comparaison en int Python
    # (précision arbitraire) pour ne pas déborder dans le calcul du check lui-même.
    max_abs = int(np.abs(Sigma_int).max())
    if max_abs * Q**2 >= INT64_SAFE_BOUND:
        raise ValueError(
            f"discretize_instance: débordement -- max|Sigma_int|*Q**2 = {max_abs * Q**2} "
            f">= 2**30 ({INT64_SAFE_BOUND}). Baisser S ou Q ("
            "S=1e6 plafonne Q à ~470)."
        )

    rank1_diag_int = np.floor(
        S * (scenarios.sigma_f[:, None] ** 2) * (scenarios.beta**2)
    ).astype(np.int64)
    diag_int = np.diagonal(Sigma_int, axis1=1, axis2=2)  # (N, n)
    psi_int = diag_int - rank1_diag_int

    # Check 2 (partie entière) -- bloquant.
    if np.any(psi_int <= 0):
        n_bad = int((psi_int <= 0).sum())
        raise RuntimeError(
            f"discretize_instance: psi_int <= 0 pour {n_bad} entrée(s) -- check 2 bloquant. "
            "Augmenter S, ou revoir noise_psi/p_shock/kappa (arrondi trop grossier relatif "
            "à psi)."
        )

    # Check 3 -- symétrie et diagonale strictement positive.
    if not np.array_equal(Sigma_int, np.transpose(Sigma_int, (0, 2, 1))):
        raise RuntimeError("discretize_instance: Sigma_int non symétrique -- check 3 bloquant.")
    if np.any(diag_int <= 0):
        raise RuntimeError(
            "discretize_instance: diagonale de Sigma_int non strictement positive -- "
            "check 3 bloquant (S trop petit relativement aux variances)."
        )

    return DiscretizedInstance(
        n=n,
        N=N,
        Q=Q,
        k=k,
        L=L,
        U=U,
        c_int=c_int,
        B=B,
        Sigma_int=Sigma_int,
        psi_int=psi_int,
        S=S,
        S_c=S_c,
        gamma=gamma,
        sigma_bar=Sigma_bar,
    )
