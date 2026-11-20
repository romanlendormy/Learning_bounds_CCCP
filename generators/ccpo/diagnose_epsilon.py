"""
diagnose_epsilon.py — Diagnostic de sensibilité à epsilon (calibration du générateur).

Outil de calibration des coefficients de la perturbation en scénarios (noise_beta,
noise_psi, noise_market, p_shock, kappa) et de gamma. Contrairement au B&B Gecode
(solver_bab/ccpo/), on résout ici par **énumération exhaustive** sur
un petit épisode -- aucune dépendance à un solveur externe.

Problème résolu, pour un episode (n, N, Q, k, L, U, c_int, B, Sigma_int) fixé :

    max  c_int' q
    s.t. sum_i q_i = Q
         q_i in {0} U {L, ..., U}
         |{i : q_i > 0}| <= k
         sum_j [q' Sigma_int[j] q <= B] >= ceil((1-eps) * N)

Lecture de la courbe (valeur optimale vs epsilon) :
    - plate                    -> bruit trop faible (augmenter p_shock/kappa/noise_market)
    - effondrement brutal       -> bruit trop fort
    - décroissance régulière    -> calibration correcte

Usage :
    python -m generators.ccpo.diagnose_epsilon
    python -m generators.ccpo.diagnose_epsilon --n 15 --N 50 --Q 20 --k 5 --L 2 --U 8 \\
        --epsilons 0.0 0.05 0.1 0.2 0.3 --seed-master 2026 --out diagnose_epsilon.png
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

from generators.ccpo.instance import draw_mother_instance
from generators.ccpo.scenarios import generate_scenarios
from generators.ccpo.discretize import GAMMA_DEFAULT, discretize_instance

DEFAULT_N = 15
DEFAULT_NSCEN = 50
DEFAULT_Q = 20
DEFAULT_K = 5
DEFAULT_L = 2
DEFAULT_U = 8
DEFAULT_EPSILONS = (0.0, 0.05, 0.1, 0.2, 0.3)


def _bounded_partitions(remainder: int, slots: int, cap: int):
    """
    Génère tous les tuples (y_1, ..., y_slots), y_i in [0, cap], sommant à `remainder`.
    Élagage : à chaque étape, les bornes [lo, hi] de y_1 garantissent que le reste est
    atteignable par les slots restants (évite de générer des branches mortes).
    """
    if slots == 0:
        if remainder == 0:
            yield ()
        return
    lo = max(0, remainder - (slots - 1) * cap)
    hi = min(cap, remainder)
    for y1 in range(lo, hi + 1):
        for rest in _bounded_partitions(remainder - y1, slots - 1, cap):
            yield (y1,) + rest


def enumerate_candidates(n: int, Q: int, k: int, L: int, U: int) -> np.ndarray:
    """
    Énumère tous les q in (N_0)^n tels que sum(q) = Q, q_i in {0} U {L,...,U},
    |{i : q_i > 0}| <= k. Retourne un tableau (M, n) int64.

    Complexité dominée par C(n, s) * (nb compositions bornées) sommé sur les tailles de
    support s valides -- exhaustif au sens strict, pas une relaxation.
    """
    if L <= 0 or U < L:
        raise ValueError(f"bornes invalides : L={L}, U={U}")
    cap = U - L
    min_support = max(1, math.ceil(Q / U))
    max_support = min(k, Q // L) if L > 0 else k
    if min_support > max_support:
        raise ValueError(
            f"aucun support de taille valide : ceil(Q/U)={min_support} > "
            f"min(k, Q//L)={max_support} (Q={Q}, k={k}, L={L}, U={U})"
        )

    blocks: list[np.ndarray] = []
    for s in range(min_support, max_support + 1):
        remainder = Q - s * L
        if remainder < 0 or remainder > s * cap:
            continue  # taille de support s infaisable pour ce Q

        value_tuples = np.array(list(_bounded_partitions(remainder, s, cap)), dtype=np.int64)
        if value_tuples.size == 0:
            continue
        value_tuples = value_tuples + L  # décalage [0,cap] -> [L,U]
        n_val = value_tuples.shape[0]

        supports = np.array(list(itertools.combinations(range(n), s)), dtype=np.int64)
        n_supp = supports.shape[0]

        # produit cartésien (supports x value_tuples), vectorisé
        idx_full = np.repeat(supports, n_val, axis=0)  # (n_supp*n_val, s)
        val_full = np.tile(value_tuples, (n_supp, 1))  # (n_supp*n_val, s)

        n_rows = n_supp * n_val
        q_block = np.zeros((n_rows, n), dtype=np.int64)
        row_idx = np.repeat(np.arange(n_rows), s)
        q_block[row_idx, idx_full.ravel()] = val_full.ravel()
        blocks.append(q_block)

    if not blocks:
        raise RuntimeError("enumerate_candidates: aucun candidat généré -- grille infaisable.")
    return np.concatenate(blocks, axis=0)


def satisfied_scenario_counts(candidates: np.ndarray, Sigma_int: np.ndarray, B: int) -> np.ndarray:
    """
    Pour chaque candidat q (ligne de `candidates`), compte le nombre de scénarios j tels
    que q' Sigma_int[j] q <= B. Vectorisé sur tous les candidats à la fois, scénario par
    scénario (évite de matérialiser un tableau (M, N) inutilement volumineux).
    """
    M = candidates.shape[0]
    N = Sigma_int.shape[0]
    Q_f = candidates.astype(np.float64)  # exact pour ces magnitudes (<< 2**53)
    counts = np.zeros(M, dtype=np.int64)
    for j in range(N):
        tmp = Q_f @ Sigma_int[j].astype(np.float64)
        port_var = np.einsum("ij,ij->i", tmp, Q_f)
        counts += port_var <= B
    return counts


@dataclass
class EpsilonPoint:
    epsilon: float
    threshold: int
    feasible: bool
    optimal_value: float | None
    best_q: np.ndarray | None


def solve_for_epsilon(
    epsilon: float,
    candidates: np.ndarray,
    counts: np.ndarray,
    c_int: np.ndarray,
    N: int,
) -> EpsilonPoint:
    threshold = math.ceil((1.0 - epsilon) * N)
    mask = counts >= threshold
    if not np.any(mask):
        return EpsilonPoint(epsilon, threshold, False, None, None)
    values = candidates[mask] @ c_int
    best_idx = int(np.argmax(values))
    return EpsilonPoint(
        epsilon, threshold, True, float(values[best_idx]), candidates[mask][best_idx]
    )


@dataclass
class DiagnosisResult:
    points: list[EpsilonPoint]
    n_candidates: int
    elapsed_seconds: float


def _check_monotonic(points: list[EpsilonPoint]) -> None:
    """
    La valeur optimale doit être non-décroissante quand epsilon augmente (le seuil
    requis diminue, donc l'ensemble réalisable ne peut que s'élargir). Une violation
    signale un bug dans l'énumération ou le calcul du seuil, pas une propriété du
    modèle à ignorer.
    """
    feasible_values = [(p.epsilon, p.optimal_value) for p in points if p.feasible]
    for (eps_a, v_a), (eps_b, v_b) in zip(feasible_values, feasible_values[1:]):
        if v_b < v_a - 1e-9:
            raise RuntimeError(
                f"valeur optimale non-monotone en epsilon ({eps_a}->{v_a} puis "
                f"{eps_b}->{v_b}) -- bug probable dans l'énumération ou le calcul du "
                "seuil SAA."
            )


def run_diagnosis(
    n: int = DEFAULT_N,
    N: int = DEFAULT_NSCEN,
    Q: int = DEFAULT_Q,
    k: int = DEFAULT_K,
    L: int = DEFAULT_L,
    U: int = DEFAULT_U,
    epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
    seed_master: int = 2026,
    gamma: float = GAMMA_DEFAULT,
) -> DiagnosisResult:
    """Construit un petit épisode fixé et résout exhaustivement pour chaque epsilon."""
    t0 = time.time()
    mother = draw_mother_instance(np.random.default_rng(seed_master), n=n)
    scenarios = generate_scenarios(np.random.default_rng(seed_master + 1), mother, N=N)
    discretized = discretize_instance(mother, scenarios, Q=Q, k=k, L=L, U=U, gamma=gamma)

    candidates = enumerate_candidates(n=n, Q=Q, k=k, L=L, U=U)
    counts = satisfied_scenario_counts(candidates, discretized.Sigma_int, discretized.B)

    points = [
        solve_for_epsilon(eps, candidates, counts, discretized.c_int, N) for eps in epsilons
    ]
    _check_monotonic(points)

    return DiagnosisResult(
        points=points, n_candidates=candidates.shape[0], elapsed_seconds=time.time() - t0
    )


def _print_report(points: list[EpsilonPoint], elapsed: float, n_candidates: int) -> None:
    print(f"# {n_candidates} candidats énumérés en {elapsed:.1f}s\n")
    print(f"{'epsilon':>8} {'seuil':>6} {'faisable':>9} {'valeur optimale':>16}")
    for p in points:
        val = f"{p.optimal_value:.1f}" if p.feasible else "infaisable"
        print(f"{p.epsilon:>8.2f} {p.threshold:>6d} {str(p.feasible):>9} {val:>16}")


def _plot(points: list[EpsilonPoint], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [p.epsilon for p in points]
    vals = [p.optimal_value if p.feasible else float("nan") for p in points]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(eps, vals, marker="o")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("valeur optimale (c_int' q)")
    ax.set_title("Sensibilité à epsilon")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"# figure écrite : {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--N", type=int, default=DEFAULT_NSCEN)
    parser.add_argument("--Q", type=int, default=DEFAULT_Q)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--U", type=int, default=DEFAULT_U)
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    parser.add_argument("--seed-master", type=int, default=2026)
    parser.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    parser.add_argument("--out", type=str, default=None, help="chemin du PNG (optionnel)")
    args = parser.parse_args()

    result = run_diagnosis(
        n=args.n,
        N=args.N,
        Q=args.Q,
        k=args.k,
        L=args.L,
        U=args.U,
        epsilons=tuple(args.epsilons),
        seed_master=args.seed_master,
        gamma=args.gamma,
    )

    _print_report(result.points, result.elapsed_seconds, result.n_candidates)
    if args.out:
        _plot(result.points, args.out)


if __name__ == "__main__":
    main()
