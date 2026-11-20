"""
Tests de diagnose_epsilon.py, sur de petites tailles pour rester rapides
(la grille de référence n=15/N=50 est documentée dans README.md).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from generators.ccpo.diagnose_epsilon import (
    DiagnosisResult,
    enumerate_candidates,
    run_diagnosis,
    satisfied_scenario_counts,
    solve_for_epsilon,
)


def test_enumerate_candidates_all_valid():
    n, Q, k, L, U = 6, 8, 3, 2, 4
    candidates = enumerate_candidates(n=n, Q=Q, k=k, L=L, U=U)
    assert candidates.shape[1] == n
    assert candidates.shape[0] > 0

    for q in candidates:
        assert q.sum() == Q
        support = q[q > 0]
        assert support.size <= k
        assert np.all((support >= L) & (support <= U))


def test_enumerate_candidates_no_duplicates():
    n, Q, k, L, U = 6, 8, 3, 2, 4
    candidates = enumerate_candidates(n=n, Q=Q, k=k, L=L, U=U)
    unique = np.unique(candidates, axis=0)
    assert unique.shape[0] == candidates.shape[0]


def test_enumerate_candidates_matches_brute_force():
    """Recoupement avec une génération naïve (itertools.product), sur un cas minuscule."""
    n, Q, k, L, U = 4, 6, 2, 2, 4
    candidates = enumerate_candidates(n=n, Q=Q, k=k, L=L, U=U)

    brute = []
    for q in itertools.product([0, 2, 3, 4], repeat=n):
        if sum(q) == Q and sum(1 for v in q if v > 0) <= k:
            brute.append(q)
    brute_set = {tuple(q) for q in brute}
    candidates_set = {tuple(int(v) for v in row) for row in candidates}
    assert candidates_set == brute_set


def test_enumerate_candidates_rejects_infeasible_grid():
    with pytest.raises(ValueError):
        enumerate_candidates(n=5, Q=100, k=2, L=2, U=4)  # k*U=8 << Q=100


def test_satisfied_scenario_counts_matches_naive_loop():
    rng = np.random.default_rng(0)
    n, N, M = 5, 7, 20
    Sigma_int = rng.integers(-100, 100, size=(N, n, n))
    Sigma_int = Sigma_int + np.transpose(Sigma_int, (0, 2, 1))  # symétrise
    candidates = rng.integers(0, 5, size=(M, n)).astype(np.int64)
    B = 500

    counts_vectorized = satisfied_scenario_counts(candidates, Sigma_int, B)

    counts_naive = np.zeros(M, dtype=np.int64)
    for m in range(M):
        q = candidates[m]
        for j in range(N):
            if q @ Sigma_int[j] @ q <= B:
                counts_naive[m] += 1

    np.testing.assert_array_equal(counts_vectorized, counts_naive)


def test_solve_for_epsilon_picks_best_feasible():
    candidates = np.array([[2, 0], [0, 2], [1, 1]])
    counts = np.array([10, 3, 8])  # scénarios satisfaits par candidat
    c_int = np.array([1, 10])  # le 2e actif rapporte plus
    N = 10

    # eps=0.0 -> seuil=10 -> seul le candidat 0 (counts=10) est faisable
    p = solve_for_epsilon(0.0, candidates, counts, c_int, N)
    assert p.feasible
    np.testing.assert_array_equal(p.best_q, [2, 0])

    # eps=0.3 -> seuil=ceil(0.7*10)=7 -> candidats 0 et 2 faisables, le 2e (1,1) rapporte 11
    p = solve_for_epsilon(0.3, candidates, counts, c_int, N)
    assert p.feasible
    np.testing.assert_array_equal(p.best_q, [1, 1])
    assert p.optimal_value == 11


def test_solve_for_epsilon_infeasible_when_no_candidate_meets_threshold():
    candidates = np.array([[2, 0]])
    counts = np.array([3])
    p = solve_for_epsilon(0.0, candidates, counts, np.array([1, 1]), N=10)
    assert not p.feasible
    assert p.optimal_value is None


def test_run_diagnosis_smoke_and_monotonic():
    """
    Instance petite (pas la grille de référence n=15/N=50 -- trop lente pour un test
    unitaire) : vérifie juste que le pipeline complet s'exécute et reste monotone (le
    garde-fou interne de run_diagnosis lèverait sinon).
    """
    result = run_diagnosis(
        n=6, N=15, Q=8, k=3, L=2, U=4, epsilons=(0.0, 0.1, 0.2, 0.3), seed_master=1
    )
    assert isinstance(result, DiagnosisResult)
    assert result.n_candidates > 0
    assert len(result.points) == 4

    feasible_values = [p.optimal_value for p in result.points if p.feasible]
    assert feasible_values == sorted(feasible_values)


def test_run_diagnosis_deterministic():
    kwargs = dict(n=6, N=15, Q=8, k=3, L=2, U=4, epsilons=(0.1, 0.2), seed_master=7)
    a = run_diagnosis(**kwargs)
    b = run_diagnosis(**kwargs)
    assert [p.optimal_value for p in a.points] == [p.optimal_value for p in b.points]
    assert [p.threshold for p in a.points] == [p.threshold for p in b.points]
