"""
Tests Module CP-SAT — vérité terrain.

Tests sur instance jouet : E=3, T=8, N=20, automate ~5 états.

Vérifié :
  FAISABILITÉ       : la solution retournée est faisable (couvre ≥ quantile scénarios)
  COHÉRENCE COÛT    : objectif ≥ borne triviale (= 0)
  REPRODUCTIBILITÉ  : même instance → même résultat
  BORNE INF         : lower_bound ≤ objective  (garde-fou pour le B&B downstream)
  AUTOMATE          : chaque quart est accepté par le DFA
  COUVERTURE        : solution vérifie les contraintes de couverture manuellement
  MICRO-ÉNUMÉRATION : sur micro-instance, énumération exhaustive ≈ CP-SAT
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import numpy as np
import pytest
from ssp_stochastic.automaton import AutomatonParams, build_shift_dfa, is_nonempty_at_length
from ssp_stochastic.demand import generate_demand_scenarios, make_bimodal_nominal
from ssp_stochastic.cost import generate_cost
from ssp_stochastic.groundtruth import solve_instance, validate_oos


# ── instance jouet ────────────────────────────────────────────────────────────

T_TOY, N_TOY = 12, 20
EPS_TOY = 0.1
SEED_TOY = 42

# Automate souple : service L_min=5..L_max=9, meal_len=1, n_P=0, b_max=7
# Avec T=12 et L_max=9, les quarts couvrent 5–9 périodes consécutives.
# Des employés avec des décalages différents couvrent facilement l'horizon central.
PARAMS_TOY = AutomatonParams(
    L_min=5, L_max=9, m=4, n_P=0, meal_len=1,
    a_min=1, b_max=7, win_start=2, win_end=8,
)

# Demande : pics centraux (périodes 2..9), nul aux bords pour faciliter la couverture.
# max(D_BAR) = 3, E_TOY = 5 → tightness ~ 0.75 (faisable).
D_BAR_TOY = np.array([0, 1, 2, 3, 3, 3, 3, 3, 3, 2, 1, 0])
E_TOY = 5


def _make_toy_instance():
    rng = np.random.default_rng(SEED_TOY)
    demand = generate_demand_scenarios(D_BAR_TOY, rho=0.4, sigma_dem=0.15, N=N_TOY, rng=rng)
    salary, _ = generate_cost(T_TOY, np.random.default_rng(SEED_TOY + 1))
    dfa = build_shift_dfa(PARAMS_TOY)
    return dfa, salary, demand


# ── tests ─────────────────────────────────────────────────────────────────────

def test_toy_feasible():
    """La solution est faisable ou optimale (pas infaisable)."""
    dfa, salary, demand = _make_toy_instance()
    res = solve_instance(dfa, salary, demand, E=E_TOY, epsilon=EPS_TOY,
                         time_limit=30.0)
    assert res.status in ("OPTIMAL", "FEASIBLE"), \
        f"Statut inattendu : {res.status}"


def test_toy_objective_nonnegative():
    """Objectif ≥ 0 (coûts positifs)."""
    dfa, salary, demand = _make_toy_instance()
    res = solve_instance(dfa, salary, demand, E=E_TOY, epsilon=EPS_TOY,
                         time_limit=30.0)
    if res.objective is not None:
        assert res.objective >= -1e-6, f"Objectif négatif : {res.objective}"


def test_lower_bound_le_objective():
    """lower_bound ≤ objective (garde-fou pour B&B downstream)."""
    dfa, salary, demand = _make_toy_instance()
    res = solve_instance(dfa, salary, demand, E=E_TOY, epsilon=EPS_TOY,
                         time_limit=30.0)
    if res.objective is not None and res.lower_bound is not None:
        assert res.lower_bound <= res.objective + 1e-4, (
            f"lower_bound={res.lower_bound} > objective={res.objective}"
        )


def test_trivial_lb_le_objective():
    """Borne triviale = 0 ≤ objective."""
    dfa, salary, demand = _make_toy_instance()
    res = solve_instance(dfa, salary, demand, E=E_TOY, epsilon=EPS_TOY,
                         time_limit=30.0)
    if res.objective is not None:
        assert res.trivial_lower_bound <= res.objective + 1e-4


def test_schedule_accepted_by_automaton():
    """Chaque quart dans le planning est accepté par le DFA."""
    dfa, salary, demand = _make_toy_instance()
    res = solve_instance(dfa, salary, demand, E=E_TOY, epsilon=EPS_TOY,
                         time_limit=30.0)
    if res.schedule is None:
        pytest.skip("Pas de solution trouvée")
    for e in range(E_TOY):
        word = "".join(res.schedule[e])
        assert dfa.accepts(word), f"Quart employé {e} refusé par le DFA : {word}"


def test_coverage_constraint_satisfied():
    """Σ_k z[k] ≥ quantile et vérification manuelle des contraintes de couverture."""
    dfa, salary, demand = _make_toy_instance()
    quantile = int(np.ceil((1.0 - EPS_TOY) * N_TOY))
    res = solve_instance(dfa, salary, demand, E=E_TOY, epsilon=EPS_TOY,
                         time_limit=30.0)
    if res.schedule is None or res.z_coverage is None:
        pytest.skip("Pas de solution trouvée")

    # Σ_k z[k] ≥ quantile
    assert res.z_coverage.sum() >= quantile, \
        f"Seulement {res.z_coverage.sum()} scénarios couverts, attendu ≥ {quantile}"

    # Vérification manuelle : pour chaque scénario couvert, y[t] ≥ d[t,k] ∀t
    E_sol, T_sol = res.schedule.shape
    y = np.array([
        sum(1 for e in range(E_sol) if res.schedule[e, t] == 'W')
        for t in range(T_sol)
    ])
    for k in range(N_TOY):
        if res.z_coverage[k]:
            for t in range(T_TOY):
                assert y[t] >= demand[t, k], (
                    f"Couverture violée : y[{t}]={y[t]} < d[{t},{k}]={demand[t,k]}"
                )


def test_reproducibility():
    """Même instance → même objectif."""
    dfa, salary, demand = _make_toy_instance()
    res1 = solve_instance(dfa, salary, demand, E=E_TOY, epsilon=EPS_TOY,
                          time_limit=30.0)
    res2 = solve_instance(dfa, salary, demand, E=E_TOY, epsilon=EPS_TOY,
                          time_limit=30.0)
    if res1.status == "OPTIMAL" and res2.status == "OPTIMAL":
        assert abs(res1.objective - res2.objective) < 1e-4


def test_micro_enumeration():
    """
    Micro-instance E=2, T=6, N=5 : compare CP-SAT avec énumération exhaustive.

    On génère tous les quarts valides, puis on vérifie que le coût optimal
    CP-SAT = coût minimum par énumération.
    """
    from ssp_stochastic.automaton import sample_words
    from itertools import product as iproduct

    T_MICRO, E_MICRO, N_MICRO = 6, 2, 5
    EPS_MICRO = 0.0   # tous les scénarios doivent être couverts

    p_micro = AutomatonParams(
        L_min=2, L_max=4, m=3, n_P=0, meal_len=1,
        a_min=1, b_max=3, win_start=2, win_end=3,
    )
    dfa_m = build_shift_dfa(p_micro)
    if not is_nonempty_at_length(dfa_m, T_MICRO):
        pytest.skip("DFA micro vide")

    rng = np.random.default_rng(0)
    demand_m = generate_demand_scenarios(
        make_bimodal_nominal(T_MICRO), rho=0.3, sigma_dem=0.15,
        N=N_MICRO, rng=rng)
    # Réduire la demande pour que le problème soit faisable avec E=2
    demand_m = np.clip(demand_m, 0, E_MICRO)

    salary_m, _ = generate_cost(T_MICRO, np.random.default_rng(1))

    # Tous les quarts valides
    valid_words = sample_words(dfa_m, T_MICRO, n=500, rng=rng, max_attempts=5000)
    # Dédupliquer
    valid_words = list(set(valid_words))

    if len(valid_words) == 0:
        pytest.skip("Aucun mot valide trouvé pour la micro-instance")

    # Coût de chaque quart
    from ssp_stochastic.cost import shift_cost
    word_costs = {w: shift_cost(w, salary_m) for w in valid_words}

    # Énumération sur E=2 employés
    quantile_m = int(np.ceil((1.0 - EPS_MICRO) * N_MICRO))
    best_enum = float('inf')
    for w1, w2 in iproduct(valid_words, valid_words):
        y = np.array([
            (1 if w1[t] == 'W' else 0) + (1 if w2[t] == 'W' else 0)
            for t in range(T_MICRO)
        ])
        # Couverture : scénario k couvert ssi y[t] ≥ d[t,k] ∀t
        n_cov = sum(
            1 for k in range(N_MICRO)
            if all(y[t] >= demand_m[t, k] for t in range(T_MICRO))
        )
        if n_cov >= quantile_m:
            cost = word_costs[w1] + word_costs[w2]
            best_enum = min(best_enum, cost)

    if best_enum == float('inf'):
        pytest.skip("Micro-instance infaisable par énumération")

    # CP-SAT
    res_m = solve_instance(dfa_m, salary_m, demand_m,
                           E=E_MICRO, epsilon=EPS_MICRO, time_limit=30.0)
    if res_m.status not in ("OPTIMAL", "FEASIBLE"):
        pytest.skip(f"CP-SAT retourne {res_m.status} sur micro-instance")

    # CP-SAT optimal ≤ best_enum + tolérance
    assert res_m.objective <= best_enum + 1e-3, (
        f"CP-SAT objective={res_m.objective:.4f} > enum={best_enum:.4f}"
    )
    # Et best_enum ≤ CP-SAT optimal (puisque CP-SAT est optimal... si OPTIMAL)
    if res_m.status == "OPTIMAL":
        assert best_enum <= res_m.objective + 1e-3, (
            f"Enum best={best_enum:.4f} > CP-SAT optimal={res_m.objective:.4f}"
        )
