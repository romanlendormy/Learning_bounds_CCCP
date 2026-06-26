"""
groundtruth.py — Vérité terrain via OR-Tools CP-SAT.

Résout le modèle SAA du §1 :

    min   Σ_e Σ_t κ(x[e,t], t)
    s.t.  x[e,:] accepté par A            ∀e   (AddAutomaton)
          y[t]   = Σ_e w(x[e,t])          ∀t
          y[t]   ≥ d[t,k] * z[k]          ∀t,k
          Σ_k z[k] ≥ ceil((1-ε)*N)

Avec z[k] ∈ {0,1} : « scénario k entièrement couvert ».

Notons que d[t,k]*z[k] est linéaire (d[t,k] est une constante par scénario).

L'automate est intégré via AddAutomaton sur l'alphabet entier Σ = {O,P,R,W}
avec des variables entières x[e,t] ∈ {0,1,2,3} (indices dans ALPHABET).

Fonctions exportées
-------------------
    solve_instance(dfa, salary, demand, E, epsilon, time_limit, verbosity)
        -> SolveResult(status, objective, schedule, z_coverage, solve_time)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ortools.sat.python import cp_model

from .automaton import DFA, ALPHABET, WORK_IND


# ── résultat ──────────────────────────────────────────────────────────────────

@dataclass
class SolveResult:
    status: str                          # 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNKNOWN'
    objective: Optional[float]           # valeur objectif ou None
    schedule: Optional[np.ndarray]       # (E, T) str, activité par employé/période
    z_coverage: Optional[np.ndarray]     # (N,) bool, scénarios couverts
    solve_time: float                    # secondes
    lower_bound: Optional[float] = None  # borne inférieure CP-SAT
    trivial_lower_bound: Optional[float] = None  # borne triviale (voir §7)


# ── solveur ───────────────────────────────────────────────────────────────────

def solve_instance(
    dfa: DFA,
    salary: np.ndarray,        # (T,) float
    demand: np.ndarray,        # (T, N) int32
    E: int,
    epsilon: float,
    time_limit: float = 60.0,
    verbosity: int = 0,
) -> SolveResult:
    """
    Résout le SSP chance-constrained (modèle SAA).

    CP-SAT utilise AddAutomaton sur les variables x[e,t] entières (indices
    dans l'alphabet trié ALPHABET).

    Paramètres
    ----------
    dfa        : automate des quarts légaux
    salary     : vecteur de coûts s[t]
    demand     : scénarios de demande (T, N)
    E          : nombre d'employés
    epsilon    : niveau de risque (contrainte chance : ≥ ceil((1-ε)*N) scénarios)
    time_limit : limite de temps en secondes
    verbosity  : 0=silencieux, 1=stats finales

    Retourne
    --------
    SolveResult
    """
    T, N = demand.shape
    quantile = int(np.ceil((1.0 - epsilon) * N))

    # ── mapping alphabet ──────────────────────────────────────────────────────
    # CP-SAT AddAutomaton travaille avec des entiers → ALPHABET[i] = i
    sym_to_int: dict[str, int] = {sym: i for i, sym in enumerate(ALPHABET)}
    int_to_sym: dict[int, str] = {i: sym for sym, i in sym_to_int.items()}
    n_sigma = len(ALPHABET)

    # Borne supérieure des coûts pour CP-SAT (entiers)
    # On scale les coûts à des entiers en multipliant par 1000
    SCALE = 1000
    s_int = np.round(salary * SCALE).astype(int)

    # ── modèle CP-SAT ─────────────────────────────────────────────────────────
    model = cp_model.CpModel()

    # Variables x[e,t] ∈ {0,..,n_sigma-1} = indices dans ALPHABET
    x = [[model.NewIntVar(0, n_sigma - 1, f"x_{e}_{t}")
          for t in range(T)]
         for e in range(E)]

    # Variables z[k] ∈ {0,1}
    z = [model.NewBoolVar(f"z_{k}") for k in range(N)]

    # Indicateurs w[e,t] ∈ {0,1} : w[e,t] = 1 ssi x[e,t] = indice(W)
    w_idx = sym_to_int['W']
    w = [[model.NewBoolVar(f"w_{e}_{t}") for t in range(T)] for e in range(E)]
    for e in range(E):
        for t in range(T):
            # w[e,t] = 1 ↔ x[e,t] == w_idx
            model.Add(x[e][t] == w_idx).OnlyEnforceIf(w[e][t])
            model.Add(x[e][t] != w_idx).OnlyEnforceIf(w[e][t].Not())

    # ── contrainte automate (AddAutomaton) ────────────────────────────────────
    # Convertir le DFA en listes (transitions) pour CP-SAT
    # Format : [(from_state, label_value, to_state), ...]
    transitions = []
    for (src, sym), dst in sorted(dfa.delta.items()):
        transitions.append((src, sym_to_int[sym], dst))

    for e in range(E):
        model.AddAutomaton(
            x[e],
            dfa.initial,
            sorted(dfa.accepting),
            transitions,
        )

    # ── couverture : y[t] = Σ_e w[e,t] ─────────────────────────────────────
    # y[t] ≥ d[t,k] * z[k]  ∀t,k
    # Équivalent : pour chaque k, z[k]=1 → y[t] ≥ d[t,k] pour tout t
    # On utilise : Σ_e w[e,t] ≥ d[t,k] * z[k]
    # d[t,k] * z[k] est linéaire (d[t,k] est constant)
    # → model.Add(Σ_e w[e,t] ≥ d[t,k]).OnlyEnforceIf(z[k])

    for k in range(N):
        for t in range(T):
            d_tk = int(demand[t, k])
            if d_tk > 0:
                model.Add(
                    sum(w[e][t] for e in range(E)) >= d_tk
                ).OnlyEnforceIf(z[k])

    # ── chance constraint : Σ_k z[k] ≥ quantile ─────────────────────────────
    model.Add(sum(z) >= quantile)

    # ── objectif : minimiser Σ_e Σ_t s[t] * w[e,t] ──────────────────────────
    obj_terms = []
    for e in range(E):
        for t in range(T):
            if s_int[t] > 0:
                obj_terms.append(s_int[t] * w[e][t])
    model.Minimize(sum(obj_terms))

    # ── résolution ─────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = (verbosity > 0)

    t0 = time.perf_counter()
    status_code = solver.Solve(model)
    elapsed = time.perf_counter() - t0

    status_map = {
        cp_model.OPTIMAL:    "OPTIMAL",
        cp_model.FEASIBLE:   "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN:    "UNKNOWN",
    }
    status = status_map.get(status_code, "UNKNOWN")

    if status in ("OPTIMAL", "FEASIBLE"):
        obj = solver.ObjectiveValue() / SCALE
        lb  = solver.BestObjectiveBound() / SCALE

        # Reconstruire le planning
        schedule = np.empty((E, T), dtype='U1')
        for e in range(E):
            for t in range(T):
                schedule[e, t] = int_to_sym[solver.Value(x[e][t])]

        z_cov = np.array([bool(solver.Value(z[k])) for k in range(N)])
    else:
        obj = lb = None
        schedule = z_cov = None

    # Borne triviale (relaxation : chaque employé peut travailler toutes les périodes)
    # Lower bound triviale = Σ_t min_coverage[t] * s[t]
    # où min_coverage[t] = max_k d[t,k]*z[k] dans la solution optimale, au minimum.
    # Plus simple : Σ_t quantile_{1-ε}(d[t,:]) * s[t] / E  (par employé)
    # On utilise : Σ_t max_t_k(d[t,k]) * s[t] * quantile/N ... trop complexe.
    # Borne triviale conservatrice : 0 (objectif ≥ 0 toujours).
    trivial_lb = 0.0

    return SolveResult(
        status=status,
        objective=obj,
        schedule=schedule,
        z_coverage=z_cov,
        solve_time=elapsed,
        lower_bound=lb,
        trivial_lower_bound=trivial_lb,
    )


# ── validation out-of-sample ──────────────────────────────────────────────────

def validate_oos(
    schedule: np.ndarray,      # (E, T) str
    demand_oos: np.ndarray,    # (T, K') int32
) -> dict:
    """
    Valide une solution (planning) sur les scénarios out-of-sample.

    Retourne
    --------
    dict avec : violation_rate (fraction de scénarios non couverts),
                n_violated, K_oos
    """
    E, T = schedule.shape
    T2, K_oos = demand_oos.shape
    assert T == T2

    # y[t] = nombre d'employés travaillant en t
    y = np.array([
        sum(1 for e in range(E) if schedule[e, t] == 'W')
        for t in range(T)
    ], dtype=int)   # (T,)

    # Scénario k est couvert ssi y[t] ≥ d[t,k] pour tout t
    covered = np.all(y[:, None] >= demand_oos, axis=0)   # (K_oos,)
    n_violated = int(K_oos - covered.sum())
    violation_rate = n_violated / K_oos

    return {
        "violation_rate": violation_rate,
        "n_violated": n_violated,
        "K_oos": K_oos,
        "n_covered": int(covered.sum()),
    }
