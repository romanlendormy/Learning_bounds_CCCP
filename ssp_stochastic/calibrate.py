"""
calibrate.py — Calibration du nombre d'employés E.

Objectif : trouver E tel que le problème soit ni trivial (E >> demande) ni
infaisable, selon un ratio de tightness cible autour de 1.0–1.2.

Stratégie
---------
1. Estimer la demande-pic au quantile (1-ε) : Q = quantile_{1-ε}(max_t d[t,:]).
2. E = ceil(Q * tightness_factor).
3. Vérifier (borne rapide) qu'il existe au moins un quart légal pour ce T.
   La vérification complète (CP-SAT) est optionnelle et coûteuse — on ne la
   fait que sur de petites instances ou si explicitement demandé.
4. Le ratio de tightness réalisé est Q / E.

Sortie : E (int), meta (dict avec ratio_tightness, E, Q_peak, statut).
"""

from __future__ import annotations
import math
import numpy as np
from .automaton import DFA, is_nonempty_at_length


def calibrate_E(
    demand: np.ndarray,                 # (T, N) int32
    epsilon: float,
    tightness_factor: float,
    dfa: DFA,
    T: int,
    min_E: int = 1,
    max_E: int = 50,
) -> tuple[int, dict]:
    """
    Calibre E à partir de la demande peak au quantile (1-ε).

    Paramètres
    ----------
    demand           : scénarios (T, N)
    epsilon          : niveau de risque (e.g. 0.1 → quantile 90%)
    tightness_factor : multiplicateur sur Q_peak (autour de 1.0–1.2)
    dfa              : automate des quarts légaux (pour vérifier la faisabilité)
    T                : horizon
    min_E, max_E     : bornes sur E

    Retourne
    --------
    E    : int
    meta : dict
    """
    T_dim, N = demand.shape
    assert T_dim == T

    # Quantile de la demande max sur toutes les périodes
    quantile_level = 1.0 - epsilon
    # Pour chaque période, quantile au niveau (1-ε) sur les N scénarios
    d_quantile = np.quantile(demand, quantile_level, axis=1)   # (T,)
    Q_peak = float(d_quantile.max())

    # E cible
    E_raw = Q_peak * tightness_factor
    E = int(math.ceil(E_raw))
    E = max(min_E, min(max_E, E))

    # Vérification rapide : le DFA a-t-il des mots acceptés de longueur T ?
    dfa_ok = is_nonempty_at_length(dfa, T)

    # Ratio de tightness réalisé
    ratio = Q_peak / E if E > 0 else float('inf')

    meta = {
        "E": E,
        "Q_peak": float(Q_peak),
        "tightness_factor": tightness_factor,
        "ratio_tightness": float(ratio),
        "epsilon_used": epsilon,
        "quantile_level": quantile_level,
        "dfa_nonempty": dfa_ok,
        "feasibility_status": "dfa_ok" if dfa_ok else "dfa_empty",
    }
    return E, meta


def adjust_E_for_feasibility(
    E: int,
    demand: np.ndarray,
    epsilon: float,
    dfa: DFA,
    T: int,
    max_E: int = 50,
) -> tuple[int, str]:
    """
    Si E est trop faible pour couvrir la chance constraint (borne triviale :
    E * 1 ≥ max_t quantile_{1-ε}(d[t,:])), augmenter jusqu'à faisabilité.

    Cette borne triviale suppose que chaque employé peut travailler toutes les
    périodes — elle est donc optimiste. Elle détecte seulement les cas
    clairement infaisables.

    Retourne (E_adjusted, status_message).
    """
    d_quantile = np.quantile(demand, 1.0 - epsilon, axis=1)
    Q_peak = float(d_quantile.max())

    # Borne inférieure triviale : E ≥ Q_peak
    E_lb = int(math.ceil(Q_peak))
    if E < E_lb:
        E = min(E_lb, max_E)
        return E, f"E ajusté à {E} (borne triviale ≥ Q_peak={Q_peak:.1f})"
    return E, "ok"
