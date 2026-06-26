"""
cost.py — Module B : coûts salariaux par période.

s[t] = b0 + b_soir * 1[t ∈ fenêtre_soir] + b_nuit * 1[t ∈ fenêtre_nuit]

κ(σ, t) = WORK_INDICATOR(σ) * s[t]
  avec WORK_INDICATOR(W)=1, WORK_INDICATOR(P)=WORK_INDICATOR(R)=WORK_INDICATOR(O)=0

Le coût d'un quart x[e,:] est Σ_t κ(x[e,t], t) = Σ_{t: x[e,t]=W} s[t].
"""

from __future__ import annotations
import numpy as np

ALPHABET = ('O', 'P', 'R', 'W')
WORK_INDICATOR: dict[str, int] = {'W': 1, 'P': 0, 'R': 0, 'O': 0}

# Fenêtres par défaut : indices [début, fin) dans {0..T-1} (fraction de T)
_SOIR_START = 0.625   # ~ 15h sur 24h
_SOIR_END   = 0.833   # ~ 20h
_NUIT_START = 0.875   # ~ 21h
_NUIT_END   = 1.0     # fin de la journée


def generate_cost(
    T: int,
    rng: np.random.Generator,
    b0_range: tuple[float, float] = (1.0, 2.0),
    b_soir_range: tuple[float, float] = (0.2, 0.8),
    b_nuit_range: tuple[float, float] = (0.5, 1.5),
    soir_window: tuple[float, float] = (_SOIR_START, _SOIR_END),
    nuit_window: tuple[float, float] = (_NUIT_START, _NUIT_END),
) -> tuple[np.ndarray, dict]:
    """
    Tire un vecteur de coûts salariaux pour T périodes.

    Retourne
    --------
    s      : (T,) float, s[t] ≥ 0 pour tout t
    params : dict avec les valeurs tirées (b0, b_soir, b_nuit, fenêtres)
    """
    b0     = float(rng.uniform(*b0_range))
    b_soir = float(rng.uniform(*b_soir_range))
    b_nuit = float(rng.uniform(*b_nuit_range))

    s = np.full(T, b0, dtype=float)

    # Fenêtre soir (indices en fraction de T)
    soir_lo = int(T * soir_window[0])
    soir_hi = int(T * soir_window[1])
    if soir_lo < soir_hi:
        s[soir_lo:soir_hi] += b_soir

    # Fenêtre nuit
    nuit_lo = int(T * nuit_window[0])
    nuit_hi = min(T, int(T * nuit_window[1]))
    if nuit_lo < nuit_hi:
        s[nuit_lo:nuit_hi] += b_nuit

    params = {
        "b0": b0,
        "b_soir": b_soir,
        "b_nuit": b_nuit,
        "soir_window_lo": soir_lo,
        "soir_window_hi": soir_hi,
        "nuit_window_lo": nuit_lo,
        "nuit_window_hi": nuit_hi,
    }
    return s, params


def shift_cost(word: str, s: np.ndarray) -> float:
    """Coût d'un quart (mot de longueur T sur l'alphabet OPWR)."""
    assert len(word) == len(s), "Longueur du mot ≠ longueur du vecteur de coûts"
    return sum(s[t] * WORK_INDICATOR[sym] for t, sym in enumerate(word))
